from __future__ import annotations

import copy
from collections import Counter, defaultdict
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import shutil
import sys
import threading
import time
import textwrap

import numpy as np

from tracking_pipeline.application.factories import (
    build_accumulator,
    build_artifact_writer,
    build_classifier,
    build_clusterer,
    build_lane_box,
    build_reader,
    build_track_postprocessors,
    build_tracker,
)
from tracking_pipeline.application.classification import classify_aggregate_results
from tracking_pipeline.application.class_normalization import ClassNormalizer
from tracking_pipeline.application.class_statistics import build_class_statistics
from tracking_pipeline.application.gt_matching import apply_gt_matches_to_results, match_saved_aggregates_to_gt
from tracking_pipeline.application.performance import (
    AGGREGATION_COMPONENT_NAMES,
    PerformanceProfiler,
    build_component_snapshot,
    derive_hz,
)
from tracking_pipeline.application.track_outcomes import build_track_outcomes
from tracking_pipeline.config.models import PipelineConfig, RuntimeConfig
from tracking_pipeline.domain.models import AggregateResult, ObjectLabelData, RunPerformance, RunSummary, Track, TrackOutcomeDebug
from tracking_pipeline.infrastructure.logging.run_logger import get_run_logger
from tracking_pipeline.infrastructure.visualization.live_frame_publisher import LiveFramePublisher
from tracking_pipeline.infrastructure.visualization.live_pcd_web_server import LivePCDWebServer

LIVE_ARTIFACT_FLUSH_INTERVAL_SEC = 2.0
LIVE_ARTIFACT_TRACKER_DEBUG_FRAME_COUNT = 1


class _LiveCliStatusWriter:
    def __init__(self, stream=None):
        self.stream = sys.stderr if stream is None else stream
        self._lock = threading.Lock()
        self._active = False
        self._last_line_count = 0

    @property
    def enabled(self) -> bool:
        isatty = getattr(self.stream, "isatty", None)
        return bool(callable(isatty) and isatty())

    def _terminal_width(self) -> int:
        fileno = getattr(self.stream, "fileno", None)
        if callable(fileno):
            try:
                return int(os.get_terminal_size(fileno()).columns)
            except Exception:
                pass
        try:
            return int(shutil.get_terminal_size(fallback=(120, 20)).columns)
        except Exception:
            return 120

    def _render_lines(self, text: str) -> list[str]:
        line = str(text).replace("\n", " ").strip()
        width = max(20, self._terminal_width())
        wrapped = textwrap.wrap(
            line,
            width=width,
            break_long_words=True,
            break_on_hyphens=False,
            drop_whitespace=True,
        )
        return wrapped or [""]

    def update(self, text: str) -> bool:
        if not self.enabled:
            return False
        lines = self._render_lines(text)
        with self._lock:
            if self._active:
                self.stream.write("\r")
                for _ in range(max(0, self._last_line_count - 1)):
                    self.stream.write("\033[2K\033[1A\r")
                self.stream.write("\033[2K\r")
            self.stream.write("\n".join(lines))
            self.stream.flush()
            self._active = True
            self._last_line_count = len(lines)
        return True

    def finish(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            if not self._active:
                return
            self.stream.write("\n")
            self.stream.flush()
            self._active = False
            self._last_line_count = 0


def _apply_runtime_limits(runtime: RuntimeConfig, logger) -> dict[str, object]:
    requested_cpu_cores = int(runtime.cpu_cores)
    if requested_cpu_cores <= 0:
        return {
            "requested_cpu_cores": 0,
            "applied_cpu_cores": 0,
            "affinity_applied": False,
            "affinity_cpus": [],
        }

    applied_cpu_cores = requested_cpu_cores
    affinity_cpus: list[int] = []
    affinity_applied = False
    try:
        if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
            available = sorted(int(cpu_id) for cpu_id in os.sched_getaffinity(0))
            if available:
                affinity_cpus = available[: min(requested_cpu_cores, len(available))]
                os.sched_setaffinity(0, set(affinity_cpus))
                applied_cpu_cores = len(affinity_cpus)
                affinity_applied = True
    except Exception as exc:
        logger.info("Runtime CPU affinity limit could not be applied: %s", exc)

    for env_name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[env_name] = str(max(1, int(applied_cpu_cores)))

    try:
        import torch
    except Exception:
        torch = None
    if torch is not None:
        try:
            torch.set_num_threads(max(1, int(applied_cpu_cores)))
        except Exception:
            pass
        try:
            torch.set_num_interop_threads(max(1, min(int(applied_cpu_cores), 4)))
        except Exception:
            pass

    logger.info(
        "Runtime CPU limit: requested=%s applied=%s affinity=%s",
        requested_cpu_cores,
        applied_cpu_cores,
        "yes" if affinity_applied else "no",
    )
    return {
        "requested_cpu_cores": int(requested_cpu_cores),
        "applied_cpu_cores": int(applied_cpu_cores),
        "affinity_applied": bool(affinity_applied),
        "affinity_cpus": [int(cpu_id) for cpu_id in affinity_cpus],
    }


def run_pipeline(config: PipelineConfig, project_root: Path) -> RunSummary:
    profiler = PerformanceProfiler()
    class_normalizer = ClassNormalizer.from_config(config.class_normalization)
    logger = get_run_logger()
    runtime_limits = _apply_runtime_limits(config.runtime, logger)
    with profiler.stage("build_components"):
        lane_box = build_lane_box(config)
        reader = build_reader(config)
        clusterer = build_clusterer(config)
        tracker = build_tracker(config)
        postprocessors = build_track_postprocessors(config)
        accumulator = build_accumulator(config)
        classifier = build_classifier(config)
        writer = build_artifact_writer(config, project_root)

    with profiler.stage("prepare_output"):
        run_dir = writer.prepare_run_dir(config)
        writer.write_config_snapshot(run_dir, config)

    live_web_runtime = _start_live_web_viewer(
        config=config,
        writer=writer,
        run_dir=run_dir,
        lane_box=lane_box,
        reader=reader,
        logger=logger,
    )
    live_status_reporter = _start_live_status_reporter(
        reader,
        writer,
        run_dir,
        config.input.paths[0],
        logger,
        config.input.format,
        runtime_limits,
    )
    live_artifact_state = _build_live_artifact_state(config.input.format)
    try:
        latest_object_labels: dict[int, ObjectLabelData] = {}
        object_label_history_by_id: dict[int, list[ObjectLabelData]] = defaultdict(list)
        object_list_seen_ids: set[int] = set()
        object_list_skipped_empty = 0
        live_web_track_outcomes: dict[int, TrackOutcomeDebug] = {}
        live_web_announced_finished_track_ids: set[int] = set()
        tracker_states = []
        frame_count = 0
        last_processed_frame_index = -1
        last_processed_frame_timestamp_ns = -1
        if config.input.format == "qb2_live":
            with profiler.stage("write_object_list"):
                _write_live_object_list_snapshot(writer, run_dir, latest_object_labels, live_status_reporter)
        frame_iterator = iter(reader.iter_frames(config.input.paths))
        interrupted = False
        try:
            with _sigterm_as_keyboard_interrupt():
                while True:
                    try:
                        with profiler.stage("read_frames"):
                            frame = next(frame_iterator)
                    except StopIteration:
                        break
                    skipped_empty, object_list_updated = _ingest_object_labels(
                        frame.object_labels,
                        latest_object_labels,
                        object_label_history_by_id,
                        object_list_seen_ids,
                        class_normalizer,
                    )
                    object_list_skipped_empty += skipped_empty
                    if object_list_updated and config.input.format == "qb2_live":
                        with profiler.stage("write_object_list"):
                            _write_live_object_list_snapshot(writer, run_dir, latest_object_labels, live_status_reporter)
                    with profiler.stage("cluster_frames"):
                        cluster_result = clusterer.cluster(frame, lane_box)
                    with profiler.stage("tracker_steps"):
                        state = tracker.step(cluster_result.detections, frame.frame_index, frame.timestamp_ns)
                    state.cluster_metrics = cluster_result.metrics
                    tracker_states.append(state)
                    frame_count += 1
                    last_processed_frame_index = int(frame.frame_index)
                    last_processed_frame_timestamp_ns = int(frame.timestamp_ns)
                    _update_live_status_reporter(
                        live_status_reporter,
                        pipeline_phase="processing_frames",
                        processed_frames=int(frame_count),
                        last_processed_frame_index=int(last_processed_frame_index),
                        last_frame_timestamp_ns=int(frame.timestamp_ns),
                        object_list_exported_count=int(len(latest_object_labels)),
                        object_list_seen_ids=int(len(object_list_seen_ids)),
                        active_track_count=int(len(state.active_tracks)),
                    )
                    _update_live_web_status(
                        live_web_runtime,
                        pipeline_phase="processing_frames",
                        processed_frames=int(frame_count),
                        last_processed_frame_index=int(last_processed_frame_index),
                        last_frame_timestamp_ns=int(frame.timestamp_ns),
                        object_list_exported_count=int(len(latest_object_labels)),
                        object_list_seen_ids=int(len(object_list_seen_ids)),
                        active_track_count=int(len(state.active_tracks)),
                    )
                    _publish_live_web_frame(live_web_runtime, frame, cluster_result, state)
                    _maybe_update_live_web_finished_track_outcomes(
                        runtime=live_web_runtime,
                        tracker=tracker,
                        lane_box=lane_box,
                        accumulator=accumulator,
                        classifier=classifier,
                        class_normalizer=class_normalizer,
                        tracker_states=tracker_states,
                        live_track_outcomes=live_web_track_outcomes,
                        announced_finished_track_ids=live_web_announced_finished_track_ids,
                        logger=logger,
                    )
                    _maybe_write_live_artifact_snapshot(
                        config=config,
                        profiler=profiler,
                        writer=writer,
                        run_dir=run_dir,
                        lane_box=lane_box,
                        tracker=tracker,
                        postprocessors=postprocessors,
                        accumulator=accumulator,
                        classifier=classifier,
                        class_normalizer=class_normalizer,
                        latest_object_labels=latest_object_labels,
                        object_label_history_by_id=object_label_history_by_id,
                        object_list_seen_ids=object_list_seen_ids,
                        object_list_skipped_empty=object_list_skipped_empty,
                        tracker_states=tracker_states,
                        frame_count=frame_count,
                        live_status_reporter=live_status_reporter,
                        live_web_runtime=live_web_runtime,
                        live_artifact_state=live_artifact_state,
                        force=False,
                        save_aggregate_intensity=config.output.save_aggregate_intensity,
                    )
        except KeyboardInterrupt:
            interrupted = True
            _update_live_status_reporter(
                live_status_reporter,
                pipeline_phase="interrupted",
                interrupted=True,
                processed_frames=int(frame_count),
            )
            _update_live_web_status(
                live_web_runtime,
                pipeline_phase="interrupted",
                interrupted=True,
                processed_frames=int(frame_count),
            )
        finally:
            _close_frame_iterator(frame_iterator)
            _close_reader(reader)

        if frame_count > 0:
            skipped_empty, object_list_updated = _ingest_object_labels(
                _drain_pending_object_labels(
                    reader,
                    last_processed_frame_index,
                    max_timestamp_ns=last_processed_frame_timestamp_ns,
                ),
                latest_object_labels,
                object_label_history_by_id,
                object_list_seen_ids,
                class_normalizer,
            )
            object_list_skipped_empty += skipped_empty
            if object_list_updated and config.input.format == "qb2_live":
                with profiler.stage("write_object_list"):
                    _write_live_object_list_snapshot(writer, run_dir, latest_object_labels, live_status_reporter)
            _update_live_status_reporter(
                live_status_reporter,
                object_list_exported_count=int(len(latest_object_labels)),
                object_list_seen_ids=int(len(object_list_seen_ids)),
            )
        elif interrupted:
            _update_live_status_reporter(live_status_reporter, pipeline_phase="stopped_without_frames", interrupted=True)
            _update_live_web_status(live_web_runtime, pipeline_phase="stopped_without_frames", interrupted=True)
            raise RuntimeError("Run interrupted before any frames were received")
        elif config.input.format == "qb2_live":
            _update_live_status_reporter(live_status_reporter, pipeline_phase="ended_without_frames")
            _update_live_web_status(live_web_runtime, pipeline_phase="ended_without_frames")
            raise RuntimeError("No QB2 frames were received before the live run ended")

        _update_live_status_reporter(
            live_status_reporter,
            pipeline_phase="finalizing",
            processed_frames=int(frame_count),
            object_list_exported_count=int(len(latest_object_labels)),
            object_list_seen_ids=int(len(object_list_seen_ids)),
        )
        _update_live_web_status(
            live_web_runtime,
            pipeline_phase="finalizing",
            processed_frames=int(frame_count),
            object_list_exported_count=int(len(latest_object_labels)),
            object_list_seen_ids=int(len(object_list_seen_ids)),
        )
        _maybe_write_live_artifact_snapshot(
            config=config,
            profiler=profiler,
            writer=writer,
            run_dir=run_dir,
            lane_box=lane_box,
            tracker=tracker,
            postprocessors=postprocessors,
            accumulator=accumulator,
            classifier=classifier,
            class_normalizer=class_normalizer,
            latest_object_labels=latest_object_labels,
            object_label_history_by_id=object_label_history_by_id,
            object_list_seen_ids=object_list_seen_ids,
            object_list_skipped_empty=object_list_skipped_empty,
            tracker_states=tracker_states,
            frame_count=frame_count,
            live_status_reporter=live_status_reporter,
            live_web_runtime=live_web_runtime,
            live_artifact_state=live_artifact_state,
            force=True,
            save_aggregate_intensity=config.output.save_aggregate_intensity,
        )

        with profiler.stage("tracker_finalize"):
            tracks = tracker.finalize()
        for processor in postprocessors:
            with profiler.stage("postprocess_tracks"):
                tracks = processor.process(tracks)

        aggregate_results: list[AggregateResult] = []
        registration_attempts = 0
        registration_accepted = 0
        registration_rejected = 0
        aggregation_component_wall = {component_name: 0.0 for component_name in AGGREGATION_COMPONENT_NAMES}
        aggregation_component_cpu = {component_name: 0.0 for component_name in AGGREGATION_COMPONENT_NAMES}
        aggregation_component_calls = {component_name: 0 for component_name in AGGREGATION_COMPONENT_NAMES}

        for track in tracks.values():
            with profiler.stage("accumulate_tracks"):
                result = accumulator.accumulate(track, lane_box)
            aggregate_results.append(result)
            metrics = result.metrics
            registration_attempts += int(metrics.get("registration_pairs", 0))
            registration_accepted += int(metrics.get("registration_accepted", 0))
            registration_rejected += int(metrics.get("registration_rejected", 0))
            _accumulate_aggregation_component_metrics(
                aggregation_component_wall,
                aggregation_component_cpu,
                aggregation_component_calls,
                result,
                config.aggregation.algorithm,
                config.aggregation.enable_tail_bridge,
            )
        if hasattr(accumulator, "merge_long_vehicle_aggregates"):
            with profiler.stage("accumulate_tracks"):
                aggregate_results = accumulator.merge_long_vehicle_aggregates(tracks, aggregate_results, lane_box)
        with profiler.stage("classify_aggregates"):
            aggregate_results = classify_aggregate_results(aggregate_results, classifier, class_normalizer)
        with profiler.stage("match_gt"):
            matched_gt, unmatched_saved_tracks, unmatched_gt_objects, gt_match_summary = match_saved_aggregates_to_gt(
                tracks,
                aggregate_results,
                dict(object_label_history_by_id),
                class_normalizer,
            )
            apply_gt_matches_to_results(aggregate_results, matched_gt, unmatched_saved_tracks)
        class_stats = build_class_statistics(aggregate_results, latest_object_labels, class_normalizer)
        _begin_writer_snapshot(writer, run_dir)
        for result in aggregate_results:
            if result.status == "saved":
                with profiler.stage("write_aggregates"):
                    writer.write_aggregate(run_dir, result, save_intensity=config.output.save_aggregate_intensity)
        track_outcomes = build_track_outcomes(tracks, aggregate_results, tracker_states)

        with profiler.stage("write_object_list"):
            writer.write_object_list(run_dir, latest_object_labels)
        with profiler.stage("write_gt_matching"):
            writer.write_gt_matching(run_dir, matched_gt, unmatched_saved_tracks, unmatched_gt_objects, gt_match_summary)

        status_counts = Counter(result.status for result in aggregate_results)
        quality_scores = [track.quality_score for track in tracks.values() if track.quality_score is not None]
        summary = RunSummary(
            input_path=config.input.paths[0],
            input_paths=list(config.input.paths),
            output_mode=str(config.output.mode),
            tracker_algorithm=config.tracking.algorithm,
            accumulator_algorithm=config.aggregation.algorithm,
            clusterer_algorithm=config.clustering.algorithm,
            frame_count=frame_count,
            finished_track_count=len(tracks),
            saved_aggregates=sum(1 for result in aggregate_results if result.status == "saved"),
            registration_attempts=registration_attempts,
            registration_accepted=registration_accepted,
            registration_rejected=registration_rejected,
            output_dir=str(run_dir),
            postprocessing_methods=[processor.name for processor in postprocessors],
            aggregate_status_counts=dict(status_counts),
            track_quality_mean=float(sum(quality_scores) / len(quality_scores)) if quality_scores else 0.0,
            object_list_exported_count=len(latest_object_labels),
            object_list_seen_ids=len(object_list_seen_ids),
            object_list_skipped_empty=int(object_list_skipped_empty),
            gt_match_saved_track_count=int(gt_match_summary["gt_match_saved_track_count"]),
            gt_match_matched_count=int(gt_match_summary["gt_match_matched_count"]),
            gt_match_unmatched_saved_count=int(gt_match_summary["gt_match_unmatched_saved_count"]),
            gt_match_unmatched_gt_count=int(gt_match_summary["gt_match_unmatched_gt_count"]),
            gt_match_mode=str(gt_match_summary["gt_match_mode"]),
            gt_match_assignment=str(gt_match_summary["gt_match_assignment"]),
            gt_match_mean_timestamp_delta_ns=float(gt_match_summary["gt_match_mean_timestamp_delta_ns"]),
            gt_match_max_timestamp_delta_ns=int(gt_match_summary["gt_match_max_timestamp_delta_ns"]),
            predicted_class_counts=dict(class_stats["predicted_class_counts"]),
            gt_class_counts=dict(class_stats["gt_class_counts"]),
            matched_gt_class_counts=dict(class_stats["matched_gt_class_counts"]),
            class_comparison_count=int(class_stats["class_comparison_count"]),
            class_match_count=int(class_stats["class_match_count"]),
            class_mismatch_count=int(class_stats["class_mismatch_count"]),
            class_count_rows=[dict(row) for row in class_stats["class_count_rows"]],
            performance=_snapshot_with_aggregation_components(
                profiler,
                aggregation_component_wall,
                aggregation_component_cpu,
                aggregation_component_calls,
                frame_count,
            ),
        )
        with profiler.stage("write_tracks"):
            writer.write_tracks(run_dir, tracks, aggregate_results)
            writer.write_tracker_debug(run_dir, tracker_states)
            writer.write_track_outcomes(run_dir, track_outcomes)
            writer.write_class_stats(run_dir, class_stats)
        _update_live_web_snapshot(live_web_runtime, track_outcomes, summary)
        summary.performance = _snapshot_with_aggregation_components(
            profiler,
            aggregation_component_wall,
            aggregation_component_cpu,
            aggregation_component_calls,
            summary.frame_count,
        )
        with profiler.stage("write_summary"):
            writer.write_summary(run_dir, summary)
        # Persist the final profile snapshot after measuring the summary write itself.
        summary.performance = _snapshot_with_aggregation_components(
            profiler,
            aggregation_component_wall,
            aggregation_component_cpu,
            aggregation_component_calls,
            summary.frame_count,
        )
        writer.write_summary(run_dir, summary)
        _update_live_status_reporter(
            live_status_reporter,
            pipeline_phase="completed",
            processed_frames=int(summary.frame_count),
            finished_track_count=int(summary.finished_track_count),
            saved_aggregates=int(summary.saved_aggregates),
            object_list_exported_count=int(summary.object_list_exported_count),
            object_list_seen_ids=int(summary.object_list_seen_ids),
            active_track_count=0,
        )
        _update_live_web_status(
            live_web_runtime,
            pipeline_phase="completed",
            processed_frames=int(summary.frame_count),
            finished_track_count=int(summary.finished_track_count),
            saved_aggregates=int(summary.saved_aggregates),
            object_list_exported_count=int(summary.object_list_exported_count),
            object_list_seen_ids=int(summary.object_list_seen_ids),
            active_track_count=0,
        )
        return summary
    finally:
        _stop_live_web_viewer(live_web_runtime)
        _stop_live_status_reporter(live_status_reporter, reader)


def _start_live_status_reporter(reader, writer, run_dir: Path, input_path: str, logger, input_format: str, runtime_limits: dict[str, object]):
    if str(input_format) != "qb2_live":
        return None
    started_monotonic = time.monotonic()
    status_path = _writer_output_path(writer, "live_status_path", run_dir, run_dir / "live_status.json")
    object_list_manifest_path = _writer_output_path(writer, "object_list_manifest_path", run_dir, run_dir / "object_list" / "manifest.jsonl")
    live_artifact_dir = _writer_output_path(writer, "live_artifact_dir", run_dir, run_dir)
    reporter = {
        "run_dir": run_dir,
        "status_path": status_path,
        "writer": writer,
        "cli_status": _LiveCliStatusWriter(),
        "state": {
            "input_path": str(input_path),
            "pipeline_phase": "waiting_for_frames",
            "processed_frames": 0,
            "last_processed_frame_index": -1,
            "last_frame_timestamp_ns": None,
            "processing_total_hz": 0.0,
            "processing_recent_hz": 0.0,
            "live_artifact_dir": str(live_artifact_dir),
            "live_artifact_flush_interval_sec": float(LIVE_ARTIFACT_FLUSH_INTERVAL_SEC),
            "live_artifact_write_count": 0,
            "last_live_artifact_write_unix_sec": None,
            "object_list_exported_count": 0,
            "object_list_seen_ids": 0,
            "object_list_manifest_path": str(object_list_manifest_path),
            "live_object_list_write_count": 0,
            "last_live_object_list_write_unix_sec": None,
            "runtime_requested_cpu_cores": int(runtime_limits.get("requested_cpu_cores", 0) or 0),
            "runtime_applied_cpu_cores": int(runtime_limits.get("applied_cpu_cores", 0) or 0),
            "runtime_affinity_applied": bool(runtime_limits.get("affinity_applied", False)),
            "runtime_affinity_cpus": [int(cpu_id) for cpu_id in runtime_limits.get("affinity_cpus", [])],
            "active_track_count": 0,
            "finished_track_count": 0,
            "saved_aggregates": 0,
            "interrupted": False,
            "_started_monotonic": float(started_monotonic),
            "_last_processed_monotonic": None,
            "_last_processed_frame_count": 0,
        },
        "lock": threading.Lock(),
        "stop_event": threading.Event(),
        "thread": None,
    }
    payload = _build_live_status_payload(reader, reporter)
    _persist_live_status_payload(reporter, payload)
    logger.info("Live run active: %s", run_dir)
    logger.info("Live status file: %s", reporter["status_path"])
    logger.info("Live artifact snapshots: %s", run_dir)
    logger.info("Waiting for QB2 raw frames; press Ctrl+C to finalize the current run")
    cli_status = reporter["cli_status"]
    if not cli_status.update(_format_live_status_line(payload)):
        logger.info("%s", _format_live_status_line(payload))
    thread = threading.Thread(
        target=_live_status_reporter_main,
        args=(reader, reporter, logger),
        name="tracking_pipeline_live_status",
        daemon=True,
    )
    reporter["thread"] = thread
    thread.start()
    return reporter


def _start_live_web_viewer(*, config: PipelineConfig, writer, run_dir: Path, lane_box, reader, logger):
    if str(config.input.format) != "qb2_live":
        return None
    if not bool(config.visualization.live_web_enabled):
        return None
    publisher = LiveFramePublisher(
        lane_box=lane_box,
        track_exit_line_axis=config.aggregation.frame_selection_line_axis,
        track_exit_edge_margin=config.output.track_exit_edge_margin,
        max_points=config.visualization.max_points,
        history_sec=config.visualization.live_web_history_sec,
        retain_all_frames=config.visualization.live_web_retain_all_frames,
        point_source=config.visualization.live_web_point_source,
        color_by_intensity=config.visualization.color_by_intensity,
        show_tracker_debug=config.visualization.show_tracker_debug,
        show_track_outcomes=config.visualization.show_track_outcome_debug,
        run_label=_resolve_live_run_label(writer, run_dir),
        reader_status_provider=lambda: _read_reader_status(reader),
    )
    server = LivePCDWebServer(
        publisher,
        host=config.visualization.live_web_host,
        port=config.visualization.live_web_port,
    )
    server.start()
    publisher.update_status(
        pipeline_phase="waiting_for_frames",
        output_dir=str(run_dir),
        live_web_host=str(config.visualization.live_web_host),
        live_web_port=int(server.port),
    )
    logger.info(
        "Live PCD web viewer listening on http://%s:%s",
        config.visualization.live_web_host,
        server.port,
    )
    return {"publisher": publisher, "server": server}


def _update_live_web_status(runtime, **updates: object) -> None:
    if runtime is None:
        return
    publisher = runtime.get("publisher")
    if isinstance(publisher, LiveFramePublisher):
        publisher.update_status(**updates)


def _publish_live_web_frame(runtime, frame, cluster_result, tracking_state) -> None:
    if runtime is None:
        return
    publisher = runtime.get("publisher")
    if isinstance(publisher, LiveFramePublisher):
        publisher.publish_frame(frame, cluster_result, tracking_state)


def _update_live_web_snapshot(runtime, track_outcomes, summary: RunSummary) -> None:
    if runtime is None:
        return
    publisher = runtime.get("publisher")
    if not isinstance(publisher, LiveFramePublisher):
        return
    publisher.update_track_outcomes(track_outcomes)
    publisher.update_summary(summary)
    publisher.update_status(
        finished_track_count=int(summary.finished_track_count),
        saved_aggregates=int(summary.saved_aggregates),
        object_list_exported_count=int(summary.object_list_exported_count),
        object_list_seen_ids=int(summary.object_list_seen_ids),
    )


def _maybe_update_live_web_finished_track_outcomes(
    *,
    runtime,
    tracker,
    lane_box,
    accumulator,
    classifier,
    class_normalizer: ClassNormalizer,
    tracker_states: list,
    live_track_outcomes: dict[int, TrackOutcomeDebug],
    announced_finished_track_ids: set[int],
    logger,
) -> None:
    if runtime is None:
        return
    publisher = runtime.get("publisher")
    if not isinstance(publisher, LiveFramePublisher):
        return
    finished_tracks = getattr(tracker, "finished_tracks", None)
    if not isinstance(finished_tracks, dict) or not finished_tracks:
        return
    new_track_ids = [
        int(track_id)
        for track_id in sorted(int(track_id) for track_id in finished_tracks)
        if int(track_id) not in announced_finished_track_ids
    ]
    if not new_track_ids:
        return

    new_tracks: dict[int, Track] = {}
    for track_id in new_track_ids:
        track = finished_tracks.get(int(track_id))
        if isinstance(track, Track):
            new_tracks[int(track_id)] = _clone_track(track)
    if not new_tracks:
        announced_finished_track_ids.update(new_track_ids)
        return

    try:
        aggregate_results = [accumulator.accumulate(track, lane_box) for track in new_tracks.values()]
        if hasattr(accumulator, "merge_long_vehicle_aggregates"):
            aggregate_results = accumulator.merge_long_vehicle_aggregates(new_tracks, aggregate_results, lane_box)
        aggregate_results = classify_aggregate_results(aggregate_results, classifier, class_normalizer)
        live_track_outcomes.update(build_track_outcomes(new_tracks, aggregate_results, tracker_states))
        announced_finished_track_ids.update(new_tracks)
        publisher.update_track_outcomes(live_track_outcomes)
        publisher.update_status(
            finished_track_count=int(len(live_track_outcomes)),
            saved_aggregates=int(sum(1 for outcome in live_track_outcomes.values() if str(outcome.status) == "saved")),
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.info("Live web track outcome update failed: %s", exc)


def _stop_live_web_viewer(runtime) -> None:
    if runtime is None:
        return
    publisher = runtime.get("publisher")
    if isinstance(publisher, LiveFramePublisher):
        publisher.mark_stopped(pipeline_phase="stopped")
        flush_pending = getattr(publisher, "flush_pending", None)
        if callable(flush_pending):
            flush_pending(timeout=0.5)
    server = runtime.get("server")
    if isinstance(server, LivePCDWebServer):
        server.stop()
    if isinstance(publisher, LiveFramePublisher):
        close = getattr(publisher, "close", None)
        if callable(close):
            close(timeout=2.0)


def _update_live_status_reporter(reporter, **updates: object) -> None:
    if reporter is None:
        return
    with reporter["lock"]:
        state = reporter["state"]
        state.update(updates)
        if "processed_frames" in updates:
            processed_frames = int(state.get("processed_frames", 0) or 0)
            now = time.monotonic()
            started_monotonic = float(state.get("_started_monotonic", now) or now)
            elapsed = max(0.0, now - started_monotonic)
            state["processing_total_hz"] = 0.0 if processed_frames <= 0 or elapsed <= 0.0 else float(processed_frames) / elapsed
            previous_monotonic = state.get("_last_processed_monotonic")
            previous_frame_count = int(state.get("_last_processed_frame_count", 0) or 0)
            if previous_monotonic is not None and processed_frames > previous_frame_count:
                delta = max(0.0, now - float(previous_monotonic))
                frame_delta = processed_frames - previous_frame_count
                state["processing_recent_hz"] = 0.0 if delta <= 0.0 else float(frame_delta) / delta
            elif processed_frames <= 1:
                state["processing_recent_hz"] = 0.0
            state["_last_processed_monotonic"] = float(now)
            state["_last_processed_frame_count"] = int(processed_frames)


def _stop_live_status_reporter(reporter, reader) -> None:
    if reporter is None:
        return
    stop_event = reporter["stop_event"]
    stop_event.set()
    thread = reporter.get("thread")
    if isinstance(thread, threading.Thread):
        thread.join(timeout=2.0)
    payload = _build_live_status_payload(reader, reporter)
    _persist_live_status_payload(reporter, payload)
    reporter["cli_status"].finish()


def _live_status_reporter_main(reader, reporter, logger) -> None:
    stop_event = reporter["stop_event"]
    next_log_at = 0.0
    cli_status = reporter["cli_status"]
    while not stop_event.wait(1.0):
        payload = _build_live_status_payload(reader, reporter)
        _persist_live_status_payload(reporter, payload)
        if cli_status.update(_format_live_status_line(payload)):
            continue
        now = time.monotonic()
        if now >= next_log_at:
            logger.info("%s", _format_live_status_line(payload))
            next_log_at = now + 2.0
    payload = _build_live_status_payload(reader, reporter)
    _persist_live_status_payload(reporter, payload)


def _build_live_status_payload(reader, reporter) -> dict[str, object]:
    with reporter["lock"]:
        state = {
            str(key): value
            for key, value in dict(reporter["state"]).items()
            if not str(key).startswith("_")
        }
    return {
        "updated_at_unix_sec": float(time.time()),
        "output_dir": str(reporter["run_dir"]),
        "status_file": str(reporter["status_path"]),
        **state,
        "reader": _read_reader_status(reader),
    }


def _read_reader_status(reader) -> dict[str, object]:
    status_snapshot = getattr(reader, "status_snapshot", None)
    if not callable(status_snapshot):
        return {}
    try:
        snapshot = status_snapshot()
    except Exception as exc:  # pragma: no cover - defensive
        return {"reader_state": "status_error", "background_error": str(exc)}
    return dict(snapshot)


def _write_live_status_file(path: Path, payload: dict[str, object]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temp_path.replace(path)


def _persist_live_status_payload(reporter, payload: dict[str, object]) -> None:
    _write_live_status_file(reporter["status_path"], payload)
    writer = reporter.get("writer")
    write_live_status = getattr(writer, "write_live_status", None)
    if callable(write_live_status):
        write_live_status(reporter["run_dir"], payload)


def _writer_output_path(writer, method_name: str, run_dir: Path, default: Path) -> Path:
    method = getattr(writer, method_name, None)
    if not callable(method):
        return default
    value = method(run_dir)
    return default if value is None else Path(value)


def _resolve_live_run_label(writer, run_dir: Path) -> str:
    status_path = _writer_output_path(writer, "live_status_path", run_dir, run_dir / "live_status.json")
    parent_name = str(status_path.parent.name or "").strip()
    if parent_name and parent_name != "_active":
        return parent_name
    run_name = str(run_dir.name or "").strip()
    return run_name or "live"


def _format_live_status_line(payload: dict[str, object]) -> str:
    reader = dict(payload.get("reader") or {})
    return (
        "live phase={phase} f={processed} hz={recent_hz:.2f}/{total_hz:.2f} tr={active_tracks} "
        "aw={artifact_writes} ow={object_writes} raw={raw} mqtt={mqtt_msgs} snap={mqtt_snapshots} "
        "pend={pending_labels} conn={mqtt_connected} wait={waiting_first_raw} "
        "raw_age={last_raw_age} mqtt_age={last_mqtt_age} state={reader_state}"
    ).format(
        phase=str(payload.get("pipeline_phase", "unknown")),
        processed=int(payload.get("processed_frames", 0) or 0),
        recent_hz=float(payload.get("processing_recent_hz", 0.0) or 0.0),
        total_hz=float(payload.get("processing_total_hz", 0.0) or 0.0),
        active_tracks=int(payload.get("active_track_count", 0) or 0),
        artifact_writes=int(payload.get("live_artifact_write_count", 0) or 0),
        object_writes=int(payload.get("live_object_list_write_count", 0) or 0),
        raw=int(reader.get("raw_frames_received", 0) or 0),
        mqtt_msgs=int(reader.get("mqtt_messages_received", 0) or 0),
        mqtt_snapshots=int(reader.get("mqtt_snapshots_received", 0) or 0),
        pending_labels=int(reader.get("pending_label_count", 0) or 0),
        mqtt_connected="yes" if bool(reader.get("mqtt_connected", False)) else "no",
        waiting_first_raw="yes" if bool(reader.get("waiting_for_first_raw_frame", False)) else "no",
        last_raw_age=_format_live_age(reader.get("last_raw_age_sec")),
        last_mqtt_age=_format_live_age(reader.get("last_mqtt_age_sec")),
        reader_state=str(reader.get("reader_state", "unknown")),
    )


def _format_live_age(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.1f}s"


def _write_live_object_list_snapshot(writer, run_dir: Path, object_labels: dict[int, ObjectLabelData], reporter) -> None:
    writer.write_object_list(run_dir, object_labels)
    if reporter is None:
        return
    flushed_at = float(time.time())
    with reporter["lock"]:
        previous_writes = int(reporter["state"].get("live_object_list_write_count", 0) or 0)
        reporter["state"].update(
            {
                "object_list_exported_count": int(len(object_labels)),
                "live_object_list_write_count": int(previous_writes + 1),
                "last_live_object_list_write_unix_sec": flushed_at,
            }
        )


def _build_live_artifact_state(input_format: str) -> dict[str, float | None] | None:
    if str(input_format) != "qb2_live":
        return None
    return {"last_flush_monotonic": None}


def _maybe_write_live_artifact_snapshot(
    *,
    config: PipelineConfig,
    profiler: PerformanceProfiler,
    writer,
    run_dir: Path,
    lane_box,
    tracker,
    postprocessors,
    accumulator,
    classifier,
    class_normalizer: ClassNormalizer,
    latest_object_labels: dict[int, ObjectLabelData],
    object_label_history_by_id: dict[int, list[ObjectLabelData]],
    object_list_seen_ids: set[int],
    object_list_skipped_empty: int,
    tracker_states: list,
    frame_count: int,
    live_status_reporter,
    live_web_runtime,
    live_artifact_state,
    force: bool,
    save_aggregate_intensity: bool,
) -> None:
    if live_artifact_state is None:
        return
    if live_web_runtime is not None and not force:
        return
    now = time.monotonic()
    last_flush_monotonic = live_artifact_state.get("last_flush_monotonic")
    if (
        not force
        and last_flush_monotonic is not None
        and max(0.0, now - float(last_flush_monotonic)) < float(LIVE_ARTIFACT_FLUSH_INTERVAL_SEC)
    ):
        return
    snapshot_tracker_states = _live_snapshot_tracker_states(tracker_states)
    _write_live_artifact_snapshot(
        config=config,
        profiler=profiler,
        writer=writer,
        run_dir=run_dir,
        lane_box=lane_box,
        tracker=tracker,
        postprocessors=postprocessors,
        accumulator=accumulator,
        classifier=classifier,
        class_normalizer=class_normalizer,
        latest_object_labels=latest_object_labels,
        object_label_history_by_id=object_label_history_by_id,
        object_list_seen_ids=object_list_seen_ids,
        object_list_skipped_empty=object_list_skipped_empty,
        tracker_states=snapshot_tracker_states,
        frame_count=frame_count,
        live_status_reporter=live_status_reporter,
        live_web_runtime=live_web_runtime,
        save_aggregate_intensity=save_aggregate_intensity,
    )
    live_artifact_state["last_flush_monotonic"] = float(time.monotonic())


def _write_live_artifact_snapshot(
    *,
    config: PipelineConfig,
    profiler: PerformanceProfiler,
    writer,
    run_dir: Path,
    lane_box,
    tracker,
    postprocessors,
    accumulator,
    classifier,
    class_normalizer: ClassNormalizer,
    latest_object_labels: dict[int, ObjectLabelData],
    object_label_history_by_id: dict[int, list[ObjectLabelData]],
    object_list_seen_ids: set[int],
    object_list_skipped_empty: int,
    tracker_states: list,
    frame_count: int,
    live_status_reporter,
    live_web_runtime,
    save_aggregate_intensity: bool,
) -> None:
    tracks = _snapshot_tracker_tracks(tracker)
    for processor in postprocessors:
        with profiler.stage("postprocess_tracks"):
            tracks = processor.process(tracks)

    aggregate_results: list[AggregateResult] = []
    registration_attempts = 0
    registration_accepted = 0
    registration_rejected = 0
    aggregation_component_wall = {component_name: 0.0 for component_name in AGGREGATION_COMPONENT_NAMES}
    aggregation_component_cpu = {component_name: 0.0 for component_name in AGGREGATION_COMPONENT_NAMES}
    aggregation_component_calls = {component_name: 0 for component_name in AGGREGATION_COMPONENT_NAMES}

    for track in tracks.values():
        with profiler.stage("accumulate_tracks"):
            result = accumulator.accumulate(track, lane_box)
        aggregate_results.append(result)
        metrics = result.metrics
        registration_attempts += int(metrics.get("registration_pairs", 0))
        registration_accepted += int(metrics.get("registration_accepted", 0))
        registration_rejected += int(metrics.get("registration_rejected", 0))
        _accumulate_aggregation_component_metrics(
            aggregation_component_wall,
            aggregation_component_cpu,
            aggregation_component_calls,
            result,
            config.aggregation.algorithm,
            config.aggregation.enable_tail_bridge,
        )
    if hasattr(accumulator, "merge_long_vehicle_aggregates"):
        with profiler.stage("accumulate_tracks"):
            aggregate_results = accumulator.merge_long_vehicle_aggregates(tracks, aggregate_results, lane_box)
    with profiler.stage("classify_aggregates"):
        aggregate_results = classify_aggregate_results(aggregate_results, classifier, class_normalizer)
    with profiler.stage("match_gt"):
        matched_gt, unmatched_saved_tracks, unmatched_gt_objects, gt_match_summary = match_saved_aggregates_to_gt(
            tracks,
            aggregate_results,
            dict(object_label_history_by_id),
            class_normalizer,
        )
        apply_gt_matches_to_results(aggregate_results, matched_gt, unmatched_saved_tracks)
    class_stats = build_class_statistics(aggregate_results, latest_object_labels, class_normalizer)
    track_outcomes = build_track_outcomes(tracks, aggregate_results, tracker_states)
    status_counts = Counter(result.status for result in aggregate_results)
    quality_scores = [track.quality_score for track in tracks.values() if track.quality_score is not None]

    _begin_writer_snapshot(writer, run_dir)
    _clear_live_artifact_outputs(writer, run_dir)
    for result in aggregate_results:
        if result.status == "saved":
            with profiler.stage("write_aggregates"):
                writer.write_aggregate(run_dir, result, save_intensity=save_aggregate_intensity)
    with profiler.stage("write_object_list"):
        writer.write_object_list(run_dir, latest_object_labels)
    with profiler.stage("write_gt_matching"):
        writer.write_gt_matching(run_dir, matched_gt, unmatched_saved_tracks, unmatched_gt_objects, gt_match_summary)
    with profiler.stage("write_tracks"):
        writer.write_tracks(run_dir, tracks, aggregate_results)
        writer.write_tracker_debug(run_dir, tracker_states)
        writer.write_track_outcomes(run_dir, track_outcomes)
        writer.write_class_stats(run_dir, class_stats)

    summary = RunSummary(
        input_path=config.input.paths[0],
        input_paths=list(config.input.paths),
        output_mode=str(config.output.mode),
        tracker_algorithm=config.tracking.algorithm,
        accumulator_algorithm=config.aggregation.algorithm,
        clusterer_algorithm=config.clustering.algorithm,
        frame_count=int(frame_count),
        finished_track_count=len(tracks),
        saved_aggregates=sum(1 for result in aggregate_results if result.status == "saved"),
        registration_attempts=registration_attempts,
        registration_accepted=registration_accepted,
        registration_rejected=registration_rejected,
        output_dir=str(run_dir),
        postprocessing_methods=[processor.name for processor in postprocessors],
        aggregate_status_counts=dict(status_counts),
        track_quality_mean=float(sum(quality_scores) / len(quality_scores)) if quality_scores else 0.0,
        object_list_exported_count=len(latest_object_labels),
        object_list_seen_ids=len(object_list_seen_ids),
        object_list_skipped_empty=int(object_list_skipped_empty),
        gt_match_saved_track_count=int(gt_match_summary["gt_match_saved_track_count"]),
        gt_match_matched_count=int(gt_match_summary["gt_match_matched_count"]),
        gt_match_unmatched_saved_count=int(gt_match_summary["gt_match_unmatched_saved_count"]),
        gt_match_unmatched_gt_count=int(gt_match_summary["gt_match_unmatched_gt_count"]),
        gt_match_mode=str(gt_match_summary["gt_match_mode"]),
        gt_match_assignment=str(gt_match_summary["gt_match_assignment"]),
        gt_match_mean_timestamp_delta_ns=float(gt_match_summary["gt_match_mean_timestamp_delta_ns"]),
        gt_match_max_timestamp_delta_ns=int(gt_match_summary["gt_match_max_timestamp_delta_ns"]),
        predicted_class_counts=dict(class_stats["predicted_class_counts"]),
        gt_class_counts=dict(class_stats["gt_class_counts"]),
        matched_gt_class_counts=dict(class_stats["matched_gt_class_counts"]),
        class_comparison_count=int(class_stats["class_comparison_count"]),
        class_match_count=int(class_stats["class_match_count"]),
        class_mismatch_count=int(class_stats["class_mismatch_count"]),
        class_count_rows=[dict(row) for row in class_stats["class_count_rows"]],
        performance=_snapshot_with_aggregation_components(
            profiler,
            aggregation_component_wall,
            aggregation_component_cpu,
            aggregation_component_calls,
            frame_count,
        ),
    )
    summary.performance = _snapshot_with_aggregation_components(
        profiler,
        aggregation_component_wall,
        aggregation_component_cpu,
        aggregation_component_calls,
        summary.frame_count,
    )
    with profiler.stage("write_summary"):
        writer.write_summary(run_dir, summary)
    _update_live_web_snapshot(live_web_runtime, track_outcomes, summary)
    _update_live_status_after_artifact_snapshot(live_status_reporter, summary)


def _live_snapshot_tracker_states(states: list) -> list:
    if not states:
        return []
    return list(states[-LIVE_ARTIFACT_TRACKER_DEBUG_FRAME_COUNT:])


def _begin_writer_snapshot(writer, run_dir: Path) -> None:
    begin_snapshot = getattr(writer, "begin_snapshot", None)
    if callable(begin_snapshot):
        begin_snapshot(run_dir)


def _clear_live_artifact_outputs(writer, run_dir: Path) -> None:
    clear_live_outputs = getattr(writer, "clear_live_outputs", None)
    if callable(clear_live_outputs):
        clear_live_outputs(run_dir)
        return
    for directory_name in ("aggregates", "object_list", "gt_matching"):
        shutil.rmtree(run_dir / directory_name, ignore_errors=True)
    for file_name in ("summary.json", "tracks.jsonl", "tracker_debug.jsonl", "track_outcomes.jsonl", "class_stats.json"):
        path = run_dir / file_name
        if path.exists():
            path.unlink()


def _snapshot_tracker_tracks(tracker) -> dict[int, Track]:
    snapshot_tracks = getattr(tracker, "snapshot_tracks", None)
    if callable(snapshot_tracks):
        tracks = snapshot_tracks()
        return {int(track_id): _clone_track(track) for track_id, track in sorted(dict(tracks).items())}

    combined: dict[int, Track] = {}
    finished_tracks = getattr(tracker, "finished_tracks", None)
    if isinstance(finished_tracks, dict):
        for track_id, track in finished_tracks.items():
            combined[int(track_id)] = track
    active_tracks = getattr(tracker, "tracks", None)
    if isinstance(active_tracks, dict):
        for track_id, track in active_tracks.items():
            combined[int(track_id)] = track
    if not combined and hasattr(tracker, "track"):
        track = getattr(tracker, "track")
        if isinstance(track, Track):
            combined[int(track.track_id)] = track
    return {int(track_id): _clone_track(track) for track_id, track in sorted(combined.items())}


def _clone_track(track: Track) -> Track:
    cloned_state = {
        str(key): copy.deepcopy(value)
        for key, value in dict(track.state).items()
        if str(key) != "kf"
    }
    return Track(
        track_id=int(track.track_id),
        centers=[np.asarray(center, dtype=np.float32).copy() for center in track.centers],
        frame_ids=[int(frame_id) for frame_id in track.frame_ids],
        frame_timestamps_ns=[int(timestamp_ns) for timestamp_ns in track.frame_timestamps_ns],
        local_points=[np.asarray(points, dtype=np.float32).copy() for points in track.local_points],
        world_points=[np.asarray(points, dtype=np.float32).copy() for points in track.world_points],
        local_intensity=[
            None if intensity is None else np.asarray(intensity, dtype=np.float32).copy() for intensity in track.local_intensity
        ],
        world_intensity=[
            None if intensity is None else np.asarray(intensity, dtype=np.float32).copy() for intensity in track.world_intensity
        ],
        point_timestamps_ns=[
            None if values is None else np.asarray(values, dtype=np.int64).copy() for values in track.point_timestamps_ns
        ],
        bbox_extents=[np.asarray(extent, dtype=np.float32).copy() for extent in track.bbox_extents],
        hit_count=int(track.hit_count),
        age=int(track.age),
        missed=int(track.missed),
        ended_by_missed=bool(track.ended_by_missed),
        source_track_ids=[int(track_id) for track_id in track.source_track_ids],
        quality_score=None if track.quality_score is None else float(track.quality_score),
        quality_metrics=copy.deepcopy(track.quality_metrics),
        state=cloned_state,
    )


def _update_live_status_after_artifact_snapshot(reporter, summary: RunSummary) -> None:
    if reporter is None:
        return
    flushed_at = float(time.time())
    with reporter["lock"]:
        previous_writes = int(reporter["state"].get("live_artifact_write_count", 0) or 0)
        reporter["state"].update(
            {
                "finished_track_count": int(summary.finished_track_count),
                "saved_aggregates": int(summary.saved_aggregates),
                "object_list_exported_count": int(summary.object_list_exported_count),
                "object_list_seen_ids": int(summary.object_list_seen_ids),
                "live_artifact_write_count": int(previous_writes + 1),
                "last_live_artifact_write_unix_sec": flushed_at,
            }
        )


@contextmanager
def _sigterm_as_keyboard_interrupt():
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous_handler = signal.getsignal(signal.SIGTERM)

    def _handle_sigterm(signum, frame):
        _ = signum, frame
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, _handle_sigterm)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


def _close_frame_iterator(frame_iterator) -> None:
    close = getattr(frame_iterator, "close", None)
    if callable(close):
        close()


def _close_reader(reader) -> None:
    close = getattr(reader, "close", None)
    if callable(close):
        close()


def _drain_pending_object_labels(
    reader,
    frame_index: int,
    max_timestamp_ns: int | None = None,
) -> list[ObjectLabelData]:
    drain = getattr(reader, "drain_pending_object_labels", None)
    if not callable(drain):
        return []
    return list(drain(frame_index, max_timestamp_ns=max_timestamp_ns))


def _ingest_object_labels(
    object_labels: list[ObjectLabelData],
    latest_object_labels: dict[int, ObjectLabelData],
    object_label_history_by_id: dict[int, list[ObjectLabelData]],
    object_list_seen_ids: set[int],
    class_normalizer: ClassNormalizer,
) -> tuple[int, bool]:
    skipped_empty = 0
    updated = False
    for object_label in object_labels:
        object_list_seen_ids.add(int(object_label.object_id))
        if len(object_label.points) == 0:
            skipped_empty += 1
            continue
        normalized_object_label = class_normalizer.normalize_object_label(object_label)
        object_label_history_by_id[int(object_label.object_id)].append(normalized_object_label)
        current = latest_object_labels.get(int(object_label.object_id))
        if _is_newer_object_label(normalized_object_label, current):
            latest_object_labels[int(object_label.object_id)] = normalized_object_label
            updated = True
    return skipped_empty, updated


def _is_newer_object_label(candidate: ObjectLabelData, current: ObjectLabelData | None) -> bool:
    if current is None:
        return True
    if int(candidate.timestamp_ns) != int(current.timestamp_ns):
        return int(candidate.timestamp_ns) > int(current.timestamp_ns)
    return int(candidate.frame_index) > int(current.frame_index)


def _accumulate_aggregation_component_metrics(
    wall_totals: dict[str, float],
    cpu_totals: dict[str, float],
    call_totals: dict[str, int],
    result: AggregateResult,
    accumulator_algorithm: str,
    enable_tail_bridge: bool,
) -> None:
    metrics = result.metrics
    for component in AGGREGATION_COMPONENT_NAMES:
        wall_totals[component] += float(metrics.get(f"{component}_wall_seconds", 0.0) or 0.0)
        cpu_totals[component] += float(metrics.get(f"{component}_cpu_seconds", 0.0) or 0.0)

    prepared_chunk_count = int(metrics.get("prepared_chunk_count", 0) or 0)
    if prepared_chunk_count <= 0:
        return
    if accumulator_algorithm == "registration_voxel_fusion":
        call_totals["registration"] += 1
    call_totals["fusion_core"] += 1
    call_totals["fusion_total"] += 1
    if result.status != "empty_fused":
        call_totals["post_filter"] += 1
        if enable_tail_bridge:
            call_totals["tail_bridge"] += 1
        call_totals["fusion_post"] += 1
    if result.status != "empty_filtered" and result.status != "empty_fused":
        call_totals["confidence_cap"] += 1
        call_totals["symmetry_completion"] += 1


def _snapshot_with_aggregation_components(
    profiler: PerformanceProfiler,
    wall_totals: dict[str, float],
    cpu_totals: dict[str, float],
    call_totals: dict[str, int],
    frame_count: int,
) -> RunPerformance:
    snapshot = profiler.snapshot()
    snapshot.aggregation_components = build_component_snapshot(wall_totals, cpu_totals, call_totals)
    snapshot.total_hz = derive_hz(frame_count, snapshot.total_wall_seconds)
    snapshot.compute_hz = derive_hz(frame_count, snapshot.compute_wall_seconds)
    return snapshot
