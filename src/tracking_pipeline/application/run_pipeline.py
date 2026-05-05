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
from tracking_pipeline.domain.models import (
    AggregateResult,
    FrameData,
    GTMatchResult,
    ObjectLabelData,
    RunPerformance,
    RunSummary,
    Track,
    TrackOutcomeDebug,
)
from tracking_pipeline.infrastructure.io.frame_segment import FrameSegmentWriter
from tracking_pipeline.infrastructure.logging.run_logger import get_run_logger
from tracking_pipeline.infrastructure.visualization.live_frame_publisher import LiveFramePublisher
from tracking_pipeline.infrastructure.visualization.live_pcd_web_server import LivePCDWebServer

LIVE_ARTIFACT_TRACKER_DEBUG_FRAME_COUNT = 1
LIVE_GT_MATCH_HISTORY_MARGIN_SEC = 5.0


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


def run_pipeline(config: PipelineConfig, project_root: Path, live_observer=None) -> RunSummary:
    profiler = PerformanceProfiler()
    class_normalizer = ClassNormalizer.from_config(config.class_normalization)
    logger = get_run_logger()
    statistics_enabled = bool(config.output.statistics_enabled)
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
    _notify_live_observer(live_observer, logger, "on_run_started", config=config, run_dir=run_dir)

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
        config.output.live_object_list_flush_interval_sec,
        config.output.live_artifact_flush_interval_sec,
        config.output.live_tracker_debug_flush_interval_sec,
        statistics_enabled,
        runtime_limits,
    )
    live_artifact_state = _build_live_artifact_state(config.input.format)
    live_object_list_state = _build_live_object_list_state(
        config.input.format,
        config.output.live_object_list_flush_interval_sec,
    )
    raw_frame_writer: FrameSegmentWriter | None = None
    try:
        raw_frame_writer = _start_raw_frame_writer(config, project_root, writer, run_dir, logger)
        latest_object_labels: dict[int, ObjectLabelData] = {}
        object_label_history_by_id: dict[int, list[ObjectLabelData]] = defaultdict(list)
        object_list_seen_ids: set[int] = set()
        object_list_skipped_empty = 0
        live_web_track_outcomes: dict[int, TrackOutcomeDebug] = {}
        live_web_announced_finished_track_ids: set[int] = set()
        live_snapshot_tracks: dict[int, Track] = {}
        live_snapshot_aggregate_results: dict[int, AggregateResult] = {}
        live_snapshot_track_outcomes: dict[int, TrackOutcomeDebug] = {}
        live_snapshot_announced_finished_track_ids: set[int] = set()
        track_outcome_frame_to_playback: dict[int, int] = {}
        track_outcome_last_active: dict[int, dict[str, object]] = {}
        tracker_states = []
        frame_count = 0
        last_processed_frame_index = -1
        last_processed_frame_timestamp_ns = -1
        if config.input.format == "qb2_live" and statistics_enabled:
            with profiler.stage("write_object_list"):
                _maybe_write_live_object_list_snapshot(
                    writer,
                    run_dir,
                    latest_object_labels,
                    live_status_reporter,
                    live_object_list_state,
                    force=True,
                )
        frame_iterator = iter(reader.iter_frames(config.input.paths))
        interrupted = False
        try:
            with _sigterm_as_keyboard_interrupt():
                while True:
                    try:
                        _set_live_pipeline_step(live_status_reporter, "read_frames", frame_count)
                        with profiler.stage("read_frames"):
                            frame = next(frame_iterator)
                    except StopIteration:
                        break
                    _notify_live_observer(live_observer, logger, "on_frame_read", frame=frame)
                    _set_live_pipeline_step(live_status_reporter, "ingest_labels", frame_count)
                    skipped_empty, object_list_updated = _ingest_object_labels(
                        frame.object_labels,
                        latest_object_labels,
                        object_label_history_by_id,
                        object_list_seen_ids,
                        class_normalizer,
                    )
                    pending_skipped_empty, pending_object_list_updated = _refresh_latest_object_labels(
                        _snapshot_pending_object_labels(reader, frame.frame_index),
                        latest_object_labels,
                        object_label_history_by_id,
                        object_list_seen_ids,
                        class_normalizer,
                    )
                    object_list_skipped_empty += skipped_empty
                    object_list_skipped_empty += pending_skipped_empty
                    if live_artifact_state is not None and (object_list_updated or pending_object_list_updated):
                        live_artifact_state["labels_dirty"] = True
                    if (object_list_updated or pending_object_list_updated) and config.input.format == "qb2_live" and statistics_enabled:
                        with profiler.stage("write_object_list"):
                            _maybe_write_live_object_list_snapshot(
                                writer,
                                run_dir,
                                latest_object_labels,
                                live_status_reporter,
                                live_object_list_state,
                            )
                    _set_live_pipeline_step(live_status_reporter, "cluster_frames", frame_count)
                    with profiler.stage("cluster_frames"):
                        cluster_result = clusterer.cluster(frame, lane_box)
                    _set_live_pipeline_step(live_status_reporter, "tracker_steps", frame_count)
                    with profiler.stage("tracker_steps"):
                        state = tracker.step(cluster_result.detections, frame.frame_index, frame.timestamp_ns)
                    state.cluster_metrics = cluster_result.metrics
                    if statistics_enabled:
                        _update_track_outcome_context(
                            state,
                            playback_index=int(frame_count),
                            frame_to_playback=track_outcome_frame_to_playback,
                            last_active_by_track=track_outcome_last_active,
                        )
                    if statistics_enabled:
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
                    if statistics_enabled:
                        _set_live_pipeline_step(live_status_reporter, "live_web_track_outcomes", frame_count)
                        _maybe_update_live_web_finished_track_outcomes(
                            runtime=live_web_runtime,
                            tracker=tracker,
                            lane_box=lane_box,
                            accumulator=accumulator,
                            classifier=classifier,
                            class_normalizer=class_normalizer,
                            frame_to_playback=track_outcome_frame_to_playback,
                            last_active_by_track=track_outcome_last_active,
                            live_track_outcomes=live_web_track_outcomes,
                            announced_finished_track_ids=live_web_announced_finished_track_ids,
                            logger=logger,
                        )
                    _set_live_pipeline_step(live_status_reporter, "write_live_artifacts", frame_count)
                    _maybe_write_incremental_live_artifact_snapshot(
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
                        frame_to_playback=track_outcome_frame_to_playback,
                        last_active_by_track=track_outcome_last_active,
                        frame_count=frame_count,
                        live_status_reporter=live_status_reporter,
                        live_web_runtime=live_web_runtime,
                        live_artifact_state=live_artifact_state,
                        live_snapshot_tracks=live_snapshot_tracks,
                        live_snapshot_aggregate_results=live_snapshot_aggregate_results,
                        live_snapshot_track_outcomes=live_snapshot_track_outcomes,
                        live_snapshot_announced_finished_track_ids=live_snapshot_announced_finished_track_ids,
                        save_aggregate_intensity=config.output.save_aggregate_intensity,
                        live_observer=live_observer,
                        logger=logger,
                    )
                    if config.input.format == "qb2_live" and statistics_enabled:
                        with profiler.stage("write_object_list"):
                            _maybe_write_live_object_list_snapshot(
                                writer,
                                run_dir,
                                latest_object_labels,
                                live_status_reporter,
                                live_object_list_state,
                                mark_dirty=False,
                            )
                    _set_live_pipeline_step(live_status_reporter, "frame_complete", frame_count)
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
            if live_artifact_state is not None and object_list_updated:
                live_artifact_state["labels_dirty"] = True
            if object_list_updated and config.input.format == "qb2_live" and statistics_enabled:
                with profiler.stage("write_object_list"):
                    _maybe_write_live_object_list_snapshot(
                        writer,
                        run_dir,
                        latest_object_labels,
                        live_status_reporter,
                        live_object_list_state,
                    )
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
        if str(config.input.format) == "qb2_live" and not bool(config.output.final_full_recompute):
            with profiler.stage("tracker_finalize"):
                finalized_tracks = tracker.finalize()
            finished_tracks = getattr(tracker, "finished_tracks", None)
            if not isinstance(finished_tracks, dict) and isinstance(finalized_tracks, dict):
                finished_tracks = finalized_tracks
            if isinstance(finished_tracks, dict):
                _discard_processed_finished_tracks(finished_tracks, live_snapshot_announced_finished_track_ids)
                new_tracks, new_results = _process_incremental_finished_tracks(
                    finished_tracks=finished_tracks,
                    lane_box=lane_box,
                    postprocessors=postprocessors,
                    accumulator=accumulator,
                    classifier=classifier,
                    class_normalizer=class_normalizer,
                    frame_to_playback=track_outcome_frame_to_playback,
                    last_active_by_track=track_outcome_last_active,
                    profiler=profiler,
                    announced_finished_track_ids=live_snapshot_announced_finished_track_ids,
                    live_snapshot_tracks=live_snapshot_tracks,
                    live_snapshot_aggregate_results=live_snapshot_aggregate_results,
                    live_snapshot_track_outcomes=live_snapshot_track_outcomes,
                    pending_snapshot_results=live_artifact_state["pending_saved_results"] if live_artifact_state is not None else None,
                    collect_track_outcomes=statistics_enabled,
                )
                _discard_processed_finished_tracks(finished_tracks, live_snapshot_announced_finished_track_ids)
                _update_live_incremental_track_metrics(
                    live_status_reporter,
                    live_web_runtime,
                    queued_finished_track_count=len(finished_tracks),
                    processed_finished_track_count=len(live_snapshot_announced_finished_track_ids),
                    snapshot_track_count=len(live_snapshot_tracks),
                    snapshot_aggregate_count=len(live_snapshot_aggregate_results),
                    saved_aggregate_count=_saved_aggregate_count(live_snapshot_aggregate_results),
                )
                if new_tracks:
                    _notify_live_observer(
                        live_observer,
                        logger,
                        "on_live_aggregates",
                        tracks=new_tracks,
                        aggregate_results=new_results,
                    )

            aggregate_results = list(live_snapshot_aggregate_results.values())
            with profiler.stage("match_gt"):
                matched_gt, unmatched_saved_tracks, unmatched_gt_objects, gt_match_summary = match_saved_aggregates_to_gt(
                    live_snapshot_tracks,
                    aggregate_results,
                    dict(object_label_history_by_id),
                    class_normalizer,
                )
                apply_gt_matches_to_results(aggregate_results, matched_gt, unmatched_saved_tracks)
            _notify_live_observer(
                live_observer,
                logger,
                "on_live_aggregates",
                tracks=live_snapshot_tracks,
                aggregate_results=aggregate_results,
            )
            class_stats = (
                build_class_statistics(aggregate_results, latest_object_labels, class_normalizer)
                if statistics_enabled
                else _empty_class_statistics()
            )
            summary = _build_incremental_live_summary(
                config=config,
                run_dir=run_dir,
                postprocessors=postprocessors,
                tracks=live_snapshot_tracks,
                aggregate_results=aggregate_results,
                latest_object_labels=latest_object_labels,
                object_list_seen_ids=object_list_seen_ids,
                object_list_skipped_empty=object_list_skipped_empty,
                class_stats=class_stats,
                frame_count=frame_count,
                gt_match_summary=gt_match_summary,
            )
            _write_selected_object_frames(raw_frame_writer, live_snapshot_tracks, aggregate_results, profiler)
            _begin_writer_snapshot(writer, run_dir)
            _clear_live_artifact_outputs(writer, run_dir)
            for result in aggregate_results:
                if str(result.status) == "saved":
                    with profiler.stage("write_aggregates"):
                        writer.write_aggregate(run_dir, result, save_intensity=config.output.save_aggregate_intensity)
            with _writer_sample_batch(writer):
                with profiler.stage("write_object_list"):
                    if statistics_enabled:
                        _maybe_write_live_object_list_snapshot(
                            writer,
                            run_dir,
                            latest_object_labels,
                            live_status_reporter,
                            live_object_list_state,
                            force=True,
                        )
                    else:
                        _write_live_object_list_snapshot(writer, run_dir, latest_object_labels, live_status_reporter)
                with profiler.stage("write_gt_matching"):
                    writer.write_gt_matching(run_dir, matched_gt, unmatched_saved_tracks, unmatched_gt_objects, gt_match_summary)
            if statistics_enabled:
                with _writer_stats_batch(writer):
                    with profiler.stage("write_tracks"):
                        writer.write_tracks(run_dir, live_snapshot_tracks, aggregate_results)
                        writer.write_tracker_debug(run_dir, tracker_states)
                        writer.write_track_outcomes(run_dir, live_snapshot_track_outcomes)
                        writer.write_class_stats(run_dir, class_stats)
                    with profiler.stage("write_summary"):
                        writer.write_summary(run_dir, summary)
                _update_live_web_snapshot(live_web_runtime, live_snapshot_track_outcomes, summary)
        else:
            if str(config.input.format) == "qb2_live" and statistics_enabled:
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
                    tracker_states=tracker_states,
                    frame_count=frame_count,
                    live_status_reporter=live_status_reporter,
                    live_web_runtime=live_web_runtime,
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
            _notify_live_observer(
                live_observer,
                logger,
                "on_live_aggregates",
                tracks=tracks,
                aggregate_results=aggregate_results,
            )
            class_stats = (
                build_class_statistics(aggregate_results, latest_object_labels, class_normalizer)
                if statistics_enabled
                else _empty_class_statistics()
            )
            _write_selected_object_frames(raw_frame_writer, tracks, aggregate_results, profiler)
            _begin_writer_snapshot(writer, run_dir)
            for result in aggregate_results:
                if result.status == "saved":
                    with profiler.stage("write_aggregates"):
                        writer.write_aggregate(run_dir, result, save_intensity=config.output.save_aggregate_intensity)
            track_outcomes = (
                build_track_outcomes(
                    tracks,
                    aggregate_results,
                    tracker_states,
                    frame_to_playback=track_outcome_frame_to_playback,
                    last_active_by_track=track_outcome_last_active,
                )
                if statistics_enabled
                else {}
            )

            with _writer_sample_batch(writer):
                with profiler.stage("write_object_list"):
                    if statistics_enabled:
                        _maybe_write_live_object_list_snapshot(
                            writer,
                            run_dir,
                            latest_object_labels,
                            live_status_reporter,
                            live_object_list_state,
                            force=True,
                        )
                    else:
                        _write_live_object_list_snapshot(writer, run_dir, latest_object_labels, live_status_reporter)
                with profiler.stage("write_gt_matching"):
                    writer.write_gt_matching(run_dir, matched_gt, unmatched_saved_tracks, unmatched_gt_objects, gt_match_summary)

            status_counts = Counter(result.status for result in aggregate_results)
            quality_scores = [track.quality_score for track in tracks.values() if track.quality_score is not None]
            articulated_summary = _build_articulated_vehicle_summary(tracks, aggregate_results)
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
                **articulated_summary,
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
                performance=(
                    _snapshot_with_aggregation_components(
                        profiler,
                        aggregation_component_wall,
                        aggregation_component_cpu,
                        aggregation_component_calls,
                        frame_count,
                    )
                    if statistics_enabled
                    else None
                ),
            )
            if statistics_enabled:
                with _writer_stats_batch(writer):
                    with profiler.stage("write_tracks"):
                        writer.write_tracks(run_dir, tracks, aggregate_results)
                        writer.write_tracker_debug(run_dir, tracker_states)
                        writer.write_track_outcomes(run_dir, track_outcomes)
                        writer.write_class_stats(run_dir, class_stats)
                    with profiler.stage("write_summary"):
                        writer.write_summary(run_dir, summary)
                _update_live_web_snapshot(live_web_runtime, track_outcomes, summary)
                summary.performance = _snapshot_with_aggregation_components(
                    profiler,
                    aggregation_component_wall,
                    aggregation_component_cpu,
                    aggregation_component_calls,
                    summary.frame_count,
                )
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
        _close_raw_frame_writer(raw_frame_writer, logger)
        _notify_live_observer(live_observer, logger, "on_run_finished")
        _stop_live_web_viewer(live_web_runtime)
        _stop_live_status_reporter(live_status_reporter, reader)


def _start_raw_frame_writer(
    config: PipelineConfig,
    project_root: Path,
    writer,
    run_dir: Path,
    logger,
) -> FrameSegmentWriter | None:
    if not bool(config.output.raw_frames_enabled):
        return None
    run_id = str(getattr(writer, "_run_id", "") or run_dir.name)
    segment_dir = _resolve_raw_frame_segment_dir(config, project_root, run_dir, run_id)
    try:
        raw_writer = FrameSegmentWriter(segment_dir)
    except OSError as exc:
        logger.warning("Raw frame recording disabled; cannot write to %s: %s", segment_dir, exc)
        return None
    logger.info("Raw frame recording enabled: %s", segment_dir)
    return raw_writer


def _resolve_raw_frame_segment_dir(config: PipelineConfig, project_root: Path, run_dir: Path, run_id: str) -> Path:
    configured_dir = str(config.output.raw_frames_dir).strip()
    if configured_dir:
        root = Path(configured_dir)
        if not root.is_absolute():
            root = (project_root / root).resolve()
        return root / run_id
    if str(config.output.mode) == "dataset":
        return run_dir / "_raw_frames" / run_id
    return run_dir / "raw_frames"


def _write_selected_object_frames(
    raw_frame_writer: FrameSegmentWriter | None,
    tracks: dict[int, Track],
    aggregate_results: list[AggregateResult],
    profiler: PerformanceProfiler,
) -> None:
    if raw_frame_writer is None:
        return
    with profiler.stage("write_raw_frames"):
        for result in aggregate_results:
            if str(result.status) != "saved":
                continue
            track = tracks.get(int(result.track_id))
            if track is None:
                continue
            for frame in _selected_object_frames(track, result):
                raw_frame_writer.write_frame(frame)


def _selected_object_frames(track: Track, result: AggregateResult) -> list[FrameData]:
    frame_ids = _raw_object_frame_ids(result)
    if not frame_ids:
        return []
    frame_to_index = {int(frame_id): index for index, frame_id in enumerate(track.frame_ids)}
    frames: list[FrameData] = []
    for frame_id in frame_ids:
        index = frame_to_index.get(int(frame_id))
        if index is None or index >= len(track.world_points):
            continue
        points = np.asarray(track.world_points[index], dtype=np.float32)
        if len(points) == 0:
            continue
        frames.append(
            FrameData(
                frame_index=int(frame_id),
                timestamp_ns=_track_frame_timestamp_at(track, index),
                points=points,
                point_intensity=_track_optional_array_at(track.world_intensity, index, np.float32),
                point_timestamp_ns=_track_optional_array_at(track.point_timestamps_ns, index, np.int64),
                source_path=f"track://{int(track.track_id)}/chunk_quality_kept",
                source_frame_index=int(frame_id),
                source_sequence_index=int(track.track_id),
            )
        )
    return frames


def _raw_object_frame_ids(result: AggregateResult) -> list[int]:
    raw_frame_ids = result.metrics.get("chunk_quality_kept_frame_ids")
    if isinstance(raw_frame_ids, (list, tuple)):
        return [int(frame_id) for frame_id in raw_frame_ids]
    return [int(frame_id) for frame_id in result.selected_frame_ids]


def _track_frame_timestamp_at(track: Track, index: int) -> int:
    if index < len(track.frame_timestamps_ns):
        return int(track.frame_timestamps_ns[index])
    return -1


def _track_optional_array_at(values: list[np.ndarray | None], index: int, dtype) -> np.ndarray | None:
    if index >= len(values):
        return None
    value = values[index]
    if value is None:
        return None
    return np.asarray(value, dtype=dtype)


def _close_raw_frame_writer(raw_frame_writer: FrameSegmentWriter | None, logger) -> None:
    if raw_frame_writer is None:
        return
    try:
        raw_frame_writer.close()
    except Exception as exc:  # pragma: no cover - best effort during shutdown
        if logger is not None:
            logger.warning("Closing raw frame recording failed: %s", exc)


def _notify_live_observer(observer, logger, method_name: str, **kwargs):
    if observer is None:
        return None
    method = getattr(observer, method_name, None)
    if not callable(method):
        return None
    try:
        return method(**kwargs)
    except Exception as exc:  # pragma: no cover - defensive isolation for live diagnostics
        if logger is not None:
            logger.info("Live observer callback %s failed: %s", method_name, exc)
        return None


def _start_live_status_reporter(
    reader,
    writer,
    run_dir: Path,
    input_path: str,
    logger,
    input_format: str,
    live_object_list_flush_interval_sec: float,
    live_artifact_flush_interval_sec: float,
    live_tracker_debug_flush_interval_sec: float,
    statistics_enabled: bool,
    runtime_limits: dict[str, object],
):
    if str(input_format) != "qb2_live":
        return None
    started_monotonic = time.monotonic()
    persist_status = bool(statistics_enabled)
    status_path = _writer_output_path(writer, "live_status_path", run_dir, run_dir / "live_status.json")
    object_list_manifest_path = _writer_output_path(writer, "object_list_manifest_path", run_dir, run_dir / "object_list" / "manifest.jsonl")
    live_artifact_dir = _writer_output_path(writer, "live_artifact_dir", run_dir, run_dir)
    reporter = {
        "run_dir": run_dir,
        "status_path": status_path,
        "persist_status": persist_status,
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
            "live_artifact_flush_interval_sec": float(live_artifact_flush_interval_sec),
            "live_tracker_debug_flush_interval_sec": float(live_tracker_debug_flush_interval_sec),
            "live_artifact_write_count": 0,
            "last_live_artifact_write_unix_sec": None,
            "live_finished_track_processed_count": 0,
            "live_finished_track_queue_count": 0,
            "live_snapshot_track_count": 0,
            "live_snapshot_aggregate_count": 0,
            "object_list_exported_count": 0,
            "object_list_seen_ids": 0,
            "object_list_manifest_path": str(object_list_manifest_path),
            "live_object_list_flush_interval_sec": float(live_object_list_flush_interval_sec),
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
            "current_pipeline_step": "starting",
            "current_pipeline_step_frame": 0,
            "_started_monotonic": float(started_monotonic),
            "_last_processed_monotonic": None,
            "_last_processed_frame_count": 0,
            "_current_pipeline_step_started_monotonic": float(started_monotonic),
        },
        "lock": threading.Lock(),
        "stop_event": threading.Event(),
        "thread": None,
    }
    payload = _build_live_status_payload(reader, reporter)
    _persist_live_status_payload(reporter, payload)
    logger.info("Live run active: %s", run_dir)
    if persist_status:
        logger.info("Live status file: %s", reporter["status_path"])
    else:
        logger.info("Live status file disabled by output.statistics_enabled=false; CLI Hz status remains active")
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
    return {
        "publisher": publisher,
        "server": server,
        "_started_monotonic": float(time.monotonic()),
        "_last_processed_monotonic": None,
        "_last_processed_frame_count": 0,
    }


def _update_live_web_status(runtime, **updates: object) -> None:
    if runtime is None:
        return
    publisher = runtime.get("publisher")
    if isinstance(publisher, LiveFramePublisher):
        publisher.update_status(**_with_live_web_processing_metrics(runtime, updates))


def _with_live_web_processing_metrics(runtime, updates: dict[str, object]) -> dict[str, object]:
    enriched = dict(updates)
    if "processed_frames" not in enriched:
        return enriched
    processed_frames = int(enriched.get("processed_frames", 0) or 0)
    now = time.monotonic()
    started_monotonic = float(runtime.get("_started_monotonic", now) or now)
    elapsed = max(0.0, now - started_monotonic)
    enriched["processing_total_hz"] = (
        0.0 if processed_frames <= 0 or elapsed <= 0.0 else float(processed_frames) / elapsed
    )
    previous_monotonic = runtime.get("_last_processed_monotonic")
    previous_frame_count = int(runtime.get("_last_processed_frame_count", 0) or 0)
    if previous_monotonic is not None and processed_frames > previous_frame_count:
        delta = max(0.0, now - float(previous_monotonic))
        frame_delta = processed_frames - previous_frame_count
        enriched["processing_recent_hz"] = 0.0 if delta <= 0.0 else float(frame_delta) / delta
    elif processed_frames <= 1:
        enriched["processing_recent_hz"] = 0.0
    runtime["_last_processed_monotonic"] = float(now)
    runtime["_last_processed_frame_count"] = int(processed_frames)
    return enriched


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
    frame_to_playback: dict[int, int],
    last_active_by_track: dict[int, dict[str, object]],
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
        live_track_outcomes.update(
            build_track_outcomes(
                new_tracks,
                aggregate_results,
                frame_to_playback=frame_to_playback,
                last_active_by_track=last_active_by_track,
            )
        )
        announced_finished_track_ids.update(new_tracks)
        publisher.update_track_outcomes(live_track_outcomes)
        publisher.update_status(
            finished_track_count=int(len(live_track_outcomes)),
            saved_aggregates=int(sum(1 for outcome in live_track_outcomes.values() if str(outcome.status) == "saved")),
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.info("Live web track outcome update failed: %s", exc)


def _update_track_outcome_context(
    state,
    *,
    playback_index: int,
    frame_to_playback: dict[int, int],
    last_active_by_track: dict[int, dict[str, object]],
) -> None:
    frame_to_playback[int(state.frame_index)] = int(playback_index)
    for active_track in getattr(state, "active_tracks", []) or []:
        last_active_by_track[int(active_track.track_id)] = {
            "playback_index": int(playback_index),
            "center": np.asarray(active_track.center, dtype=np.float32).copy(),
        }


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


def _set_live_pipeline_step(reporter, step: str, frame_count: int) -> None:
    if reporter is None:
        return
    with reporter["lock"]:
        state = reporter["state"]
        current_step = str(state.get("current_pipeline_step", "") or "")
        current_frame = int(state.get("current_pipeline_step_frame", -1) or -1)
        next_frame = int(frame_count)
        if current_step == str(step) and current_frame == next_frame:
            return
        state["current_pipeline_step"] = str(step)
        state["current_pipeline_step_frame"] = next_frame
        state["_current_pipeline_step_started_monotonic"] = float(time.monotonic())


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
        step_started_monotonic = reporter["state"].get("_current_pipeline_step_started_monotonic")
    if step_started_monotonic is None:
        state["current_pipeline_step_age_sec"] = None
    else:
        state["current_pipeline_step_age_sec"] = max(0.0, time.monotonic() - float(step_started_monotonic))
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
    if not bool(reporter.get("persist_status", True)):
        return
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
        "finq={finished_queue} snaptr={snapshot_tracks} "
        "aw={artifact_writes} ow={object_writes} raw={raw} mqtt={mqtt_msgs} snap={mqtt_snapshots} "
        "q={pending_labels}/{pending_snapshots} drop={dropped_overflow_labels}/{dropped_stale_labels} "
        "conn={mqtt_connected} wait={waiting_first_raw} reconn={raw_reconnects} "
        "raw_age={last_raw_age} mqtt_age={last_mqtt_age} state={reader_state} "
        "step={pipeline_step} step_age={pipeline_step_age}"
    ).format(
        phase=str(payload.get("pipeline_phase", "unknown")),
        processed=int(payload.get("processed_frames", 0) or 0),
        recent_hz=float(payload.get("processing_recent_hz", 0.0) or 0.0),
        total_hz=float(payload.get("processing_total_hz", 0.0) or 0.0),
        active_tracks=int(payload.get("active_track_count", 0) or 0),
        finished_queue=int(payload.get("live_finished_track_queue_count", 0) or 0),
        snapshot_tracks=int(payload.get("live_snapshot_track_count", 0) or 0),
        artifact_writes=int(payload.get("live_artifact_write_count", 0) or 0),
        object_writes=int(payload.get("live_object_list_write_count", 0) or 0),
        raw=int(reader.get("raw_frames_received", 0) or 0),
        mqtt_msgs=int(reader.get("mqtt_messages_received", 0) or 0),
        mqtt_snapshots=int(reader.get("mqtt_snapshots_received", 0) or 0),
        pending_labels=int(reader.get("pending_label_count", 0) or 0),
        pending_snapshots=int(reader.get("pending_snapshot_count", 0) or 0),
        dropped_overflow_labels=int(reader.get("dropped_overflow_label_count", 0) or 0),
        dropped_stale_labels=int(reader.get("dropped_stale_label_count", 0) or 0),
        mqtt_connected="yes" if bool(reader.get("mqtt_connected", False)) else "no",
        waiting_first_raw="yes" if bool(reader.get("waiting_for_first_raw_frame", False)) else "no",
        raw_reconnects=int(reader.get("raw_stream_reconnect_count", 0) or 0),
        last_raw_age=_format_live_age(reader.get("last_raw_age_sec")),
        last_mqtt_age=_format_live_age(reader.get("last_mqtt_age_sec")),
        reader_state=str(reader.get("reader_state", "unknown")),
        pipeline_step=str(payload.get("current_pipeline_step", "unknown")),
        pipeline_step_age=_format_live_age(payload.get("current_pipeline_step_age_sec")),
    )


def _format_live_age(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.1f}s"


def _build_live_object_list_state(input_format: str, flush_interval_sec: float) -> dict[str, float | bool | None] | None:
    if str(input_format) != "qb2_live":
        return None
    return {
        "dirty": False,
        "last_flush_monotonic": None,
        "flush_interval_sec": float(max(0.0, float(flush_interval_sec))),
    }


def _maybe_write_live_object_list_snapshot(
    writer,
    run_dir: Path,
    object_labels: dict[int, ObjectLabelData],
    reporter,
    state,
    *,
    force: bool = False,
    mark_dirty: bool = True,
) -> bool:
    if state is None:
        _write_live_object_list_snapshot(writer, run_dir, object_labels, reporter)
        return True
    if mark_dirty and not force:
        state["dirty"] = True
    last_flush_monotonic = state.get("last_flush_monotonic")
    flush_interval_sec = float(state.get("flush_interval_sec", 0.0) or 0.0)
    dirty = bool(state.get("dirty"))
    should_write = bool(force and (bool(state.get("dirty")) or last_flush_monotonic is None))
    if not should_write:
        if not dirty and last_flush_monotonic is not None:
            return False
        now = time.monotonic()
        should_write = dirty and (
            last_flush_monotonic is None or flush_interval_sec <= 0.0 or (now - float(last_flush_monotonic)) >= flush_interval_sec
        )
        if not should_write:
            return False
    _write_live_object_list_snapshot(writer, run_dir, object_labels, reporter)
    state["dirty"] = False
    state["last_flush_monotonic"] = float(time.monotonic())
    return True


def _write_live_object_list_snapshot(writer, run_dir: Path, object_labels: dict[int, ObjectLabelData], reporter) -> None:
    write_live_object_list_snapshot = getattr(writer, "write_live_object_list_snapshot", None)
    if callable(write_live_object_list_snapshot):
        write_live_object_list_snapshot(run_dir, object_labels)
    else:
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


def _build_live_artifact_state(input_format: str) -> dict[str, object] | None:
    if str(input_format) != "qb2_live":
        return None
    return {
        "last_flush_monotonic": None,
        "last_tracker_debug_flush_monotonic": None,
        "pending_flush": False,
        "pending_saved_results": {},
        "labels_dirty": False,
        "cached_matches": [],
        "cached_unmatched_saved_tracks": [],
        "cached_unmatched_gt_objects": [],
        "cached_gt_match_summary": None,
        "cached_class_stats": None,
    }


def _discard_processed_finished_tracks(finished_tracks: dict[int, Track], processed_track_ids: set[int]) -> int:
    removed_count = 0
    processed_ids = {int(track_id) for track_id in processed_track_ids}
    for track_id in list(finished_tracks):
        if int(track_id) in processed_ids:
            finished_tracks.pop(track_id, None)
            removed_count += 1
    return int(removed_count)


def _update_live_incremental_track_metrics(
    live_status_reporter,
    live_web_runtime,
    *,
    queued_finished_track_count: int,
    processed_finished_track_count: int,
    snapshot_track_count: int,
    snapshot_aggregate_count: int,
    saved_aggregate_count: int,
) -> None:
    updates = {
        "live_finished_track_queue_count": int(queued_finished_track_count),
        "live_finished_track_processed_count": int(processed_finished_track_count),
        "live_snapshot_track_count": int(snapshot_track_count),
        "live_snapshot_aggregate_count": int(snapshot_aggregate_count),
        "saved_aggregates": int(saved_aggregate_count),
    }
    _update_live_status_reporter(live_status_reporter, **updates)
    _update_live_web_status(live_web_runtime, **updates)


def _saved_aggregate_count(aggregate_results_by_track: dict[int, AggregateResult]) -> int:
    return int(sum(1 for result in aggregate_results_by_track.values() if str(result.status) == "saved"))


def _maybe_write_incremental_live_artifact_snapshot(
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
    frame_to_playback: dict[int, int],
    last_active_by_track: dict[int, dict[str, object]],
    frame_count: int,
    live_status_reporter,
    live_web_runtime,
    live_artifact_state,
    live_snapshot_tracks: dict[int, Track],
    live_snapshot_aggregate_results: dict[int, AggregateResult],
    live_snapshot_track_outcomes: dict[int, TrackOutcomeDebug],
    live_snapshot_announced_finished_track_ids: set[int],
    save_aggregate_intensity: bool,
    live_observer=None,
    logger=None,
    force: bool = False,
) -> None:
    if live_artifact_state is None:
        return
    statistics_enabled = bool(config.output.statistics_enabled)
    finished_tracks = getattr(tracker, "finished_tracks", None)
    if isinstance(finished_tracks, dict):
        _discard_processed_finished_tracks(finished_tracks, live_snapshot_announced_finished_track_ids)
        _update_live_incremental_track_metrics(
            live_status_reporter,
            live_web_runtime,
            queued_finished_track_count=len(finished_tracks),
            processed_finished_track_count=len(live_snapshot_announced_finished_track_ids),
            snapshot_track_count=len(live_snapshot_tracks),
            snapshot_aggregate_count=len(live_snapshot_aggregate_results),
            saved_aggregate_count=_saved_aggregate_count(live_snapshot_aggregate_results),
        )

    if isinstance(finished_tracks, dict) and finished_tracks:
        _set_live_pipeline_step(live_status_reporter, "accumulate_finished_tracks", frame_count)
        new_tracks, aggregate_results = _process_incremental_finished_tracks(
            finished_tracks=finished_tracks,
            lane_box=lane_box,
            postprocessors=postprocessors,
            accumulator=accumulator,
            classifier=classifier,
            class_normalizer=class_normalizer,
            frame_to_playback=frame_to_playback,
            last_active_by_track=last_active_by_track,
            profiler=profiler,
            announced_finished_track_ids=live_snapshot_announced_finished_track_ids,
            live_snapshot_tracks=live_snapshot_tracks,
            live_snapshot_aggregate_results=live_snapshot_aggregate_results,
            live_snapshot_track_outcomes=live_snapshot_track_outcomes,
            pending_snapshot_results=live_artifact_state["pending_saved_results"],
            collect_track_outcomes=statistics_enabled,
        )
        _discard_processed_finished_tracks(finished_tracks, live_snapshot_announced_finished_track_ids)
        _update_live_incremental_track_metrics(
            live_status_reporter,
            live_web_runtime,
            queued_finished_track_count=len(finished_tracks),
            processed_finished_track_count=len(live_snapshot_announced_finished_track_ids),
            snapshot_track_count=len(live_snapshot_tracks),
            snapshot_aggregate_count=len(live_snapshot_aggregate_results),
            saved_aggregate_count=_saved_aggregate_count(live_snapshot_aggregate_results),
        )
        if new_tracks:
            live_artifact_state["pending_flush"] = True
            _notify_live_observer(
                live_observer,
                logger,
                "on_live_aggregates",
                tracks=new_tracks,
                aggregate_results=aggregate_results,
            )

    dataset_snapshot_dirty = (
        bool(force)
        or bool(live_artifact_state.get("pending_flush"))
        or (statistics_enabled and bool(live_artifact_state.get("labels_dirty")))
    )
    if not dataset_snapshot_dirty:
        return

    now = time.monotonic()
    last_flush_monotonic = live_artifact_state.get("last_flush_monotonic")
    flush_interval_sec = float(max(0.0, float(config.output.live_artifact_flush_interval_sec)))
    if not force and last_flush_monotonic is not None and flush_interval_sec > 0.0:
        if (now - float(last_flush_monotonic)) < flush_interval_sec:
            return
    aggregate_results = list(live_snapshot_aggregate_results.values())
    tracks_to_write = _snapshot_tracker_tracks(tracker) if statistics_enabled else dict(live_snapshot_tracks)
    if statistics_enabled:
        for track_id, track in live_snapshot_tracks.items():
            tracks_to_write[int(track_id)] = _clone_track_metadata(track)
    if not tracks_to_write and not aggregate_results and not latest_object_labels:
        return
    cached_gt_match_summary = live_artifact_state.get("cached_gt_match_summary")
    cached_class_stats = live_artifact_state.get("cached_class_stats")
    if dataset_snapshot_dirty or cached_gt_match_summary is None or (statistics_enabled and cached_class_stats is None):
        pending_results_by_track = {
            int(track_id): result
            for track_id, result in dict(live_artifact_state.get("pending_saved_results") or {}).items()
        }
        pending_saved_results_by_track = {
            int(track_id): result
            for track_id, result in pending_results_by_track.items()
            if str(getattr(result, "status", "")) == "saved"
        }
        cached_matches = list(live_artifact_state.get("cached_matches") or [])
        cached_unmatched_saved_tracks = list(live_artifact_state.get("cached_unmatched_saved_tracks") or [])
        cached_unmatched_gt_objects = list(live_artifact_state.get("cached_unmatched_gt_objects") or [])
        replaced_track_ids = set(pending_results_by_track)
        rematch_track_ids = set(pending_saved_results_by_track)
        if cached_gt_match_summary is None and not rematch_track_ids:
            rematch_track_ids = {int(result.track_id) for result in aggregate_results if str(result.status) == "saved"}
            replaced_track_ids.update(rematch_track_ids)

        if rematch_track_ids:
            match_tracks = {
                int(track_id): live_snapshot_tracks[int(track_id)]
                for track_id in sorted(rematch_track_ids)
                if int(track_id) in live_snapshot_tracks
            }
            match_results = [
                live_snapshot_aggregate_results[int(track_id)]
                for track_id in sorted(match_tracks)
                if int(track_id) in live_snapshot_aggregate_results
            ]
            scoped_label_history = _live_gt_match_label_history(
                match_tracks,
                object_label_history_by_id,
                margin_sec=LIVE_GT_MATCH_HISTORY_MARGIN_SEC,
            )
            _set_live_pipeline_step(live_status_reporter, "match_gt", frame_count)
            with profiler.stage("match_gt"):
                new_matches, new_unmatched_saved, new_unmatched_gt, _ = match_saved_aggregates_to_gt(
                    match_tracks,
                    match_results,
                    scoped_label_history,
                    class_normalizer,
                )
                apply_gt_matches_to_results(match_results, new_matches, new_unmatched_saved)
            matched_gt = [match for match in cached_matches if int(match.track_id) not in replaced_track_ids] + new_matches
            unmatched_saved_tracks = [
                match for match in cached_unmatched_saved_tracks if int(match.track_id) not in replaced_track_ids
            ] + new_unmatched_saved
            replaced_gt_object_ids = {
                int(match.gt_object_id)
                for match in cached_matches
                if int(match.track_id) in replaced_track_ids and match.gt_object_id is not None
            }
            cached_unmatched_gt_by_object = {
                int(match.gt_object_id): match
                for match in cached_unmatched_gt_objects
                if match.gt_object_id is not None
                and int(match.gt_object_id) not in replaced_gt_object_ids
            }
            for match in new_unmatched_gt:
                if match.gt_object_id is not None:
                    cached_unmatched_gt_by_object[int(match.gt_object_id)] = match
            unmatched_gt_objects = [cached_unmatched_gt_by_object[object_id] for object_id in sorted(cached_unmatched_gt_by_object)]
        else:
            matched_gt = [match for match in cached_matches if int(match.track_id) not in replaced_track_ids]
            unmatched_saved_tracks = [
                match for match in cached_unmatched_saved_tracks if int(match.track_id) not in replaced_track_ids
            ]
            replaced_gt_object_ids = {
                int(match.gt_object_id)
                for match in cached_matches
                if int(match.track_id) in replaced_track_ids and match.gt_object_id is not None
            }
            unmatched_gt_objects = [
                match
                for match in cached_unmatched_gt_objects
                if match.gt_object_id is None or int(match.gt_object_id) not in replaced_gt_object_ids
            ]
        gt_match_summary = _live_gt_match_summary(matched_gt, unmatched_saved_tracks, unmatched_gt_objects)

        class_stats = (
            build_class_statistics(
                aggregate_results,
                latest_object_labels,
                class_normalizer,
            )
            if statistics_enabled
            else _empty_class_statistics()
        )
        live_artifact_state["cached_matches"] = list(matched_gt)
        live_artifact_state["cached_unmatched_saved_tracks"] = list(unmatched_saved_tracks)
        live_artifact_state["cached_unmatched_gt_objects"] = list(unmatched_gt_objects)
        live_artifact_state["cached_gt_match_summary"] = dict(gt_match_summary)
        live_artifact_state["cached_class_stats"] = dict(class_stats)
        live_artifact_state["labels_dirty"] = False
    else:
        matched_gt = list(live_artifact_state.get("cached_matches") or [])
        unmatched_saved_tracks = list(live_artifact_state.get("cached_unmatched_saved_tracks") or [])
        unmatched_gt_objects = list(live_artifact_state.get("cached_unmatched_gt_objects") or [])
        gt_match_summary = dict(cached_gt_match_summary)
        class_stats = dict(cached_class_stats) if statistics_enabled else _empty_class_statistics()

    summary = _build_incremental_live_summary(
        config=config,
        run_dir=run_dir,
        postprocessors=postprocessors,
        tracks=tracks_to_write,
        aggregate_results=aggregate_results,
        latest_object_labels=latest_object_labels,
        object_list_seen_ids=object_list_seen_ids,
        object_list_skipped_empty=object_list_skipped_empty,
        class_stats=class_stats,
        frame_count=frame_count,
        gt_match_summary=gt_match_summary,
    )
    track_outcomes_to_write = dict(live_snapshot_track_outcomes) if statistics_enabled else {}
    unsummarized_tracks = {
        int(track_id): track
        for track_id, track in tracks_to_write.items()
        if int(track_id) not in track_outcomes_to_write
    }
    if statistics_enabled and unsummarized_tracks:
        track_outcomes_to_write.update(
            build_track_outcomes(
                unsummarized_tracks,
                {},
                frame_to_playback=frame_to_playback,
                last_active_by_track=last_active_by_track,
            )
        )

    _begin_writer_snapshot(writer, run_dir)
    if dataset_snapshot_dirty:
        _set_live_pipeline_step(live_status_reporter, "write_aggregates", frame_count)
        for result in aggregate_results:
            if str(result.status) == "saved":
                with profiler.stage("write_aggregates"):
                    writer.write_aggregate(run_dir, result, save_intensity=save_aggregate_intensity)
        _set_live_pipeline_step(live_status_reporter, "write_object_list_gt", frame_count)
        with _writer_sample_batch(writer):
            with profiler.stage("write_object_list"):
                writer.write_object_list(run_dir, latest_object_labels)
            with profiler.stage("write_gt_matching"):
                writer.write_gt_matching(run_dir, matched_gt, unmatched_saved_tracks, unmatched_gt_objects, gt_match_summary)
    if statistics_enabled:
        with _writer_stats_batch(writer):
            with profiler.stage("write_tracks"):
                writer.write_tracks(run_dir, tracks_to_write, aggregate_results)
                tracker_debug_interval_sec = float(max(0.0, float(config.output.live_tracker_debug_flush_interval_sec)))
                last_tracker_debug_flush = live_artifact_state.get("last_tracker_debug_flush_monotonic")
                should_write_tracker_debug = bool(force)
                if not should_write_tracker_debug:
                    should_write_tracker_debug = last_tracker_debug_flush is None or tracker_debug_interval_sec <= 0.0
                    if not should_write_tracker_debug:
                        should_write_tracker_debug = (now - float(last_tracker_debug_flush)) >= tracker_debug_interval_sec
                if should_write_tracker_debug:
                    writer.write_tracker_debug(run_dir, _live_snapshot_tracker_states(tracker_states))
                    live_artifact_state["last_tracker_debug_flush_monotonic"] = float(now)
                writer.write_track_outcomes(run_dir, track_outcomes_to_write)
                writer.write_class_stats(run_dir, class_stats)
            with profiler.stage("write_summary"):
                writer.write_summary(run_dir, summary)

    live_artifact_state["last_flush_monotonic"] = float(now)
    live_artifact_state["pending_flush"] = False
    live_artifact_state["pending_saved_results"] = {}
    if statistics_enabled:
        _update_live_status_after_artifact_snapshot(live_status_reporter, summary)


def _live_gt_match_label_history(
    tracks: dict[int, Track],
    object_label_history_by_id: dict[int, list[ObjectLabelData]],
    *,
    margin_sec: float,
) -> dict[int, list[ObjectLabelData]]:
    if not tracks:
        return {}
    timestamp_values = [
        int(timestamp)
        for track in tracks.values()
        for timestamp in track.frame_timestamps_ns
    ]
    if not timestamp_values:
        return dict(object_label_history_by_id)
    margin_ns = int(round(max(0.0, float(margin_sec)) * 1_000_000_000))
    min_timestamp_ns = int(min(timestamp_values)) - margin_ns
    max_timestamp_ns = int(max(timestamp_values)) + margin_ns
    scoped: dict[int, list[ObjectLabelData]] = {}
    for object_id, history in object_label_history_by_id.items():
        labels = [
            label
            for label in history
            if min_timestamp_ns <= int(label.timestamp_ns) <= max_timestamp_ns
        ]
        if labels:
            scoped[int(object_id)] = labels
    return scoped


def _live_gt_match_summary(
    matched_gt: list[GTMatchResult],
    unmatched_saved_tracks: list[GTMatchResult],
    unmatched_gt_objects: list[GTMatchResult],
) -> dict[str, int | float | str]:
    saved_results = list(matched_gt) + list(unmatched_saved_tracks)
    matched_deltas = [
        int(match.timestamp_delta_ns)
        for match in saved_results
        if bool(match.matched) and match.timestamp_delta_ns is not None
    ]
    return {
        "gt_match_saved_track_count": int(len(saved_results)),
        "gt_match_matched_count": int(sum(1 for match in saved_results if bool(match.matched))),
        "gt_match_unmatched_saved_count": int(sum(1 for match in saved_results if not bool(match.matched))),
        "gt_match_unmatched_gt_count": int(len(unmatched_gt_objects)),
        "gt_match_mode": "track_center_trajectory",
        "gt_match_assignment": "one_to_one",
        "gt_match_mean_timestamp_delta_ns": float(sum(matched_deltas) / len(matched_deltas)) if matched_deltas else 0.0,
        "gt_match_max_timestamp_delta_ns": int(max(matched_deltas)) if matched_deltas else 0,
    }


def _process_incremental_finished_tracks(
    *,
    finished_tracks: dict[int, Track],
    lane_box,
    postprocessors,
    accumulator,
    classifier,
    class_normalizer: ClassNormalizer,
    frame_to_playback: dict[int, int],
    last_active_by_track: dict[int, dict[str, object]],
    profiler: PerformanceProfiler,
    announced_finished_track_ids: set[int],
    live_snapshot_tracks: dict[int, Track],
    live_snapshot_aggregate_results: dict[int, AggregateResult],
    live_snapshot_track_outcomes: dict[int, TrackOutcomeDebug],
    pending_snapshot_results: dict[int, AggregateResult] | None = None,
    collect_track_outcomes: bool = True,
) -> tuple[dict[int, Track], list[AggregateResult]]:
    new_track_ids = [
        int(track_id)
        for track_id in sorted(int(track_id) for track_id in finished_tracks)
        if int(track_id) not in announced_finished_track_ids
    ]
    if not new_track_ids:
        return {}, []

    new_tracks: dict[int, Track] = {}
    for track_id in new_track_ids:
        track = finished_tracks.get(int(track_id))
        if isinstance(track, Track):
            new_tracks[int(track_id)] = _clone_track(track)
    if not new_tracks:
        announced_finished_track_ids.update(new_track_ids)
        return {}, []

    candidate_tracks = _select_incremental_postprocess_candidates(
        live_snapshot_tracks=live_snapshot_tracks,
        new_tracks=new_tracks,
        postprocessors=postprocessors,
    )
    before_signatures = {
        int(track_id): _incremental_postprocess_signature(track)
        for track_id, track in candidate_tracks.items()
    }
    processed_tracks = candidate_tracks
    for processor in postprocessors:
        with profiler.stage("postprocess_tracks"):
            processed_tracks = processor.process(processed_tracks)

    affected_track_ids = set(int(track_id) for track_id in new_tracks)
    for track_id, track in processed_tracks.items():
        previous_signature = before_signatures.get(int(track_id))
        current_signature = _incremental_postprocess_signature(track)
        if previous_signature != current_signature:
            affected_track_ids.add(int(track_id))
            affected_track_ids.update(_long_vehicle_component_ids(track))
    for track_id in list(affected_track_ids):
        track = processed_tracks.get(int(track_id))
        if track is not None:
            affected_track_ids.update(_long_vehicle_component_ids(track))
    affected_tracks = {
        int(track_id): processed_tracks[int(track_id)]
        for track_id in sorted(affected_track_ids)
        if int(track_id) in processed_tracks
    }
    if not affected_tracks:
        announced_finished_track_ids.update(new_tracks)
        return {}, []

    aggregate_results: list[AggregateResult] = []
    for track in affected_tracks.values():
        with profiler.stage("accumulate_tracks"):
            result = accumulator.accumulate(track, lane_box)
        aggregate_results.append(result)
    if hasattr(accumulator, "merge_long_vehicle_aggregates"):
        with profiler.stage("accumulate_tracks"):
            aggregate_results = accumulator.merge_long_vehicle_aggregates(affected_tracks, aggregate_results, lane_box)
    with profiler.stage("classify_aggregates"):
        aggregate_results = classify_aggregate_results(aggregate_results, classifier, class_normalizer)

    announced_finished_track_ids.update(new_tracks)
    missing_candidate_ids = set(candidate_tracks) - set(processed_tracks)
    for track_id in missing_candidate_ids:
        live_snapshot_tracks.pop(int(track_id), None)
        live_snapshot_aggregate_results.pop(int(track_id), None)
        live_snapshot_track_outcomes.pop(int(track_id), None)
        if pending_snapshot_results is not None:
            pending_snapshot_results.pop(int(track_id), None)
    live_snapshot_tracks.update({int(track_id): _clone_track(track) for track_id, track in processed_tracks.items()})
    for result in aggregate_results:
        live_snapshot_aggregate_results[int(result.track_id)] = result
        if pending_snapshot_results is not None:
            pending_snapshot_results[int(result.track_id)] = result
    if bool(collect_track_outcomes):
        live_snapshot_track_outcomes.update(
            build_track_outcomes(
                affected_tracks,
                aggregate_results,
                frame_to_playback=frame_to_playback,
                last_active_by_track=last_active_by_track,
            )
        )
    return affected_tracks, aggregate_results


def _select_incremental_postprocess_candidates(
    *,
    live_snapshot_tracks: dict[int, Track],
    new_tracks: dict[int, Track],
    postprocessors,
) -> dict[int, Track]:
    candidate_tracks = {int(track_id): _clone_track(track) for track_id, track in new_tracks.items()}
    if not live_snapshot_tracks or not _postprocessors_need_cross_track_context(postprocessors):
        return candidate_tracks

    max_frame_gap = _incremental_candidate_frame_gap(postprocessors)
    new_ranges = [
        frame_range
        for track in new_tracks.values()
        if (frame_range := _track_frame_range(track)) is not None
    ]
    if not new_ranges:
        return candidate_tracks

    for track_id, track in live_snapshot_tracks.items():
        frame_range = _track_frame_range(track)
        if frame_range is None:
            continue
        if any(_frame_ranges_are_close(frame_range, new_range, max_frame_gap=max_frame_gap) for new_range in new_ranges):
            candidate_tracks[int(track_id)] = _clone_track(track)
    return candidate_tracks


def _postprocessors_need_cross_track_context(postprocessors) -> bool:
    cross_track_processors = {
        "articulated_vehicle_merge",
        "co_moving_track_merge",
        "tracklet_stitching",
    }
    return any(str(getattr(processor, "name", "")) in cross_track_processors for processor in postprocessors)


def _incremental_candidate_frame_gap(postprocessors) -> int:
    max_gap = 0
    for processor in postprocessors:
        config = getattr(processor, "config", None)
        for attr_name in (
            "stitching_max_gap",
            "articulated_gap_eval_window_frames",
            "parallel_merge_min_overlap_frames",
        ):
            if config is None or not hasattr(config, attr_name):
                continue
            try:
                max_gap = max(max_gap, int(getattr(config, attr_name)))
            except (TypeError, ValueError):
                continue
    return int(max(0, max_gap))


def _track_frame_range(track: Track) -> tuple[int, int] | None:
    if track.frame_ids:
        return int(min(track.frame_ids)), int(max(track.frame_ids))
    first_frame = int(getattr(track, "first_frame", -1))
    last_frame = int(getattr(track, "last_frame", -1))
    if first_frame < 0 or last_frame < 0:
        return None
    return min(first_frame, last_frame), max(first_frame, last_frame)


def _frame_ranges_are_close(
    left: tuple[int, int],
    right: tuple[int, int],
    *,
    max_frame_gap: int,
) -> bool:
    left_min, left_max = left
    right_min, right_max = right
    if left_min <= right_max and right_min <= left_max:
        return True
    gap = max(left_min - right_max, right_min - left_max)
    return int(gap) <= int(max_frame_gap)


def _incremental_postprocess_signature(track: Track) -> tuple[object, ...]:
    relevant_state = {
        str(key): value
        for key, value in dict(track.state).items()
        if str(key).startswith("articulated_")
        or str(key).startswith("long_vehicle_component")
        or str(key) == "object_kind"
    }
    return (
        None if track.quality_score is None else float(track.quality_score),
        _signature_value(track.quality_metrics),
        _signature_value(relevant_state),
    )


def _long_vehicle_component_ids(track: Track) -> set[int]:
    raw_ids = (
        track.state.get("long_vehicle_component_track_ids")
        or track.state.get("articulated_component_track_ids")
        or []
    )
    component_ids: set[int] = set()
    for raw_id in raw_ids:
        try:
            component_ids.add(int(raw_id))
        except Exception:
            continue
    if component_ids:
        component_ids.add(int(track.track_id))
    return component_ids


def _signature_value(value):
    if isinstance(value, np.ndarray):
        return tuple(_signature_value(item) for item in value.tolist())
    if isinstance(value, dict):
        return tuple((str(key), _signature_value(value[key])) for key in sorted(value))
    if isinstance(value, (list, tuple)):
        return tuple(_signature_value(item) for item in value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _empty_class_statistics() -> dict[str, object]:
    return {
        "predicted_class_counts": {},
        "gt_class_counts": {},
        "matched_gt_class_counts": {},
        "class_comparison_count": 0,
        "class_match_count": 0,
        "class_mismatch_count": 0,
        "class_count_rows": [],
    }


def _build_articulated_vehicle_summary(
    tracks: dict[int, Track],
    aggregate_results: list[AggregateResult],
) -> dict[str, int]:
    pair_ids: set[tuple[int, ...] | int] = set()
    articulated_track_ids: set[int] = set()
    for track_id, track in tracks.items():
        if not bool(track.state.get("articulated_vehicle")):
            continue
        articulated_track_ids.add(int(track_id))
        component_ids = track.state.get("articulated_component_track_ids")
        if component_ids:
            pair_ids.add(tuple(sorted(int(component_id) for component_id in component_ids)))
        elif track.state.get("articulated_pair_id") is not None:
            pair_ids.add(int(track.state.get("articulated_pair_id")))

    merged_component_count = 0
    saved_count = 0
    for result in aggregate_results:
        if str(result.status) == "merged_into_long_vehicle_group":
            merged_component_count += 1
        if str(result.status) == "saved" and bool(result.metrics.get("articulated_vehicle")):
            saved_count += 1

    return {
        "articulated_vehicle_pair_count": int(len(pair_ids)),
        "articulated_vehicle_track_count": int(len(articulated_track_ids)),
        "articulated_vehicle_merged_component_count": int(merged_component_count),
        "articulated_vehicle_saved_count": int(saved_count),
    }


def _build_incremental_live_summary(
    *,
    config: PipelineConfig,
    run_dir: Path,
    postprocessors,
    tracks: dict[int, Track],
    aggregate_results: list[AggregateResult],
    latest_object_labels: dict[int, ObjectLabelData],
    object_list_seen_ids: set[int],
    object_list_skipped_empty: int,
    class_stats: dict[str, object],
    frame_count: int,
    gt_match_summary: dict[str, int | float | str] | None = None,
) -> RunSummary:
    status_counts = Counter(result.status for result in aggregate_results)
    quality_scores = [track.quality_score for track in tracks.values() if track.quality_score is not None]
    registration_attempts = int(sum(int(result.metrics.get("registration_pairs", 0) or 0) for result in aggregate_results))
    registration_accepted = int(sum(int(result.metrics.get("registration_accepted", 0) or 0) for result in aggregate_results))
    registration_rejected = int(sum(int(result.metrics.get("registration_rejected", 0) or 0) for result in aggregate_results))
    input_paths = list(config.input.paths)
    input_path = str(input_paths[0]) if input_paths else ""
    gt_match_summary = {} if gt_match_summary is None else dict(gt_match_summary)
    articulated_summary = _build_articulated_vehicle_summary(tracks, aggregate_results)
    return RunSummary(
        input_path=input_path,
        input_paths=input_paths,
        output_mode=str(config.output.mode),
        tracker_algorithm=config.tracking.algorithm,
        accumulator_algorithm=config.aggregation.algorithm,
        clusterer_algorithm=config.clustering.algorithm,
        frame_count=int(frame_count),
        finished_track_count=int(len(tracks)),
        saved_aggregates=int(sum(1 for result in aggregate_results if str(result.status) == "saved")),
        registration_attempts=registration_attempts,
        registration_accepted=registration_accepted,
        registration_rejected=registration_rejected,
        output_dir=str(run_dir),
        postprocessing_methods=[processor.name for processor in postprocessors],
        aggregate_status_counts=dict(status_counts),
        **articulated_summary,
        track_quality_mean=float(sum(quality_scores) / len(quality_scores)) if quality_scores else 0.0,
        object_list_exported_count=int(len(latest_object_labels)),
        object_list_seen_ids=int(len(object_list_seen_ids)),
        object_list_skipped_empty=int(object_list_skipped_empty),
        gt_match_saved_track_count=int(gt_match_summary.get("gt_match_saved_track_count", 0)),
        gt_match_matched_count=int(gt_match_summary.get("gt_match_matched_count", 0)),
        gt_match_unmatched_saved_count=int(gt_match_summary.get("gt_match_unmatched_saved_count", 0)),
        gt_match_unmatched_gt_count=int(gt_match_summary.get("gt_match_unmatched_gt_count", 0)),
        gt_match_mode=str(gt_match_summary.get("gt_match_mode", "")),
        gt_match_assignment=str(gt_match_summary.get("gt_match_assignment", "")),
        gt_match_mean_timestamp_delta_ns=float(gt_match_summary.get("gt_match_mean_timestamp_delta_ns", 0.0)),
        gt_match_max_timestamp_delta_ns=int(gt_match_summary.get("gt_match_max_timestamp_delta_ns", 0)),
        predicted_class_counts=dict(class_stats["predicted_class_counts"]),
        gt_class_counts=dict(class_stats["gt_class_counts"]),
        matched_gt_class_counts=dict(class_stats["matched_gt_class_counts"]),
        class_comparison_count=int(class_stats["class_comparison_count"]),
        class_match_count=int(class_stats["class_match_count"]),
        class_mismatch_count=int(class_stats["class_mismatch_count"]),
        class_count_rows=[dict(row) for row in class_stats["class_count_rows"]],
        performance=None,
    )


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
    with _writer_sample_batch(writer):
        with profiler.stage("write_object_list"):
            writer.write_object_list(run_dir, latest_object_labels)
        with profiler.stage("write_gt_matching"):
            writer.write_gt_matching(run_dir, matched_gt, unmatched_saved_tracks, unmatched_gt_objects, gt_match_summary)
    with _writer_stats_batch(writer):
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
        **_build_articulated_vehicle_summary(tracks, aggregate_results),
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
    with _writer_stats_batch(writer):
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


@contextmanager
def _writer_sample_batch(writer):
    begin_sample_batch = getattr(writer, "begin_sample_batch", None)
    end_sample_batch = getattr(writer, "end_sample_batch", None)
    if callable(begin_sample_batch):
        begin_sample_batch()
    try:
        yield
    finally:
        if callable(end_sample_batch):
            end_sample_batch()


@contextmanager
def _writer_stats_batch(writer):
    begin_stats_batch = getattr(writer, "begin_stats_batch", None)
    end_stats_batch = getattr(writer, "end_stats_batch", None)
    if callable(begin_stats_batch):
        begin_stats_batch()
    try:
        yield
    finally:
        if callable(end_stats_batch):
            end_stats_batch()


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
        return {int(track_id): _clone_track_metadata(track) for track_id, track in sorted(dict(tracks).items())}

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
    return {int(track_id): _clone_track_metadata(track) for track_id, track in sorted(combined.items())}


def _clone_track_metadata(track: Track) -> Track:
    cloned_state = {
        str(key): copy.deepcopy(value)
        for key, value in dict(track.state).items()
        if str(key) != "kf"
    }
    return Track(
        track_id=int(track.track_id),
        centers=[] if not track.centers else [np.asarray(track.centers[-1], dtype=np.float32).copy()],
        frame_ids=[int(frame_id) for frame_id in track.frame_ids],
        frame_timestamps_ns=[int(timestamp_ns) for timestamp_ns in track.frame_timestamps_ns],
        bbox_extents=[] if not track.bbox_extents else [np.asarray(track.bbox_extents[-1], dtype=np.float32).copy()],
        hit_count=int(track.hit_count),
        age=int(track.age),
        missed=int(track.missed),
        ended_by_missed=bool(track.ended_by_missed),
        source_track_ids=[int(track_id) for track_id in track.source_track_ids],
        quality_score=None if track.quality_score is None else float(track.quality_score),
        quality_metrics=copy.deepcopy(track.quality_metrics),
        state=cloned_state,
    )


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


def _snapshot_pending_object_labels(reader, frame_index: int) -> list[ObjectLabelData]:
    snapshot = getattr(reader, "snapshot_pending_object_labels", None)
    if not callable(snapshot):
        return []
    return list(snapshot(frame_index))


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
        _upsert_object_label_history(object_label_history_by_id, normalized_object_label)
        current = latest_object_labels.get(int(object_label.object_id))
        if _is_newer_object_label(normalized_object_label, current):
            latest_object_labels[int(object_label.object_id)] = normalized_object_label
            updated = True
    return skipped_empty, updated


def _refresh_latest_object_labels(
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
        _upsert_object_label_history(object_label_history_by_id, normalized_object_label)
        current = latest_object_labels.get(int(object_label.object_id))
        if _is_newer_object_label(normalized_object_label, current):
            latest_object_labels[int(object_label.object_id)] = normalized_object_label
            updated = True
    return skipped_empty, updated


def _upsert_object_label_history(
    object_label_history_by_id: dict[int, list[ObjectLabelData]],
    object_label: ObjectLabelData,
) -> None:
    history = object_label_history_by_id.setdefault(int(object_label.object_id), [])
    if not history:
        history.append(object_label)
        return

    last_label = history[-1]
    last_timestamp_ns = int(last_label.timestamp_ns)
    current_timestamp_ns = int(object_label.timestamp_ns)
    if current_timestamp_ns < last_timestamp_ns:
        return
    if current_timestamp_ns == last_timestamp_ns:
        if int(object_label.frame_index) <= int(last_label.frame_index):
            return
        history[-1] = object_label
        return
    history.append(object_label)


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
