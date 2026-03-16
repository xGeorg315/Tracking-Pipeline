from __future__ import annotations

from collections import Counter
from pathlib import Path

from tracking_pipeline.application.factories import (
    build_accumulator,
    build_artifact_writer,
    build_clusterer,
    build_lane_box,
    build_reader,
    build_track_postprocessors,
    build_tracker,
)
from tracking_pipeline.application.performance import PerformanceProfiler
from tracking_pipeline.config.models import PipelineConfig
from tracking_pipeline.domain.models import AggregateResult, ObjectLabelData, RunSummary


def run_pipeline(config: PipelineConfig, project_root: Path) -> RunSummary:
    profiler = PerformanceProfiler()
    with profiler.stage("build_components"):
        lane_box = build_lane_box(config)
        reader = build_reader(config)
        clusterer = build_clusterer(config)
        tracker = build_tracker(config)
        postprocessors = build_track_postprocessors(config)
        accumulator = build_accumulator(config)
        writer = build_artifact_writer(project_root)

    with profiler.stage("prepare_output"):
        run_dir = writer.prepare_run_dir(config)
        writer.write_config_snapshot(run_dir, config)

    latest_object_labels: dict[int, ObjectLabelData] = {}
    object_list_seen_ids: set[int] = set()
    object_list_skipped_empty = 0
    with profiler.stage("read_frames"):
        frames = reader.iter_frames(config.input.paths)
    for frame in frames:
        for object_label in frame.object_labels:
            object_list_seen_ids.add(int(object_label.object_id))
            if len(object_label.points) == 0:
                object_list_skipped_empty += 1
                continue
            current = latest_object_labels.get(int(object_label.object_id))
            if _is_newer_object_label(object_label, current):
                latest_object_labels[int(object_label.object_id)] = object_label
        with profiler.stage("cluster_frames"):
            cluster_result = clusterer.cluster(frame, lane_box)
        with profiler.stage("tracker_steps"):
            tracker.step(cluster_result.detections, frame.frame_index)

    with profiler.stage("tracker_finalize"):
        tracks = tracker.finalize()
    for processor in postprocessors:
        with profiler.stage("postprocess_tracks"):
            tracks = processor.process(tracks)

    aggregate_results: list[AggregateResult] = []
    registration_attempts = 0
    registration_accepted = 0
    registration_rejected = 0

    for track in tracks.values():
        with profiler.stage("accumulate_tracks"):
            result = accumulator.accumulate(track, lane_box)
        aggregate_results.append(result)
        metrics = result.metrics
        registration_attempts += int(metrics.get("registration_pairs", 0))
        registration_accepted += int(metrics.get("registration_accepted", 0))
        registration_rejected += int(metrics.get("registration_rejected", 0))
        if result.status == "saved":
            with profiler.stage("write_aggregates"):
                writer.write_aggregate(run_dir, result, save_intensity=config.output.save_aggregate_intensity)

    with profiler.stage("write_object_list"):
        writer.write_object_list(run_dir, latest_object_labels)

    status_counts = Counter(result.status for result in aggregate_results)
    quality_scores = [track.quality_score for track in tracks.values() if track.quality_score is not None]
    summary = RunSummary(
        input_path=config.input.paths[0],
        input_paths=list(config.input.paths),
        tracker_algorithm=config.tracking.algorithm,
        accumulator_algorithm=config.aggregation.algorithm,
        clusterer_algorithm=config.clustering.algorithm,
        frame_count=len(frames),
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
        performance=profiler.snapshot(),
    )
    with profiler.stage("write_tracks"):
        writer.write_tracks(run_dir, tracks, aggregate_results)
    summary.performance = profiler.snapshot()
    with profiler.stage("write_summary"):
        writer.write_summary(run_dir, summary)
    # Persist the final profile snapshot after measuring the summary write itself.
    summary.performance = profiler.snapshot()
    writer.write_summary(run_dir, summary)
    return summary


def _is_newer_object_label(candidate: ObjectLabelData, current: ObjectLabelData | None) -> bool:
    if current is None:
        return True
    if int(candidate.timestamp_ns) != int(current.timestamp_ns):
        return int(candidate.timestamp_ns) > int(current.timestamp_ns)
    return int(candidate.frame_index) > int(current.frame_index)
