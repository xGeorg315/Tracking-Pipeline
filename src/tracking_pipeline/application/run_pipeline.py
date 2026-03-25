from __future__ import annotations

from collections import Counter
from pathlib import Path

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
from tracking_pipeline.config.models import PipelineConfig
from tracking_pipeline.domain.models import AggregateResult, ObjectLabelData, RunPerformance, RunSummary


def run_pipeline(config: PipelineConfig, project_root: Path) -> RunSummary:
    profiler = PerformanceProfiler()
    class_normalizer = ClassNormalizer.from_config(config.class_normalization)
    with profiler.stage("build_components"):
        lane_box = build_lane_box(config)
        reader = build_reader(config)
        clusterer = build_clusterer(config)
        tracker = build_tracker(config)
        postprocessors = build_track_postprocessors(config)
        accumulator = build_accumulator(config)
        classifier = build_classifier(config)
        writer = build_artifact_writer(project_root)

    with profiler.stage("prepare_output"):
        run_dir = writer.prepare_run_dir(config)
        writer.write_config_snapshot(run_dir, config)

    latest_object_labels: dict[int, ObjectLabelData] = {}
    object_list_seen_ids: set[int] = set()
    object_list_skipped_empty = 0
    tracker_states = []
    with profiler.stage("read_frames"):
        frames = reader.iter_frames(config.input.paths)
    for frame in frames:
        for object_label in frame.object_labels:
            object_list_seen_ids.add(int(object_label.object_id))
            if len(object_label.points) == 0:
                object_list_skipped_empty += 1
                continue
            normalized_object_label = class_normalizer.normalize_object_label(object_label)
            current = latest_object_labels.get(int(object_label.object_id))
            if _is_newer_object_label(normalized_object_label, current):
                latest_object_labels[int(object_label.object_id)] = normalized_object_label
        with profiler.stage("cluster_frames"):
            cluster_result = clusterer.cluster(frame, lane_box)
        with profiler.stage("tracker_steps"):
            state = tracker.step(cluster_result.detections, frame.frame_index, frame.timestamp_ns)
        state.cluster_metrics = cluster_result.metrics
        tracker_states.append(state)

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
            latest_object_labels,
            class_normalizer,
        )
        apply_gt_matches_to_results(aggregate_results, matched_gt, unmatched_saved_tracks)
    class_stats = build_class_statistics(aggregate_results, latest_object_labels, class_normalizer)
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
            len(frames),
        ),
    )
    with profiler.stage("write_tracks"):
        writer.write_tracks(run_dir, tracks, aggregate_results)
        writer.write_tracker_debug(run_dir, tracker_states)
        writer.write_track_outcomes(run_dir, track_outcomes)
        writer.write_class_stats(run_dir, class_stats)
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
    return summary


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
