from __future__ import annotations

from pathlib import Path

import numpy as np

from tracking_pipeline.application.factories import (
    build_accumulator,
    build_clusterer,
    build_lane_box,
    build_reader,
    build_track_postprocessors,
    build_tracker,
    build_viewer,
)
from tracking_pipeline.application.track_outcomes import build_track_outcomes
from tracking_pipeline.config.models import PipelineConfig
from tracking_pipeline.domain.models import ArticulatedMergeDebugEvent, FrameTrackingState
from tracking_pipeline.infrastructure.postprocessing.articulated_vehicle_merge import ArticulatedVehicleMergePostprocessor


def replay_run(config: PipelineConfig, project_root: Path) -> None:
    _ = project_root
    lane_box = build_lane_box(config)
    reader = build_reader(config)
    clusterer = build_clusterer(config)
    tracker = build_tracker(config)
    postprocessors = build_track_postprocessors(config)
    accumulator = build_accumulator(config)
    viewer = build_viewer(config)

    states = []
    frames = reader.iter_frames(config.input.paths)
    for frame in frames:
        cluster_result = clusterer.cluster(frame, lane_box)
        state = tracker.step(cluster_result.detections, frame.frame_index, frame.timestamp_ns)
        state.full_frame_points = frame.points
        state.full_frame_intensity = frame.point_intensity
        state.lane_points = cluster_result.lane_points
        state.lane_intensity = cluster_result.lane_intensity
        state.detections = cluster_result.detections
        state.cluster_metrics = cluster_result.metrics
        states.append(state)

    tracks = tracker.finalize()
    articulated_merge_debug_events: list[ArticulatedMergeDebugEvent] = []
    for processor in postprocessors:
        tracks = processor.process(tracks)
        if isinstance(processor, ArticulatedVehicleMergePostprocessor):
            articulated_merge_debug_events.extend(_build_articulated_merge_debug_events(states, processor.debug_records))
    aggregate_results = [accumulator.accumulate(track, lane_box) for track in tracks.values()]
    if hasattr(accumulator, "merge_long_vehicle_aggregates"):
        aggregate_results = accumulator.merge_long_vehicle_aggregates(tracks, aggregate_results, lane_box)
    aggregate_result_map = {int(result.track_id): result for result in aggregate_results}
    track_outcomes = build_track_outcomes(tracks, aggregate_result_map, states)
    viewer.replay(states, lane_box, aggregate_result_map, track_outcomes, articulated_merge_debug_events)


def _build_articulated_merge_debug_events(
    states: list[FrameTrackingState],
    debug_records,
) -> list[ArticulatedMergeDebugEvent]:
    frame_to_playback = {int(state.frame_index): int(index) for index, state in enumerate(states)}
    events: list[ArticulatedMergeDebugEvent] = []
    for record in debug_records:
        playback_start_index = int(frame_to_playback.get(int(record.overlap_start_frame_id), -1))
        playback_end_index = int(frame_to_playback.get(int(record.overlap_end_frame_id), -1))
        if playback_start_index < 0 or playback_end_index < 0:
            continue
        center = _merge_debug_center(states, int(record.lead_track_id), int(record.rear_track_id), playback_end_index)
        events.append(
            ArticulatedMergeDebugEvent(
                lead_track_id=int(record.lead_track_id),
                rear_track_id=int(record.rear_track_id),
                accepted=bool(record.accepted),
                rejection_reason=str(record.rejection_reason),
                playback_start_index=playback_start_index,
                playback_end_index=playback_end_index,
                full_gap_mean=float(record.full_gap_mean),
                full_gap_std=float(record.full_gap_std),
                tail_gap_mean=float(record.tail_gap_mean),
                tail_gap_std=float(record.tail_gap_std),
                tail_window_frame_count=int(record.tail_window_frame_count),
                mean_lateral_offset=float(record.mean_lateral_offset),
                mean_vertical_offset=float(record.mean_vertical_offset),
                center=center,
            )
        )
    return events


def _merge_debug_center(
    states: list[FrameTrackingState],
    lead_track_id: int,
    rear_track_id: int,
    playback_end_index: int,
):
    if playback_end_index < 0 or playback_end_index >= len(states):
        return None
    state = states[playback_end_index]
    centers = []
    for active_track in state.active_tracks:
        if int(active_track.track_id) in {int(lead_track_id), int(rear_track_id)}:
            centers.append(active_track.center)
    if not centers:
        return None
    return np.mean(np.asarray(centers, dtype=np.float32), axis=0)
