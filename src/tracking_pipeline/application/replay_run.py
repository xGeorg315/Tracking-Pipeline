from __future__ import annotations

from pathlib import Path

from tracking_pipeline.application.factories import (
    build_accumulator,
    build_clusterer,
    build_lane_box,
    build_reader,
    build_track_postprocessors,
    build_tracker,
    build_viewer,
)
from tracking_pipeline.config.models import PipelineConfig


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
        state.lane_points = cluster_result.lane_points
        state.lane_intensity = cluster_result.lane_intensity
        state.detections = cluster_result.detections
        state.cluster_metrics = cluster_result.metrics
        states.append(state)

    tracks = tracker.finalize()
    for processor in postprocessors:
        tracks = processor.process(tracks)
    aggregate_results = {track_id: accumulator.accumulate(track, lane_box) for track_id, track in tracks.items()}
    viewer.replay(states, lane_box, aggregate_results)
