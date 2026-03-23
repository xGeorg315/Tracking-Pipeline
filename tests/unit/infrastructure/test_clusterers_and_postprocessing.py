from __future__ import annotations

import numpy as np
import pytest

from tracking_pipeline.config.models import ClusteringConfig, PostprocessingConfig
from tracking_pipeline.domain.models import FrameData, LidarScanData, SensorCalibrationData, Track
from tracking_pipeline.domain.value_objects import LaneBox
from tracking_pipeline.infrastructure.clustering.beam_neighbor_region_growing import BeamNeighborRegionGrowingClusterer
from tracking_pipeline.infrastructure.clustering.euclidean_clustering import EuclideanClusteringClusterer
from tracking_pipeline.infrastructure.clustering.ground_removed_dbscan import GroundRemovedDBSCANClusterer
from tracking_pipeline.infrastructure.clustering.hdbscan_clusterer import HDBSCANClusterer
from tracking_pipeline.infrastructure.clustering.range_image_connected_components import RangeImageConnectedComponentsClusterer
from tracking_pipeline.infrastructure.clustering.range_image_depth_jump import RangeImageDepthJumpClusterer
from tracking_pipeline.infrastructure.postprocessing.articulated_vehicle_merge import ArticulatedVehicleMergePostprocessor
from tracking_pipeline.infrastructure.postprocessing.co_moving_track_merge import CoMovingTrackMergePostprocessor
from tracking_pipeline.infrastructure.postprocessing.track_quality_scoring import TrackQualityScoringPostprocessor
from tracking_pipeline.infrastructure.postprocessing.tracklet_stitching import TrackletStitchingPostprocessor
from tracking_pipeline.infrastructure.postprocessing.trajectory_smoothing import TrajectorySmoothingPostprocessor


def _cluster_frame() -> FrameData:
    cluster_a = np.random.default_rng(4).normal(loc=[0.0, 0.0, 1.0], scale=0.03, size=(30, 3)).astype(np.float32)
    cluster_b = np.random.default_rng(5).normal(loc=[1.5, 1.5, 1.1], scale=0.03, size=(30, 3)).astype(np.float32)
    points = np.vstack([cluster_a, cluster_b]).astype(np.float32)
    point_intensity = np.linspace(0.0, 1.0, len(points), dtype=np.float32)
    point_timestamp_ns = np.arange(10_000, 10_000 + len(points), dtype=np.int64)
    return FrameData(frame_index=0, timestamp_ns=1, points=points, point_intensity=point_intensity, point_timestamp_ns=point_timestamp_ns)


def _ground_frame() -> FrameData:
    xs = np.linspace(-1.0, 1.0, 12, dtype=np.float32)
    ys = np.linspace(0.0, 3.0, 12, dtype=np.float32)
    ground = np.array([[x, y, 0.0] for x in xs for y in ys], dtype=np.float32)
    object_points = np.random.default_rng(6).normal(loc=[0.2, 1.5, 1.0], scale=0.03, size=(40, 3)).astype(np.float32)
    return FrameData(frame_index=0, timestamp_ns=1, points=np.vstack([ground, object_points]).astype(np.float32))


def _sensor_frame() -> FrameData:
    calibration = SensorCalibrationData(
        sensor_name="sensor_a",
        vertical_scanlines=4,
        horizontal_scanlines=8,
        beam_altitude_angles=np.array([-0.3, -0.1, 0.1, 0.3], dtype=np.float32),
    )
    rows = np.array([2, 2, 2, 2, 2, 2], dtype=np.int32)
    cols = np.array([0, 1, 2, 5, 6, 7], dtype=np.int32)
    xyz = np.array(
        [
            [5.0, 0.0, 1.0],
            [5.1, 0.1, 1.0],
            [5.2, 0.2, 1.0],
            [10.0, 2.0, 1.0],
            [10.1, 2.1, 1.0],
            [10.2, 2.2, 1.0],
        ],
        dtype=np.float32,
    )
    ranges = np.linalg.norm(xyz, axis=1).astype(np.float32)
    scan = LidarScanData(
        sensor_name="sensor_a",
        timestamp_ns=1,
        xyz=xyz,
        ranges=ranges,
        row_index=rows,
        col_index=cols,
        calibration=calibration,
        intensity=np.linspace(0.1, 0.9, len(xyz), dtype=np.float32),
        point_timestamp_ns=np.arange(20_000, 20_000 + len(xyz), dtype=np.int64),
    )
    return FrameData(
        frame_index=0,
        timestamp_ns=1,
        points=xyz,
        point_intensity=np.linspace(0.1, 0.9, len(xyz), dtype=np.float32),
        point_timestamp_ns=np.arange(20_000, 20_000 + len(xyz), dtype=np.int64),
        scans=[scan],
    )


def _track(track_id: int, frame_ids: list[int], centers: list[np.ndarray]) -> Track:
    track = Track(track_id=track_id, age=len(frame_ids), hit_count=len(frame_ids), ended_by_missed=True)
    for frame_id, center in zip(frame_ids, centers):
        points = np.asarray([center, center + np.array([0.1, 0.0, 0.0], dtype=np.float32)], dtype=np.float32)
        track.centers.append(center.astype(np.float32))
        track.frame_ids.append(frame_id)
        track.world_points.append(points.copy())
        track.local_points.append(points - center)
        track.bbox_extents.append(np.array([0.1, 0.1, 0.1], dtype=np.float32))
    return track


def _box_points(center: np.ndarray, *, width: float, length: float, height: float) -> np.ndarray:
    signs = np.asarray(
        [
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    scale = np.asarray([width * 0.5, length * 0.5, height * 0.5], dtype=np.float32)
    return center.astype(np.float32) + signs * scale


def _articulated_track(
    track_id: int,
    frame_ids: list[int],
    longitudinal_centers: list[float],
    *,
    lateral_center: float = 0.0,
    vertical_center: float = 1.0,
    width: float = 0.9,
    length: float = 3.4,
    height: float = 1.4,
) -> Track:
    track = Track(track_id=track_id, age=len(frame_ids), hit_count=len(frame_ids), ended_by_missed=True, source_track_ids=[track_id])
    extent = np.asarray([width, length, height], dtype=np.float32)
    for frame_id, longitudinal_center in zip(frame_ids, longitudinal_centers):
        center = np.asarray([lateral_center, longitudinal_center, vertical_center], dtype=np.float32)
        points = _box_points(center, width=width, length=length, height=height)
        intensity = np.linspace(float(track_id), float(track_id) + 0.7, len(points), dtype=np.float32)
        point_timestamps_ns = np.arange(frame_id * 1000, frame_id * 1000 + len(points), dtype=np.int64)
        track.add_observation(
            center,
            points,
            frame_id,
            frame_id * 1_000_000,
            extent,
            intensity=intensity,
            point_timestamp_ns=point_timestamps_ns,
        )
    track.age = max(track.age, track.last_frame - track.first_frame + 1)
    track.hit_count = len(track.frame_ids)
    return track


def test_euclidean_clusterer_detects_two_clusters() -> None:
    frame = _cluster_frame()
    lane_box = LaneBox.from_values([-2.0, 3.0, -2.0, 3.0, 0.0, 2.0])
    clusterer = EuclideanClusteringClusterer(
        ClusteringConfig(algorithm="euclidean_clustering", eps=0.25, vehicle_min_points=10, vehicle_max_points=100)
    )
    result = clusterer.cluster(frame, lane_box)
    assert len(result.detections) == 2
    assert result.metrics["accepted_cluster_count"] == 2
    assert result.lane_intensity is not None
    assert all(detection.intensity is not None for detection in result.detections)
    assert all(detection.point_timestamp_ns is not None for detection in result.detections)


def test_ground_removed_dbscan_reports_removed_ground_points() -> None:
    frame = _ground_frame()
    lane_box = LaneBox.from_values([-2.0, 2.0, -1.0, 4.0, -0.5, 2.0])
    clusterer = GroundRemovedDBSCANClusterer(
        ClusteringConfig(
            algorithm="ground_removed_dbscan",
            eps=0.18,
            min_points=8,
            vehicle_min_points=10,
            plane_distance_threshold=0.03,
            plane_num_iterations=80,
        )
    )
    result = clusterer.cluster(frame, lane_box)
    assert result.metrics["ground_removed_count"] > 0
    assert len(result.detections) >= 1


def test_hdbscan_clusterer_requires_optional_dependency_when_missing() -> None:
    clusterer = HDBSCANClusterer(ClusteringConfig(algorithm="hdbscan", vehicle_min_points=5, hdbscan_min_cluster_size=5))
    if clusterer._hdbscan is not None:
        pytest.skip("Optional hdbscan dependency is installed in this environment")
    frame = _cluster_frame()
    lane_box = LaneBox.from_values([-2.0, 3.0, -2.0, 3.0, 0.0, 2.0])
    with pytest.raises(RuntimeError, match="hdbscan is not installed"):
        clusterer.cluster(frame, lane_box)


def test_range_image_connected_components_detects_two_sensor_clusters() -> None:
    frame = _sensor_frame()
    lane_box = LaneBox.from_values([0.0, 12.0, -1.0, 4.0, 0.0, 2.0])
    clusterer = RangeImageConnectedComponentsClusterer(
        ClusteringConfig(
            algorithm="range_image_connected_components",
            vehicle_min_points=2,
            sensor_min_component_size=2,
            sensor_neighbor_rows=0,
            sensor_neighbor_cols=1,
            sensor_depth_jump_abs=1.0,
            sensor_depth_jump_ratio=0.1,
        )
    )
    result = clusterer.cluster(frame, lane_box)
    assert len(result.detections) == 2
    assert result.metrics["sensor_component_count_kept"] == 2


def test_range_image_depth_jump_splits_large_range_gap() -> None:
    frame = _sensor_frame()
    lane_box = LaneBox.from_values([0.0, 12.0, -1.0, 4.0, 0.0, 2.0])
    clusterer = RangeImageDepthJumpClusterer(
        ClusteringConfig(
            algorithm="range_image_depth_jump",
            vehicle_min_points=2,
            sensor_min_component_size=2,
            sensor_neighbor_rows=0,
            sensor_neighbor_cols=1,
            sensor_depth_jump_abs=0.2,
            sensor_depth_jump_ratio=0.01,
        )
    )
    result = clusterer.cluster(frame, lane_box)
    assert len(result.detections) == 2


def test_beam_neighbor_region_growing_uses_sensor_neighbors() -> None:
    frame = _sensor_frame()
    lane_box = LaneBox.from_values([0.0, 12.0, -1.0, 4.0, 0.0, 2.0])
    clusterer = BeamNeighborRegionGrowingClusterer(
        ClusteringConfig(
            algorithm="beam_neighbor_region_growing",
            eps=0.6,
            vehicle_min_points=2,
            sensor_min_component_size=2,
            sensor_neighbor_rows=0,
            sensor_neighbor_cols=1,
        )
    )
    result = clusterer.cluster(frame, lane_box)
    assert len(result.detections) == 2


def test_tracklet_stitching_merges_adjacent_tracks() -> None:
    left = _track(1, [0, 1], [np.array([0.0, 0.0, 1.0], dtype=np.float32), np.array([0.4, 0.0, 1.0], dtype=np.float32)])
    right = _track(2, [3, 4], [np.array([0.8, 0.0, 1.0], dtype=np.float32), np.array([1.2, 0.0, 1.0], dtype=np.float32)])
    processor = TrackletStitchingPostprocessor(PostprocessingConfig(enable_tracklet_stitching=True, stitching_max_gap=3, stitching_max_center_dist=0.6))
    merged = processor.process({1: left, 2: right})
    assert list(merged.keys()) == [1]
    assert merged[1].source_track_ids == [1, 2]
    assert merged[1].hit_count == 4


def test_trajectory_smoothing_updates_local_points() -> None:
    track = _track(
        3,
        [0, 1, 2],
        [
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
            np.array([1.0, 0.0, 1.0], dtype=np.float32),
            np.array([2.0, 0.0, 1.0], dtype=np.float32),
        ],
    )
    original_local = [points.copy() for points in track.local_points]
    processor = TrajectorySmoothingPostprocessor(PostprocessingConfig(enable_trajectory_smoothing=True, smoothing_window=3))
    smoothed = processor.process({track.track_id: track})[track.track_id]
    assert np.allclose(smoothed.centers[1], np.array([1.0, 0.0, 1.0], dtype=np.float32))
    assert not np.allclose(smoothed.local_points[0], original_local[0])


def test_co_moving_track_merge_merges_parallel_tracks() -> None:
    left = _track(
        10,
        [0, 1, 2, 3],
        [
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
            np.array([0.0, 1.0, 1.0], dtype=np.float32),
            np.array([0.0, 2.0, 1.0], dtype=np.float32),
            np.array([0.0, 3.0, 1.0], dtype=np.float32),
        ],
    )
    right = _track(
        11,
        [0, 1, 2, 3],
        [
            np.array([0.4, 0.7, 1.0], dtype=np.float32),
            np.array([0.4, 1.7, 1.0], dtype=np.float32),
            np.array([0.4, 2.7, 1.0], dtype=np.float32),
            np.array([0.4, 3.7, 1.0], dtype=np.float32),
        ],
    )
    processor = CoMovingTrackMergePostprocessor(
        PostprocessingConfig(
            enable_co_moving_track_merge=True,
            parallel_merge_min_overlap_frames=3,
            parallel_merge_min_overlap_ratio=0.7,
            parallel_merge_max_lateral_offset=0.8,
            parallel_merge_max_longitudinal_gap=1.2,
        ),
        longitudinal_axis="y",
    )
    merged = processor.process({left.track_id: left, right.track_id: right})
    assert list(merged.keys()) == [10]
    assert merged[10].source_track_ids == [10, 11]
    assert merged[10].hit_count == 4


def test_co_moving_track_merge_rejects_large_lateral_offset() -> None:
    left = _track(
        12,
        [0, 1, 2, 3],
        [
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
            np.array([0.0, 1.0, 1.0], dtype=np.float32),
            np.array([0.0, 2.0, 1.0], dtype=np.float32),
            np.array([0.0, 3.0, 1.0], dtype=np.float32),
        ],
    )
    right = _track(
        13,
        [0, 1, 2, 3],
        [
            np.array([1.5, 0.7, 1.0], dtype=np.float32),
            np.array([1.5, 1.7, 1.0], dtype=np.float32),
            np.array([1.5, 2.7, 1.0], dtype=np.float32),
            np.array([1.5, 3.7, 1.0], dtype=np.float32),
        ],
    )
    processor = CoMovingTrackMergePostprocessor(
        PostprocessingConfig(
            enable_co_moving_track_merge=True,
            parallel_merge_min_overlap_frames=3,
            parallel_merge_min_overlap_ratio=0.7,
            parallel_merge_max_lateral_offset=0.8,
            parallel_merge_max_longitudinal_gap=1.2,
        ),
        longitudinal_axis="y",
    )
    merged = processor.process({left.track_id: left, right.track_id: right})
    assert set(merged.keys()) == {12, 13}


def test_articulated_vehicle_merge_merges_front_and_rear_tracks() -> None:
    front = _articulated_track(21, [0, 1, 2, 3], [10.0, 11.0, 12.0, 13.0], lateral_center=0.0)
    rear = _articulated_track(22, [0, 1, 2, 3], [6.8, 7.8, 8.8, 9.8], lateral_center=0.1)
    processor = ArticulatedVehicleMergePostprocessor(PostprocessingConfig(enable_articulated_vehicle_merge=True), longitudinal_axis="y")

    merged = processor.process({front.track_id: front, rear.track_id: rear})

    assert list(merged.keys()) == [21]
    merged_track = merged[21]
    assert merged_track.source_track_ids == [21, 22]
    assert merged_track.hit_count == 4
    assert merged_track.state["articulated_vehicle"] is True
    assert merged_track.state["articulated_component_track_ids"] == [21, 22]
    assert merged_track.state["object_kind"] == "truck_with_trailer"
    assert merged_track.state["articulated_rear_gap_mean"] >= -0.5
    assert merged_track.state["articulated_rear_gap_mean"] <= 0.1
    assert merged_track.state["articulated_rear_gap_std"] < 0.1
    assert all(len(points) == 16 for points in merged_track.world_points)
    assert processor.debug_records[-1].accepted is True
    assert processor.debug_records[-1].rejection_reason == "tail_gap"


def test_articulated_vehicle_merge_rejects_side_by_side_neighbor() -> None:
    left = _articulated_track(23, [0, 1, 2, 3], [10.0, 11.0, 12.0, 13.0], lateral_center=-0.3)
    right = _articulated_track(24, [0, 1, 2, 3], [10.2, 11.2, 12.2, 13.2], lateral_center=0.3)
    processor = ArticulatedVehicleMergePostprocessor(PostprocessingConfig(enable_articulated_vehicle_merge=True), longitudinal_axis="y")

    merged = processor.process({left.track_id: left, right.track_id: right})

    assert set(merged.keys()) == {23, 24}


def test_articulated_vehicle_merge_rejects_large_or_unstable_hitch_gap() -> None:
    front = _articulated_track(25, [0, 1, 2, 3], [10.0, 11.0, 12.0, 13.0], lateral_center=0.0)
    far_rear = _articulated_track(26, [0, 1, 2, 3], [4.0, 5.0, 6.0, 7.0], lateral_center=0.1)
    unstable_rear = _articulated_track(27, [0, 1, 2, 3], [6.8, 8.6, 7.6, 10.7], lateral_center=0.1)
    processor = ArticulatedVehicleMergePostprocessor(PostprocessingConfig(enable_articulated_vehicle_merge=True), longitudinal_axis="y")

    merged_far = processor.process({front.track_id: front, far_rear.track_id: far_rear})
    merged_unstable = processor.process({front.track_id: front, unstable_rear.track_id: unstable_rear})

    assert set(merged_far.keys()) == {25, 26}
    assert set(merged_unstable.keys()) == {25, 27}


def test_articulated_vehicle_merge_rejects_large_vertical_offset() -> None:
    front = _articulated_track(28, [0, 1, 2, 3], [10.0, 11.0, 12.0, 13.0], lateral_center=0.0, vertical_center=1.0)
    rear = _articulated_track(29, [0, 1, 2, 3], [6.8, 7.8, 8.8, 9.8], lateral_center=0.1, vertical_center=1.9)
    processor = ArticulatedVehicleMergePostprocessor(PostprocessingConfig(enable_articulated_vehicle_merge=True), longitudinal_axis="y")

    merged = processor.process({front.track_id: front, rear.track_id: rear})

    assert set(merged.keys()) == {28, 29}


def test_articulated_vehicle_merge_uses_tail_window_for_gap_acceptance() -> None:
    front = _articulated_track(32, [0, 1, 2, 3, 4, 5], [10.0, 11.0, 12.0, 13.0, 14.0, 15.0], lateral_center=0.0, length=5.0)
    rear = _articulated_track(33, [0, 1, 2, 3, 4, 5], [2.5, 4.8, 5.8, 6.8, 7.8, 8.8], lateral_center=0.1, length=5.0)
    processor = ArticulatedVehicleMergePostprocessor(
        PostprocessingConfig(
            enable_articulated_vehicle_merge=True,
            articulated_gap_eval_window_frames=5,
            articulated_max_hitch_gap=1.25,
            articulated_max_hitch_gap_std=0.6,
            articulated_min_combined_length=8.0,
        ),
        longitudinal_axis="y",
    )

    merged = processor.process({front.track_id: front, rear.track_id: rear})

    assert list(merged.keys()) == [32]
    debug = processor.debug_records[-1]
    assert debug.accepted is True
    assert debug.rejection_reason == "tail_gap"
    assert debug.tail_window_frame_count == 5
    assert debug.full_gap_mean > 1.25
    assert debug.tail_gap_mean < 1.25


def test_articulated_vehicle_merge_rejects_unstable_tail_window_gap() -> None:
    front = _articulated_track(34, [0, 1, 2, 3, 4, 5], [10.0, 11.0, 12.0, 13.0, 14.0, 15.0], lateral_center=0.0, length=5.0)
    rear = _articulated_track(35, [0, 1, 2, 3, 4, 5], [4.8, 4.2, 6.8, 6.2, 8.8, 8.2], lateral_center=0.1, length=5.0)
    processor = ArticulatedVehicleMergePostprocessor(
        PostprocessingConfig(
            enable_articulated_vehicle_merge=True,
            articulated_gap_eval_window_frames=5,
            articulated_max_hitch_gap=2.0,
            articulated_max_hitch_gap_std=0.6,
        ),
        longitudinal_axis="y",
    )

    merged = processor.process({front.track_id: front, rear.track_id: rear})

    assert set(merged.keys()) == {34, 35}
    debug = processor.debug_records[-1]
    assert debug.accepted is False
    assert debug.rejection_reason == "gap_tail_std"
    assert debug.tail_gap_std > 0.6


def test_articulated_vehicle_merge_falls_back_to_full_gap_for_short_tail_window() -> None:
    front = _articulated_track(36, [0, 1], [10.0, 11.0], lateral_center=0.0, length=5.0)
    rear = _articulated_track(37, [0, 1], [3.6, 4.6], lateral_center=0.1, length=5.0)
    processor = ArticulatedVehicleMergePostprocessor(
        PostprocessingConfig(
            enable_articulated_vehicle_merge=True,
            articulated_min_overlap_frames=2,
            articulated_gap_eval_window_frames=5,
            articulated_max_hitch_gap=1.3,
            articulated_max_hitch_gap_std=0.6,
        ),
        longitudinal_axis="y",
    )

    merged = processor.process({front.track_id: front, rear.track_id: rear})

    assert set(merged.keys()) == {36, 37}
    debug = processor.debug_records[-1]
    assert debug.accepted is False
    assert debug.rejection_reason == "gap_full"
    assert debug.tail_window_frame_count == 2
    assert debug.full_gap_mean > 1.3


def test_articulated_vehicle_merge_recomputes_quality_metrics() -> None:
    front = _articulated_track(30, [0, 1, 2, 3], [10.0, 11.0, 12.0, 13.0], lateral_center=0.0)
    rear = _articulated_track(31, [0, 1, 2, 3], [6.8, 7.8, 8.8, 9.8], lateral_center=0.1)
    merge_processor = ArticulatedVehicleMergePostprocessor(PostprocessingConfig(enable_articulated_vehicle_merge=True), longitudinal_axis="y")
    score_processor = TrackQualityScoringPostprocessor(PostprocessingConfig(enable_track_quality_scoring=True), longitudinal_axis="y")

    merged_tracks = merge_processor.process({front.track_id: front, rear.track_id: rear})
    scored_track = score_processor.process(merged_tracks)[30]

    assert scored_track.quality_score is not None
    assert scored_track.quality_metrics["is_articulated_vehicle"] is True
    assert scored_track.quality_metrics["is_long_vehicle"] is True
    assert scored_track.quality_metrics["object_kind"] == "truck_with_trailer"


def test_track_quality_scoring_populates_quality_fields() -> None:
    track = _track(
        4,
        [0, 1, 2, 3],
        [
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
            np.array([0.5, 0.0, 1.0], dtype=np.float32),
            np.array([1.0, 0.0, 1.0], dtype=np.float32),
            np.array([1.5, 0.0, 1.0], dtype=np.float32),
        ],
    )
    scored = TrackQualityScoringPostprocessor(PostprocessingConfig(enable_track_quality_scoring=True)).process({track.track_id: track})[
        track.track_id
    ]
    assert scored.quality_score is not None
    assert 0.0 <= scored.quality_score <= 1.0
    assert scored.quality_metrics["observation_count"] == 4
    assert "cross_section_cv" in scored.quality_metrics
    assert "length_cv" in scored.quality_metrics
    assert scored.quality_metrics["is_articulated_vehicle"] is False
    assert scored.quality_metrics["is_long_vehicle"] is False
