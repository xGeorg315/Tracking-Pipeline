from __future__ import annotations

import numpy as np

from tracking_pipeline.config.models import AggregationConfig, OutputConfig, TrackingConfig
from tracking_pipeline.domain.models import Track
from tracking_pipeline.domain.rules import compute_extent
from tracking_pipeline.domain.value_objects import LaneBox
from tracking_pipeline.infrastructure.aggregation.occupancy_consensus_fusion import OccupancyConsensusFusionAccumulator
from tracking_pipeline.infrastructure.aggregation.registration_backends import _BaseRegistrationBackend
from tracking_pipeline.infrastructure.aggregation.registration_voxel_fusion import RegistrationVoxelFusionAccumulator
from tracking_pipeline.infrastructure.aggregation.voxel_fusion import VoxelFusionAccumulator
from tracking_pipeline.infrastructure.aggregation.weighted_voxel_fusion import WeightedVoxelFusionAccumulator


def _track() -> Track:
    track = Track(track_id=1, hit_count=5, age=5, missed=1, ended_by_missed=True)
    for frame_id, center_x in enumerate([0.95, 0.92, 0.90]):
        center = np.array([center_x, 1.0 + frame_id, 1.0], dtype=np.float32)
        world = np.array([[center_x, 1.0 + frame_id, 1.0], [center_x + 0.1, 1.0 + frame_id, 1.0]], dtype=np.float32)
        track.centers.append(center)
        track.frame_ids.append(frame_id)
        track.world_points.append(world.copy())
        track.local_points.append((world - center).copy())
        track.bbox_extents.append(np.array([0.1, 0.1, 0.1], dtype=np.float32))
    return track


def _track_with_intensity(points: np.ndarray, intensity: np.ndarray) -> Track:
    track = Track(track_id=21, hit_count=1, age=1, missed=0, ended_by_missed=True)
    center = np.mean(points, axis=0).astype(np.float32)
    track.centers.append(center)
    track.frame_ids.append(0)
    track.world_points.append(np.asarray(points, dtype=np.float32))
    track.local_points.append(np.asarray(points, dtype=np.float32) - center)
    track.world_intensity.append(np.asarray(intensity, dtype=np.float32))
    track.local_intensity.append(np.asarray(intensity, dtype=np.float32))
    track.bbox_extents.append(compute_extent(np.asarray(points, dtype=np.float32)))
    return track


def _points(point_count: int, center_y: float, extent_y: float) -> np.ndarray:
    x = np.linspace(-0.15, 0.15, point_count, dtype=np.float32)
    y = np.linspace(center_y - (extent_y * 0.5), center_y + (extent_y * 0.5), point_count, dtype=np.float32)
    z = np.linspace(0.9, 1.1, point_count, dtype=np.float32)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def _track_from_profile(point_counts: list[int], extent_ys: list[float], start_frame: int = 100) -> Track:
    track = Track(track_id=9, hit_count=len(point_counts), age=len(point_counts), missed=1, ended_by_missed=True)
    for index, (point_count, extent_y) in enumerate(zip(point_counts, extent_ys)):
        frame_id = start_frame + index
        center_y = 10.0 + index
        points = _points(point_count, center_y, extent_y)
        center = np.mean(points, axis=0).astype(np.float32)
        track.centers.append(center)
        track.frame_ids.append(frame_id)
        track.world_points.append(points.copy())
        track.local_points.append((points - center).astype(np.float32))
        track.bbox_extents.append(np.array([0.3, extent_y, 0.2], dtype=np.float32))
    return track


def _constant_chunk(x_value: float) -> np.ndarray:
    return np.stack(
        [
            np.full((8,), x_value, dtype=np.float32),
            np.linspace(0.0, 0.35, 8, dtype=np.float32),
            np.zeros((8,), dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32)


class _ScriptedRegistrationBackend(_BaseRegistrationBackend):
    name = "scripted"

    def __init__(self, config: AggregationConfig, scripted_results: list[tuple[float, float]]):
        super().__init__(config)
        self.scripted_results = list(scripted_results)
        self.seen_targets: list[np.ndarray] = []

    def _register_pair(self, source_xyz: np.ndarray, target_xyz: np.ndarray) -> tuple[np.ndarray, float, float]:
        self.seen_targets.append(np.asarray(target_xyz, dtype=np.float32).copy())
        fitness, translation_x = self.scripted_results.pop(0)
        transform = np.eye(4, dtype=np.float64)
        transform[0, 3] = float(translation_x)
        return transform, float(fitness), 0.01


class _SubsetRegistrationAccumulator(VoxelFusionAccumulator):
    fusion_method = "registration_voxel_fusion"

    def _prepare_for_fusion(self, chunks: list[np.ndarray]) -> tuple[list[np.ndarray], dict[str, object]]:
        return [chunks[0], chunks[2]], {
            "alignment_method": "scripted",
            "registration_backend": "scripted",
            "registration_pairs": 2,
            "registration_accepted": 1,
            "registration_rejected": 1,
            "registration_input_chunk_count": len(chunks),
            "registration_output_chunk_count": 2,
            "registration_dropped_count": 1,
            "registration_keep_indices": [0, 2],
            "registration_chunk_weights": [1.0, 0.8],
            "registration_skipped": False,
        }


def test_voxel_fusion_accumulator_saves_track() -> None:
    accumulator = VoxelFusionAccumulator(AggregationConfig(min_saved_aggregate_points=0), OutputConfig(), TrackingConfig())
    result = accumulator.accumulate(_track(), LaneBox.from_values([-1.0, 1.0, 0.0, 10.0, 0.0, 2.0]))
    assert result.status == "saved"
    assert len(result.points) > 0


def test_voxel_fusion_accumulator_averages_intensity_per_voxel() -> None:
    track = _track_with_intensity(
        np.array(
            [
                [0.00, 0.00, 0.00],
                [0.01, 0.01, 0.00],
                [0.02, 0.00, 0.00],
            ],
            dtype=np.float32,
        ),
        np.array([0.0, 0.5, 1.0], dtype=np.float32),
    )
    accumulator = VoxelFusionAccumulator(
        AggregationConfig(
            frame_selection_method="all_track_frames",
            frame_downsample_voxel=0.0,
            fusion_voxel_size=0.1,
            aggregate_voxel=0.0,
            post_filter_stat_nb_neighbors=999,
            min_saved_aggregate_points=0,
        ),
        OutputConfig(require_track_exit=False, save_world=True),
        TrackingConfig(min_track_hits=1),
    )

    result = accumulator.accumulate(track, LaneBox.from_values([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]))

    assert result.status == "saved"
    assert result.intensity is not None
    assert len(result.points) == 1
    assert np.allclose(result.intensity, np.array([0.5], dtype=np.float32))


def test_registration_voxel_fusion_accumulator_falls_back_without_backend() -> None:
    accumulator = RegistrationVoxelFusionAccumulator(
        AggregationConfig(algorithm="registration_voxel_fusion", min_saved_aggregate_points=0),
        OutputConfig(),
        TrackingConfig(),
    )
    if hasattr(accumulator.backend, "_small_gicp"):
        accumulator.backend._small_gicp = None
    result = accumulator.accumulate(_track(), LaneBox.from_values([-1.0, 1.0, 0.0, 10.0, 0.0, 2.0]))
    assert result.status == "saved"
    assert result.metrics["registration_backend"] == "unavailable"
    assert result.metrics["registration_keep_indices"] == [0, 1, 2]
    assert result.metrics["registration_input_chunk_count"] == 3
    assert result.metrics["registration_output_chunk_count"] == 3
    assert result.metrics["registration_dropped_count"] == 0


def test_registration_backend_keeps_only_anchor_and_accepted_chunks() -> None:
    backend = _ScriptedRegistrationBackend(
        AggregationConfig(registration_min_fitness=0.25, registration_max_translation=3.2, frame_downsample_voxel=0.0),
        scripted_results=[(0.10, 0.0), (0.80, 0.0)],
    )
    chunks = [_constant_chunk(0.0), _constant_chunk(10.0), _constant_chunk(20.0)]

    aligned, metrics = backend.align_chunks(chunks)

    assert len(aligned) == 2
    assert metrics["registration_keep_indices"] == [0, 2]
    assert metrics["registration_input_chunk_count"] == 3
    assert metrics["registration_output_chunk_count"] == 2
    assert metrics["registration_dropped_count"] == 1
    assert metrics["registration_accepted"] == 1
    assert metrics["registration_rejected"] == 1
    assert np.allclose(np.mean(backend.seen_targets[0], axis=0), np.mean(chunks[0], axis=0))
    assert np.allclose(np.mean(backend.seen_targets[1], axis=0), np.mean(chunks[0], axis=0))
    assert np.allclose(aligned[1], chunks[2])


def test_registration_subset_keeps_frame_ids_intensity_and_chunk_weights_aligned() -> None:
    track = Track(track_id=31, hit_count=3, age=3, ended_by_missed=True)
    for frame_id, x_value, intensity_value in ((100, 0.0, 0.1), (101, 1.0, 0.5), (102, 2.0, 0.9)):
        points = np.array([[x_value, 0.00, 0.0], [x_value, 0.01, 0.0]], dtype=np.float32)
        center = np.mean(points, axis=0).astype(np.float32)
        intensity = np.full((len(points),), intensity_value, dtype=np.float32)
        track.centers.append(center)
        track.frame_ids.append(frame_id)
        track.world_points.append(points.copy())
        track.local_points.append((points - center).astype(np.float32))
        track.world_intensity.append(intensity.copy())
        track.local_intensity.append(intensity.copy())
        track.bbox_extents.append(compute_extent(points))
    accumulator = _SubsetRegistrationAccumulator(
        AggregationConfig(
            frame_selection_method="all_track_frames",
            frame_downsample_voxel=0.0,
            fusion_voxel_size=0.1,
            aggregate_voxel=0.0,
            post_filter_stat_nb_neighbors=999,
            min_saved_aggregate_points=0,
        ),
        OutputConfig(require_track_exit=False, save_world=True),
        TrackingConfig(min_track_hits=1),
    )

    result = accumulator.accumulate(track, LaneBox.from_values([-1.0, 3.0, -1.0, 1.0, -1.0, 1.0]))

    assert result.status == "saved"
    assert result.selected_frame_ids == [100, 102]
    assert result.metrics["registration_keep_indices"] == [0, 2]
    assert result.metrics["registration_input_chunk_count"] == 3
    assert result.metrics["registration_output_chunk_count"] == 2
    assert result.metrics["registration_dropped_count"] == 1
    assert result.metrics["registration_chunk_weights"] == [1.0, 0.8]
    assert result.intensity is not None
    assert len(result.intensity) == 2
    assert np.allclose(np.sort(result.intensity), np.array([0.1, 0.9], dtype=np.float32))


def test_voxel_fusion_stat_filter_keeps_intensity_aligned() -> None:
    cluster = np.array(
        [[0.02 * idx, 0.0, 0.0] for idx in range(16)] + [[8.0, 8.0, 8.0]],
        dtype=np.float32,
    )
    intensity = np.linspace(0.0, 1.0, len(cluster), dtype=np.float32)
    track = _track_with_intensity(cluster, intensity)
    accumulator = VoxelFusionAccumulator(
        AggregationConfig(
            frame_selection_method="all_track_frames",
            frame_downsample_voxel=0.0,
            fusion_voxel_size=0.01,
            aggregate_voxel=0.0,
            post_filter_stat_nb_neighbors=8,
            post_filter_stat_std_ratio=1.0,
            min_saved_aggregate_points=0,
        ),
        OutputConfig(require_track_exit=False, save_world=True),
        TrackingConfig(min_track_hits=1),
    )

    result = accumulator.accumulate(track, LaneBox.from_values([-1.0, 9.0, -1.0, 9.0, -1.0, 9.0]))

    assert result.status == "saved"
    assert result.intensity is not None
    assert len(result.intensity) == len(result.points)
    assert len(result.points) < len(cluster)


def test_weighted_voxel_fusion_reports_chunk_weights() -> None:
    track = _track()
    track.world_points = [
        np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32),
        np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.4, 0.0, 0.0]], dtype=np.float32),
    ]
    track.local_points = [points.copy() for points in track.world_points]
    accumulator = WeightedVoxelFusionAccumulator(
        AggregationConfig(
            algorithm="weighted_voxel_fusion",
            fusion_weight_mode="point_count",
            frame_downsample_voxel=0.0,
            aggregate_voxel=0.0,
            post_filter_stat_nb_neighbors=999,
            min_saved_aggregate_points=0,
        ),
        OutputConfig(require_track_exit=False, save_world=True),
        TrackingConfig(),
    )
    result = accumulator.accumulate(track, LaneBox.from_values([-1.0, 1.0, 0.0, 10.0, 0.0, 2.0]))
    assert result.status == "saved"
    assert result.metrics["fusion_method"] == "weighted_voxel_fusion"
    assert result.metrics["chunk_weights"] == [1.0, 2.0, 3.0]


def test_weighted_voxel_fusion_quality_mode_uses_track_quality_and_chunk_consistency() -> None:
    track = _track()
    track.quality_score = 0.5
    track.world_points = [
        np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32),
        np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.6, 0.0, 0.0]], dtype=np.float32),
        np.array([[0.0, 0.0, 0.0], [1.8, 0.0, 0.0], [3.6, 0.0, 0.0], [5.4, 0.0, 0.0]], dtype=np.float32),
    ]
    track.local_points = [points.copy() for points in track.world_points]
    accumulator = WeightedVoxelFusionAccumulator(
        AggregationConfig(
            algorithm="weighted_voxel_fusion",
            fusion_weight_mode="quality",
            frame_downsample_voxel=0.0,
            aggregate_voxel=0.0,
            post_filter_stat_nb_neighbors=999,
            min_saved_aggregate_points=0,
        ),
        OutputConfig(require_track_exit=False, save_world=True),
        TrackingConfig(),
    )
    result = accumulator.accumulate(track, LaneBox.from_values([-1.0, 1.0, 0.0, 10.0, 0.0, 2.0]))
    assert result.status == "saved"
    assert len(result.metrics["chunk_weights"]) == 3
    assert result.metrics["chunk_weights"][2] < result.metrics["chunk_weights"][1]
    assert max(result.metrics["chunk_weights"]) <= 0.5 + 1e-6


def test_quality_threshold_skips_artifact_save_status() -> None:
    track = _track()
    track.quality_score = 0.2
    accumulator = WeightedVoxelFusionAccumulator(
        AggregationConfig(
            algorithm="weighted_voxel_fusion",
            fusion_weight_mode="quality",
            min_track_quality_for_save=0.4,
            min_saved_aggregate_points=0,
        ),
        OutputConfig(require_track_exit=False),
        TrackingConfig(),
    )
    result = accumulator.accumulate(track, LaneBox.from_values([-1.0, 1.0, 0.0, 10.0, 0.0, 2.0]))
    assert result.status == "skipped_quality_threshold"
    assert result.metrics["quality_threshold"] == 0.4


def test_long_vehicle_quality_threshold_uses_long_vehicle_override() -> None:
    track = _track()
    track.quality_score = 0.1
    track.quality_metrics = {"is_long_vehicle": True}
    accumulator = WeightedVoxelFusionAccumulator(
        AggregationConfig(
            algorithm="weighted_voxel_fusion",
            fusion_weight_mode="quality",
            min_track_quality_for_save=0.4,
            min_track_quality_for_save_long_vehicle=0.0,
            min_saved_aggregate_points=0,
        ),
        OutputConfig(require_track_exit=False),
        TrackingConfig(),
    )
    result = accumulator.accumulate(track, LaneBox.from_values([-1.0, 1.0, 0.0, 10.0, 0.0, 2.0]))
    assert result.status == "saved"
    assert result.metrics["long_vehicle_mode_applied"] is True


def test_weighted_voxel_fusion_long_vehicle_quality_mode_ignores_longitudinal_extent_variation() -> None:
    track = _track()
    track.quality_score = 0.6
    track.quality_metrics = {"is_long_vehicle": True}
    track.world_points = [
        np.array([[0.0, 0.0, 0.0], [0.1, 2.0, 0.0]], dtype=np.float32),
        np.array([[0.0, 0.0, 0.0], [0.1, 4.0, 0.0]], dtype=np.float32),
        np.array([[0.0, 0.0, 0.0], [0.1, 6.0, 0.0]], dtype=np.float32),
    ]
    track.local_points = [points.copy() for points in track.world_points]
    accumulator = WeightedVoxelFusionAccumulator(
        AggregationConfig(
            algorithm="weighted_voxel_fusion",
            fusion_weight_mode="quality",
            frame_selection_line_axis="y",
            frame_downsample_voxel=0.0,
            aggregate_voxel=0.0,
            post_filter_stat_nb_neighbors=999,
            min_saved_aggregate_points=0,
        ),
        OutputConfig(require_track_exit=False, save_world=True),
        TrackingConfig(),
    )
    result = accumulator.accumulate(track, LaneBox.from_values([-1.0, 1.0, 0.0, 10.0, 0.0, 2.0]))
    assert result.status == "saved"
    assert len(result.metrics["chunk_weights"]) == 3
    assert max(result.metrics["chunk_weights"]) - min(result.metrics["chunk_weights"]) < 1e-6


def test_tail_bridge_connects_longitudinal_components_for_long_vehicle_mode() -> None:
    track = Track(track_id=8, hit_count=1, age=1, ended_by_missed=True)
    points = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.2, 1.0],
            [0.0, 0.4, 1.0],
            [0.0, 0.6, 1.0],
            [0.0, 1.6, 1.0],
            [0.0, 1.8, 1.0],
            [0.0, 2.0, 1.0],
            [0.0, 2.2, 1.0],
        ],
        dtype=np.float32,
    )
    center = np.mean(points, axis=0).astype(np.float32)
    track.centers.append(center)
    track.frame_ids.append(0)
    track.world_points.append(points.copy())
    track.local_points.append(points.copy())
    track.world_intensity.append(np.concatenate([np.full((4,), 0.2, dtype=np.float32), np.full((4,), 0.8, dtype=np.float32)]))
    track.local_intensity.append(np.concatenate([np.full((4,), 0.2, dtype=np.float32), np.full((4,), 0.8, dtype=np.float32)]))
    track.bbox_extents.append(np.array([0.0, 2.2, 0.0], dtype=np.float32))
    track.quality_score = 0.5
    track.quality_metrics = {"is_long_vehicle": True}
    accumulator = WeightedVoxelFusionAccumulator(
        AggregationConfig(
            algorithm="weighted_voxel_fusion",
            frame_selection_method="all_track_frames",
            frame_selection_line_axis="y",
            long_vehicle_mode=True,
            frame_downsample_voxel=0.0,
            fusion_voxel_size=0.05,
            aggregate_voxel=0.0,
            post_filter_stat_nb_neighbors=999,
            tail_bridge_longitudinal_gap_max=1.5,
            tail_bridge_lateral_gap_max=0.5,
            tail_bridge_vertical_gap_max=0.5,
            min_saved_aggregate_points=0,
        ),
        OutputConfig(require_track_exit=False, save_world=True),
        TrackingConfig(min_track_hits=1),
    )
    result = accumulator.accumulate(track, LaneBox.from_values([-1.0, 1.0, -1.0, 4.0, 0.0, 2.0]))
    assert result.status == "saved"
    assert result.metrics["tail_bridge_count"] >= 1
    assert result.metrics["component_count_post_fusion"] == 1
    assert result.metrics["longitudinal_extent"] >= 2.0
    assert result.intensity is not None
    assert np.any((result.intensity > 0.2) & (result.intensity < 0.8))


def test_occupancy_consensus_keeps_only_consistent_voxels() -> None:
    track = Track(track_id=7, hit_count=4, age=4, ended_by_missed=True)
    for frame_id, points in enumerate(
        [
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
            np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32),
        ]
    ):
        track.centers.append(np.zeros((3,), dtype=np.float32))
        track.frame_ids.append(frame_id)
        track.world_points.append(points)
        track.local_points.append(points.copy())
        track.bbox_extents.append(np.array([2.0, 0.1, 0.1], dtype=np.float32))
    track.hit_count = len(track.frame_ids)
    accumulator = OccupancyConsensusFusionAccumulator(
        AggregationConfig(
            algorithm="occupancy_consensus_fusion",
            frame_downsample_voxel=0.0,
            fusion_voxel_size=0.25,
            consensus_ratio=0.7,
            fusion_min_observations=1,
            aggregate_voxel=0.0,
            post_filter_stat_nb_neighbors=999,
            min_saved_aggregate_points=0,
        ),
        OutputConfig(require_track_exit=False, save_world=True),
        TrackingConfig(min_track_hits=1),
    )
    result = accumulator.accumulate(track, LaneBox.from_values([-1.0, 1.0, 0.0, 10.0, 0.0, 2.0]))
    assert result.status == "saved"
    assert len(result.points) == 1
    assert np.allclose(result.points[0], np.array([0.0, 0.0, 0.0], dtype=np.float32))


def test_chunk_quality_filter_removes_weak_tail_before_keyframe_motion() -> None:
    track = _track_from_profile(
        point_counts=[35, 70, 120, 130, 125, 30, 20, 15],
        extent_ys=[1.0, 2.0, 4.0, 4.2, 4.1, 0.9, 0.7, 0.5],
    )
    accumulator = VoxelFusionAccumulator(
        AggregationConfig(
            frame_selection_method="keyframe_motion",
            keyframe_keep=3,
            frame_downsample_voxel=0.0,
            aggregate_voxel=0.0,
            post_filter_stat_nb_neighbors=999,
            min_saved_aggregate_points=0,
        ),
        OutputConfig(require_track_exit=False, save_world=True),
        TrackingConfig(min_track_hits=1),
    )

    result = accumulator.accumulate(track, LaneBox.from_values([-1.0, 1.0, 0.0, 20.0, 0.0, 2.0]))

    assert result.status == "saved"
    assert result.metrics["chunk_quality_total"] == 8
    assert result.metrics["chunk_quality_kept"] == 4
    assert result.metrics["chunk_quality_segment_start_frame"] == 101
    assert result.metrics["chunk_quality_segment_end_frame"] == 104
    assert 107 not in result.selected_frame_ids
    assert max(result.selected_frame_ids) <= 104


def test_chunk_quality_filter_expands_short_peak_segment() -> None:
    track = _track_from_profile(
        point_counts=[15, 20, 25, 100, 28, 18, 12],
        extent_ys=[0.3, 0.4, 0.5, 3.0, 0.55, 0.4, 0.3],
    )
    accumulator = VoxelFusionAccumulator(
        AggregationConfig(
            frame_selection_method="all_track_frames",
            chunk_min_segment_length=4,
            frame_downsample_voxel=0.0,
            aggregate_voxel=0.0,
            post_filter_stat_nb_neighbors=999,
            min_saved_aggregate_points=0,
        ),
        OutputConfig(require_track_exit=False, save_world=True),
        TrackingConfig(min_track_hits=1),
    )

    result = accumulator.accumulate(track, LaneBox.from_values([-1.0, 1.0, 0.0, 20.0, 0.0, 2.0]))

    assert result.status == "saved"
    assert result.selected_frame_ids == [101, 102, 103, 104]
    assert result.metrics["chunk_quality_kept"] == 4
    assert result.metrics["chunk_quality_segment_start_frame"] == 101
    assert result.metrics["chunk_quality_segment_end_frame"] == 104


def test_min_saved_aggregate_points_skips_small_artifacts() -> None:
    track = _track_from_profile(point_counts=[120], extent_ys=[2.0], start_frame=200)
    accumulator = VoxelFusionAccumulator(
        AggregationConfig(
            frame_selection_method="all_track_frames",
            frame_downsample_voxel=0.0,
            fusion_voxel_size=0.005,
            aggregate_voxel=0.0,
            post_filter_stat_nb_neighbors=999,
            min_saved_aggregate_points=180,
        ),
        OutputConfig(require_track_exit=False, save_world=True),
        TrackingConfig(min_track_hits=1),
    )

    result = accumulator.accumulate(track, LaneBox.from_values([-1.0, 1.0, 0.0, 20.0, 0.0, 2.0]))

    assert result.status == "skipped_min_saved_points"
    assert result.metrics["min_saved_aggregate_points"] == 180
    assert result.metrics["point_count_after_downsample"] < 180


def test_min_saved_aggregate_points_keeps_large_artifacts() -> None:
    track = _track_from_profile(point_counts=[360], extent_ys=[2.5], start_frame=300)
    accumulator = VoxelFusionAccumulator(
        AggregationConfig(
            frame_selection_method="all_track_frames",
            frame_downsample_voxel=0.0,
            fusion_voxel_size=0.005,
            aggregate_voxel=0.0,
            post_filter_stat_nb_neighbors=999,
            min_saved_aggregate_points=180,
        ),
        OutputConfig(require_track_exit=False, save_world=True),
        TrackingConfig(min_track_hits=1),
    )

    result = accumulator.accumulate(track, LaneBox.from_values([-1.0, 1.0, 0.0, 20.0, 0.0, 2.0]))

    assert result.status == "saved"
    assert result.metrics["point_count_after_downsample"] >= 180
