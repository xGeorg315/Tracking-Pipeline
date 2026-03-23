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


def _single_chunk_track(
    points: np.ndarray,
    *,
    center: np.ndarray | None = None,
    intensity: np.ndarray | None = None,
    frame_id: int = 0,
) -> Track:
    track = Track(track_id=42, hit_count=1, age=1, missed=0, ended_by_missed=True)
    world_points = np.asarray(points, dtype=np.float32)
    track_center = np.zeros((3,), dtype=np.float32) if center is None else np.asarray(center, dtype=np.float32)
    track.centers.append(track_center.copy())
    track.frame_ids.append(int(frame_id))
    track.world_points.append(world_points.copy())
    track.local_points.append((world_points - track_center).astype(np.float32))
    if intensity is not None:
        intensity_values = np.asarray(intensity, dtype=np.float32)
        track.world_intensity.append(intensity_values.copy())
        track.local_intensity.append(intensity_values.copy())
    track.bbox_extents.append(compute_extent(world_points))
    return track


def _track_from_chunks(
    chunks: list[np.ndarray],
    *,
    intensities: list[np.ndarray | None] | None = None,
    start_frame: int = 0,
    track_id: int = 52,
) -> Track:
    track = Track(track_id=track_id, hit_count=len(chunks), age=len(chunks), missed=0, ended_by_missed=True)
    intensity_values = intensities if intensities is not None else [None for _ in chunks]
    for offset, (chunk, chunk_intensity) in enumerate(zip(chunks, intensity_values)):
        world_points = np.asarray(chunk, dtype=np.float32)
        center = np.mean(world_points, axis=0).astype(np.float32) if len(world_points) > 0 else np.zeros((3,), dtype=np.float32)
        track.centers.append(center.copy())
        track.frame_ids.append(int(start_frame + offset))
        track.world_points.append(world_points.copy())
        track.local_points.append((world_points - center).astype(np.float32))
        if chunk_intensity is not None:
            scalar = np.asarray(chunk_intensity, dtype=np.float32)
            track.world_intensity.append(scalar.copy())
            track.local_intensity.append(scalar.copy())
        track.bbox_extents.append(compute_extent(world_points))
    return track


def _symmetry_accumulator(
    *,
    enabled: bool = True,
    save_world: bool = False,
    min_saved_aggregate_points: int = 0,
    fusion_voxel_size: float = 0.10,
) -> VoxelFusionAccumulator:
    return VoxelFusionAccumulator(
        AggregationConfig(
            symmetry_completion=enabled,
            frame_selection_method="all_track_frames",
            frame_downsample_voxel=0.0,
            fusion_voxel_size=fusion_voxel_size,
            aggregate_voxel=0.0,
            post_filter_stat_nb_neighbors=999,
            min_saved_aggregate_points=min_saved_aggregate_points,
        ),
        OutputConfig(require_track_exit=False, save_world=save_world),
        TrackingConfig(min_track_hits=1),
    )


def _motion_deskew_accumulator(
    *,
    enabled: bool,
    save_world: bool = False,
    long_vehicle_mode: bool = True,
) -> VoxelFusionAccumulator:
    return VoxelFusionAccumulator(
        AggregationConfig(
            motion_deskew=enabled,
            long_vehicle_mode=long_vehicle_mode,
            frame_selection_method="all_track_frames",
            frame_downsample_voxel=0.0,
            fusion_voxel_size=0.05,
            aggregate_voxel=0.0,
            post_filter_stat_nb_neighbors=999,
            min_saved_aggregate_points=0,
        ),
        OutputConfig(require_track_exit=False, save_world=save_world),
        TrackingConfig(min_track_hits=1),
    )


def _motion_distorted_track(*, with_point_timestamps: bool = True) -> Track:
    track = Track(track_id=77, hit_count=3, age=3, missed=0, ended_by_missed=True)
    point_time_offsets_ns = np.array([0, 50_000_000, 100_000_000], dtype=np.int64)
    x_values = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    speed_mps = 10.0
    for index, center_y in enumerate([0.0, 10.0, 20.0]):
        frame_timestamp_ns = int(index * 1_000_000_000)
        absolute_times = frame_timestamp_ns + point_time_offsets_ns
        dt_seconds = (point_time_offsets_ns.astype(np.float64) - float(np.median(point_time_offsets_ns))) * 1e-9
        points = np.stack(
            [
                x_values,
                np.full((len(x_values),), center_y, dtype=np.float32) + (speed_mps * dt_seconds).astype(np.float32),
                np.zeros((len(x_values),), dtype=np.float32),
            ],
            axis=1,
        ).astype(np.float32)
        center = np.array([0.0, center_y, 0.0], dtype=np.float32)
        track.centers.append(center.copy())
        track.frame_ids.append(index)
        track.frame_timestamps_ns.append(frame_timestamp_ns)
        track.world_points.append(points.copy())
        track.local_points.append((points - center).astype(np.float32))
        track.bbox_extents.append(np.array([2.0, 1.0, 0.0], dtype=np.float32))
        track.point_timestamps_ns.append(absolute_times.copy() if with_point_timestamps else None)
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


class _ScriptedPrepareAccumulator(VoxelFusionAccumulator):
    fusion_method = "registration_voxel_fusion"

    def __init__(
        self,
        config: AggregationConfig,
        output_config: OutputConfig,
        tracking_config: TrackingConfig,
        *,
        keep_indices: list[int],
        chunk_weights: list[float] | None = None,
    ):
        super().__init__(config, output_config, tracking_config)
        self.keep_indices = list(keep_indices)
        self.chunk_weights = None if chunk_weights is None else [float(weight) for weight in chunk_weights]

    def _prepare_for_fusion(self, chunks: list[np.ndarray]) -> tuple[list[np.ndarray], dict[str, object]]:
        keep_indices = [int(index) for index in self.keep_indices if 0 <= int(index) < len(chunks)]
        kept_chunks = [chunks[index] for index in keep_indices]
        chunk_weights = self.chunk_weights if self.chunk_weights is not None and len(self.chunk_weights) == len(keep_indices) else [1.0 for _ in keep_indices]
        pair_count = max(0, len(chunks) - 1)
        accepted = max(0, len(keep_indices) - (1 if 0 in keep_indices else 0))
        return kept_chunks, {
            "alignment_method": "scripted",
            "registration_backend": "scripted",
            "registration_pairs": pair_count,
            "registration_accepted": accepted,
            "registration_rejected": max(0, pair_count - accepted),
            "registration_input_chunk_count": len(chunks),
            "registration_output_chunk_count": len(keep_indices),
            "registration_dropped_count": max(0, len(chunks) - len(keep_indices)),
            "registration_keep_indices": keep_indices,
            "registration_chunk_weights": chunk_weights,
            "registration_skipped": False,
        }


class _CapturingAccumulator(VoxelFusionAccumulator):
    def __init__(self, config: AggregationConfig, output_config: OutputConfig, tracking_config: TrackingConfig):
        super().__init__(config, output_config, tracking_config)
        self.captured: dict[str, object] = {}

    def _apply_registration_subset(
        self,
        accumulation_input: list[np.ndarray],
        prepared_chunks: list[np.ndarray],
        prepared_intensity: list[np.ndarray | None],
        selected_centers: list[np.ndarray],
        selected_frame_ids: list[int],
        registration_metrics: dict[str, object],
    ) -> tuple[list[np.ndarray], list[np.ndarray | None], list[np.ndarray], list[int], dict[str, object]]:
        self.captured["prepared_intensity"] = [
            None if values is None else np.asarray(values, dtype=np.float32).copy() for values in prepared_intensity
        ]
        self.captured["selected_centers"] = [np.asarray(center, dtype=np.float32).copy() for center in selected_centers]
        self.captured["selected_frame_ids"] = list(selected_frame_ids)
        return super()._apply_registration_subset(
            accumulation_input,
            prepared_chunks,
            prepared_intensity,
            selected_centers,
            selected_frame_ids,
            registration_metrics,
        )


def test_voxel_fusion_accumulator_saves_track() -> None:
    accumulator = VoxelFusionAccumulator(
        AggregationConfig(min_saved_aggregate_points=0),
        OutputConfig(require_track_exit=False),
        TrackingConfig(),
    )
    result = accumulator.accumulate(_track(), LaneBox.from_values([-1.0, 1.0, 0.0, 10.0, 0.0, 2.0]))
    assert result.status == "saved"
    assert len(result.points) > 0
    assert result.metrics["decision_stage"] == "saved"
    assert result.metrics["decision_reason_code"] == "saved"
    assert result.metrics["decision_summary"].startswith("saved points=")
    extent = compute_extent(result.points)
    assert result.metrics["vehicle_length_axis"] == "y"
    assert result.metrics["vehicle_width_axis"] == "x"
    assert result.metrics["vehicle_height_axis"] == "z"
    assert np.isclose(result.metrics["vehicle_length"], float(extent[1]))
    assert np.isclose(result.metrics["vehicle_width"], float(extent[0]))
    assert np.isclose(result.metrics["vehicle_height"], float(extent[2]))
    assert np.isclose(result.metrics["extent_x"], float(extent[0]))
    assert np.isclose(result.metrics["extent_y"], float(extent[1]))
    assert np.isclose(result.metrics["extent_z"], float(extent[2]))


def test_voxel_fusion_accumulator_reports_min_hits_decision() -> None:
    accumulator = VoxelFusionAccumulator(
        AggregationConfig(min_saved_aggregate_points=0),
        OutputConfig(require_track_exit=False),
        TrackingConfig(min_track_hits=6),
    )

    result = accumulator.accumulate(_track(), LaneBox.from_values([-1.0, 1.0, 0.0, 10.0, 0.0, 2.0]))

    assert result.status == "skipped_min_hits"
    assert result.metrics["decision_stage"] == "tracking_gate"
    assert result.metrics["decision_reason_code"] == "min_hits"
    assert result.metrics["decision_summary"] == "min_hits 5/6"
    assert result.metrics["hit_count"] == 5
    assert result.metrics["min_track_hits"] == 6


def test_voxel_fusion_accumulator_reports_track_exit_decision() -> None:
    accumulator = VoxelFusionAccumulator(
        AggregationConfig(min_saved_aggregate_points=0),
        OutputConfig(require_track_exit=True, track_exit_edge_margin=0.9),
        TrackingConfig(min_track_hits=1),
    )

    result = accumulator.accumulate(_track(), LaneBox.from_values([-10.0, 10.0, -10.0, 10.0, 0.0, 2.0]))

    assert result.status == "skipped_track_exit"
    assert result.metrics["decision_stage"] == "exit_gate"
    assert result.metrics["decision_reason_code"] == "track_exit"
    assert result.metrics["decision_summary"] == "track_exit line=-9.10 center=3.00"
    assert result.metrics["track_exited"] is False
    assert result.metrics["track_exit_line_axis"] == "y"
    assert result.metrics["track_exit_line_side"] == "min"
    assert result.metrics["track_exit_line_value"] == -9.1
    assert result.metrics["track_center_line_coordinate"] == 3.0
    assert result.metrics["track_passed_exit_line"] is False
    assert result.metrics["distance_to_exit_line"] == 12.1
    assert result.metrics["closest_edge_distance"] == 12.1


def test_voxel_fusion_accumulator_reports_empty_selection_decision() -> None:
    track = Track(track_id=99, hit_count=1, age=1, missed=1, ended_by_missed=True)
    accumulator = VoxelFusionAccumulator(
        AggregationConfig(min_saved_aggregate_points=0),
        OutputConfig(require_track_exit=False),
        TrackingConfig(min_track_hits=1),
    )

    result = accumulator.accumulate(track, LaneBox.from_values([-1.0, 1.0, 0.0, 10.0, 0.0, 2.0]))

    assert result.status == "empty_selection"
    assert result.metrics["decision_stage"] == "selection"
    assert result.metrics["decision_reason_code"] == "empty_selection"
    assert result.metrics["decision_summary"] == "empty_selection selected=0"
    assert result.metrics["selected_frame_count"] == 0


def test_voxel_fusion_accumulator_reports_min_saved_points_decision() -> None:
    track = _single_chunk_track(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.01, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    accumulator = VoxelFusionAccumulator(
        AggregationConfig(
            min_saved_aggregate_points=5,
            frame_selection_method="all_track_frames",
            frame_downsample_voxel=0.0,
            fusion_voxel_size=0.1,
            aggregate_voxel=0.0,
            post_filter_stat_nb_neighbors=999,
        ),
        OutputConfig(require_track_exit=False, save_world=True),
        TrackingConfig(min_track_hits=1),
    )

    result = accumulator.accumulate(track, LaneBox.from_values([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]))

    assert result.status == "skipped_min_saved_points"
    assert result.metrics["decision_stage"] == "save_gate"
    assert result.metrics["decision_reason_code"] == "min_saved_points"
    assert result.metrics["decision_summary"] == "min_saved_points 1/5"
    assert result.metrics["point_count_after_downsample"] == 1
    assert result.metrics["min_saved_aggregate_points"] == 5


def test_voxel_fusion_accumulator_reports_quality_threshold_decision() -> None:
    track = _single_chunk_track(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.2, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    track.quality_score = 0.2
    accumulator = VoxelFusionAccumulator(
        AggregationConfig(
            min_saved_aggregate_points=0,
            min_track_quality_for_save=0.5,
            frame_selection_method="all_track_frames",
            frame_downsample_voxel=0.0,
            fusion_voxel_size=0.1,
            aggregate_voxel=0.0,
            post_filter_stat_nb_neighbors=999,
        ),
        OutputConfig(require_track_exit=False, save_world=True),
        TrackingConfig(min_track_hits=1),
    )

    result = accumulator.accumulate(track, LaneBox.from_values([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]))

    assert result.status == "skipped_quality_threshold"
    assert result.metrics["decision_stage"] == "quality_gate"
    assert result.metrics["decision_reason_code"] == "quality_threshold"
    assert result.metrics["decision_summary"] == "quality 0.20<0.50"
    assert result.metrics["quality_score"] == 0.2
    assert result.metrics["quality_threshold"] == 0.5


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


def test_symmetry_completion_disabled_leaves_saved_aggregate_unchanged() -> None:
    track = _single_chunk_track(
        np.array(
            [
                [0.22, 0.00, 1.0],
                [0.22, 0.12, 1.0],
                [0.22, 0.24, 1.0],
            ],
            dtype=np.float32,
        ),
        center=np.zeros((3,), dtype=np.float32),
    )
    accumulator = _symmetry_accumulator(enabled=False, save_world=False, fusion_voxel_size=0.10)

    result = accumulator.accumulate(track, LaneBox.from_values([-1.0, 1.0, -1.0, 2.0, 0.0, 2.0]))

    assert result.status == "saved"
    assert len(result.points) == 3
    assert result.metrics["symmetry_completion_enabled"] is False
    assert result.metrics["symmetry_completion_applied"] is False
    assert result.metrics["point_count_before_symmetry_completion"] == 3
    assert result.metrics["point_count_after_symmetry_completion"] == 3
    assert result.metrics["symmetry_completion_generated_points"] == 0


def test_lane_end_touch_filter_disabled_leaves_all_frames_available() -> None:
    track = Track(track_id=88, hit_count=4, age=4, missed=1, ended_by_missed=True)
    for frame_id, center_y in ((10, 2.8), (11, 1.5), (12, 0.08), (13, -0.5)):
        points = np.array(
            [
                [0.0, center_y - 0.1, 0.0],
                [0.0, center_y + 0.1, 0.0],
            ],
            dtype=np.float32,
        )
        center = np.mean(points, axis=0).astype(np.float32)
        intensity = np.full((len(points),), frame_id / 100.0, dtype=np.float32)
        track.centers.append(center)
        track.frame_ids.append(frame_id)
        track.world_points.append(points.copy())
        track.local_points.append((points - center).astype(np.float32))
        track.world_intensity.append(intensity.copy())
        track.local_intensity.append(intensity.copy())
        track.bbox_extents.append(compute_extent(points))
    accumulator = VoxelFusionAccumulator(
        AggregationConfig(
            frame_selection_method="all_track_frames",
            frame_downsample_voxel=0.0,
            fusion_voxel_size=0.1,
            aggregate_voxel=0.0,
            post_filter_stat_nb_neighbors=999,
            min_saved_aggregate_points=0,
            truncate_after_lane_end_touch=False,
        ),
        OutputConfig(require_track_exit=False, save_world=False),
        TrackingConfig(min_track_hits=1),
    )

    result = accumulator.accumulate(track, LaneBox.from_values([-1.0, 1.0, 0.0, 10.0, -1.0, 1.0]))

    assert result.status == "saved"
    assert result.selected_frame_ids == [10, 11, 12, 13]
    assert result.metrics["lane_end_touch_filter_enabled"] is False
    assert result.metrics["lane_end_touch_found"] is False
    assert result.metrics["lane_end_touch_frame_id"] == -1
    assert result.metrics["lane_end_touch_index"] == -1
    assert result.metrics["lane_end_touch_axis"] == "y"
    assert result.metrics["lane_end_touch_value"] == 0.0
    assert result.metrics["lane_end_touch_kept_frame_count"] == 4


def test_lane_end_touch_filter_truncates_all_track_frames_using_world_points() -> None:
    track = Track(track_id=89, hit_count=4, age=4, missed=1, ended_by_missed=True)
    for frame_id, center_y in ((10, 2.8), (11, 1.5), (12, 0.08), (13, -0.5)):
        points = np.array(
            [
                [0.0, center_y - 0.1, 0.0],
                [0.0, center_y + 0.1, 0.0],
            ],
            dtype=np.float32,
        )
        center = np.mean(points, axis=0).astype(np.float32)
        intensity = np.full((len(points),), frame_id / 100.0, dtype=np.float32)
        track.centers.append(center)
        track.frame_ids.append(frame_id)
        track.world_points.append(points.copy())
        track.local_points.append((points - center).astype(np.float32))
        track.world_intensity.append(intensity.copy())
        track.local_intensity.append(intensity.copy())
        track.bbox_extents.append(compute_extent(points))
    accumulator = VoxelFusionAccumulator(
        AggregationConfig(
            frame_selection_method="all_track_frames",
            frame_downsample_voxel=0.0,
            fusion_voxel_size=0.1,
            aggregate_voxel=0.0,
            post_filter_stat_nb_neighbors=999,
            min_saved_aggregate_points=0,
            truncate_after_lane_end_touch=True,
        ),
        OutputConfig(require_track_exit=False, save_world=False),
        TrackingConfig(min_track_hits=1),
    )

    result = accumulator.accumulate(track, LaneBox.from_values([-1.0, 1.0, 0.0, 10.0, -1.0, 1.0]))

    assert result.status == "saved"
    assert result.selected_frame_ids == [10, 11, 12]
    assert result.metrics["lane_end_touch_filter_enabled"] is True
    assert result.metrics["lane_end_touch_found"] is True
    assert result.metrics["lane_end_touch_frame_id"] == 12
    assert result.metrics["lane_end_touch_index"] == 2
    assert result.metrics["lane_end_touch_axis"] == "y"
    assert result.metrics["lane_end_touch_value"] == 0.0
    assert result.metrics["lane_end_touch_kept_frame_count"] == 3


def test_lane_end_touch_filter_limits_keyframe_motion_and_keeps_intensity_centers_aligned() -> None:
    track = Track(track_id=90, hit_count=4, age=4, missed=1, ended_by_missed=True)
    centers_by_frame: list[np.ndarray] = []
    for frame_id, center_y in ((10, 3.0), (11, 1.8), (12, 0.05), (13, -0.6)):
        points = np.array(
            [
                [0.0, center_y - 0.15, 0.0],
                [0.0, center_y + 0.15, 0.0],
            ],
            dtype=np.float32,
        )
        center = np.mean(points, axis=0).astype(np.float32)
        intensity = np.full((len(points),), frame_id / 100.0, dtype=np.float32)
        centers_by_frame.append(center.copy())
        track.centers.append(center)
        track.frame_ids.append(frame_id)
        track.world_points.append(points.copy())
        track.local_points.append((points - center).astype(np.float32))
        track.world_intensity.append(intensity.copy())
        track.local_intensity.append(intensity.copy())
        track.bbox_extents.append(compute_extent(points))
    accumulator = _CapturingAccumulator(
        AggregationConfig(
            frame_selection_method="keyframe_motion",
            use_all_frames=False,
            keyframe_keep=2,
            frame_downsample_voxel=0.0,
            fusion_voxel_size=0.1,
            aggregate_voxel=0.0,
            post_filter_stat_nb_neighbors=999,
            min_saved_aggregate_points=0,
            truncate_after_lane_end_touch=True,
        ),
        OutputConfig(require_track_exit=False, save_world=False),
        TrackingConfig(min_track_hits=1),
    )

    result = accumulator.accumulate(track, LaneBox.from_values([-1.0, 1.0, 0.0, 10.0, -1.0, 1.0]))

    assert result.status == "saved"
    assert result.selected_frame_ids == [10, 11, 12]
    assert accumulator.captured["selected_frame_ids"] == [10, 11, 12]
    captured_centers = accumulator.captured["selected_centers"]
    assert isinstance(captured_centers, list)
    assert np.allclose(captured_centers[0], centers_by_frame[0])
    assert np.allclose(captured_centers[1], centers_by_frame[1])
    assert np.allclose(captured_centers[2], centers_by_frame[2])
    captured_intensity = accumulator.captured["prepared_intensity"]
    assert isinstance(captured_intensity, list)
    assert np.isclose(float(np.mean(captured_intensity[0])), 0.10)
    assert np.isclose(float(np.mean(captured_intensity[1])), 0.11)
    assert np.isclose(float(np.mean(captured_intensity[2])), 0.12)
    assert 13 not in result.selected_frame_ids


def test_lane_end_touch_filter_limits_length_coverage_to_prefix() -> None:
    track = _track_from_profile([30, 40, 50, 60], [1.0, 2.0, 3.0, 4.0], start_frame=200)
    touching_points = track.world_points[2].copy()
    touching_points[:, 1] -= float(np.min(touching_points[:, 1])) - 0.02
    track.world_points[2] = touching_points
    track.local_points[2] = (touching_points - track.centers[2]).astype(np.float32)
    post_touch_points = track.world_points[3].copy()
    post_touch_points[:, 1] -= 4.5
    track.world_points[3] = post_touch_points
    track.local_points[3] = (post_touch_points - track.centers[3]).astype(np.float32)
    accumulator = VoxelFusionAccumulator(
        AggregationConfig(
            frame_selection_method="length_coverage",
            use_all_frames=False,
            keyframe_keep=2,
            length_coverage_bins=4,
            frame_downsample_voxel=0.0,
            fusion_voxel_size=0.1,
            aggregate_voxel=0.0,
            post_filter_stat_nb_neighbors=999,
            min_saved_aggregate_points=0,
            truncate_after_lane_end_touch=True,
        ),
        OutputConfig(require_track_exit=False, save_world=True),
        TrackingConfig(min_track_hits=1),
    )

    result = accumulator.accumulate(track, LaneBox.from_values([-1.0, 1.0, 0.0, 10.0, -1.0, 1.0]))

    assert result.status == "saved"
    assert result.selected_frame_ids == [201, 202]
    assert result.metrics["lane_end_touch_found"] is True
    assert result.metrics["lane_end_touch_frame_id"] == 202
    assert 203 not in result.selected_frame_ids


def test_lane_end_touch_filter_limits_tail_coverage_to_prefix() -> None:
    track = Track(track_id=91, hit_count=4, age=4, missed=1, ended_by_missed=True)
    for frame_id, center_y in ((10, 3.0), (11, 1.8), (12, 0.05), (13, -0.6)):
        points = np.array(
            [
                [0.0, center_y - 0.15, 0.0],
                [0.0, center_y + 0.15, 0.0],
            ],
            dtype=np.float32,
        )
        center = np.mean(points, axis=0).astype(np.float32)
        track.centers.append(center)
        track.frame_ids.append(frame_id)
        track.world_points.append(points.copy())
        track.local_points.append((points - center).astype(np.float32))
        track.bbox_extents.append(compute_extent(points))
    accumulator = VoxelFusionAccumulator(
        AggregationConfig(
            frame_selection_method="tail_coverage",
            use_all_frames=False,
            top_k_frames=2,
            keyframe_keep=2,
            length_coverage_bins=4,
            frame_downsample_voxel=0.0,
            fusion_voxel_size=0.1,
            aggregate_voxel=0.0,
            post_filter_stat_nb_neighbors=999,
            min_saved_aggregate_points=0,
            truncate_after_lane_end_touch=True,
        ),
        OutputConfig(require_track_exit=False, save_world=True),
        TrackingConfig(min_track_hits=1),
    )

    result = accumulator.accumulate(track, LaneBox.from_values([-1.0, 1.0, 0.0, 10.0, -1.0, 1.0]))

    assert result.status == "saved"
    assert result.selected_frame_ids == [11, 12]
    assert result.metrics["lane_end_touch_found"] is True
    assert result.metrics["tail_window_start_index"] == 1
    assert result.metrics["tail_window_size"] == 2
    assert 13 not in result.selected_frame_ids


def test_motion_deskew_disabled_leaves_distorted_local_extent_unchanged() -> None:
    track = _motion_distorted_track()
    accumulator = _motion_deskew_accumulator(enabled=False, save_world=False, long_vehicle_mode=True)

    result = accumulator.accumulate(track, LaneBox.from_values([-5.0, 5.0, -5.0, 25.0, -1.0, 1.0]))

    assert result.status == "saved"
    assert result.metrics["motion_deskew_enabled"] is False
    assert result.metrics["motion_deskew_applied"] is False
    assert result.metrics["motion_deskew_skipped_reason"] == "disabled"
    assert np.isclose(result.metrics["extent_y"], 1.0, atol=1e-4)


def test_motion_deskew_corrects_distorted_local_chunks_for_long_vehicle() -> None:
    track = _motion_distorted_track()
    accumulator = _motion_deskew_accumulator(enabled=True, save_world=False, long_vehicle_mode=True)

    result = accumulator.accumulate(track, LaneBox.from_values([-5.0, 5.0, -5.0, 25.0, -1.0, 1.0]))

    assert result.status == "saved"
    assert result.metrics["motion_deskew_enabled"] is True
    assert result.metrics["motion_deskew_applied"] is True
    assert result.metrics["motion_deskew_skipped_reason"] == "applied"
    assert result.metrics["motion_deskew_corrected_chunk_count"] == 3
    assert result.metrics["motion_deskew_confident_chunk_count"] == 3
    assert np.isclose(result.metrics["motion_deskew_mean_speed_mps"], 10.0, atol=1e-3)
    assert np.isclose(result.metrics["motion_deskew_mean_time_span_ms"], 100.0, atol=1e-3)
    assert result.metrics["motion_deskew_mean_abs_shift_m"] > 0.0
    assert result.metrics["extent_y"] < 1e-4


def test_motion_deskew_skips_non_elongated_candidate() -> None:
    track = _motion_distorted_track()
    accumulator = _motion_deskew_accumulator(enabled=True, save_world=False, long_vehicle_mode=False)

    result = accumulator.accumulate(track, LaneBox.from_values([-5.0, 5.0, -5.0, 25.0, -1.0, 1.0]))

    assert result.status == "saved"
    assert result.metrics["motion_deskew_applied"] is False
    assert result.metrics["motion_deskew_skipped_reason"] == "not_elongated_candidate"
    assert np.isclose(result.metrics["extent_y"], 1.0, atol=1e-4)


def test_motion_deskew_skips_when_point_timestamps_are_missing() -> None:
    track = _motion_distorted_track(with_point_timestamps=False)
    accumulator = _motion_deskew_accumulator(enabled=True, save_world=False, long_vehicle_mode=True)

    result = accumulator.accumulate(track, LaneBox.from_values([-5.0, 5.0, -5.0, 25.0, -1.0, 1.0]))

    assert result.status == "saved"
    assert result.metrics["motion_deskew_applied"] is False
    assert result.metrics["motion_deskew_skipped_reason"] == "no_point_timestamps"
    assert np.isclose(result.metrics["extent_y"], 1.0, atol=1e-4)


def test_symmetry_completion_cuts_at_plane_and_mirrors_stronger_half_with_intensity() -> None:
    track = _single_chunk_track(
        np.array(
            [
                [0.22, 0.00, 1.0],
                [0.22, 0.12, 1.0],
                [0.22, 0.24, 1.0],
            ],
            dtype=np.float32,
        ),
        center=np.zeros((3,), dtype=np.float32),
        intensity=np.array([0.2, 0.5, 0.8], dtype=np.float32),
    )
    accumulator = _symmetry_accumulator(enabled=True, save_world=False, fusion_voxel_size=0.10)

    result = accumulator.accumulate(track, LaneBox.from_values([-1.0, 1.0, -1.0, 2.0, 0.0, 2.0]))

    assert result.status == "saved"
    assert len(result.points) == 6
    assert result.intensity is not None
    negative_points = result.points[result.points[:, 0] < 0.0]
    positive_points = result.points[result.points[:, 0] > 0.0]
    assert len(negative_points) == 3
    assert len(positive_points) == 3
    assert np.allclose(np.sort(negative_points[:, 0]), np.array([-0.22, -0.22, -0.22], dtype=np.float32), atol=1e-6)
    assert np.allclose(result.intensity[result.points[:, 0] < 0.0], np.array([0.2, 0.5, 0.8], dtype=np.float32), atol=1e-6)
    assert result.metrics["symmetry_completion_enabled"] is True
    assert result.metrics["symmetry_completion_applied"] is True
    assert result.metrics["symmetry_completion_lateral_axis"] == "x"
    assert result.metrics["symmetry_completion_plane_coordinate"] == 0.0
    assert result.metrics["symmetry_completion_source_side"] == "positive"
    assert result.metrics["point_count_before_symmetry_completion"] == 3
    assert result.metrics["symmetry_completion_source_slice_count"] == 3
    assert result.metrics["symmetry_completion_target_slice_count"] == 0
    assert result.metrics["symmetry_completion_candidate_count"] == 3
    assert result.metrics["symmetry_completion_generated_points"] == 3
    assert result.metrics["point_count_after_symmetry_completion"] == 6
    assert result.metrics["point_count_after_downsample"] == 6
    assert result.metrics["symmetry_completion_skipped_reason"] == "applied"


def test_symmetry_completion_replaces_weaker_half_instead_of_overlapping_with_it() -> None:
    track = _single_chunk_track(
        np.array(
            [
                [0.22, 0.00, 1.0],
                [0.22, 0.12, 1.0],
                [0.22, 0.24, 1.0],
                [-0.18, 0.00, 1.0],
                [-0.18, 0.12, 1.0],
            ],
            dtype=np.float32,
        ),
        center=np.zeros((3,), dtype=np.float32),
    )
    accumulator = _symmetry_accumulator(enabled=True, save_world=False, fusion_voxel_size=0.10)

    result = accumulator.accumulate(track, LaneBox.from_values([-1.0, 1.0, -1.0, 1.0, 0.0, 2.0]))

    assert result.status == "saved"
    assert len(result.points) == 6
    assert result.metrics["symmetry_completion_applied"] is True
    assert result.metrics["symmetry_completion_source_side"] == "positive"
    assert result.metrics["symmetry_completion_candidate_count"] == 3
    assert result.metrics["symmetry_completion_generated_points"] == 3
    assert result.metrics["point_count_after_symmetry_completion"] == 6
    assert not np.any(np.isclose(result.points[:, 0], -0.18, atol=1e-6))
    assert np.allclose(np.sort(result.points[result.points[:, 0] < 0.0][:, 0]), np.array([-0.22, -0.22, -0.22], dtype=np.float32), atol=1e-6)


def test_symmetry_completion_uses_world_center_median_for_plane_coordinate() -> None:
    track = _single_chunk_track(
        np.array(
            [
                [5.22, 0.00, 1.0],
                [5.22, 0.12, 1.0],
                [5.22, 0.24, 1.0],
            ],
            dtype=np.float32,
        ),
        center=np.array([5.0, 0.5, 1.0], dtype=np.float32),
    )
    accumulator = _symmetry_accumulator(enabled=True, save_world=True, fusion_voxel_size=0.10)

    result = accumulator.accumulate(track, LaneBox.from_values([4.0, 6.0, -1.0, 2.0, 0.0, 2.0]))

    assert result.status == "saved"
    assert result.metrics["symmetry_completion_plane_coordinate"] == 5.0
    assert np.allclose(
        np.sort(result.points[:, 0]),
        np.array([4.78, 4.78, 4.78, 5.22, 5.22, 5.22], dtype=np.float32),
        atol=1e-6,
    )


def test_symmetry_completion_refines_local_plane_for_asymmetric_two_sided_aggregate() -> None:
    track = _single_chunk_track(
        np.array(
            [
                [-1.00, 0.00, 1.0],
                [-1.00, 0.12, 1.0],
                [0.50, 0.24, 1.0],
            ],
            dtype=np.float32,
        ),
        center=np.zeros((3,), dtype=np.float32),
    )
    accumulator = _symmetry_accumulator(enabled=True, save_world=False, fusion_voxel_size=0.10)

    result = accumulator.accumulate(track, LaneBox.from_values([-2.0, 2.0, -1.0, 3.0, 0.0, 2.0]))

    assert result.status == "saved"
    assert result.metrics["symmetry_completion_plane_coordinate"] < -0.1
    assert result.metrics["symmetry_completion_plane_coordinate"] > -0.6
    assert result.metrics["symmetry_completion_source_side"] == "negative"
    assert np.any(result.points[:, 0] > 0.3)


def test_symmetry_completion_does_not_change_min_saved_points_status() -> None:
    track = _single_chunk_track(
        np.array([[0.30, 0.00, 1.0]], dtype=np.float32),
        center=np.zeros((3,), dtype=np.float32),
    )
    accumulator = _symmetry_accumulator(enabled=True, save_world=False, min_saved_aggregate_points=2, fusion_voxel_size=0.10)

    result = accumulator.accumulate(track, LaneBox.from_values([-1.0, 1.0, -1.0, 1.0, 0.0, 2.0]))

    assert result.status == "skipped_min_saved_points"
    assert len(result.points) == 0
    assert result.metrics["symmetry_completion_enabled"] is True
    assert result.metrics["symmetry_completion_applied"] is False
    assert result.metrics["point_count_before_symmetry_completion"] == 1
    assert result.metrics["point_count_after_symmetry_completion"] == 1


def test_symmetry_completion_prefers_more_extended_half_when_counts_match() -> None:
    track = _single_chunk_track(
        np.array(
            [
                [0.22, 0.00, 1.0],
                [0.22, 0.12, 1.0],
                [0.22, 0.24, 1.0],
                [-0.18, 0.12, 1.0],
                [-0.18, 0.24, 1.0],
                [-0.18, 0.36, 1.0],
            ],
            dtype=np.float32,
        ),
        center=np.zeros((3,), dtype=np.float32),
    )
    accumulator = _symmetry_accumulator(enabled=True, save_world=False, fusion_voxel_size=0.10)

    result = accumulator.accumulate(track, LaneBox.from_values([-2.0, 2.0, -1.0, 1.0, 0.0, 2.0]))

    assert result.status == "saved"
    assert result.metrics["symmetry_completion_applied"] is True
    assert result.metrics["symmetry_completion_source_side"] == "positive"
    assert result.metrics["symmetry_completion_source_slice_count"] == 3
    assert result.metrics["symmetry_completion_target_slice_count"] == 3
    assert result.metrics["symmetry_completion_candidate_count"] == 3
    assert result.metrics["symmetry_completion_generated_points"] == 3
    assert np.allclose(np.sort(result.points[result.points[:, 0] < 0.0][:, 0]), np.array([-0.22, -0.22, -0.22], dtype=np.float32), atol=1e-6)


def test_symmetry_completion_skips_when_all_points_stay_in_center_strip() -> None:
    centered_track = _single_chunk_track(
        np.array(
            [
                [0.00, 0.00, 1.0],
                [0.02, 0.12, 1.0],
                [-0.02, 0.24, 1.0],
            ],
            dtype=np.float32,
        ),
        center=np.zeros((3,), dtype=np.float32),
    )
    accumulator = _symmetry_accumulator(enabled=True, save_world=False, fusion_voxel_size=0.10)

    result = accumulator.accumulate(centered_track, LaneBox.from_values([-2.0, 2.0, -1.0, 2.0, 0.0, 2.0]))

    assert result.status == "saved"
    assert result.metrics["symmetry_completion_applied"] is False
    assert result.metrics["symmetry_completion_skipped_reason"] == "no_off_plane_points"
    assert len(result.points) == 3


def test_registration_voxel_fusion_accumulator_falls_back_without_backend() -> None:
    accumulator = RegistrationVoxelFusionAccumulator(
        AggregationConfig(
            algorithm="registration_voxel_fusion",
            min_saved_aggregate_points=0,
            enable_registration_underfill_fallback=True,
            registration_min_kept_chunks=4,
        ),
        OutputConfig(require_track_exit=False),
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
    assert result.metrics["registration_fallback_applied"] is False
    assert result.metrics["registration_fallback_min_kept_chunks"] == 4
    assert result.metrics["registration_attempt_output_chunk_count"] == 3
    assert result.metrics["registration_attempt_dropped_count"] == 0
    assert result.metrics["registration_attempt_keep_indices"] == [0, 1, 2]
    assert result.metrics["registration_attempt_chunk_weights"] == [1.0, 1.0, 1.0]


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


def test_registration_underfill_fallback_restores_all_selected_chunks_and_attempt_metrics() -> None:
    chunks = [_constant_chunk(float(x_value)) for x_value in (0.0, 10.0, 20.0, 30.0)]
    intensities = [np.full((len(chunk),), value, dtype=np.float32) for chunk, value in zip(chunks, (0.1, 0.2, 0.3, 0.4))]
    track = _track_from_chunks(chunks, intensities=intensities, start_frame=200, track_id=71)
    accumulator = _ScriptedPrepareAccumulator(
        AggregationConfig(
            frame_selection_method="all_track_frames",
            frame_downsample_voxel=0.0,
            fusion_voxel_size=0.1,
            aggregate_voxel=0.0,
            post_filter_stat_nb_neighbors=999,
            min_saved_aggregate_points=0,
            enable_registration_underfill_fallback=True,
            registration_min_kept_chunks=4,
        ),
        OutputConfig(require_track_exit=False, save_world=True),
        TrackingConfig(min_track_hits=1),
        keep_indices=[0],
        chunk_weights=[1.0],
    )

    result = accumulator.accumulate(track, LaneBox.from_values([-1.0, 31.0, -1.0, 1.0, -1.0, 1.0]))

    assert result.status == "saved"
    assert result.selected_frame_ids == [200, 201, 202, 203]
    assert result.metrics["registration_fallback_applied"] is True
    assert result.metrics["registration_fallback_min_kept_chunks"] == 4
    assert result.metrics["registration_input_chunk_count"] == 4
    assert result.metrics["registration_output_chunk_count"] == 4
    assert result.metrics["registration_dropped_count"] == 0
    assert result.metrics["registration_keep_indices"] == [0, 1, 2, 3]
    assert result.metrics["registration_chunk_weights"] == [1.0, 1.0, 1.0, 1.0]
    assert result.metrics["registration_attempt_output_chunk_count"] == 1
    assert result.metrics["registration_attempt_dropped_count"] == 3
    assert result.metrics["registration_attempt_keep_indices"] == [0]
    assert result.metrics["registration_attempt_chunk_weights"] == [1.0]
    assert result.intensity is not None
    assert np.allclose(np.unique(np.round(result.intensity, decimals=1)), np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32))


def test_registration_underfill_fallback_does_not_apply_when_threshold_is_met() -> None:
    chunks = [_constant_chunk(float(x_value)) for x_value in (0.0, 10.0, 20.0, 30.0)]
    track = _track_from_chunks(chunks, start_frame=300, track_id=72)
    accumulator = _ScriptedPrepareAccumulator(
        AggregationConfig(
            frame_selection_method="all_track_frames",
            frame_downsample_voxel=0.0,
            fusion_voxel_size=0.1,
            aggregate_voxel=0.0,
            post_filter_stat_nb_neighbors=999,
            min_saved_aggregate_points=0,
            enable_registration_underfill_fallback=True,
            registration_min_kept_chunks=4,
        ),
        OutputConfig(require_track_exit=False, save_world=True),
        TrackingConfig(min_track_hits=1),
        keep_indices=[0, 1, 2, 3],
        chunk_weights=[1.0, 0.9, 0.8, 0.7],
    )

    result = accumulator.accumulate(track, LaneBox.from_values([-1.0, 31.0, -1.0, 1.0, -1.0, 1.0]))

    assert result.status == "saved"
    assert result.selected_frame_ids == [300, 301, 302, 303]
    assert result.metrics["registration_fallback_applied"] is False
    assert result.metrics["registration_output_chunk_count"] == 4
    assert result.metrics["registration_keep_indices"] == [0, 1, 2, 3]
    assert result.metrics["registration_chunk_weights"] == [1.0, 0.9, 0.8, 0.7]
    assert result.metrics["registration_attempt_output_chunk_count"] == 4
    assert result.metrics["registration_attempt_keep_indices"] == [0, 1, 2, 3]
    assert result.metrics["registration_attempt_chunk_weights"] == [1.0, 0.9, 0.8, 0.7]


def test_registration_underfill_fallback_restores_all_preselected_chunks_even_below_threshold() -> None:
    chunks = [_constant_chunk(float(x_value)) for x_value in (0.0, 10.0, 20.0)]
    track = _track_from_chunks(chunks, start_frame=400, track_id=73)
    accumulator = _ScriptedPrepareAccumulator(
        AggregationConfig(
            frame_selection_method="all_track_frames",
            frame_downsample_voxel=0.0,
            fusion_voxel_size=0.1,
            aggregate_voxel=0.0,
            post_filter_stat_nb_neighbors=999,
            min_saved_aggregate_points=0,
            enable_registration_underfill_fallback=True,
            registration_min_kept_chunks=4,
        ),
        OutputConfig(require_track_exit=False, save_world=True),
        TrackingConfig(min_track_hits=1),
        keep_indices=[0],
    )

    result = accumulator.accumulate(track, LaneBox.from_values([-1.0, 21.0, -1.0, 1.0, -1.0, 1.0]))

    assert result.status == "saved"
    assert result.selected_frame_ids == [400, 401, 402]
    assert result.metrics["registration_fallback_applied"] is True
    assert result.metrics["registration_output_chunk_count"] == 3
    assert result.metrics["registration_dropped_count"] == 0
    assert result.metrics["registration_keep_indices"] == [0, 1, 2]
    assert result.metrics["registration_chunk_weights"] == [1.0, 1.0, 1.0]
    assert result.metrics["registration_attempt_output_chunk_count"] == 1
    assert result.metrics["registration_attempt_keep_indices"] == [0]


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


def test_explicit_center_diversity_selection_is_not_overridden_for_long_vehicle() -> None:
    track = _track_from_profile([80, 90, 100, 110], [5.0, 5.2, 5.4, 5.6], start_frame=500)
    accumulator = WeightedVoxelFusionAccumulator(
        AggregationConfig(
            algorithm="weighted_voxel_fusion",
            frame_selection_method="center_diversity",
            keyframe_keep=3,
            long_vehicle_mode=True,
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
    assert result.metrics["long_vehicle_mode_applied"] is True
    assert result.metrics["frame_selection_method"] == "center_diversity"


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


def test_chunk_quality_filter_removes_weak_tail_before_quality_coverage() -> None:
    track = _track_from_profile(
        point_counts=[35, 70, 120, 130, 125, 30, 20, 15],
        extent_ys=[1.0, 2.0, 4.0, 4.2, 4.1, 0.9, 0.7, 0.5],
    )
    accumulator = VoxelFusionAccumulator(
        AggregationConfig(
            frame_selection_method="quality_coverage",
            keyframe_keep=3,
            length_coverage_bins=4,
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
    assert result.metrics["frame_selection_method"] == "quality_coverage"
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


def test_confidence_point_cap_disabled_leaves_saved_aggregate_unclipped() -> None:
    points = np.stack(
        [
            np.zeros((32,), dtype=np.float32),
            np.linspace(0.0, 3.1, 32, dtype=np.float32),
            np.zeros((32,), dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    track = _single_chunk_track(points)
    accumulator = VoxelFusionAccumulator(
        AggregationConfig(
            frame_selection_method="all_track_frames",
            frame_downsample_voxel=0.0,
            fusion_voxel_size=0.01,
            aggregate_voxel=0.0,
            post_filter_stat_nb_neighbors=999,
            min_saved_aggregate_points=0,
            enable_confidence_point_cap=False,
            confidence_point_cap_max_points=8,
            confidence_point_cap_bins=4,
        ),
        OutputConfig(require_track_exit=False, save_world=True),
        TrackingConfig(min_track_hits=1),
    )

    result = accumulator.accumulate(track, LaneBox.from_values([-1.0, 1.0, -1.0, 5.0, -1.0, 1.0]))

    assert result.status == "saved"
    assert len(result.points) == 32
    assert result.metrics["confidence_point_cap_enabled"] is False
    assert result.metrics["confidence_point_cap_applied"] is False
    assert result.metrics["point_count_before_confidence_cap"] == 32
    assert result.metrics["point_count_after_confidence_cap"] == 32


def test_confidence_point_cap_limits_saved_aggregate_to_target_points() -> None:
    points = np.stack(
        [
            np.zeros((32,), dtype=np.float32),
            np.linspace(0.0, 3.1, 32, dtype=np.float32),
            np.zeros((32,), dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    track = _single_chunk_track(points)
    accumulator = VoxelFusionAccumulator(
        AggregationConfig(
            frame_selection_method="all_track_frames",
            frame_downsample_voxel=0.0,
            fusion_voxel_size=0.01,
            aggregate_voxel=0.0,
            post_filter_stat_nb_neighbors=999,
            min_saved_aggregate_points=0,
            enable_confidence_point_cap=True,
            confidence_point_cap_max_points=8,
            confidence_point_cap_bins=4,
        ),
        OutputConfig(require_track_exit=False, save_world=True),
        TrackingConfig(min_track_hits=1),
    )

    result = accumulator.accumulate(track, LaneBox.from_values([-1.0, 1.0, -1.0, 5.0, -1.0, 1.0]))

    assert result.status == "saved"
    assert len(result.points) == 8
    assert result.metrics["confidence_point_cap_enabled"] is True
    assert result.metrics["confidence_point_cap_applied"] is True
    assert result.metrics["confidence_point_cap_target"] == 8
    assert result.metrics["confidence_point_cap_bins"] == 4
    assert result.metrics["point_count_before_confidence_cap"] == 32
    assert result.metrics["point_count_after_confidence_cap"] == 8
    assert result.metrics["point_count_after_downsample"] == 8


def test_confidence_point_cap_does_not_fill_when_aggregate_is_under_budget() -> None:
    points = np.stack(
        [
            np.zeros((6,), dtype=np.float32),
            np.linspace(0.0, 1.0, 6, dtype=np.float32),
            np.zeros((6,), dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    track = _single_chunk_track(points)
    accumulator = VoxelFusionAccumulator(
        AggregationConfig(
            frame_selection_method="all_track_frames",
            frame_downsample_voxel=0.0,
            fusion_voxel_size=0.01,
            aggregate_voxel=0.0,
            post_filter_stat_nb_neighbors=999,
            min_saved_aggregate_points=0,
            enable_confidence_point_cap=True,
            confidence_point_cap_max_points=8,
            confidence_point_cap_bins=4,
        ),
        OutputConfig(require_track_exit=False, save_world=True),
        TrackingConfig(min_track_hits=1),
    )

    result = accumulator.accumulate(track, LaneBox.from_values([-1.0, 1.0, -1.0, 5.0, -1.0, 1.0]))

    assert result.status == "saved"
    assert len(result.points) == 6
    assert result.metrics["confidence_point_cap_applied"] is False
    assert result.metrics["point_count_before_confidence_cap"] == 6
    assert result.metrics["point_count_after_confidence_cap"] == 6


def test_confidence_point_cap_runs_before_symmetry_completion() -> None:
    track = _single_chunk_track(
        np.array(
            [
                [0.22, 0.00, 1.0],
                [0.22, 0.12, 1.0],
                [0.22, 0.24, 1.0],
                [0.22, 0.36, 1.0],
            ],
            dtype=np.float32,
        ),
        center=np.zeros((3,), dtype=np.float32),
    )
    accumulator = VoxelFusionAccumulator(
        AggregationConfig(
            symmetry_completion=True,
            frame_selection_method="all_track_frames",
            frame_downsample_voxel=0.0,
            fusion_voxel_size=0.01,
            aggregate_voxel=0.0,
            post_filter_stat_nb_neighbors=999,
            min_saved_aggregate_points=0,
            enable_confidence_point_cap=True,
            confidence_point_cap_max_points=2,
            confidence_point_cap_bins=1,
        ),
        OutputConfig(require_track_exit=False, save_world=False),
        TrackingConfig(min_track_hits=1),
    )

    result = accumulator.accumulate(track, LaneBox.from_values([-1.0, 1.0, -1.0, 1.0, 0.0, 2.0]))

    assert result.status == "saved"
    assert result.metrics["confidence_point_cap_applied"] is True
    assert result.metrics["point_count_before_confidence_cap"] == 4
    assert result.metrics["point_count_after_confidence_cap"] == 2
    assert result.metrics["symmetry_completion_applied"] is True
    assert result.metrics["point_count_before_symmetry_completion"] == 2
    assert result.metrics["point_count_after_symmetry_completion"] == 4
    assert result.metrics["point_count_after_downsample"] == 4
    assert len(result.points) == 4


def test_confidence_point_cap_preserves_longitudinal_coverage_with_bins_plus_rest() -> None:
    first_bin = np.stack(
        [np.zeros((12,), dtype=np.float32), np.linspace(0.0, 0.9, 12, dtype=np.float32), np.zeros((12,), dtype=np.float32)],
        axis=1,
    ).astype(np.float32)
    second_bin = np.stack(
        [np.zeros((3,), dtype=np.float32), np.linspace(2.0, 2.2, 3, dtype=np.float32), np.zeros((3,), dtype=np.float32)],
        axis=1,
    ).astype(np.float32)
    third_bin = np.stack(
        [np.zeros((3,), dtype=np.float32), np.linspace(4.0, 4.2, 3, dtype=np.float32), np.zeros((3,), dtype=np.float32)],
        axis=1,
    ).astype(np.float32)
    fourth_bin = np.stack(
        [np.zeros((3,), dtype=np.float32), np.linspace(6.0, 6.2, 3, dtype=np.float32), np.zeros((3,), dtype=np.float32)],
        axis=1,
    ).astype(np.float32)
    track = _single_chunk_track(np.vstack([first_bin, second_bin, third_bin, fourth_bin]).astype(np.float32))
    accumulator = VoxelFusionAccumulator(
        AggregationConfig(
            frame_selection_method="all_track_frames",
            frame_selection_line_axis="y",
            frame_downsample_voxel=0.0,
            fusion_voxel_size=0.01,
            aggregate_voxel=0.0,
            post_filter_stat_nb_neighbors=999,
            min_saved_aggregate_points=0,
            enable_confidence_point_cap=True,
            confidence_point_cap_max_points=8,
            confidence_point_cap_bins=4,
        ),
        OutputConfig(require_track_exit=False, save_world=True),
        TrackingConfig(min_track_hits=1),
    )

    result = accumulator.accumulate(track, LaneBox.from_values([-1.0, 1.0, -1.0, 8.0, -1.0, 1.0]))

    assert result.status == "saved"
    assert len(result.points) == 8
    selected_y = np.asarray(result.points[:, 1], dtype=np.float32)
    assert np.sum(selected_y < 1.5) == 2
    assert np.sum((selected_y >= 1.5) & (selected_y < 3.0)) == 2
    assert np.sum((selected_y >= 3.0) & (selected_y < 5.0)) == 2
    assert np.sum(selected_y >= 5.0) == 2


def test_confidence_point_cap_prefers_points_with_higher_weighted_support() -> None:
    weak_chunk = np.array([[0.0, 0.2, 0.0]], dtype=np.float32)
    strong_chunk = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 1.1, 0.0],
            [0.0, 1.2, 0.0],
            [0.0, 1.3, 0.0],
        ],
        dtype=np.float32,
    )
    track = _track_from_chunks([weak_chunk, strong_chunk], track_id=77)
    accumulator = WeightedVoxelFusionAccumulator(
        AggregationConfig(
            algorithm="weighted_voxel_fusion",
            fusion_weight_mode="point_count",
            frame_selection_method="all_track_frames",
            frame_selection_line_axis="y",
            frame_downsample_voxel=0.0,
            fusion_voxel_size=0.01,
            aggregate_voxel=0.0,
            post_filter_stat_nb_neighbors=999,
            min_saved_aggregate_points=0,
            enable_confidence_point_cap=True,
            confidence_point_cap_max_points=2,
            confidence_point_cap_bins=1,
        ),
        OutputConfig(require_track_exit=False, save_world=True),
        TrackingConfig(min_track_hits=1),
    )

    result = accumulator.accumulate(track, LaneBox.from_values([-1.0, 1.0, -1.0, 4.0, -1.0, 1.0]))

    assert result.status == "saved"
    assert len(result.points) == 2
    assert np.all(np.asarray(result.points[:, 1], dtype=np.float32) >= 1.0)
