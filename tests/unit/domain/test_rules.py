from __future__ import annotations

import numpy as np

from tracking_pipeline.domain.models import Track
from tracking_pipeline.domain.rules import (
    filter_chunks_by_shape_consistency,
    find_lane_end_touch_index,
    is_valid_transform,
    select_best_frames_for_aggregation,
    track_exited_lane_box,
)
from tracking_pipeline.domain.value_objects import LaneBox


def test_lane_box_mask_filters_points() -> None:
    lane_box = LaneBox.from_values([-1.0, 1.0, -2.0, 2.0, 0.0, 2.0])
    points = np.array([[0.0, 0.0, 1.0], [2.0, 0.0, 1.0], [0.0, 1.0, -1.0]], dtype=np.float32)
    mask = lane_box.mask(points)
    assert mask.tolist() == [True, False, False]


def test_select_best_frames_for_aggregation_limits_to_last_k_before_touch() -> None:
    lane_box = LaneBox.from_values([0.0, 10.0, 0.0, 20.0, 0.0, 2.0])
    chunks = [
        np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
        np.array([[0.0, 2.0, 0.0]], dtype=np.float32),
        np.array([[0.0, 3.0, 0.0]], dtype=np.float32),
        np.array([[0.0, 4.0, 0.0]], dtype=np.float32),
    ]
    centers = [chunk.mean(axis=0) for chunk in chunks]
    selected_chunks, _, frame_ids, info = select_best_frames_for_aggregation(
        chunks=chunks,
        centers=centers,
        frame_ids=[10, 11, 12, 13],
        frame_selection_method="line_touch_last_k",
        use_all_frames=False,
        top_k=2,
        keyframe_keep=2,
        length_coverage_bins=4,
        lane_box=lane_box,
        line_axis="y",
        line_ratio=0.20,
        line_touch_margin=0.05,
    )
    assert len(selected_chunks) == 2
    assert frame_ids == [12, 13]
    assert info["touch_found"] is True


def test_select_best_frames_for_aggregation_keyframe_motion_keeps_endpoints() -> None:
    lane_box = LaneBox.from_values([0.0, 10.0, 0.0, 20.0, 0.0, 2.0])
    chunks = [np.array([[0.0, float(y), 0.0]], dtype=np.float32) for y in [1.0, 2.0, 5.0, 6.0, 10.0]]
    centers = [chunk.mean(axis=0) for chunk in chunks]
    _, _, frame_ids, info = select_best_frames_for_aggregation(
        chunks=chunks,
        centers=centers,
        frame_ids=[10, 11, 12, 13, 14],
        frame_selection_method="keyframe_motion",
        use_all_frames=False,
        top_k=2,
        keyframe_keep=3,
        length_coverage_bins=4,
        lane_box=lane_box,
        line_axis="y",
        line_ratio=0.20,
        line_touch_margin=0.05,
    )
    assert frame_ids[0] == 10
    assert frame_ids[-1] == 14
    assert len(frame_ids) == 3
    assert info["strategy"] == "keyframe_motion"


def test_select_best_frames_for_aggregation_length_coverage_preserves_longitudinal_span() -> None:
    lane_box = LaneBox.from_values([0.0, 4.0, 0.0, 20.0, 0.0, 2.0])
    chunks = [
        np.array([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32),
        np.array([[0.0, 2.0, 0.0], [0.0, 4.0, 0.0]], dtype=np.float32),
        np.array([[0.0, 6.0, 0.0], [0.0, 9.5, 0.0]], dtype=np.float32),
        np.array([[0.0, 10.0, 0.0], [0.0, 14.0, 0.0]], dtype=np.float32),
    ]
    centers = [chunk.mean(axis=0) for chunk in chunks]
    _, _, frame_ids, info = select_best_frames_for_aggregation(
        chunks=chunks,
        centers=centers,
        frame_ids=[20, 21, 22, 23],
        frame_selection_method="length_coverage",
        use_all_frames=False,
        top_k=2,
        keyframe_keep=3,
        length_coverage_bins=6,
        lane_box=lane_box,
        line_axis="y",
        line_ratio=0.2,
        line_touch_margin=0.05,
    )
    assert frame_ids[0] == 20
    assert frame_ids[-1] == 23
    assert 22 in frame_ids
    assert info["strategy"] == "length_coverage"


def test_select_best_frames_for_aggregation_quality_coverage_prefers_higher_quality_with_equal_coverage() -> None:
    lane_box = LaneBox.from_values([0.0, 4.0, 0.0, 20.0, 0.0, 2.0])
    chunks = [
        np.stack([np.zeros((8,), dtype=np.float32), np.linspace(0.0, 0.8, 8, dtype=np.float32), np.zeros((8,), dtype=np.float32)], axis=1),
        np.stack([np.zeros((30,), dtype=np.float32), np.linspace(1.0, 1.8, 30, dtype=np.float32), np.zeros((30,), dtype=np.float32)], axis=1),
        np.stack([np.zeros((10,), dtype=np.float32), np.linspace(2.0, 2.8, 10, dtype=np.float32), np.zeros((10,), dtype=np.float32)], axis=1),
        np.stack([np.zeros((8,), dtype=np.float32), np.linspace(3.0, 3.8, 8, dtype=np.float32), np.zeros((8,), dtype=np.float32)], axis=1),
    ]
    centers = [chunk.mean(axis=0) for chunk in chunks]
    _, _, frame_ids, info = select_best_frames_for_aggregation(
        chunks=chunks,
        centers=centers,
        frame_ids=[30, 31, 32, 33],
        frame_selection_method="quality_coverage",
        use_all_frames=False,
        top_k=2,
        keyframe_keep=3,
        length_coverage_bins=4,
        lane_box=lane_box,
        line_axis="y",
        line_ratio=0.2,
        line_touch_margin=0.05,
    )
    assert frame_ids == [30, 31, 33]
    assert info["strategy"] == "quality_coverage"
    assert info["candidate_count"] == 4


def test_select_best_frames_for_aggregation_tail_coverage_uses_tail_window() -> None:
    lane_box = LaneBox.from_values([0.0, 4.0, 0.0, 20.0, 0.0, 2.0])
    chunks = [
        np.stack([np.zeros((6,), dtype=np.float32), np.linspace(float(index), float(index) + 0.6, 6, dtype=np.float32), np.zeros((6,), dtype=np.float32)], axis=1)
        for index in range(6)
    ]
    centers = [chunk.mean(axis=0) for chunk in chunks]
    _, _, frame_ids, info = select_best_frames_for_aggregation(
        chunks=chunks,
        centers=centers,
        frame_ids=[10, 11, 12, 13, 14, 15],
        frame_selection_method="tail_coverage",
        use_all_frames=False,
        top_k=3,
        keyframe_keep=2,
        length_coverage_bins=3,
        lane_box=lane_box,
        line_axis="y",
        line_ratio=0.2,
        line_touch_margin=0.05,
    )
    assert frame_ids == [13, 15]
    assert info["strategy"] == "tail_coverage"
    assert info["tail_window_start_index"] == 3
    assert info["tail_window_size"] == 3


def test_select_best_frames_for_aggregation_center_diversity_keeps_endpoints_and_spreads_centers() -> None:
    lane_box = LaneBox.from_values([0.0, 4.0, 0.0, 20.0, 0.0, 2.0])
    centers = [
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([10.0, 0.0, 0.0], dtype=np.float32),
        np.array([11.0, 0.0, 0.0], dtype=np.float32),
        np.array([20.0, 0.0, 0.0], dtype=np.float32),
    ]
    chunks = [
        np.array([[center[0], 0.0, 0.0], [center[0], 0.2, 0.0]], dtype=np.float32)
        for center in centers
    ]
    _, _, frame_ids, info = select_best_frames_for_aggregation(
        chunks=chunks,
        centers=centers,
        frame_ids=[40, 41, 42, 43, 44],
        frame_selection_method="center_diversity",
        use_all_frames=False,
        top_k=2,
        keyframe_keep=3,
        length_coverage_bins=4,
        lane_box=lane_box,
        line_axis="y",
        line_ratio=0.2,
        line_touch_margin=0.05,
    )
    assert frame_ids == [40, 42, 44]
    assert info["strategy"] == "center_diversity"


def test_find_lane_end_touch_index_uses_first_point_touch_on_axis_min() -> None:
    lane_box = LaneBox.from_values([-1.0, 1.0, 0.0, 20.0, 0.0, 2.0])
    chunks = [
        np.array([[0.0, 2.0, 0.0], [0.0, 2.5, 0.0]], dtype=np.float32),
        np.array([[0.0, 0.03, 0.0], [0.0, 0.30, 0.0]], dtype=np.float32),
        np.array([[0.0, -0.4, 0.0], [0.0, 0.1, 0.0]], dtype=np.float32),
    ]
    centers = [chunk.mean(axis=0) for chunk in chunks]

    touch_idx = find_lane_end_touch_index(chunks, centers, lane_box, axis_idx=1, touch_margin=0.05)

    assert touch_idx == 1


def test_find_lane_end_touch_index_returns_none_without_touch() -> None:
    lane_box = LaneBox.from_values([-1.0, 1.0, 0.0, 20.0, 0.0, 2.0])
    chunks = [
        np.array([[0.0, 1.0, 0.0], [0.0, 1.2, 0.0]], dtype=np.float32),
        np.array([[0.0, 0.4, 0.0], [0.0, 0.6, 0.0]], dtype=np.float32),
    ]
    centers = [chunk.mean(axis=0) for chunk in chunks]

    touch_idx = find_lane_end_touch_index(chunks, centers, lane_box, axis_idx=1, touch_margin=0.05)

    assert touch_idx is None


def test_find_lane_end_touch_index_falls_back_to_center_when_points_miss() -> None:
    lane_box = LaneBox.from_values([-1.0, 1.0, 0.0, 20.0, 0.0, 2.0])
    chunks = [
        np.array([[0.0, 1.0, 0.0], [0.0, 1.2, 0.0]], dtype=np.float32),
        np.array([[0.0, 0.20, 0.0], [0.0, 0.22, 0.0]], dtype=np.float32),
    ]
    centers = [
        np.array([0.0, 1.1, 0.0], dtype=np.float32),
        np.array([0.0, 0.03, 0.0], dtype=np.float32),
    ]

    touch_idx = find_lane_end_touch_index(chunks, centers, lane_box, axis_idx=1, touch_margin=0.05)

    assert touch_idx == 1


def test_filter_chunks_by_shape_consistency_removes_large_outlier() -> None:
    chunks = [
        np.array([[0.0, 0.0, 0.0], [0.2, 0.2, 0.2]], dtype=np.float32),
        np.array([[0.0, 0.0, 0.0], [0.25, 0.2, 0.2]], dtype=np.float32),
        np.array([[0.0, 0.0, 0.0], [4.0, 4.0, 4.0]], dtype=np.float32),
    ]
    centers = [chunk.mean(axis=0) for chunk in chunks]
    kept_chunks, _, kept_frames, info = filter_chunks_by_shape_consistency(chunks, centers, [1, 2, 3], max_extent_ratio=1.5)
    assert len(kept_chunks) == 2
    assert kept_frames == [1, 2]
    assert info["shape_consistency_kept"] == 2


def test_track_exited_lane_box_requires_crossing_exit_line_and_missed() -> None:
    lane_box = LaneBox.from_values([-1.0, 1.0, 0.0, 10.0, 0.0, 2.0])
    track = Track(track_id=1, hit_count=5, age=6, missed=1, ended_by_missed=True)
    track.centers.append(np.array([0.95, 0.05, 1.0], dtype=np.float32))
    assert track_exited_lane_box(track, lane_box, edge_margin=0.1) is True


def test_track_exited_lane_box_rejects_center_before_exit_line() -> None:
    lane_box = LaneBox.from_values([-1.0, 1.0, 0.0, 10.0, 0.0, 2.0])
    track = Track(track_id=1, hit_count=5, age=6, missed=1, ended_by_missed=True)
    track.centers.append(np.array([0.95, 0.25, 1.0], dtype=np.float32))
    assert track_exited_lane_box(track, lane_box, edge_margin=0.1) is False


def test_track_exited_lane_box_rejects_active_track_even_after_crossing_line() -> None:
    lane_box = LaneBox.from_values([-1.0, 1.0, 0.0, 10.0, 0.0, 2.0])
    track = Track(track_id=1, hit_count=5, age=6, missed=0, ended_by_missed=False)
    track.centers.append(np.array([0.95, 0.05, 1.0], dtype=np.float32))
    assert track_exited_lane_box(track, lane_box, edge_margin=0.1) is False


def test_is_valid_transform_rejects_large_translation() -> None:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = np.array([10.0, 0.0, 0.0])
    assert is_valid_transform(transform, max_translation=3.0) is False
