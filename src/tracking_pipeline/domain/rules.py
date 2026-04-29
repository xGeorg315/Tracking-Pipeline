from __future__ import annotations

from typing import Any

import numpy as np

from tracking_pipeline.domain.models import Track
from tracking_pipeline.domain.value_objects import LaneBox


def axis_to_index(axis: str | int) -> int:
    if isinstance(axis, str):
        axis = axis.strip().lower()
        return {"x": 0, "y": 1, "z": 2}.get(axis, 1)
    return int(np.clip(int(axis), 0, 2))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def lane_axis_bounds(lane_box: LaneBox, axis_idx: int) -> tuple[float, float]:
    if axis_idx == 0:
        return lane_box.x_min, lane_box.x_max
    if axis_idx == 1:
        return lane_box.y_min, lane_box.y_max
    return lane_box.z_min, lane_box.z_max


def orthogonal_axes(axis_idx: int) -> tuple[int, int]:
    remaining = [axis for axis in (0, 1, 2) if axis != int(axis_idx)]
    return int(remaining[0]), int(remaining[1])


def compute_extent(points: np.ndarray) -> np.ndarray:
    if points is None or len(points) == 0:
        return np.zeros((3,), dtype=np.float32)
    return (np.max(points, axis=0) - np.min(points, axis=0)).astype(np.float32)


def find_touch_start_index(
    chunks: list[np.ndarray],
    centers: list[np.ndarray],
    axis_idx: int,
    line_value: float,
    touch_margin: float = 0.12,
) -> int | None:
    margin = abs(float(touch_margin))
    for index, points in enumerate(chunks):
        if points is None or len(points) == 0:
            continue
        values = np.asarray(points, dtype=np.float64)[:, axis_idx]
        values = values[np.isfinite(values)]
        if len(values) == 0:
            continue
        if float(np.min(values)) - margin <= line_value <= float(np.max(values)) + margin:
            return index

    for index, center in enumerate(centers):
        if center is None:
            continue
        arr = np.asarray(center, dtype=np.float64)
        if arr.shape[0] <= axis_idx:
            continue
        if abs(float(arr[axis_idx]) - line_value) <= margin:
            return index
    return None


def find_lane_end_touch_index(
    chunks: list[np.ndarray],
    centers: list[np.ndarray],
    lane_box: LaneBox,
    axis_idx: int,
    touch_margin: float = 0.12,
) -> int | None:
    lane_end_value, _ = lane_axis_bounds(lane_box, axis_idx)
    return find_touch_start_index(chunks, centers, axis_idx, lane_end_value, touch_margin)


def select_keyframes_by_motion(centers: list[np.ndarray], keep: int) -> list[int]:
    if not centers:
        return []
    if len(centers) <= keep:
        return list(range(len(centers)))
    motion_scores = np.zeros((len(centers),), dtype=np.float64)
    arr = np.asarray(centers, dtype=np.float64)
    if len(arr) > 1:
        deltas = np.linalg.norm(arr[1:] - arr[:-1], axis=1)
        motion_scores[1:] += deltas
        motion_scores[:-1] += deltas
    candidate_indices = np.argsort(motion_scores)[::-1].tolist()
    chosen = {0, len(centers) - 1}
    for index in candidate_indices:
        chosen.add(int(index))
        if len(chosen) >= keep:
            break
    return sorted(chosen)


def select_keyframes_by_length_coverage(chunks: list[np.ndarray], axis_idx: int, keep: int, bins: int) -> list[int]:
    if not chunks:
        return []
    if len(chunks) <= keep:
        return list(range(len(chunks)))

    intervals = []
    for points in chunks:
        if points is None or len(points) == 0:
            intervals.append((0.0, 0.0))
            continue
        values = np.asarray(points, dtype=np.float64)[:, axis_idx]
        values = values[np.isfinite(values)]
        if len(values) == 0:
            intervals.append((0.0, 0.0))
        else:
            intervals.append((float(np.min(values)), float(np.max(values))))

    global_min = min(lo for lo, _ in intervals)
    global_max = max(hi for _, hi in intervals)
    if abs(global_max - global_min) <= 1e-6:
        spans = [hi - lo for lo, hi in intervals]
        return sorted(np.argsort(spans)[::-1][:keep].tolist())

    bin_edges = np.linspace(global_min, global_max, max(2, int(bins)) + 1)
    coverage: list[set[int]] = []
    spans = []
    for lo, hi in intervals:
        spans.append(hi - lo)
        covered = set()
        for index in range(len(bin_edges) - 1):
            left = float(bin_edges[index])
            right = float(bin_edges[index + 1])
            if hi < left or lo > right:
                continue
            covered.add(index)
        coverage.append(covered)

    start_idx = int(np.argmin([lo for lo, _ in intervals]))
    end_idx = int(np.argmax([hi for _, hi in intervals]))
    chosen = {start_idx, end_idx}
    covered_bins = set().union(*(coverage[index] for index in chosen))

    while len(chosen) < keep:
        best_index = None
        best_gain = -1
        best_span = -1.0
        for index, bins_covered in enumerate(coverage):
            if index in chosen:
                continue
            gain = len(bins_covered - covered_bins)
            span = float(spans[index])
            if gain > best_gain or (gain == best_gain and span > best_span):
                best_index = index
                best_gain = gain
                best_span = span
        if best_index is None:
            break
        chosen.add(int(best_index))
        covered_bins.update(coverage[best_index])

    if len(chosen) < keep:
        remaining = [index for index in np.argsort(spans)[::-1].tolist() if index not in chosen]
        for index in remaining:
            chosen.add(int(index))
            if len(chosen) >= keep:
                break
    return sorted(chosen)


def _chunk_axis_intervals(chunks: list[np.ndarray], axis_idx: int) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    for points in chunks:
        if points is None or len(points) == 0:
            intervals.append((0.0, 0.0))
            continue
        values = np.asarray(points, dtype=np.float64)[:, axis_idx]
        values = values[np.isfinite(values)]
        if len(values) == 0:
            intervals.append((0.0, 0.0))
        else:
            intervals.append((float(np.min(values)), float(np.max(values))))
    return intervals


def _chunk_quality_scores(chunks: list[np.ndarray], axis_idx: int) -> list[tuple[float, float, float]]:
    intervals = _chunk_axis_intervals(chunks, axis_idx)
    scores: list[tuple[float, float, float]] = []
    for points, (lo, hi) in zip(chunks, intervals):
        arr = np.asarray(points, dtype=np.float32)
        point_count = 0.0 if points is None else float(len(points))
        span = float(hi - lo)
        extent_norm = float(np.linalg.norm(compute_extent(arr)))
        scores.append((point_count, span, extent_norm))
    return scores


def _interval_coverage(intervals: list[tuple[float, float]], bins: int) -> tuple[list[set[int]], list[float]]:
    if not intervals:
        return [], []
    global_min = min(lo for lo, _ in intervals)
    global_max = max(hi for _, hi in intervals)
    spans = [float(hi - lo) for lo, hi in intervals]
    if abs(global_max - global_min) <= 1e-6:
        return [set() for _ in intervals], spans

    bin_edges = np.linspace(global_min, global_max, max(2, int(bins)) + 1)
    coverage: list[set[int]] = []
    for lo, hi in intervals:
        covered = set()
        for index in range(len(bin_edges) - 1):
            left = float(bin_edges[index])
            right = float(bin_edges[index + 1])
            if hi < left or lo > right:
                continue
            covered.add(index)
        coverage.append(covered)
    return coverage, spans


def select_keyframes_by_quality_coverage(chunks: list[np.ndarray], axis_idx: int, keep: int, bins: int) -> list[int]:
    if not chunks:
        return []
    if len(chunks) <= keep:
        return list(range(len(chunks)))

    intervals = _chunk_axis_intervals(chunks, axis_idx)
    quality_scores = _chunk_quality_scores(chunks, axis_idx)
    coverage, _ = _interval_coverage(intervals, bins)
    start_idx = int(np.argmin([lo for lo, _ in intervals]))
    end_idx = int(np.argmax([hi for _, hi in intervals]))
    chosen = {start_idx, end_idx}
    covered_bins = set().union(*(coverage[index] for index in chosen))

    while len(chosen) < keep:
        best_index = None
        best_gain = -1
        best_score = None
        for index, bins_covered in enumerate(coverage):
            if index in chosen:
                continue
            gain = len(bins_covered - covered_bins)
            score = (*quality_scores[index], index)
            if gain > best_gain or (gain == best_gain and (best_score is None or score > best_score)):
                best_index = index
                best_gain = gain
                best_score = score
        if best_index is None or best_gain <= 0:
            break
        chosen.add(int(best_index))
        covered_bins.update(coverage[best_index])

    if len(chosen) < keep:
        remaining = [index for index in range(len(chunks)) if index not in chosen]
        remaining.sort(key=lambda index: (*quality_scores[index], index), reverse=True)
        for index in remaining:
            chosen.add(int(index))
            if len(chosen) >= keep:
                break
    return sorted(chosen)


def select_keyframes_by_tail_coverage(chunks: list[np.ndarray], axis_idx: int, keep: int, bins: int, top_k: int) -> tuple[list[int], int]:
    if not chunks:
        return [], 0
    window_size = min(len(chunks), max(1, int(top_k)))
    start_index = max(0, len(chunks) - window_size)
    chosen = select_keyframes_by_quality_coverage(chunks[start_index:], axis_idx, keep, bins)
    return [start_index + int(index) for index in chosen], start_index


def select_keyframes_by_center_diversity(
    chunks: list[np.ndarray],
    centers: list[np.ndarray],
    axis_idx: int,
    keep: int,
) -> list[int]:
    if not centers:
        return []
    if len(centers) <= keep:
        return list(range(len(centers)))

    arr = np.asarray(centers, dtype=np.float64)
    quality_scores = _chunk_quality_scores(chunks, axis_idx)
    spans = [score[1] for score in quality_scores]
    point_counts = [score[0] for score in quality_scores]
    chosen = {0, len(centers) - 1}

    while len(chosen) < keep:
        best_index = None
        best_key = None
        for index in range(len(centers)):
            if index in chosen:
                continue
            min_dist = min(float(np.linalg.norm(arr[index] - arr[selected])) for selected in chosen)
            key = (min_dist, point_counts[index], spans[index], index)
            if best_key is None or key > best_key:
                best_index = index
                best_key = key
        if best_index is None:
            break
        chosen.add(int(best_index))
    return sorted(chosen)


def select_best_frames_for_aggregation(
    chunks: list[np.ndarray],
    centers: list[np.ndarray],
    frame_ids: list[int],
    frame_selection_method: str,
    use_all_frames: bool,
    top_k: int,
    keyframe_keep: int,
    length_coverage_bins: int,
    lane_box: LaneBox,
    line_axis: str,
    line_ratio: float,
    line_touch_margin: float,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int], dict[str, Any]]:
    if not chunks:
        return [], [], [], {}

    method = frame_selection_method
    if method == "auto":
        method = "all_track_frames" if use_all_frames else "line_touch_last_k"

    if method == "all_track_frames":
        return list(chunks), list(centers), list(frame_ids), {
            "strategy": "all_track_frames",
            "touch_found": None,
            "candidate_count": len(chunks),
        }

    if method == "keyframe_motion":
        chosen = select_keyframes_by_motion(centers, max(1, int(keyframe_keep)))
        return (
            [chunks[index] for index in chosen],
            [centers[index] for index in chosen],
            [frame_ids[index] for index in chosen],
            {
                "strategy": "keyframe_motion",
                "candidate_count": len(chunks),
                "selected_end_index": int(chosen[-1]),
            },
        )

    if method == "length_coverage":
        axis_idx = axis_to_index(line_axis)
        chosen = select_keyframes_by_length_coverage(chunks, axis_idx, max(1, int(keyframe_keep)), max(2, int(length_coverage_bins)))
        return (
            [chunks[index] for index in chosen],
            [centers[index] for index in chosen],
            [frame_ids[index] for index in chosen],
            {
                "strategy": "length_coverage",
                "line_axis": ["x", "y", "z"][axis_idx],
                "candidate_count": len(chunks),
                "selected_end_index": int(chosen[-1]),
            },
        )

    if method == "quality_coverage":
        axis_idx = axis_to_index(line_axis)
        chosen = select_keyframes_by_quality_coverage(chunks, axis_idx, max(1, int(keyframe_keep)), max(2, int(length_coverage_bins)))
        return (
            [chunks[index] for index in chosen],
            [centers[index] for index in chosen],
            [frame_ids[index] for index in chosen],
            {
                "strategy": "quality_coverage",
                "line_axis": ["x", "y", "z"][axis_idx],
                "candidate_count": len(chunks),
                "selected_end_index": int(chosen[-1]),
            },
        )

    if method == "tail_coverage":
        axis_idx = axis_to_index(line_axis)
        chosen, tail_window_start_index = select_keyframes_by_tail_coverage(
            chunks,
            axis_idx,
            max(1, int(keyframe_keep)),
            max(2, int(length_coverage_bins)),
            max(1, int(top_k)),
        )
        return (
            [chunks[index] for index in chosen],
            [centers[index] for index in chosen],
            [frame_ids[index] for index in chosen],
            {
                "strategy": "tail_coverage",
                "line_axis": ["x", "y", "z"][axis_idx],
                "candidate_count": min(len(chunks), max(1, int(top_k))),
                "selected_end_index": int(chosen[-1]),
                "tail_window_start_index": int(tail_window_start_index),
                "tail_window_size": min(len(chunks), max(1, int(top_k))),
            },
        )

    if method == "center_diversity":
        axis_idx = axis_to_index(line_axis)
        chosen = select_keyframes_by_center_diversity(chunks, centers, axis_idx, max(1, int(keyframe_keep)))
        return (
            [chunks[index] for index in chosen],
            [centers[index] for index in chosen],
            [frame_ids[index] for index in chosen],
            {
                "strategy": "center_diversity",
                "line_axis": ["x", "y", "z"][axis_idx],
                "candidate_count": len(chunks),
                "selected_end_index": int(chosen[-1]),
            },
        )

    axis_idx = axis_to_index(line_axis)
    lo, hi = lane_axis_bounds(lane_box, axis_idx)
    line_value = lo + clamp01(line_ratio) * (hi - lo)
    touch_idx = find_touch_start_index(chunks, centers, axis_idx, line_value, line_touch_margin)
    candidate_indices = list(range(0, touch_idx + 1)) if touch_idx is not None else list(range(len(chunks)))
    chosen = candidate_indices[-max(1, int(top_k)) :]
    return (
        [chunks[i] for i in chosen],
        [centers[i] for i in chosen],
        [frame_ids[i] for i in chosen],
        {
            "strategy": "line_touch_last_k",
            "line_axis": ["x", "y", "z"][axis_idx],
            "line_ratio": float(clamp01(line_ratio)),
            "line_value": float(line_value),
            "line_touch_margin": float(abs(line_touch_margin)),
            "touch_found": touch_idx is not None,
            "touch_index": -1 if touch_idx is None else int(touch_idx),
            "candidate_count": len(candidate_indices),
            "selected_end_index": -1 if not chosen else int(chosen[-1]),
        },
    )


def filter_chunks_by_shape_consistency(
    chunks: list[np.ndarray],
    centers: list[np.ndarray],
    frame_ids: list[int],
    max_extent_ratio: float,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int], dict[str, Any]]:
    if not chunks:
        return [], [], [], {"shape_consistency_filter": False, "shape_consistency_kept": 0}

    extents = np.asarray([compute_extent(chunk) for chunk in chunks], dtype=np.float64)
    norms = np.linalg.norm(extents, axis=1)
    median_norm = float(np.median(norms)) if len(norms) else 0.0
    if median_norm <= 1e-6:
        kept_indices = list(range(len(chunks)))
    else:
        kept_indices = [
            index
            for index, norm in enumerate(norms)
            if (median_norm / float(max_extent_ratio)) <= float(norm) <= (median_norm * float(max_extent_ratio))
        ]
    if not kept_indices:
        kept_indices = [int(np.argmax(norms))] if len(norms) else []
    return (
        [chunks[index] for index in kept_indices],
        [centers[index] for index in kept_indices],
        [frame_ids[index] for index in kept_indices],
        {
            "shape_consistency_filter": True,
            "shape_consistency_kept": len(kept_indices),
            "shape_consistency_total": len(chunks),
        },
    )


def moving_average_centers(centers: list[np.ndarray], window: int) -> list[np.ndarray]:
    if window <= 1 or len(centers) <= 2:
        return [np.asarray(center, dtype=np.float32).copy() for center in centers]
    half_window = window // 2
    smoothed = []
    for index in range(len(centers)):
        start = max(0, index - half_window)
        end = min(len(centers), index + half_window + 1)
        smoothed.append(np.mean(np.asarray(centers[start:end], dtype=np.float32), axis=0).astype(np.float32))
    return smoothed


def track_exit_line_value(lane_box: LaneBox, axis: str | int, edge_margin: float = 0.9) -> float:
    axis_idx = axis_to_index(axis)
    axis_min, _ = lane_axis_bounds(lane_box, axis_idx)
    return float(axis_min) + float(edge_margin)


def track_exited_lane_box(
    track: Track,
    lane_box: LaneBox,
    edge_margin: float = 0.9,
    axis: str | int = "y",
) -> bool:
    if not track.centers:
        return False
    axis_idx = axis_to_index(axis)
    last = np.asarray(track.centers[-1], dtype=np.float64)
    if last.shape[0] <= axis_idx:
        return False
    line_value = track_exit_line_value(lane_box, axis_idx, edge_margin=edge_margin)
    return float(last[axis_idx]) <= float(line_value) and bool(track.ended_by_missed or track.missed > 0)


def is_valid_transform(transform: np.ndarray, max_translation: float | None) -> bool:
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        return False
    if not np.allclose(transform[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-3):
        return False
    det = float(np.linalg.det(transform[:3, :3]))
    if abs(det) < 1e-3 or abs(det) > 5.0:
        return False
    if max_translation is not None and np.linalg.norm(transform[:3, 3]) > float(max_translation):
        return False
    return True
