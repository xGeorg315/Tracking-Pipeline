from __future__ import annotations

from dataclasses import asdict

import numpy as np

from tracking_pipeline.application.class_normalization import ClassNormalizer
from tracking_pipeline.domain.models import AggregateResult, GTMatchResult, ObjectLabelData, Track
from tracking_pipeline.infrastructure.tracking.assignment import assign_cost_matrix


GT_MATCH_MODE = "track_center_trajectory"
GT_MATCH_ASSIGNMENT = "one_to_one"


def match_saved_aggregates_to_gt(
    tracks: dict[int, Track],
    aggregate_results: list[AggregateResult],
    object_labels_by_id: dict[int, ObjectLabelData] | dict[int, list[ObjectLabelData]],
    class_normalizer: ClassNormalizer | None = None,
) -> tuple[list[GTMatchResult], list[GTMatchResult], list[GTMatchResult], dict[str, int | float | str]]:
    saved_results = sorted(
        [result for result in aggregate_results if str(result.status) == "saved"],
        key=lambda result: int(result.track_id),
    )
    saved_tracks = [tracks[int(result.track_id)] for result in saved_results if int(result.track_id) in tracks]
    label_histories = _normalize_object_label_histories(object_labels_by_id, class_normalizer)
    gt_object_ids = sorted(label_histories)
    gt_histories = [label_histories[object_id] for object_id in gt_object_ids]
    gt_labels = [history[-1] for history in gt_histories]

    if not saved_tracks and not gt_labels:
        return [], [], [], _summary_metrics([])
    if not saved_tracks:
        unmatched_gt = [
            GTMatchResult(
                track_id=-1,
                gt_object_id=int(label.object_id),
                our_last_timestamp_ns=-1,
                gt_timestamp_ns=int(label.timestamp_ns),
                timestamp_delta_ns=None,
                our_last_frame_id=-1,
                gt_frame_index=int(label.frame_index),
                assignment_cost=None,
                matched=False,
                unmatched_reason="unmatched_gt",
                gt_obj_class=_normalize_gt_class_name(str(label.obj_class or ""), class_normalizer),
                gt_obj_class_score=float(label.obj_class_score),
            )
            for label in gt_labels
        ]
        return [], [], unmatched_gt, _summary_metrics([])
    if not gt_labels:
        unmatched_saved = [
            GTMatchResult(
                track_id=int(track.track_id),
                gt_object_id=None,
                our_last_timestamp_ns=_track_last_timestamp_ns(track),
                gt_timestamp_ns=None,
                timestamp_delta_ns=None,
                our_last_frame_id=int(track.last_frame),
                gt_frame_index=None,
                assignment_cost=None,
                matched=False,
                unmatched_reason="unmatched_no_gt_available",
            )
            for track in saved_tracks
        ]
        return [], unmatched_saved, [], _summary_metrics(unmatched_saved)

    match_cost, pair_metadata = _trajectory_cost_matrix(saved_tracks, gt_object_ids, gt_histories)
    valid_mask = np.ones_like(match_cost, dtype=bool)
    assignment_rows, unmatched_rows, unmatched_cols = assign_cost_matrix(match_cost, valid_mask, method="hungarian")

    matched_results: list[GTMatchResult] = []
    for row in sorted(assignment_rows):
        col = int(assignment_rows[row])
        track = saved_tracks[int(row)]
        pair_info = pair_metadata[(int(row), col)]
        report_label = pair_info["report_label"]
        timestamp_delta_ns = _track_timestamp_delta_ns(track, report_label)
        matched_results.append(
            GTMatchResult(
                track_id=int(track.track_id),
                gt_object_id=int(report_label.object_id),
                our_last_timestamp_ns=_track_last_timestamp_ns(track),
                gt_timestamp_ns=int(report_label.timestamp_ns),
                timestamp_delta_ns=timestamp_delta_ns,
                our_last_frame_id=int(track.last_frame),
                gt_frame_index=int(report_label.frame_index),
                assignment_cost=float(match_cost[int(row), col]),
                matched=True,
                gt_obj_class=_normalize_gt_class_name(str(report_label.obj_class or ""), class_normalizer),
                gt_obj_class_score=float(report_label.obj_class_score),
            )
        )

    unmatched_saved = [
        GTMatchResult(
            track_id=int(saved_tracks[int(row)].track_id),
            gt_object_id=None,
            our_last_timestamp_ns=_track_last_timestamp_ns(saved_tracks[int(row)]),
            gt_timestamp_ns=None,
            timestamp_delta_ns=None,
            our_last_frame_id=int(saved_tracks[int(row)].last_frame),
            gt_frame_index=None,
            assignment_cost=None,
            matched=False,
            unmatched_reason="unmatched_no_gt_available",
        )
        for row in sorted(unmatched_rows)
    ]
    unmatched_gt = [
        GTMatchResult(
            track_id=-1,
            gt_object_id=int(gt_labels[int(col)].object_id),
            our_last_timestamp_ns=-1,
            gt_timestamp_ns=int(gt_labels[int(col)].timestamp_ns),
            timestamp_delta_ns=None,
            our_last_frame_id=-1,
            gt_frame_index=int(gt_labels[int(col)].frame_index),
            assignment_cost=None,
            matched=False,
            unmatched_reason="unmatched_gt",
            gt_obj_class=_normalize_gt_class_name(str(gt_labels[int(col)].obj_class or ""), class_normalizer),
            gt_obj_class_score=float(gt_labels[int(col)].obj_class_score),
        )
        for col in sorted(unmatched_cols)
    ]
    all_saved_results = matched_results + unmatched_saved
    return matched_results, unmatched_saved, unmatched_gt, _summary_metrics(all_saved_results, unmatched_gt)


def apply_gt_matches_to_results(
    aggregate_results: list[AggregateResult],
    matches: list[GTMatchResult],
    unmatched_saved: list[GTMatchResult],
) -> None:
    match_by_track_id = {int(match.track_id): match for match in matches + unmatched_saved}
    for result in aggregate_results:
        if str(result.status) != "saved":
            continue
        match = match_by_track_id.get(int(result.track_id))
        if match is None:
            continue
        result.metrics["gt_match_mode"] = GT_MATCH_MODE
        result.metrics["gt_match_assignment"] = GT_MATCH_ASSIGNMENT
        result.metrics["gt_matched"] = bool(match.matched)
        if match.matched:
            result.metrics["gt_object_id"] = int(match.gt_object_id)
            result.metrics["gt_timestamp_ns"] = int(match.gt_timestamp_ns)
            if match.timestamp_delta_ns is not None:
                result.metrics["gt_timestamp_delta_ns"] = int(match.timestamp_delta_ns)
            else:
                result.metrics.pop("gt_timestamp_delta_ns", None)
            result.metrics["gt_frame_index"] = int(match.gt_frame_index)
            result.metrics["gt_assignment_cost"] = float(match.assignment_cost)
            result.metrics.pop("gt_unmatched_reason", None)
            if match.gt_obj_class:
                result.metrics["gt_obj_class"] = str(match.gt_obj_class)
            else:
                result.metrics.pop("gt_obj_class", None)
            if match.gt_obj_class_score is not None:
                result.metrics["gt_obj_class_score"] = float(match.gt_obj_class_score)
            else:
                result.metrics.pop("gt_obj_class_score", None)
        else:
            result.metrics["gt_unmatched_reason"] = str(match.unmatched_reason)
            result.metrics.pop("gt_obj_class", None)
            result.metrics.pop("gt_obj_class_score", None)


def match_rows(matches: list[GTMatchResult]) -> list[dict[str, object]]:
    return [asdict(match) for match in matches]


def _trajectory_cost_matrix(
    saved_tracks: list[Track],
    gt_object_ids: list[int],
    gt_histories: list[list[ObjectLabelData]],
) -> tuple[np.ndarray, dict[tuple[int, int], dict[str, ObjectLabelData]]]:
    cost_matrix = np.zeros((len(saved_tracks), len(gt_histories)), dtype=np.float64)
    pair_metadata: dict[tuple[int, int], dict[str, ObjectLabelData]] = {}
    for row, track in enumerate(saved_tracks):
        track_alignment_axis, use_timestamp_axis = _track_alignment_axis(track)
        track_centers = _track_centers(track)
        track_extents = _track_extents(track, len(track_centers))
        last_alignment_value = int(track_alignment_axis[-1])
        weights = np.linspace(0.5, 1.0, num=len(track_centers), dtype=np.float64)
        weights = weights / max(np.sum(weights), 1e-9)
        for col, history in enumerate(gt_histories):
            gt_alignment_axis = _label_alignment_axis(history, use_timestamp_axis)
            gt_centers = np.asarray([_label_center(label) for label in history], dtype=np.float64)
            gt_extents = np.asarray([_label_extent(label) for label in history], dtype=np.float64)
            nearest_indices = _nearest_history_indices(track_alignment_axis, gt_alignment_axis)
            center_distances = np.linalg.norm(track_centers - gt_centers[nearest_indices], axis=1)
            mean_center_distance = float(np.sum(weights * center_distances)) if len(center_distances) > 0 else float("inf")
            safe_den = np.maximum(np.maximum(track_extents, gt_extents[nearest_indices]), 1e-3)
            extent_distances = np.mean(np.abs(track_extents - gt_extents[nearest_indices]) / safe_den, axis=1)
            mean_extent_distance = float(np.sum(weights * extent_distances)) if len(extent_distances) > 0 else 0.0
            mean_time_distance = (
                float(np.sum(weights * (np.abs(track_alignment_axis - gt_alignment_axis[nearest_indices]).astype(np.float64) * 1e-9)))
                if use_timestamp_axis
                else 0.0
            )
            report_index = int(np.argmin(np.abs(gt_alignment_axis - last_alignment_value)))
            report_label = history[report_index]
            last_center_distance = float(np.linalg.norm(track_centers[-1] - gt_centers[report_index]))
            last_time_distance = (
                float(abs(last_alignment_value - int(gt_alignment_axis[report_index])) * 1e-9)
                if use_timestamp_axis
                else 0.0
            )
            frame_delta = abs(int(track.last_frame) - int(report_label.frame_index))
            object_rank = int(gt_object_ids[col])
            cost = (
                (2.0 * mean_center_distance)
                + (0.75 * last_center_distance)
                + (0.25 * mean_extent_distance)
                + (0.35 * mean_time_distance)
                + (0.20 * last_time_distance)
                + (1e-6 * float(frame_delta))
                + (1e-9 * float(object_rank))
            )
            cost_matrix[row, col] = float(cost)
            pair_metadata[(row, col)] = {"report_label": report_label}
    return cost_matrix, pair_metadata


def _normalize_object_label_histories(
    object_labels_by_id: dict[int, ObjectLabelData] | dict[int, list[ObjectLabelData]],
    class_normalizer: ClassNormalizer | None,
) -> dict[int, list[ObjectLabelData]]:
    histories: dict[int, list[ObjectLabelData]] = {}
    for raw_object_id, raw_value in object_labels_by_id.items():
        labels = raw_value if isinstance(raw_value, list) else [raw_value]
        normalized_labels: list[ObjectLabelData] = []
        for label in labels:
            normalized = label if class_normalizer is None else class_normalizer.normalize_object_label(label)
            if len(normalized.points) == 0:
                continue
            normalized_labels.append(normalized)
        if not normalized_labels:
            continue
        normalized_labels.sort(key=lambda label: (int(label.timestamp_ns), int(label.frame_index)))
        histories[int(raw_object_id)] = normalized_labels
    return histories


def _track_alignment_axis(track: Track) -> tuple[np.ndarray, bool]:
    if track.frame_timestamps_ns and len(track.frame_timestamps_ns) == len(track.centers):
        return np.asarray(track.frame_timestamps_ns, dtype=np.int64), True
    if track.frame_ids:
        return np.asarray(track.frame_ids, dtype=np.int64), False
    return np.zeros((1,), dtype=np.int64), False


def _track_centers(track: Track) -> np.ndarray:
    if track.centers:
        return np.asarray(track.centers, dtype=np.float64)
    return np.asarray([track.current_center()], dtype=np.float64)


def _track_extents(track: Track, expected_count: int) -> np.ndarray:
    if track.bbox_extents and len(track.bbox_extents) == expected_count:
        return np.asarray(track.bbox_extents, dtype=np.float64)
    current_extent = np.asarray(track.current_extent(), dtype=np.float64)
    return np.repeat(current_extent[None, :], expected_count, axis=0)


def _nearest_history_indices(track_timestamps: np.ndarray, gt_timestamps: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(gt_timestamps, track_timestamps)
    positions = np.clip(positions, 0, len(gt_timestamps) - 1)
    prev_positions = np.clip(positions - 1, 0, len(gt_timestamps) - 1)
    use_prev = np.abs(track_timestamps - gt_timestamps[prev_positions]) <= np.abs(track_timestamps - gt_timestamps[positions])
    return np.where(use_prev, prev_positions, positions).astype(np.int64, copy=False)


def _label_center(label: ObjectLabelData) -> np.ndarray:
    points = np.asarray(label.points, dtype=np.float64)
    if len(points) == 0:
        return np.zeros((3,), dtype=np.float64)
    return np.mean(points, axis=0)


def _label_extent(label: ObjectLabelData) -> np.ndarray:
    points = np.asarray(label.points, dtype=np.float64)
    if len(points) == 0:
        return np.zeros((3,), dtype=np.float64)
    return np.ptp(points, axis=0)


def _label_alignment_axis(history: list[ObjectLabelData], use_timestamp_axis: bool) -> np.ndarray:
    if use_timestamp_axis:
        return np.asarray([int(label.timestamp_ns) for label in history], dtype=np.int64)
    return np.asarray([int(label.frame_index) for label in history], dtype=np.int64)


def _track_last_timestamp_ns(track: Track) -> int:
    if track.frame_timestamps_ns:
        return int(track.frame_timestamps_ns[-1])
    return int(track.last_frame)


def _track_timestamp_delta_ns(track: Track, label: ObjectLabelData) -> int | None:
    if not track.frame_timestamps_ns:
        return None
    return int(abs(int(track.frame_timestamps_ns[-1]) - int(label.timestamp_ns)))


def _normalize_gt_class_name(class_name: str, class_normalizer: ClassNormalizer | None) -> str:
    if class_normalizer is None:
        return str(class_name or "")
    return class_normalizer.normalize(class_name)


def _summary_metrics(saved_results: list[GTMatchResult], unmatched_gt: list[GTMatchResult] | None = None) -> dict[str, int | float | str]:
    unmatched_gt = [] if unmatched_gt is None else unmatched_gt
    matched_deltas = [int(match.timestamp_delta_ns) for match in saved_results if match.matched and match.timestamp_delta_ns is not None]
    return {
        "gt_match_saved_track_count": int(len(saved_results)),
        "gt_match_matched_count": int(sum(1 for match in saved_results if match.matched)),
        "gt_match_unmatched_saved_count": int(sum(1 for match in saved_results if not match.matched)),
        "gt_match_unmatched_gt_count": int(len(unmatched_gt)),
        "gt_match_mode": GT_MATCH_MODE,
        "gt_match_assignment": GT_MATCH_ASSIGNMENT,
        "gt_match_mean_timestamp_delta_ns": float(sum(matched_deltas) / len(matched_deltas)) if matched_deltas else 0.0,
        "gt_match_max_timestamp_delta_ns": int(max(matched_deltas)) if matched_deltas else 0,
    }
