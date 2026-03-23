from __future__ import annotations

from dataclasses import asdict

import numpy as np

from tracking_pipeline.domain.models import AggregateResult, GTMatchResult, ObjectLabelData, Track
from tracking_pipeline.infrastructure.tracking.assignment import assign_cost_matrix


def match_saved_aggregates_to_gt(
    tracks: dict[int, Track],
    aggregate_results: list[AggregateResult],
    latest_object_labels: dict[int, ObjectLabelData],
) -> tuple[list[GTMatchResult], list[GTMatchResult], list[GTMatchResult], dict[str, int | float | str]]:
    saved_results = sorted(
        [result for result in aggregate_results if str(result.status) == "saved"],
        key=lambda result: int(result.track_id),
    )
    saved_tracks = [tracks[int(result.track_id)] for result in saved_results if int(result.track_id) in tracks]
    gt_labels = [latest_object_labels[object_id] for object_id in sorted(latest_object_labels)]

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

    timestamp_cost = _timestamp_cost_matrix(saved_tracks, gt_labels)
    valid_mask = np.ones_like(timestamp_cost, dtype=bool)
    assignment_rows, unmatched_rows, unmatched_cols = assign_cost_matrix(timestamp_cost, valid_mask, method="hungarian")

    matched_results: list[GTMatchResult] = []
    for row in sorted(assignment_rows):
        col = int(assignment_rows[row])
        track = saved_tracks[int(row)]
        label = gt_labels[col]
        matched_results.append(
            GTMatchResult(
                track_id=int(track.track_id),
                gt_object_id=int(label.object_id),
                our_last_timestamp_ns=_track_last_timestamp_ns(track),
                gt_timestamp_ns=int(label.timestamp_ns),
                timestamp_delta_ns=int(abs(_track_last_timestamp_ns(track) - int(label.timestamp_ns))),
                our_last_frame_id=int(track.last_frame),
                gt_frame_index=int(label.frame_index),
                assignment_cost=float(timestamp_cost[int(row), col]),
                matched=True,
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
        result.metrics["gt_match_mode"] = "timestamp_only"
        result.metrics["gt_match_assignment"] = "one_to_one"
        result.metrics["gt_matched"] = bool(match.matched)
        if match.matched:
            result.metrics["gt_object_id"] = int(match.gt_object_id)
            result.metrics["gt_timestamp_ns"] = int(match.gt_timestamp_ns)
            result.metrics["gt_timestamp_delta_ns"] = int(match.timestamp_delta_ns)
            result.metrics["gt_frame_index"] = int(match.gt_frame_index)
            result.metrics["gt_assignment_cost"] = float(match.assignment_cost)
        else:
            result.metrics["gt_unmatched_reason"] = str(match.unmatched_reason)


def match_rows(matches: list[GTMatchResult]) -> list[dict[str, object]]:
    return [asdict(match) for match in matches]


def _timestamp_cost_matrix(saved_tracks: list[Track], gt_labels: list[ObjectLabelData]) -> np.ndarray:
    assign_count = max(1, min(len(saved_tracks), len(gt_labels)))
    last_timestamps = np.asarray([_track_last_timestamp_ns(track) for track in saved_tracks], dtype=np.float64)
    last_frame_ids = np.asarray([int(track.last_frame) for track in saved_tracks], dtype=np.float64)
    gt_timestamps = np.asarray([int(label.timestamp_ns) for label in gt_labels], dtype=np.float64)
    gt_frame_indices = np.asarray([int(label.frame_index) for label in gt_labels], dtype=np.float64)

    timestamp_delta = np.abs(last_timestamps[:, None] - gt_timestamps[None, :])
    frame_delta = np.abs(last_frame_ids[:, None] - gt_frame_indices[None, :])
    frame_den = max(1.0, float(np.max(frame_delta)) + 1.0)
    gt_rank_delta = np.abs(
        np.arange(len(saved_tracks), dtype=np.float64).reshape(-1, 1)
        - np.arange(len(gt_labels), dtype=np.float64).reshape(1, -1)
    )
    rank_den = max(1.0, float(np.max(gt_rank_delta)) + 1.0)
    secondary_eps = 0.25 / float(assign_count) / frame_den
    tertiary_eps = 0.0625 / float(assign_count * assign_count) / rank_den
    return (
        timestamp_delta
        + secondary_eps * frame_delta
        + tertiary_eps * gt_rank_delta
    ).astype(np.float64)


def _track_last_timestamp_ns(track: Track) -> int:
    if track.frame_timestamps_ns:
        return int(track.frame_timestamps_ns[-1])
    return int(track.last_frame)


def _summary_metrics(saved_results: list[GTMatchResult], unmatched_gt: list[GTMatchResult] | None = None) -> dict[str, int | float | str]:
    unmatched_gt = [] if unmatched_gt is None else unmatched_gt
    matched_deltas = [int(match.timestamp_delta_ns) for match in saved_results if match.matched and match.timestamp_delta_ns is not None]
    return {
        "gt_match_saved_track_count": int(len(saved_results)),
        "gt_match_matched_count": int(sum(1 for match in saved_results if match.matched)),
        "gt_match_unmatched_saved_count": int(sum(1 for match in saved_results if not match.matched)),
        "gt_match_unmatched_gt_count": int(len(unmatched_gt)),
        "gt_match_mode": "timestamp_only",
        "gt_match_assignment": "one_to_one",
        "gt_match_mean_timestamp_delta_ns": float(sum(matched_deltas) / len(matched_deltas)) if matched_deltas else 0.0,
        "gt_match_max_timestamp_delta_ns": int(max(matched_deltas)) if matched_deltas else 0,
    }
