from __future__ import annotations

import numpy as np

from tracking_pipeline.domain.models import AggregateResult, FrameTrackingState, Track, TrackOutcomeDebug


def build_track_outcomes(
    tracks: dict[int, Track],
    aggregate_results: list[AggregateResult] | dict[int, AggregateResult],
    states: list[FrameTrackingState],
) -> dict[int, TrackOutcomeDebug]:
    if isinstance(aggregate_results, dict):
        by_track_id = {int(track_id): result for track_id, result in aggregate_results.items()}
    else:
        by_track_id = {int(result.track_id): result for result in aggregate_results}
    frame_to_playback = {int(state.frame_index): int(index) for index, state in enumerate(states)}
    last_active_by_track = _last_active_track_states(states)

    outcomes: dict[int, TrackOutcomeDebug] = {}
    for track_id, track in sorted(tracks.items()):
        result = by_track_id.get(int(track_id))
        metrics = {} if result is None else dict(result.metrics)
        last_frame_id = int(track.last_frame)
        last_active_state = last_active_by_track.get(int(track_id))
        last_playback_index = int(frame_to_playback.get(last_frame_id, -1))
        last_center = None if not track.centers else track.current_center().copy()
        if last_active_state is not None:
            last_playback_index = int(last_active_state["playback_index"])
            last_center = np.asarray(last_active_state["center"], dtype=np.float32).copy()
        outcomes[int(track_id)] = TrackOutcomeDebug(
            track_id=int(track_id),
            status="missing_result" if result is None else str(result.status),
            decision_stage=str(metrics.get("decision_stage", "missing_result")),
            decision_reason_code=str(metrics.get("decision_reason_code", "missing_result")),
            decision_summary=str(metrics.get("decision_summary", "missing result")),
            last_frame_id=last_frame_id,
            last_playback_index=last_playback_index,
            last_center=last_center,
            hit_count=int(track.hit_count),
            age=int(track.age),
            missed=int(track.missed),
            ended_by_missed=bool(track.ended_by_missed),
            quality_score=None if track.quality_score is None else float(track.quality_score),
            selected_frame_ids=[] if result is None else [int(frame_id) for frame_id in result.selected_frame_ids],
            tracker_debug_summary=_track_debug_summary(track),
            predicted_class_id=None if metrics.get("predicted_class_id") is None else int(metrics.get("predicted_class_id")),
            predicted_class_name=str(metrics.get("predicted_class_name", "")),
            predicted_class_score=None
            if metrics.get("predicted_class_score") is None
            else float(metrics.get("predicted_class_score")),
            classification_backend=str(metrics.get("classification_backend", "")),
            classification_point_source=str(metrics.get("classification_point_source", "")),
            classification_input_point_count=int(metrics.get("classification_input_point_count", 0) or 0),
            gt_obj_class=str(metrics.get("gt_obj_class", "")),
            gt_obj_class_score=None
            if metrics.get("gt_obj_class_score") is None
            else float(metrics.get("gt_obj_class_score")),
        )
    return outcomes


def _last_active_track_states(states: list[FrameTrackingState]) -> dict[int, dict[str, int | np.ndarray]]:
    last_active_by_track: dict[int, dict[str, int | np.ndarray]] = {}
    for playback_index, state in enumerate(states):
        for active_track in state.active_tracks:
            last_active_by_track[int(active_track.track_id)] = {
                "playback_index": int(playback_index),
                "center": np.asarray(active_track.center, dtype=np.float32).copy(),
            }
    return last_active_by_track


def _track_debug_summary(track: Track) -> dict[str, int]:
    summary = track.state.get("tracker_debug_summary")
    if not isinstance(summary, dict):
        return {
            "spawned_count": 0,
            "matched_count": 0,
            "missed_count": 0,
            "suppressed_spawn_count": 0,
        }
    return {
        "spawned_count": int(summary.get("spawned_count", 0)),
        "matched_count": int(summary.get("matched_count", 0)),
        "missed_count": int(summary.get("missed_count", 0)),
        "suppressed_spawn_count": int(summary.get("suppressed_spawn_count", 0)),
    }
