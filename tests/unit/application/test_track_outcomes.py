from __future__ import annotations

import numpy as np

from tracking_pipeline.application.track_outcomes import build_track_outcomes
from tracking_pipeline.domain.models import ActiveTrackState, AggregateResult, FrameTrackingState, Track


def test_build_track_outcomes_uses_track_and_aggregate_result_fields() -> None:
    track = Track(track_id=7, hit_count=4, age=5, missed=1, ended_by_missed=True, quality_score=0.42)
    for frame_id, center_y in ((100, 1.0), (101, 2.0), (102, 3.0)):
        center = np.array([0.0, center_y, 0.0], dtype=np.float32)
        points = np.array([center, center + np.array([0.1, 0.0, 0.0], dtype=np.float32)], dtype=np.float32)
        track.centers.append(center.copy())
        track.frame_ids.append(frame_id)
        track.world_points.append(points.copy())
        track.local_points.append((points - center).astype(np.float32))
        track.bbox_extents.append(np.array([0.1, 0.1, 0.1], dtype=np.float32))
    track.state["tracker_debug_summary"] = {
        "spawned_count": 1,
        "matched_count": 3,
        "missed_count": 1,
        "suppressed_spawn_count": 0,
    }

    result = AggregateResult(
        track_id=7,
        points=np.zeros((0, 3), dtype=np.float32),
        selected_frame_ids=[101, 102],
        status="skipped_quality_threshold",
        metrics={
            "decision_stage": "quality_gate",
            "decision_reason_code": "quality_threshold",
            "decision_summary": "quality 0.42<0.50",
            "predicted_class_id": 3,
            "predicted_class_name": "truck",
            "predicted_class_score": 0.91,
            "classification_backend": "pointnext",
            "classification_point_source": "candidate_points_world",
            "classification_input_point_count": 128,
            "gt_obj_class": "car",
            "gt_obj_class_score": 0.97,
        },
    )
    states = [
        FrameTrackingState(frame_index=100, lane_points=np.zeros((0, 3), dtype=np.float32), detections=[], active_tracks=[]),
        FrameTrackingState(frame_index=101, lane_points=np.zeros((0, 3), dtype=np.float32), detections=[], active_tracks=[]),
        FrameTrackingState(frame_index=102, lane_points=np.zeros((0, 3), dtype=np.float32), detections=[], active_tracks=[]),
    ]

    outcomes = build_track_outcomes({7: track}, [result], states)

    assert set(outcomes.keys()) == {7}
    outcome = outcomes[7]
    assert outcome.status == "skipped_quality_threshold"
    assert outcome.decision_stage == "quality_gate"
    assert outcome.decision_reason_code == "quality_threshold"
    assert outcome.decision_summary == "quality 0.42<0.50"
    assert outcome.last_frame_id == 102
    assert outcome.last_playback_index == 2
    assert np.allclose(outcome.last_center, np.array([0.0, 3.0, 0.0], dtype=np.float32))
    assert outcome.hit_count == 4
    assert outcome.age == 5
    assert outcome.missed == 1
    assert outcome.ended_by_missed is True
    assert outcome.quality_score == 0.42
    assert outcome.selected_frame_ids == [101, 102]
    assert outcome.tracker_debug_summary["matched_count"] == 3
    assert outcome.predicted_class_id == 3
    assert outcome.predicted_class_name == "truck"
    assert outcome.predicted_class_score == 0.91
    assert outcome.classification_backend == "pointnext"
    assert outcome.classification_point_source == "candidate_points_world"
    assert outcome.classification_input_point_count == 128
    assert outcome.gt_obj_class == "car"
    assert outcome.gt_obj_class_score == 0.97


def test_build_track_outcomes_prefers_last_active_predicted_state_for_playback_position() -> None:
    track = Track(track_id=7, hit_count=3, age=5, missed=2, ended_by_missed=False)
    for frame_id, center_y in ((100, 1.0), (101, 2.0), (102, 3.0)):
        center = np.array([0.0, center_y, 0.0], dtype=np.float32)
        points = np.array([center, center + np.array([0.1, 0.0, 0.0], dtype=np.float32)], dtype=np.float32)
        track.centers.append(center.copy())
        track.frame_ids.append(frame_id)
        track.world_points.append(points.copy())
        track.local_points.append((points - center).astype(np.float32))
        track.bbox_extents.append(np.array([0.1, 0.1, 0.1], dtype=np.float32))

    result = AggregateResult(
        track_id=7,
        points=np.zeros((0, 3), dtype=np.float32),
        selected_frame_ids=[101, 102],
        status="saved",
        metrics={
            "decision_stage": "saved",
            "decision_reason_code": "saved",
            "decision_summary": "saved points=32",
        },
    )
    predicted_center = np.array([0.0, 3.8, 0.0], dtype=np.float32)
    states = [
        FrameTrackingState(frame_index=100, lane_points=np.zeros((0, 3), dtype=np.float32), detections=[], active_tracks=[]),
        FrameTrackingState(frame_index=101, lane_points=np.zeros((0, 3), dtype=np.float32), detections=[], active_tracks=[]),
        FrameTrackingState(frame_index=102, lane_points=np.zeros((0, 3), dtype=np.float32), detections=[], active_tracks=[]),
        FrameTrackingState(
            frame_index=103,
            lane_points=np.zeros((0, 3), dtype=np.float32),
            detections=[],
            active_tracks=[
                ActiveTrackState(
                    track_id=7,
                    points=np.array([predicted_center], dtype=np.float32),
                    center=predicted_center,
                    status="predicted",
                )
            ],
        ),
    ]

    outcomes = build_track_outcomes({7: track}, [result], states)

    outcome = outcomes[7]
    assert outcome.last_frame_id == 102
    assert outcome.last_playback_index == 3
    assert np.allclose(outcome.last_center, predicted_center)
