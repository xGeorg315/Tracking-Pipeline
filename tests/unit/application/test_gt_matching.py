from __future__ import annotations

import numpy as np

from tracking_pipeline.application.gt_matching import apply_gt_matches_to_results, match_saved_aggregates_to_gt
from tracking_pipeline.domain.models import AggregateResult, ObjectLabelData, Track


def _saved_track(track_id: int, *, frame_id: int, timestamp_ns: int) -> Track:
    track = Track(track_id=track_id, hit_count=1, age=1, ended_by_missed=True)
    center = np.array([0.0, float(frame_id), 0.0], dtype=np.float32)
    track.centers.append(center)
    track.frame_ids.append(int(frame_id))
    track.frame_timestamps_ns.append(int(timestamp_ns))
    track.world_points.append(np.array([[0.0, float(frame_id), 0.0]], dtype=np.float32))
    track.local_points.append(np.array([[0.0, 0.0, 0.0]], dtype=np.float32))
    track.bbox_extents.append(np.array([0.1, 0.1, 0.1], dtype=np.float32))
    return track


def _saved_result(track_id: int) -> AggregateResult:
    return AggregateResult(
        track_id=track_id,
        points=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        selected_frame_ids=[track_id],
        status="saved",
        metrics={},
    )


def _gt_label(object_id: int, *, frame_index: int, timestamp_ns: int) -> ObjectLabelData:
    return ObjectLabelData(
        object_id=int(object_id),
        timestamp_ns=int(timestamp_ns),
        points=np.array([[float(object_id), 0.0, 0.0]], dtype=np.float32),
        frame_index=int(frame_index),
        source_path="fixture.pb",
        sensor_name="sensor_a",
    )


def test_gt_matching_assigns_saved_tracks_one_to_one_by_min_timestamp_delta() -> None:
    tracks = {
        1: _saved_track(1, frame_id=10, timestamp_ns=100),
        2: _saved_track(2, frame_id=20, timestamp_ns=300),
    }
    results = [_saved_result(1), _saved_result(2)]
    labels = {
        7: _gt_label(7, frame_index=9, timestamp_ns=98),
        8: _gt_label(8, frame_index=19, timestamp_ns=305),
    }

    matches, unmatched_saved, unmatched_gt, summary = match_saved_aggregates_to_gt(tracks, results, labels)

    assert unmatched_saved == []
    assert unmatched_gt == []
    assert {(match.track_id, match.gt_object_id) for match in matches} == {(1, 7), (2, 8)}
    assert summary["gt_match_saved_track_count"] == 2
    assert summary["gt_match_matched_count"] == 2


def test_gt_matching_marks_extra_saved_tracks_as_unmatched() -> None:
    tracks = {
        1: _saved_track(1, frame_id=10, timestamp_ns=100),
        2: _saved_track(2, frame_id=20, timestamp_ns=200),
        3: _saved_track(3, frame_id=30, timestamp_ns=300),
    }
    results = [_saved_result(1), _saved_result(2), _saved_result(3)]
    labels = {
        7: _gt_label(7, frame_index=10, timestamp_ns=99),
        8: _gt_label(8, frame_index=20, timestamp_ns=198),
    }

    matches, unmatched_saved, unmatched_gt, summary = match_saved_aggregates_to_gt(tracks, results, labels)

    assert len(matches) == 2
    assert len(unmatched_saved) == 1
    assert unmatched_saved[0].track_id == 3
    assert unmatched_saved[0].unmatched_reason == "unmatched_no_gt_available"
    assert unmatched_gt == []
    assert summary["gt_match_unmatched_saved_count"] == 1


def test_gt_matching_marks_extra_gt_objects_as_unmatched() -> None:
    tracks = {
        1: _saved_track(1, frame_id=10, timestamp_ns=100),
        2: _saved_track(2, frame_id=20, timestamp_ns=200),
    }
    results = [_saved_result(1), _saved_result(2)]
    labels = {
        7: _gt_label(7, frame_index=10, timestamp_ns=99),
        8: _gt_label(8, frame_index=20, timestamp_ns=198),
        9: _gt_label(9, frame_index=30, timestamp_ns=500),
    }

    matches, unmatched_saved, unmatched_gt, summary = match_saved_aggregates_to_gt(tracks, results, labels)

    assert len(matches) == 2
    assert unmatched_saved == []
    assert len(unmatched_gt) == 1
    assert unmatched_gt[0].gt_object_id == 9
    assert unmatched_gt[0].unmatched_reason == "unmatched_gt"
    assert summary["gt_match_unmatched_gt_count"] == 1


def test_gt_matching_uses_frame_delta_then_gt_id_as_tie_break() -> None:
    tracks = {
        1: _saved_track(1, frame_id=10, timestamp_ns=100),
        2: _saved_track(2, frame_id=10, timestamp_ns=100),
    }
    results = [_saved_result(1), _saved_result(2)]
    labels = {
        8: _gt_label(8, frame_index=10, timestamp_ns=100),
        7: _gt_label(7, frame_index=10, timestamp_ns=100),
    }

    matches, _, _, _ = match_saved_aggregates_to_gt(tracks, results, labels)

    assert [(match.track_id, match.gt_object_id) for match in matches] == [(1, 7), (2, 8)]


def test_gt_matching_only_annotates_saved_results() -> None:
    tracks = {
        1: _saved_track(1, frame_id=10, timestamp_ns=100),
        2: _saved_track(2, frame_id=20, timestamp_ns=200),
    }
    saved = _saved_result(1)
    skipped = AggregateResult(
        track_id=2,
        points=np.zeros((0, 3), dtype=np.float32),
        selected_frame_ids=[2],
        status="skipped_quality_threshold",
        metrics={},
    )
    labels = {7: _gt_label(7, frame_index=10, timestamp_ns=99)}

    matches, unmatched_saved, _, _ = match_saved_aggregates_to_gt(tracks, [saved, skipped], labels)
    apply_gt_matches_to_results([saved, skipped], matches, unmatched_saved)

    assert saved.metrics["gt_matched"] is True
    assert saved.metrics["gt_object_id"] == 7
    assert "gt_matched" not in skipped.metrics
