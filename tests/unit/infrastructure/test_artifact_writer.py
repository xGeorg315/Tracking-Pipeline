from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import open3d as o3d

from tracking_pipeline.domain.models import AggregateResult, GTMatchResult, Track, TrackOutcomeDebug
from tracking_pipeline.infrastructure.io.artifact_writer import JsonArtifactWriter


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_artifact_writer_writes_track_outcomes_and_track_failure_fields(tmp_path: Path) -> None:
    writer = JsonArtifactWriter(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    track = Track(
        track_id=7,
        hit_count=4,
        age=6,
        missed=2,
        ended_by_missed=True,
        quality_score=0.28,
        state={"tracker_debug_summary": {"spawned_count": 1, "matched_count": 4, "missed_count": 2, "suppressed_spawn_count": 0}},
    )
    track.centers = [
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        np.array([1.5, 0.1, 0.0], dtype=np.float32),
    ]
    track.frame_ids = [100, 101]

    result = AggregateResult(
        track_id=7,
        points=np.zeros((0, 3), dtype=np.float32),
        selected_frame_ids=[100],
        status="skipped_quality_threshold",
        metrics={
            "decision_stage": "quality_gate",
            "decision_reason_code": "quality_threshold",
            "decision_summary": "quality 0.28<0.40",
        },
    )
    outcome = TrackOutcomeDebug(
        track_id=7,
        status="skipped_quality_threshold",
        decision_stage="quality_gate",
        decision_reason_code="quality_threshold",
        decision_summary="quality 0.28<0.40",
        last_frame_id=101,
        last_playback_index=5,
        last_center=np.array([1.5, 0.1, 0.0], dtype=np.float32),
        hit_count=4,
        age=6,
        missed=2,
        ended_by_missed=True,
        quality_score=0.28,
        selected_frame_ids=[100],
        tracker_debug_summary={"spawned_count": 1, "matched_count": 4, "missed_count": 2, "suppressed_spawn_count": 0},
    )

    writer.write_tracks(run_dir, {7: track}, [result])
    writer.write_track_outcomes(run_dir, {7: outcome})

    track_rows = _load_jsonl(run_dir / "tracks.jsonl")
    outcome_rows = _load_jsonl(run_dir / "track_outcomes.jsonl")

    assert track_rows == [
        {
            "age": 6,
            "aggregate_status": "skipped_quality_threshold",
            "aggregation_metrics": {
                "decision_reason_code": "quality_threshold",
                "decision_stage": "quality_gate",
                "decision_summary": "quality 0.28<0.40",
            },
            "decision_reason_code": "quality_threshold",
            "decision_stage": "quality_gate",
            "decision_summary": "quality 0.28<0.40",
            "ended_by_missed": True,
            "frame_ids": [100, 101],
            "hit_count": 4,
            "last_center": [1.5, 0.10000000149011612, 0.0],
            "last_frame_id": 101,
            "missed": 2,
            "quality_metrics": {},
            "quality_score": 0.28,
            "selected_frame_ids": [100],
            "source_track_ids": [7],
            "track_id": 7,
            "tracker_debug_summary": {
                "matched_count": 4,
                "missed_count": 2,
                "spawned_count": 1,
                "suppressed_spawn_count": 0,
            },
        }
    ]
    assert outcome_rows == [
        {
            "age": 6,
            "decision_reason_code": "quality_threshold",
            "decision_stage": "quality_gate",
            "decision_summary": "quality 0.28<0.40",
            "ended_by_missed": True,
            "hit_count": 4,
            "last_center": [1.5, 0.10000000149011612, 0.0],
            "last_frame_id": 101,
            "last_playback_index": 5,
            "missed": 2,
            "quality_score": 0.28,
            "selected_frame_ids": [100],
            "status": "skipped_quality_threshold",
            "track_id": 7,
            "tracker_debug_summary": {
                "matched_count": 4,
                "missed_count": 2,
                "spawned_count": 1,
                "suppressed_spawn_count": 0,
            },
        }
    ]


def test_artifact_writer_writes_aggregate_with_reflectivity_field(tmp_path: Path) -> None:
    writer = JsonArtifactWriter(tmp_path)
    run_dir = tmp_path / "run"
    (run_dir / "aggregates").mkdir(parents=True)

    result = AggregateResult(
        track_id=3,
        points=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        selected_frame_ids=[100],
        status="saved",
        metrics={},
        intensity=np.array([1.5, 4.0], dtype=np.float32),
    )

    writer.write_aggregate(run_dir, result, save_intensity=True)

    loaded = o3d.t.io.read_point_cloud(str(run_dir / "aggregates" / "track_0003.pcd"))

    assert "reflectivity" in loaded.point
    assert "intensity" not in loaded.point
    assert np.allclose(loaded.point["reflectivity"].numpy().reshape(-1), np.array([1.5, 4.0], dtype=np.float32))


def test_artifact_writer_writes_articulated_vehicle_fields(tmp_path: Path) -> None:
    writer = JsonArtifactWriter(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    track = Track(
        track_id=12,
        hit_count=4,
        age=4,
        ended_by_missed=True,
        source_track_ids=[12, 13],
        state={
            "articulated_vehicle": True,
            "articulated_component_track_ids": [12, 13],
            "articulated_rear_gap_mean": 0.2,
            "articulated_rear_gap_std": 0.05,
            "object_kind": "truck_with_trailer",
        },
    )
    track.centers = [np.array([0.0, 12.0, 1.0], dtype=np.float32)]
    track.frame_ids = [101]
    track.quality_metrics = {"is_articulated_vehicle": True, "is_long_vehicle": True}

    result = AggregateResult(
        track_id=12,
        points=np.zeros((0, 3), dtype=np.float32),
        selected_frame_ids=[101],
        status="saved",
        metrics={
            "articulated_vehicle": True,
            "articulated_component_track_ids": [12, 13],
            "articulated_rear_gap_mean": 0.2,
            "articulated_rear_gap_std": 0.05,
            "object_kind": "truck_with_trailer",
        },
    )

    writer.write_tracks(run_dir, {12: track}, [result])

    rows = _load_jsonl(run_dir / "tracks.jsonl")

    assert rows[0]["articulated_vehicle"] is True
    assert rows[0]["articulated_component_track_ids"] == [12, 13]
    assert rows[0]["object_kind"] == "truck_with_trailer"
    assert rows[0]["aggregation_metrics"]["articulated_rear_gap_mean"] == 0.2
    assert rows[0]["aggregation_metrics"]["articulated_rear_gap_std"] == 0.05


def test_artifact_writer_writes_long_vehicle_anchor_fields(tmp_path: Path) -> None:
    writer = JsonArtifactWriter(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    track = Track(track_id=21, hit_count=5, age=5, ended_by_missed=True)
    track.centers = [np.array([0.0, 12.0, 0.0], dtype=np.float32)]
    track.frame_ids = [88]
    result = AggregateResult(
        track_id=21,
        points=np.zeros((0, 3), dtype=np.float32),
        selected_frame_ids=[88],
        status="saved",
        metrics={
            "long_vehicle_local_anchor_mode": "lead_front",
            "long_vehicle_local_anchor_axis": "y",
            "long_vehicle_lead_track_id": 21,
            "long_vehicle_local_anchor_value_world": 14.5,
        },
    )

    writer.write_tracks(run_dir, {21: track}, [result])

    rows = _load_jsonl(run_dir / "tracks.jsonl")

    assert rows[0]["long_vehicle_local_anchor_mode"] == "lead_front"
    assert rows[0]["long_vehicle_local_anchor_axis"] == "y"
    assert rows[0]["long_vehicle_lead_track_id"] == 21
    assert rows[0]["long_vehicle_local_anchor_value_world"] == 14.5


def test_artifact_writer_writes_classification_fields(tmp_path: Path) -> None:
    writer = JsonArtifactWriter(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    track = Track(track_id=4, hit_count=3, age=3, ended_by_missed=True)
    track.centers = [np.array([0.0, 1.0, 0.0], dtype=np.float32)]
    track.frame_ids = [42]
    result = AggregateResult(
        track_id=4,
        points=np.zeros((1, 3), dtype=np.float32),
        selected_frame_ids=[42],
        status="saved",
        metrics={
            "predicted_class_id": 6,
            "predicted_class_name": "trailer",
            "predicted_class_score": 0.88,
            "classification_backend": "pointnext",
            "classification_point_source": "result_points",
            "classification_input_point_count": 512,
            "gt_obj_class": "LKW-Anhaenger",
            "gt_obj_class_score": 0.97,
        },
    )
    outcome = TrackOutcomeDebug(
        track_id=4,
        status="saved",
        decision_stage="saved",
        decision_reason_code="saved",
        decision_summary="saved points=512",
        last_frame_id=42,
        last_playback_index=1,
        last_center=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        predicted_class_id=6,
        predicted_class_name="trailer",
        predicted_class_score=0.88,
        classification_backend="pointnext",
        classification_point_source="result_points",
        classification_input_point_count=512,
        gt_obj_class="LKW-Anhaenger",
        gt_obj_class_score=0.97,
    )

    writer.write_tracks(run_dir, {4: track}, [result])
    writer.write_track_outcomes(run_dir, {4: outcome})

    track_rows = _load_jsonl(run_dir / "tracks.jsonl")
    outcome_rows = _load_jsonl(run_dir / "track_outcomes.jsonl")

    assert track_rows[0]["predicted_class_id"] == 6
    assert track_rows[0]["predicted_class_name"] == "trailer"
    assert track_rows[0]["predicted_class_score"] == 0.88
    assert track_rows[0]["classification_backend"] == "pointnext"
    assert track_rows[0]["classification_point_source"] == "result_points"
    assert track_rows[0]["classification_input_point_count"] == 512
    assert track_rows[0]["gt_obj_class"] == "LKW-Anhaenger"
    assert track_rows[0]["gt_obj_class_score"] == 0.97
    assert outcome_rows[0]["predicted_class_id"] == 6
    assert outcome_rows[0]["predicted_class_name"] == "trailer"
    assert outcome_rows[0]["predicted_class_score"] == 0.88
    assert outcome_rows[0]["classification_backend"] == "pointnext"
    assert outcome_rows[0]["classification_point_source"] == "result_points"
    assert outcome_rows[0]["classification_input_point_count"] == 512
    assert outcome_rows[0]["gt_obj_class"] == "LKW-Anhaenger"
    assert outcome_rows[0]["gt_obj_class_score"] == 0.97


def test_artifact_writer_writes_gt_matching_artifacts_and_track_fields(tmp_path: Path) -> None:
    writer = JsonArtifactWriter(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    track = Track(track_id=5, hit_count=3, age=3, ended_by_missed=True)
    track.centers = [np.array([0.0, 1.0, 0.0], dtype=np.float32)]
    track.frame_ids = [42]
    result = AggregateResult(
        track_id=5,
        points=np.zeros((1, 3), dtype=np.float32),
        selected_frame_ids=[42],
        status="saved",
        metrics={
            "gt_matched": True,
            "gt_match_mode": "timestamp_only",
            "gt_match_assignment": "one_to_one",
            "gt_object_id": 17,
            "gt_obj_class": "truck",
            "gt_obj_class_score": 0.83,
            "gt_timestamp_ns": 1005,
            "gt_timestamp_delta_ns": 5,
            "gt_frame_index": 7,
        },
    )
    match = GTMatchResult(
        track_id=5,
        gt_object_id=17,
        our_last_timestamp_ns=1000,
        gt_timestamp_ns=1005,
        timestamp_delta_ns=5,
        our_last_frame_id=42,
        gt_frame_index=7,
        assignment_cost=5.0,
        matched=True,
        gt_obj_class="truck",
        gt_obj_class_score=0.83,
    )
    writer.write_tracks(run_dir, {5: track}, [result])
    writer.write_gt_matching(
        run_dir,
        [match],
        [],
        [],
        {
            "gt_match_saved_track_count": 1,
            "gt_match_matched_count": 1,
            "gt_match_unmatched_saved_count": 0,
            "gt_match_unmatched_gt_count": 0,
            "gt_match_mode": "timestamp_only",
            "gt_match_assignment": "one_to_one",
            "gt_match_mean_timestamp_delta_ns": 5.0,
            "gt_match_max_timestamp_delta_ns": 5,
        },
    )

    track_rows = _load_jsonl(run_dir / "tracks.jsonl")
    match_rows = _load_jsonl(run_dir / "gt_matching" / "matches.jsonl")
    summary_payload = json.loads((run_dir / "gt_matching" / "summary.json").read_text(encoding="utf-8"))

    assert track_rows[0]["gt_matched"] is True
    assert track_rows[0]["gt_object_id"] == 17
    assert track_rows[0]["matched_gt_pcd_path"] == "object_list/object_0017.pcd"
    assert track_rows[0]["gt_obj_class"] == "truck"
    assert track_rows[0]["gt_obj_class_score"] == 0.83
    assert track_rows[0]["gt_timestamp_ns"] == 1005
    assert track_rows[0]["gt_timestamp_delta_ns"] == 5
    assert track_rows[0]["gt_match_mode"] == "timestamp_only"
    assert track_rows[0]["gt_match_assignment"] == "one_to_one"
    assert match_rows[0]["track_id"] == 5
    assert match_rows[0]["gt_object_id"] == 17
    assert match_rows[0]["gt_obj_class"] == "truck"
    assert match_rows[0]["gt_obj_class_score"] == 0.83
    assert summary_payload["gt_match_matched_count"] == 1


def test_artifact_writer_writes_class_stats_json(tmp_path: Path) -> None:
    writer = JsonArtifactWriter(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    writer.write_class_stats(
        run_dir,
        {
            "predicted_class_counts": {"PKW": 3, "Van": 1},
            "gt_class_counts": {"PKW": 4, "Van": 2},
            "matched_gt_class_counts": {"PKW": 2, "Van": 1},
            "class_comparison_count": 3,
            "class_match_count": 2,
            "class_mismatch_count": 1,
            "class_count_rows": [
                {"class_name": "PKW", "predicted_count": 3, "gt_match_count": 2},
                {"class_name": "Van", "predicted_count": 1, "gt_match_count": 1},
                {"class_name": "TOTAL", "predicted_count": 4, "gt_match_count": 3},
            ],
        },
    )

    payload = json.loads((run_dir / "class_stats.json").read_text(encoding="utf-8"))

    assert payload["predicted_class_counts"] == {"PKW": 3, "Van": 1}
    assert payload["gt_class_counts"] == {"PKW": 4, "Van": 2}
    assert payload["matched_gt_class_counts"] == {"PKW": 2, "Van": 1}
    assert payload["class_comparison_count"] == 3
    assert payload["class_match_count"] == 2
    assert payload["class_mismatch_count"] == 1
    assert payload["class_count_rows"] == [
        {"class_name": "PKW", "predicted_count": 3, "gt_match_count": 2},
        {"class_name": "Van", "predicted_count": 1, "gt_match_count": 1},
        {"class_name": "TOTAL", "predicted_count": 4, "gt_match_count": 3},
    ]
