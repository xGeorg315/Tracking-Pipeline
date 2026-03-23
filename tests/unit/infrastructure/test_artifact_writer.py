from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import open3d as o3d

from tracking_pipeline.domain.models import AggregateResult, Track, TrackOutcomeDebug
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
