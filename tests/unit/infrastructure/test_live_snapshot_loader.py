from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import yaml

from tracking_pipeline.config.models import (
    AggregationConfig,
    InputConfig,
    OutputConfig,
    PipelineConfig,
    PreprocessingConfig,
    VisualizationConfig,
)
from tracking_pipeline.infrastructure.visualization.live_snapshot_loader import LiveSnapshotLoader


def _config(dataset_root: Path) -> PipelineConfig:
    return PipelineConfig(
        input=InputConfig(paths=["dummy.pb"]),
        preprocessing=PreprocessingConfig(lane_box=[-2.0, 2.0, 0.0, 12.0, -1.0, 3.0]),
        aggregation=AggregationConfig(frame_selection_line_axis="y"),
        output=OutputConfig(
            mode="dataset",
            dataset_root_dir=str(dataset_root),
            require_track_exit=True,
            track_exit_edge_margin=0.9,
        ),
        visualization=VisualizationConfig(
            show_tracker_debug=True,
            show_track_outcome_debug=True,
            max_assoc_dist=4.2,
        ),
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def _write_config_snapshot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "preprocessing": {"lane_box": [-3.0, 3.0, 1.0, 18.0, -0.5, 4.0]},
                "visualization": {
                    "show_tracker_debug": True,
                    "show_track_outcome_debug": False,
                    "max_assoc_dist": 5.6,
                },
                "output": {
                    "require_track_exit": False,
                    "track_exit_edge_margin": 1.4,
                },
                "aggregation": {"frame_selection_line_axis": "x"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _live_status(processed_frames: int, frame_index: int) -> dict[str, object]:
    return {
        "pipeline_phase": "streaming",
        "processed_frames": processed_frames,
        "last_processed_frame_index": frame_index,
        "active_track_count": 2,
        "finished_track_count": 5,
        "saved_aggregates": 3,
        "object_list_exported_count": 7,
        "processing_recent_hz": 9.5,
        "processing_total_hz": 8.1,
    }


def _tracker_row(frame_index: int, track_id: int = 11, detection_id: int = 101) -> dict[str, object]:
    return {
        "frame_index": frame_index,
        "cluster_metrics": {"cluster_count": 2},
        "tracker_metrics": {"assignment_method": "hungarian"},
        "tracker_debug": {
            "assignment_method": "hungarian",
            "matched_count": 1,
            "missed_count": 0,
            "spawned_count": 1,
            "suppressed_count": 0,
            "halo_detection_count": 0,
            "track_states": [
                {
                    "track_id": track_id,
                    "predicted_center": [1.0, 2.0, 0.5],
                    "output_center": [1.2, 2.1, 0.5],
                    "status": "matched",
                    "matched_detection_id": detection_id,
                    "missed_before": 0,
                    "missed_after": 0,
                }
            ],
            "detection_states": [
                {
                    "detection_id": detection_id,
                    "center": [1.2, 2.15, 0.5],
                    "status": "matched",
                    "matched_track_id": track_id,
                    "spawned_track_id": None,
                    "spawn_suppressed": False,
                    "tracking_halo_only": False,
                }
            ],
        },
    }


def _track_row(track_id: int) -> dict[str, object]:
    return {"track_id": track_id, "hit_count": 5, "age": 6, "missed": 0}


def _track_outcome_row(track_id: int, status: str = "saved") -> dict[str, object]:
    return {
        "track_id": track_id,
        "status": status,
        "decision_stage": "save_gate" if status != "saved" else "saved",
        "decision_reason_code": "saved" if status == "saved" else "min_hits",
        "decision_summary": "saved points=248" if status == "saved" else "min_hits 2/4",
        "last_frame_id": 41,
        "last_playback_index": 12,
        "last_center": [2.0, 4.0, 0.4],
        "hit_count": 5,
        "age": 6,
        "missed": 1,
        "ended_by_missed": True,
        "quality_score": 0.74,
        "selected_frame_ids": [39, 40, 41],
        "tracker_debug_summary": {"matched": 4, "spawned": 1},
        "predicted_class_name": "car",
        "predicted_class_score": 0.92,
        "gt_obj_class": "PKW",
    }


def test_live_snapshot_loader_returns_waiting_snapshot_without_active_run(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    loader = LiveSnapshotLoader(dataset_root, _config(dataset_root))

    snapshot = loader.load()

    assert snapshot.waiting is True
    assert snapshot.run_id == ""
    assert snapshot.tracker_frame is None
    assert snapshot.lane_box is not None
    assert snapshot.lane_box.x_min == -2.0
    assert snapshot.require_track_exit is True
    assert snapshot.track_exit_edge_margin == 0.9
    assert snapshot.track_exit_line_axis == "y"


def test_live_snapshot_loader_discovers_latest_active_run_and_latest_stats_dir(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    loader = LiveSnapshotLoader(dataset_root, _config(dataset_root))

    active_old = dataset_root / "_stats" / "_active" / "run_old" / "live_status.json"
    active_new = dataset_root / "_stats" / "_active" / "run_new" / "live_status.json"
    _write_json(active_old, _live_status(processed_frames=2, frame_index=1))
    _write_json(active_new, _live_status(processed_frames=14, frame_index=13))
    os.utime(active_old, ns=(10, 10))
    os.utime(active_new, ns=(20, 20))

    old_stats_dir = dataset_root / "_stats" / "2026-04-06" / "run_new"
    new_stats_dir = dataset_root / "_stats" / "2026-04-07" / "run_new"
    _write_jsonl(old_stats_dir / "tracker_debug.jsonl", [_tracker_row(8)])
    _write_jsonl(new_stats_dir / "tracker_debug.jsonl", [_tracker_row(12), _tracker_row(13)])
    _write_jsonl(new_stats_dir / "tracks.jsonl", [_track_row(11)])
    _write_jsonl(new_stats_dir / "track_outcomes.jsonl", [_track_outcome_row(11)])
    _write_json(new_stats_dir / "summary.json", {"gt_match_matched_count": 9, "gt_match_unmatched_gt_count": 4})
    _write_jsonl(
        dataset_root / "_stats" / "_active" / "run_new" / "object_list_manifest.jsonl",
        [{"object_id": 9001, "obj_class": "PKW"}],
    )
    _write_config_snapshot(new_stats_dir / "config.snapshot.yaml")
    os.utime(old_stats_dir, ns=(30, 30))
    os.utime(new_stats_dir, ns=(40, 40))

    snapshot = loader.load()

    assert snapshot.waiting is False
    assert snapshot.run_id == "run_new"
    assert snapshot.active_dir == dataset_root / "_stats" / "_active" / "run_new"
    assert snapshot.stats_dir == new_stats_dir
    assert snapshot.live_status["processed_frames"] == 14
    assert snapshot.summary["gt_match_matched_count"] == 9
    assert len(snapshot.object_list_rows) == 1
    assert snapshot.tracker_frame is not None
    assert snapshot.tracker_frame.frame_index == 13
    assert snapshot.tracker_frame.tracker_debug is not None
    assert snapshot.tracker_frame.tracker_debug.track_states[0].track_id == 11
    assert snapshot.track_rows[11]["age"] == 6
    assert snapshot.track_outcomes[11].predicted_class_name == "car"
    assert np.allclose(snapshot.track_outcomes[11].last_center, np.array([2.0, 4.0, 0.4], dtype=np.float32))
    assert snapshot.lane_box is not None
    assert snapshot.lane_box.x_min == -3.0
    assert snapshot.require_track_exit is False
    assert snapshot.track_exit_edge_margin == 1.4
    assert snapshot.track_exit_line_axis == "x"
    assert snapshot.visualization_config is not None
    assert snapshot.visualization_config.max_assoc_dist == 5.6
    assert snapshot.visualization_config.show_track_outcome_debug is False


def test_live_snapshot_loader_respects_explicit_run_id_and_force_reload(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    loader = LiveSnapshotLoader(dataset_root, _config(dataset_root))
    stats_dir = dataset_root / "_stats" / "2026-04-07" / "run_force"
    tracker_path = stats_dir / "tracker_debug.jsonl"

    _write_jsonl(tracker_path, [_tracker_row(5)])
    initial_mtime = tracker_path.stat().st_mtime_ns

    first = loader.load(run_id="run_force")
    assert first.run_id == "run_force"
    assert first.tracker_frame is not None
    assert first.tracker_frame.frame_index == 5

    _write_jsonl(tracker_path, [_tracker_row(9)])
    os.utime(tracker_path, ns=(initial_mtime, initial_mtime))

    cached = loader.load(run_id="run_force")
    forced = loader.load(run_id="run_force", force=True)

    assert cached.tracker_frame is not None
    assert cached.tracker_frame.frame_index == 5
    assert forced.tracker_frame is not None
    assert forced.tracker_frame.frame_index == 9


def test_live_snapshot_loader_keeps_last_valid_snapshot_on_partial_jsonl_update(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    loader = LiveSnapshotLoader(dataset_root, _config(dataset_root))
    active_dir = dataset_root / "_stats" / "_active" / "run_partial"
    stats_dir = dataset_root / "_stats" / "2026-04-07" / "run_partial"
    tracker_path = stats_dir / "tracker_debug.jsonl"

    _write_json(active_dir / "live_status.json", _live_status(processed_frames=4, frame_index=4))
    _write_jsonl(tracker_path, [_tracker_row(4)])

    first = loader.load(run_id="run_partial")
    assert first.tracker_frame is not None
    assert first.tracker_frame.frame_index == 4

    tracker_path.write_text("{not-json}\n", encoding="utf-8")
    os.utime(tracker_path, ns=(200, 200))
    stale = loader.load(run_id="run_partial")
    assert stale.tracker_frame is not None
    assert stale.tracker_frame.frame_index == 4
    assert any("tracker debug updating" in warning for warning in stale.warnings)

    _write_jsonl(tracker_path, [_tracker_row(8)])
    os.utime(tracker_path, ns=(300, 300))
    fresh = loader.load(run_id="run_partial")
    assert fresh.tracker_frame is not None
    assert fresh.tracker_frame.frame_index == 8
