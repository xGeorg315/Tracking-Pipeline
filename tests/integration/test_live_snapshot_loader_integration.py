from __future__ import annotations

import json
import os
from pathlib import Path

from tracking_pipeline.config.models import InputConfig, OutputConfig, PipelineConfig, PreprocessingConfig
from tracking_pipeline.infrastructure.visualization.live_snapshot_loader import LiveSnapshotLoader


def _config(dataset_root: Path) -> PipelineConfig:
    return PipelineConfig(
        input=InputConfig(paths=["dummy.pb"]),
        preprocessing=PreprocessingConfig(lane_box=[-1.0, 1.0, 0.0, 10.0, 0.0, 2.0]),
        output=OutputConfig(mode="dataset", dataset_root_dir=str(dataset_root)),
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_live_snapshot_loader_tracks_incremental_snapshot_updates(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    loader = LiveSnapshotLoader(dataset_root, _config(dataset_root))
    run_id = "run_incremental"
    active_dir = dataset_root / "_stats" / "_active" / run_id
    stats_dir = dataset_root / "_stats" / "2026-04-07" / run_id

    _write_json(
        active_dir / "live_status.json",
        {
            "pipeline_phase": "streaming",
            "processed_frames": 1,
            "last_processed_frame_index": 0,
            "active_track_count": 1,
            "finished_track_count": 0,
            "saved_aggregates": 0,
        },
    )
    _write_jsonl(
        stats_dir / "tracker_debug.jsonl",
        [
            {
                "frame_index": 0,
                "tracker_debug": {
                    "assignment_method": "hungarian",
                    "matched_count": 0,
                    "missed_count": 0,
                    "spawned_count": 1,
                    "track_states": [{"track_id": 1, "status": "spawned", "output_center": [0.0, 1.0, 0.0]}],
                    "detection_states": [{"detection_id": 1, "center": [0.0, 1.0, 0.0], "status": "spawned"}],
                },
            }
        ],
    )
    os.utime(active_dir / "live_status.json", ns=(100, 100))
    os.utime(stats_dir / "tracker_debug.jsonl", ns=(110, 110))

    first = loader.load()

    assert first.run_id == run_id
    assert first.tracker_frame is not None
    assert first.tracker_frame.frame_index == 0
    assert first.live_status["processed_frames"] == 1

    _write_json(
        active_dir / "live_status.json",
        {
            "pipeline_phase": "streaming",
            "processed_frames": 3,
            "last_processed_frame_index": 2,
            "active_track_count": 2,
            "finished_track_count": 1,
            "saved_aggregates": 1,
        },
    )
    _write_jsonl(
        stats_dir / "tracker_debug.jsonl",
        [
            {
                "frame_index": 0,
                "tracker_debug": {
                    "assignment_method": "hungarian",
                    "matched_count": 0,
                    "missed_count": 0,
                    "spawned_count": 1,
                    "track_states": [{"track_id": 1, "status": "spawned", "output_center": [0.0, 1.0, 0.0]}],
                    "detection_states": [{"detection_id": 1, "center": [0.0, 1.0, 0.0], "status": "spawned"}],
                },
            },
            {
                "frame_index": 2,
                "tracker_debug": {
                    "assignment_method": "hungarian",
                    "matched_count": 1,
                    "missed_count": 0,
                    "spawned_count": 1,
                    "track_states": [{"track_id": 2, "status": "matched", "output_center": [1.0, 2.0, 0.0]}],
                    "detection_states": [{"detection_id": 2, "center": [1.0, 2.0, 0.0], "status": "matched"}],
                },
            },
        ],
    )
    _write_json(stats_dir / "summary.json", {"gt_match_matched_count": 1, "gt_match_unmatched_gt_count": 2})
    os.utime(active_dir / "live_status.json", ns=(200, 200))
    os.utime(stats_dir / "tracker_debug.jsonl", ns=(210, 210))
    os.utime(stats_dir / "summary.json", ns=(220, 220))

    second = loader.load()

    assert second.run_id == run_id
    assert second.tracker_frame is not None
    assert second.tracker_frame.frame_index == 2
    assert second.live_status["processed_frames"] == 3
    assert second.summary["gt_match_unmatched_gt_count"] == 2
