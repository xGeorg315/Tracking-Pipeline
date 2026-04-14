from __future__ import annotations

from pathlib import Path

import numpy as np

from tracking_pipeline.config.models import VisualizationConfig
from tracking_pipeline.domain.models import FrameTrackerDebug, TrackOutcomeDebug
from tracking_pipeline.infrastructure.visualization.live_snapshot_loader import LiveSnapshot, LiveTrackerFrameSnapshot
from tracking_pipeline.infrastructure.visualization.open3d_live_viewer import Open3DLiveViewer


def _snapshot(
    *,
    waiting: bool = False,
    live_status: dict[str, object] | None = None,
    summary: dict[str, object] | None = None,
    track_outcomes: dict[int, TrackOutcomeDebug] | None = None,
    tracker_debug: FrameTrackerDebug | None = None,
    warnings: list[str] | None = None,
) -> LiveSnapshot:
    return LiveSnapshot(
        dataset_root=Path("/tmp/dataset"),
        run_id="run_live",
        waiting=waiting,
        live_status=live_status or {},
        summary=summary or {},
        track_outcomes=track_outcomes or {},
        tracker_frame=None if tracker_debug is None else LiveTrackerFrameSnapshot(frame_index=13, tracker_debug=tracker_debug),
        warnings=warnings or [],
    )


def test_live_viewer_outcome_events_reuse_replay_status_colors_and_labels() -> None:
    viewer = Open3DLiveViewer(VisualizationConfig(), loader=None)  # type: ignore[arg-type]
    snapshot = _snapshot(
        track_outcomes={
            1: TrackOutcomeDebug(
                track_id=1,
                status="saved",
                decision_stage="saved",
                decision_reason_code="saved",
                decision_summary="saved points=248",
                last_frame_id=42,
                last_playback_index=7,
                last_center=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                predicted_class_name="car",
                predicted_class_score=0.88,
                gt_obj_class="PKW",
            ),
            2: TrackOutcomeDebug(
                track_id=2,
                status="skipped_min_hits",
                decision_stage="save_gate",
                decision_reason_code="min_hits",
                decision_summary="min_hits 2/4",
                last_frame_id=42,
                last_playback_index=7,
                last_center=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            ),
        }
    )

    events = viewer._build_outcome_events(snapshot)

    assert [event.track_id for event in events] == [1, 2]
    assert viewer._outcome_color(events[0]) == (0.20, 1.00, 0.35)
    assert viewer._outcome_color(events[1]) == (0.60, 0.60, 0.60)
    assert viewer._outcome_label_text(events[0]) == "saved #1 car 0.88 | gt:PKW"
    assert viewer._outcome_label_text(events[1]) == "skip #2 min_hits 2/4"


def test_live_viewer_status_summary_and_tracker_text_cover_waiting_and_live_states() -> None:
    waiting_snapshot = _snapshot(waiting=True)
    live_snapshot = _snapshot(
        live_status={
            "pipeline_phase": "streaming",
            "last_processed_frame_index": 18,
            "processed_frames": 19,
            "active_track_count": 3,
            "finished_track_count": 8,
            "saved_aggregates": 5,
            "object_list_exported_count": 12,
            "processing_recent_hz": 8.2,
            "processing_total_hz": 7.9,
        },
        summary={"gt_match_matched_count": 6, "gt_match_unmatched_gt_count": 4},
        tracker_debug=FrameTrackerDebug(assignment_method="hungarian", matched_count=2, missed_count=1, spawned_count=1),
        warnings=["stale/partial snapshot: tracker debug updating"],
    )

    assert Open3DLiveViewer._build_status_text(waiting_snapshot, paused=False) == "Waiting for live run (live)"
    assert Open3DLiveViewer._build_summary_text(waiting_snapshot) == "Waiting for active snapshot under /tmp/dataset"
    assert Open3DLiveViewer._build_status_text(live_snapshot, paused=True) == (
        "Run run_live\nphase=streaming frame=18 processed=19 refresh=paused"
    )
    assert Open3DLiveViewer._build_summary_text(live_snapshot) == (
        "active=3 finished=8 saved=5\n"
        "gt=12 hz=8.20/7.90\n"
        "gt_match matched=6 unmatched_gt=4\n"
        "stale/partial snapshot: tracker debug updating"
    )
    assert Open3DLiveViewer._build_tracker_debug_text(live_snapshot, enabled=True) == (
        "Tracker Debug (hungarian)\nmatched: 2\nmissed: 1\nspawned: 1\nsuppressed: 0"
    )
