from __future__ import annotations

from pathlib import Path

import numpy as np

from tracking_pipeline.domain.models import FrameTrackerDebug, TrackOutcomeDebug, DetectionDebugState, TrackDebugState
from tracking_pipeline.infrastructure.visualization.live_snapshot_loader import LiveSnapshot, LiveTrackerFrameSnapshot
from tracking_pipeline.infrastructure.visualization.live_web_server import LiveWebViewerServer, build_live_web_payload


def _snapshot(
    *,
    waiting: bool = False,
    tracker_debug: FrameTrackerDebug | None = None,
    track_outcomes: dict[int, TrackOutcomeDebug] | None = None,
    warnings: list[str] | None = None,
) -> LiveSnapshot:
    return LiveSnapshot(
        dataset_root=Path("/tmp/dataset"),
        run_id="run_web",
        waiting=waiting,
        warnings=warnings or [],
        live_status={
            "pipeline_phase": "streaming",
            "last_processed_frame_index": 12,
            "processed_frames": 13,
            "active_track_count": 2,
            "finished_track_count": 5,
            "saved_aggregates": 3,
        },
        tracker_frame=None if tracker_debug is None else LiveTrackerFrameSnapshot(frame_index=12, tracker_debug=tracker_debug),
        track_outcomes=track_outcomes or {},
    )


def test_build_live_web_payload_serializes_waiting_snapshot() -> None:
    payload = build_live_web_payload(_snapshot(waiting=True))

    assert payload["waiting"] is True
    assert payload["run_id"] == "run_web"
    assert payload["status_text"] == "Waiting for live run (live)"
    assert payload["poll_interval_ms"] == 1000
    assert payload["detections"] == []
    assert payload["tracks"] == []
    assert payload["outcomes"] == []


def test_build_live_web_payload_serializes_tracker_and_outcomes() -> None:
    payload = build_live_web_payload(
        _snapshot(
            tracker_debug=FrameTrackerDebug(
                assignment_method="hungarian",
                matched_count=1,
                missed_count=0,
                spawned_count=1,
                track_states=[
                    TrackDebugState(
                        track_id=7,
                        status="matched",
                        predicted_center=np.array([1.0, 2.0, 0.5], dtype=np.float32),
                        output_center=np.array([1.2, 2.1, 0.5], dtype=np.float32),
                        matched_detection_id=101,
                    )
                ],
                detection_states=[
                    DetectionDebugState(
                        detection_id=101,
                        center=np.array([1.2, 2.15, 0.5], dtype=np.float32),
                        status="matched",
                        matched_track_id=7,
                    )
                ],
            ),
            track_outcomes={
                7: TrackOutcomeDebug(
                    track_id=7,
                    status="saved",
                    decision_stage="saved",
                    decision_reason_code="saved",
                    decision_summary="saved points=248",
                    last_frame_id=12,
                    last_playback_index=12,
                    last_center=np.array([1.2, 2.1, 0.5], dtype=np.float32),
                    predicted_class_name="car",
                    predicted_class_score=0.91,
                    gt_obj_class="PKW",
                )
            },
            warnings=["stale/partial snapshot: tracker debug updating"],
        )
    )

    assert payload["waiting"] is False
    assert payload["tracker_text"].startswith("Tracker Debug (hungarian)")
    assert payload["warnings"] == ["stale/partial snapshot: tracker debug updating"]
    assert payload["detections"][0]["detection_id"] == 101
    assert payload["detections"][0]["color"] == [0.2, 1.0, 0.35]
    assert payload["tracks"][0]["track_id"] == 7
    assert payload["tracks"][0]["predicted_label"] == "pred #7"
    assert payload["tracks"][0]["output_label"] == "track #7"
    assert payload["outcomes"][0]["label"] == "saved #7 car 0.91 | gt:PKW"
    assert payload["outcomes"][0]["color"] == [0.2, 1.0, 0.35]


def test_live_web_server_force_refresh_invalidates_loader_cache() -> None:
    class _FakeLoader:
        def __init__(self) -> None:
            self.invalidated = 0
            self.calls: list[tuple[str | None, bool]] = []

        def invalidate_cache(self) -> None:
            self.invalidated += 1

        def load(self, run_id: str | None = None, force: bool = False) -> LiveSnapshot:
            self.calls.append((run_id, force))
            return _snapshot()

    loader = _FakeLoader()
    server = LiveWebViewerServer(loader)

    server.snapshot_payload(run_id="run_web", force=True)

    assert loader.invalidated == 1
    assert loader.calls == [("run_web", True)]
