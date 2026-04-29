from __future__ import annotations

import base64
import threading
import numpy as np

from tracking_pipeline.domain.models import (
    ActiveTrackState,
    ClusterResult,
    Detection,
    DetectionDebugState,
    FrameData,
    FrameTrackerDebug,
    FrameTrackingState,
    RunSummary,
    TrackDebugState,
    TrackOutcomeDebug,
)
from tracking_pipeline.domain.value_objects import LaneBox
from tracking_pipeline.infrastructure.visualization.live_frame_publisher import LiveFramePublisher
from tracking_pipeline.infrastructure.visualization import live_frame_publisher as live_frame_publisher_module


def _publisher(
    *,
    max_points: int = 3,
    history_sec: float = 1.0,
    max_frames: int | None = None,
    async_publish: bool = False,
) -> LiveFramePublisher:
    return LiveFramePublisher(
        lane_box=LaneBox.from_values([-2.0, 2.0, 0.0, 12.0, 0.0, 3.0]),
        track_exit_line_axis="y",
        track_exit_edge_margin=0.9,
        max_points=max_points,
        history_sec=history_sec,
        point_source="lane",
        color_by_intensity=True,
        show_tracker_debug=True,
        show_track_outcomes=True,
        run_label="run_live_001",
        max_frames=max_frames,
        async_publish=async_publish,
    )


def _frame(frame_index: int, timestamp_ns: int, point_count: int = 5) -> FrameData:
    points = np.stack(
        [
            np.linspace(0.0, float(point_count - 1), point_count, dtype=np.float32),
            np.full((point_count,), float(frame_index), dtype=np.float32),
            np.linspace(0.0, 1.0, point_count, dtype=np.float32),
        ],
        axis=1,
    )
    return FrameData(
        frame_index=frame_index,
        timestamp_ns=timestamp_ns,
        points=points,
        point_intensity=np.linspace(0.1, 0.9, point_count, dtype=np.float32),
        source_path="qb2_live://sensor@10.0.0.1",
    )


def _cluster_result() -> ClusterResult:
    detection = Detection(
        detection_id=7,
        points=np.array([[0.0, 1.0, 0.5]], dtype=np.float32),
        center=np.array([0.5, 1.5, 0.5], dtype=np.float32),
        min_bound=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        max_bound=np.array([1.0, 2.0, 1.0], dtype=np.float32),
    )
    return ClusterResult(
        lane_points=np.array(
            [
                [10.0, 20.0, 0.5],
                [11.0, 21.0, 0.6],
                [12.0, 22.0, 0.7],
            ],
            dtype=np.float32,
        ),
        detections=[detection],
        lane_intensity=np.array([0.2, 0.4, 0.6], dtype=np.float32),
    )


def _tracking_state(frame_index: int) -> FrameTrackingState:
    return FrameTrackingState(
        frame_index=frame_index,
        lane_points=np.array(
            [
                [10.0, 20.0, 0.5],
                [11.0, 21.0, 0.6],
                [12.0, 22.0, 0.7],
            ],
            dtype=np.float32,
        ),
        detections=[],
        active_tracks=[
            ActiveTrackState(
                track_id=9,
                points=np.array([[0.5, 1.5, 0.5]], dtype=np.float32),
                center=np.array([0.5, 1.5, 0.5], dtype=np.float32),
                status="matched",
            )
        ],
        tracker_debug=FrameTrackerDebug(
            assignment_method="hungarian",
            detection_states=[
                DetectionDebugState(
                    detection_id=7,
                    center=np.array([0.5, 1.5, 0.5], dtype=np.float32),
                    status="matched",
                    matched_track_id=9,
                )
            ],
            track_states=[
                TrackDebugState(
                    track_id=9,
                    predicted_center=np.array([0.6, 1.6, 0.5], dtype=np.float32),
                    output_center=np.array([0.5, 1.5, 0.5], dtype=np.float32),
                    status="matched",
                    matched_detection_id=7,
                    gate_radius=1.4,
                    missed_before=0,
                    missed_after=0,
                )
            ],
            matched_count=1,
        ),
        lane_intensity=np.array([0.2, 0.4, 0.6], dtype=np.float32),
    )


def test_live_frame_publisher_caps_points_and_prunes_frames_in_rolling_window() -> None:
    publisher = _publisher(max_points=3, history_sec=1.0, max_frames=2)

    seq_1 = publisher.publish_frame(_frame(0, 0, point_count=6), _cluster_result(), _tracking_state(0))
    seq_2 = publisher.publish_frame(_frame(1, 400_000_000, point_count=6), _cluster_result(), _tracking_state(1))
    seq_3 = publisher.publish_frame(_frame(2, 800_000_000, point_count=6), _cluster_result(), _tracking_state(2))

    assert seq_1 == 1
    assert seq_2 == 2
    assert seq_3 == 3
    assert publisher.get_frame(1) is None
    kept = publisher.get_frame(3)
    assert kept is not None
    assert kept["point_count"] == 3
    assert kept["points_xyz_encoding"] == "f16"
    assert isinstance(kept["points_xyz_b64"], str)
    assert len(base64.b64decode(kept["points_xyz_b64"])) == 9 * 2
    assert "point_intensity_b64" not in kept
    assert "point_intensity_encoding" not in kept

    publisher.publish_frame(_frame(3, 2_200_000_000, point_count=4), _cluster_result(), _tracking_state(3))

    meta = publisher.current_meta()
    assert publisher.get_frame(2) is None
    assert publisher.get_frame(3) is None
    assert publisher.get_frame(4) is not None
    assert meta["sequence_window"]["oldest_sequence_id"] == 4
    assert meta["sequence_window"]["latest_sequence_id"] == 4
    assert meta["sequence_window"]["frame_count"] == 1
    assert meta["sequence_window"]["pruned_frame_count"] == 3
    assert "retain_all_frames" not in meta


def test_live_frame_publisher_prunes_by_hard_frame_limit() -> None:
    publisher = _publisher(max_points=3, history_sec=10.0, max_frames=2)

    publisher.publish_frame(_frame(0, 0, point_count=6), _cluster_result(), _tracking_state(0))
    publisher.publish_frame(_frame(1, 400_000_000, point_count=6), _cluster_result(), _tracking_state(1))
    publisher.publish_frame(_frame(2, 800_000_000, point_count=6), _cluster_result(), _tracking_state(2))

    meta = publisher.current_meta()

    assert publisher.get_frame(1) is None
    assert publisher.get_frame(2) is not None
    assert publisher.get_frame(3) is not None
    assert meta["sequence_window"]["oldest_sequence_id"] == 2
    assert meta["sequence_window"]["latest_sequence_id"] == 3
    assert meta["sequence_window"]["frame_count"] == 2
    assert meta["sequence_window"]["pruned_frame_count"] == 1


def test_live_frame_publisher_returns_frame_batches_in_sequence_order() -> None:
    publisher = _publisher(max_points=3, history_sec=10.0, max_frames=2)

    publisher.publish_frame(_frame(0, 0, point_count=6), _cluster_result(), _tracking_state(0))
    publisher.publish_frame(_frame(1, 400_000_000, point_count=6), _cluster_result(), _tracking_state(1))
    publisher.publish_frame(_frame(2, 800_000_000, point_count=6), _cluster_result(), _tracking_state(2))
    publisher.publish_frame(_frame(3, 1_200_000_000, point_count=6), _cluster_result(), _tracking_state(3))

    batch = publisher.get_frames(1, limit=4)

    assert [int(row["sequence_id"]) for row in batch] == [3, 4]
    assert [int(row["frame_index"]) for row in batch] == [2, 3]


def test_live_frame_publisher_serializes_tracker_status_summary_and_outcomes() -> None:
    publisher = _publisher(max_points=4, history_sec=1.5, max_frames=4)
    publisher.update_status(pipeline_phase="processing_frames", processed_frames=3, active_track_count=1)
    publisher.update_track_outcomes(
        {
            9: TrackOutcomeDebug(
                track_id=9,
                status="saved",
                decision_stage="save_gate",
                decision_reason_code="saved",
                decision_summary="saved after live snapshot",
                last_frame_id=2,
                last_playback_index=2,
                last_center=np.array([0.5, 1.5, 0.5], dtype=np.float32),
                predicted_class_name="car",
            )
        }
    )
    publisher.update_summary(
        RunSummary(
            input_path="qb2_live://sensor@10.0.0.1",
            input_paths=["qb2_live://sensor@10.0.0.1"],
            output_mode="dataset",
            tracker_algorithm="kalman_nn",
            accumulator_algorithm="registration_voxel_fusion",
            clusterer_algorithm="dbscan",
            frame_count=3,
            finished_track_count=1,
            saved_aggregates=1,
            registration_attempts=2,
            registration_accepted=2,
            registration_rejected=0,
            output_dir="/tmp/run_live_001",
        )
    )

    publisher.publish_frame(_frame(2, 2_000_000_000, point_count=4), _cluster_result(), _tracking_state(2))

    meta = publisher.current_meta()
    frame_payload = publisher.get_frame(1)

    assert meta["run_label"] == "run_live_001"
    assert meta["status"]["pipeline_phase"] == "processing_frames"
    assert meta["status"]["processed_frames"] == 3
    assert meta["summary"]["saved_aggregates"] == 1
    assert meta["point_source"] == "lane"
    assert meta["monitoring"]["status_line"].startswith("live phase=processing_frames f=3")
    assert "hz=0.00/0.00" in meta["monitoring"]["status_line"]
    assert "saved=0" in meta["monitoring"]["status_line"]
    assert "raw=0 mqtt=0" in meta["monitoring"]["status_line"]
    assert meta["track_outcome_version"] == 1
    assert meta["track_outcomes"][0]["track_id"] == 9
    assert meta["track_outcomes"][0]["status"] == "saved"
    assert meta["track_outcomes"][0]["updated_at_unix_sec"] > 0.0
    assert meta["overlay_defaults"]["show_tracker_debug"] is True
    assert meta["overlay_defaults"]["show_track_outcomes"] is True
    assert frame_payload is not None
    assert frame_payload["detections"][0]["matched_track_id"] == 9
    assert frame_payload["detections"][0]["min_bound"] == [0.0, 1.0, 0.0]
    assert frame_payload["detections"][0]["max_bound"] == [1.0, 2.0, 1.0]
    assert frame_payload["detections"][0]["extent"] == [1.0, 1.0, 1.0]
    assert frame_payload["detections"][0]["point_count"] == 1
    assert frame_payload["track_states"][0]["matched_detection_id"] == 7
    assert frame_payload["track_states"][0]["status"] == "matched"


def test_live_frame_publisher_uses_full_frame_points_when_configured() -> None:
    publisher = LiveFramePublisher(
        lane_box=LaneBox.from_values([-2.0, 2.0, 0.0, 12.0, 0.0, 3.0]),
        track_exit_line_axis="y",
        track_exit_edge_margin=0.9,
        max_points=4,
        history_sec=1.0,
        point_source="all",
        color_by_intensity=True,
        show_tracker_debug=True,
        show_track_outcomes=True,
        run_label="run_live_001",
        max_frames=2,
        async_publish=False,
    )

    publisher.publish_frame(_frame(0, 0, point_count=5), _cluster_result(), _tracking_state(0))

    payload = publisher.get_frame(1)
    assert payload is not None
    assert payload["points_xyz_encoding"] == "f16"
    assert payload["point_count"] == 4
    assert len(base64.b64decode(payload["points_xyz_b64"])) == 12 * 2


def test_live_frame_publisher_async_mode_drops_stale_pending_frames() -> None:
    publisher = _publisher(max_points=3, history_sec=1.0, max_frames=4, async_publish=True)
    original_serializer = live_frame_publisher_module._serialize_float16_base64
    serializer_started = threading.Event()
    serializer_release = threading.Event()

    def _slow_serializer(values):
        serializer_started.set()
        serializer_release.wait(timeout=2.0)
        return original_serializer(values)

    try:
        live_frame_publisher_module._serialize_float16_base64 = _slow_serializer
        first_seq = publisher.publish_frame(_frame(0, 0, point_count=6), _cluster_result(), _tracking_state(0))
        assert serializer_started.wait(timeout=2.0)
        second_seq = publisher.publish_frame(_frame(1, 400_000_000, point_count=6), _cluster_result(), _tracking_state(1))
        third_seq = publisher.publish_frame(_frame(2, 800_000_000, point_count=6), _cluster_result(), _tracking_state(2))
    finally:
        serializer_release.set()
        live_frame_publisher_module._serialize_float16_base64 = original_serializer

    publisher.flush_pending(timeout=2.0)
    meta = publisher.current_meta()
    publisher.close(timeout=2.0)

    assert first_seq == 1
    assert second_seq == 2
    assert third_seq == 3
    assert publisher.get_frame(first_seq) is not None
    assert publisher.get_frame(second_seq) is None
    assert publisher.get_frame(third_seq) is not None
    assert meta["status"]["live_web_dropped_frame_count"] >= 1


def test_live_frame_publisher_preserves_outcome_timestamp_for_unchanged_rows() -> None:
    publisher = _publisher(max_points=4, history_sec=1.0, max_frames=2)
    outcome = TrackOutcomeDebug(
        track_id=9,
        status="saved",
        decision_stage="save_gate",
        decision_reason_code="saved",
        decision_summary="saved after live snapshot",
        last_frame_id=2,
        last_playback_index=2,
        last_center=np.array([0.5, 1.5, 0.5], dtype=np.float32),
        predicted_class_name="car",
    )

    original_time = live_frame_publisher_module.time.time
    try:
        live_frame_publisher_module.time.time = lambda: 100.0
        publisher.update_track_outcomes({9: outcome})
        first_updated_at = publisher.current_meta()["track_outcomes"][0]["updated_at_unix_sec"]

        live_frame_publisher_module.time.time = lambda: 105.0
        publisher.update_track_outcomes({9: outcome})
        second_updated_at = publisher.current_meta()["track_outcomes"][0]["updated_at_unix_sec"]

        changed_outcome = TrackOutcomeDebug(
            track_id=9,
            status="saved",
            decision_stage="save_gate",
            decision_reason_code="saved",
            decision_summary="saved after live snapshot updated",
            last_frame_id=2,
            last_playback_index=2,
            last_center=np.array([0.5, 1.5, 0.5], dtype=np.float32),
            predicted_class_name="car",
        )
        live_frame_publisher_module.time.time = lambda: 110.0
        publisher.update_track_outcomes({9: changed_outcome})
        third_updated_at = publisher.current_meta()["track_outcomes"][0]["updated_at_unix_sec"]
    finally:
        live_frame_publisher_module.time.time = original_time

    assert first_updated_at == 100.0
    assert second_updated_at == 100.0
    assert third_updated_at == 110.0
