from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from tracking_pipeline.config.models import TrackingConfig
from tracking_pipeline.domain.models import Detection
from tracking_pipeline.infrastructure.readers.a42_proto import common, data, frame, label, sensors
from tracking_pipeline.infrastructure.readers.a42_pb_reader import A42PBReader
from tracking_pipeline.infrastructure.tracking.assignment import assign_cost_matrix
from tracking_pipeline.infrastructure.tracking.euclidean_nn import EuclideanNNTracker
from tracking_pipeline.infrastructure.tracking.hungarian_kalman import HungarianKalmanTracker
from tracking_pipeline.infrastructure.tracking.kalman_nn import KalmanNNTracker


def _detection(center_x: float, detection_id: int = 1, tracking_halo_only: bool = False, spawn_suppressed: bool = False) -> Detection:
    points = np.array([[center_x, 0.0, 0.0], [center_x + 0.1, 0.0, 0.0]], dtype=np.float32)
    return Detection(
        detection_id=detection_id,
        points=points,
        center=points.mean(axis=0),
        min_bound=points.min(axis=0),
        max_bound=points.max(axis=0),
        metadata={
            "tracking_halo_only": bool(tracking_halo_only),
            "spawn_suppressed": bool(spawn_suppressed),
        },
    )


def _pack_xyz(points: list[list[float]]) -> bytes:
    return np.asarray(points, dtype="<f4").reshape(-1, 3).tobytes()


def _pack_u16(values: list[int]) -> bytes:
    return np.asarray(values, dtype="<u2").tobytes()


def _pack_u64(values: list[int]) -> bytes:
    return np.asarray(values, dtype="<u8").tobytes()


def _write_object_list_fixture(path: Path) -> Path:
    calibration = common.SensorCalibration(sensor_name="sensor_a")
    frames = [
        frame.Frame(
            frame_timestamp_ns=1000,
            frame_id=1,
            lidars=[
                sensors.LidarScan(
                    scan_timestamp_ns=1000,
                    pointcloud=data.PointCloud(cartesian=_pack_xyz([[0.0, 0.0, 0.0]])),
                    calibration=calibration,
                    object_list=[
                        label.ObjectBBox(
                            id=7,
                            timestamp_ns=1000,
                            pointcloud=data.PointCloud(cartesian=_pack_xyz([[1.0, 0.0, 0.0]])),
                            obj_class="car",
                            obj_class_score=0.9,
                        ),
                        label.ObjectBBox(
                            id=9,
                            timestamp_ns=1001,
                            pointcloud=data.PointCloud(cartesian=b""),
                            obj_class="truck",
                            obj_class_score=0.7,
                        ),
                    ],
                )
            ],
        ),
        frame.Frame(
            frame_timestamp_ns=1001,
            frame_id=2,
            lidars=[
                sensors.LidarScan(
                    scan_timestamp_ns=1001,
                    pointcloud=data.PointCloud(cartesian=_pack_xyz([[0.1, 0.0, 0.0]])),
                    calibration=calibration,
                    object_list=[
                        label.ObjectBBox(
                            id=7,
                            timestamp_ns=1002,
                            pointcloud=data.PointCloud(cartesian=_pack_xyz([[2.0, 0.0, 0.0], [2.5, 0.0, 0.0]])),
                            obj_class="car",
                            obj_class_score=0.95,
                        )
                    ],
                )
            ],
        ),
    ]
    with path.open("wb") as handle:
        for item in frames:
            payload = bytes(item)
            handle.write(struct.pack("<I", len(payload)))
            handle.write(payload)
    return path


def test_a42_reader_reads_fixture() -> None:
    fixture = Path(__file__).resolve().parents[2] / "fixtures" / "sample_a42.pb"
    frames = A42PBReader().iter_frames([str(fixture)])
    assert len(frames) == 10
    assert frames[0].points.shape[1] == 3
    assert frames[0].source_path == str(fixture.resolve())
    assert frames[0].source_frame_index == 0
    assert frames[0].source_sequence_index == 0
    assert frames[0].scans
    assert frames[0].scans[0].xyz.shape[1] == 3
    assert len(frames[0].scans[0].ranges) == len(frames[0].scans[0].xyz)
    assert len(frames[0].scans[0].row_index) == len(frames[0].scans[0].xyz)
    assert len(frames[0].scans[0].col_index) == len(frames[0].scans[0].xyz)


def test_a42_reader_concatenates_multiple_files_with_global_frame_index() -> None:
    fixture = Path(__file__).resolve().parents[2] / "fixtures" / "sample_a42.pb"
    frames = A42PBReader().iter_frames([str(fixture), str(fixture)])

    assert len(frames) == 20
    assert frames[0].frame_index == 0
    assert frames[9].frame_index == 9
    assert frames[10].frame_index == 10
    assert frames[19].frame_index == 19
    assert frames[9].source_sequence_index == 0
    assert frames[10].source_sequence_index == 1
    assert frames[9].source_frame_index == 9
    assert frames[10].source_frame_index == 0


def test_a42_reader_reads_object_list_labels(tmp_path: Path) -> None:
    fixture = _write_object_list_fixture(tmp_path / "object_list.pb")
    frames = A42PBReader().iter_frames([str(fixture)])

    assert len(frames) == 2
    assert len(frames[0].object_labels) == 2
    assert frames[0].object_labels[0].object_id == 7
    assert frames[0].object_labels[0].sensor_name == "sensor_a"
    assert frames[0].object_labels[0].frame_index == 0
    assert frames[0].object_labels[0].source_path == str(fixture.resolve())
    assert frames[0].object_labels[0].points.shape == (1, 3)
    assert frames[0].object_labels[1].object_id == 9
    assert frames[0].object_labels[1].points.shape == (0, 3)
    assert frames[1].object_labels[0].timestamp_ns == 1002
    assert frames[1].object_labels[0].points.shape == (2, 3)


def test_a42_reader_reads_range_corrected_reflectivity_from_reflectivity_field(tmp_path: Path) -> None:
    calibration = common.SensorCalibration(sensor_name="sensor_a")
    fixture = tmp_path / "reflectivity.pb"
    payload_frame = frame.Frame(
        frame_timestamp_ns=1000,
        frame_id=1,
        lidars=[
            sensors.LidarScan(
                scan_timestamp_ns=1000,
                pointcloud=data.PointCloud(
                    cartesian=_pack_xyz([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
                    reflectivity=_pack_u16([0, 32768, 65535]),
                ),
                calibration=calibration,
            )
        ],
    )
    with fixture.open("wb") as handle:
        payload = bytes(payload_frame)
        handle.write(struct.pack("<I", len(payload)))
        handle.write(payload)

    frames = A42PBReader(read_intensity=True).iter_frames([str(fixture)])

    assert len(frames) == 1
    assert frames[0].point_intensity is not None
    assert np.allclose(
        frames[0].point_intensity,
        np.array([0.0, 32768.0 / 65535.0, 4.0], dtype=np.float32),
        atol=1e-6,
    )
    assert frames[0].scans[0].intensity is not None
    assert np.allclose(frames[0].scans[0].intensity, frames[0].point_intensity)


def test_a42_reader_falls_back_to_intensity_for_range_corrected_reflectivity(tmp_path: Path) -> None:
    calibration = common.SensorCalibration(sensor_name="sensor_a")
    fixture = tmp_path / "intensity.pb"
    payload_frame = frame.Frame(
        frame_timestamp_ns=1000,
        frame_id=1,
        lidars=[
            sensors.LidarScan(
                scan_timestamp_ns=1000,
                pointcloud=data.PointCloud(
                    cartesian=_pack_xyz([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
                    intensity=_pack_u16([0, 32768, 65535]),
                ),
                calibration=calibration,
            )
        ],
    )
    with fixture.open("wb") as handle:
        payload = bytes(payload_frame)
        handle.write(struct.pack("<I", len(payload)))
        handle.write(payload)

    frames = A42PBReader(read_intensity=True).iter_frames([str(fixture)])

    assert len(frames) == 1
    assert frames[0].point_intensity is not None
    assert np.allclose(
        frames[0].point_intensity,
        np.array([0.0, 32768.0 / 65535.0, 4.0], dtype=np.float32),
        atol=1e-6,
    )


def test_a42_reader_prefers_reflectivity_over_intensity_for_range_corrected_output(tmp_path: Path) -> None:
    calibration = common.SensorCalibration(sensor_name="sensor_a")
    fixture = tmp_path / "priority.pb"
    payload_frame = frame.Frame(
        frame_timestamp_ns=1000,
        frame_id=1,
        lidars=[
            sensors.LidarScan(
                scan_timestamp_ns=1000,
                pointcloud=data.PointCloud(
                    cartesian=_pack_xyz([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
                    intensity=_pack_u16([65535, 65535]),
                    reflectivity=_pack_u16([32768, 32768]),
                ),
                calibration=calibration,
            )
        ],
    )
    with fixture.open("wb") as handle:
        payload = bytes(payload_frame)
        handle.write(struct.pack("<I", len(payload)))
        handle.write(payload)

    frames = A42PBReader(read_intensity=True).iter_frames([str(fixture)])

    assert len(frames) == 1
    assert frames[0].point_intensity is not None
    assert np.allclose(
        frames[0].point_intensity,
        np.array([32768.0 / 65535.0, 4.0 * (32768.0 / 65535.0)], dtype=np.float32),
        atol=1e-6,
    )


def test_a42_reader_reads_current_dataset_reflectivity_as_range_corrected_output() -> None:
    fixture = Path(__file__).resolve().parents[2] / "data" / "20260316_120647_01401-01600.pb"
    if not fixture.exists():
        pytest.skip("Current dataset not available in workspace")

    with fixture.open("rb") as handle:
        size = struct.unpack("<I", handle.read(4))[0]
        payload = handle.read(size)
    raw_frame = frame.Frame().parse(payload)
    raw_scan = raw_frame.lidars[0]
    raw_xyz = np.frombuffer(raw_scan.pointcloud.cartesian, dtype="<f4").reshape(-1, 3)
    finite_mask = np.isfinite(raw_xyz).all(axis=1)
    raw_signal = np.frombuffer(raw_scan.pointcloud.reflectivity, dtype="<u2").astype(np.float32) / 65535.0
    expected = raw_signal[finite_mask] * np.sum(np.square(raw_xyz[finite_mask]), axis=1, dtype=np.float32)

    frames = A42PBReader(read_intensity=True).iter_frames([str(fixture)])

    assert len(raw_scan.pointcloud.intensity) == 0
    assert len(raw_scan.pointcloud.reflectivity) > 0
    assert frames[0].scans[0].intensity is not None
    assert np.allclose(frames[0].scans[0].intensity, expected.astype(np.float32, copy=False), atol=1e-6)


def test_a42_reader_reads_absolute_point_timestamps_from_timestamp_offset(tmp_path: Path) -> None:
    calibration = common.SensorCalibration(sensor_name="sensor_a")
    fixture = tmp_path / "timestamp_offset.pb"
    payload_frame = frame.Frame(
        frame_timestamp_ns=1000,
        frame_id=1,
        lidars=[
            sensors.LidarScan(
                scan_timestamp_ns=5000,
                pointcloud=data.PointCloud(
                    cartesian=_pack_xyz([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
                    timestamp_offset=_pack_u64([0, 10, 25]),
                ),
                calibration=calibration,
            )
        ],
    )
    with fixture.open("wb") as handle:
        payload = bytes(payload_frame)
        handle.write(struct.pack("<I", len(payload)))
        handle.write(payload)

    frames = A42PBReader(read_intensity=False).iter_frames([str(fixture)])

    assert len(frames) == 1
    assert frames[0].point_timestamp_ns is not None
    assert np.array_equal(frames[0].point_timestamp_ns, np.array([5000, 5010, 5025], dtype=np.int64))
    assert frames[0].scans[0].point_timestamp_ns is not None
    assert np.array_equal(frames[0].scans[0].point_timestamp_ns, frames[0].point_timestamp_ns)


def test_euclidean_tracker_keeps_same_track_for_close_detections() -> None:
    tracker = EuclideanNNTracker(TrackingConfig(algorithm="euclidean_nn", max_dist=2.0, max_missed=2))
    tracker.step([_detection(0.0)], 0, 1000)
    tracker.step([_detection(0.5)], 1, 2000)
    tracks = tracker.finalize()
    assert list(tracks.keys()) == [1]
    assert tracks[1].hit_count == 2


def test_kalman_tracker_survives_short_gap() -> None:
    tracker = KalmanNNTracker(TrackingConfig(max_dist=2.0, max_missed=2, sticky_max_dist=3.0))
    tracker.step([_detection(0.0)], 0, 1000)
    tracker.step([], 1, 2000)
    tracker.step([_detection(0.4)], 2, 3000)
    tracks = tracker.finalize()
    assert list(tracks.keys()) == [1]
    assert tracks[1].hit_count == 2


def test_tracker_propagates_detection_intensity_to_active_state_and_track() -> None:
    tracker = KalmanNNTracker(TrackingConfig(max_dist=2.0, max_missed=1, sticky_max_dist=3.0))
    detection = _detection(0.0)
    detection.intensity = np.array([0.2, 0.8], dtype=np.float32)
    detection.point_timestamp_ns = np.array([1000, 1030], dtype=np.int64)

    state = tracker.step([detection], 0, 1000)
    tracks = tracker.finalize()

    assert state.active_tracks[0].intensity is not None
    assert np.allclose(state.active_tracks[0].intensity, np.array([0.2, 0.8], dtype=np.float32))
    assert tracks[1].world_intensity[0] is not None
    assert np.allclose(tracks[1].world_intensity[0], np.array([0.2, 0.8], dtype=np.float32))
    assert np.allclose(tracks[1].local_intensity[0], np.array([0.2, 0.8], dtype=np.float32))
    assert tracks[1].point_timestamps_ns[0] is not None
    assert np.array_equal(tracks[1].point_timestamps_ns[0], np.array([1000, 1030], dtype=np.int64))
    assert tracks[1].frame_timestamps_ns == [1000]


def test_kalman_tracker_emits_match_debug_state() -> None:
    tracker = KalmanNNTracker(TrackingConfig(max_dist=2.0, max_missed=1, sticky_max_dist=3.0))
    tracker.step([_detection(0.0, detection_id=10)], 0, 1000)

    state = tracker.step([_detection(0.4, detection_id=11)], 1, 2000)

    assert state.tracker_debug is not None
    assert state.tracker_metrics["matched_count"] == 1
    assert state.tracker_debug.track_states[0].status == "matched"
    assert state.tracker_debug.track_states[0].matched_detection_id == 11
    assert state.tracker_debug.track_states[0].predicted_center is not None
    assert state.tracker_debug.detection_states[0].status == "matched"
    assert state.tracker_debug.detection_states[0].matched_track_id == 1


def test_kalman_tracker_emits_missed_debug_state() -> None:
    tracker = KalmanNNTracker(TrackingConfig(max_dist=2.0, max_missed=2, sticky_max_dist=3.0))
    tracker.step([_detection(0.0, detection_id=10)], 0, 1000)

    state = tracker.step([], 1, 2000)

    assert state.tracker_debug is not None
    assert state.tracker_metrics["unmatched_track_count"] == 1
    assert state.tracker_debug.track_states[0].status == "missed"
    assert state.tracker_debug.track_states[0].missed_before == 0
    assert state.tracker_debug.track_states[0].missed_after == 1


def test_kalman_tracker_keeps_predicted_active_track_until_max_missed() -> None:
    tracker = KalmanNNTracker(TrackingConfig(max_dist=2.0, max_missed=2, sticky_max_dist=3.0))
    tracker.step([_detection(0.0, detection_id=10)], 0, 1000)

    first_gap = tracker.step([], 1, 2000)
    second_gap = tracker.step([], 2, 3000)

    assert len(first_gap.active_tracks) == 1
    assert first_gap.active_tracks[0].track_id == 1
    assert first_gap.active_tracks[0].status == "predicted"
    assert len(second_gap.active_tracks) == 1
    assert second_gap.active_tracks[0].track_id == 1
    assert second_gap.active_tracks[0].status == "predicted"
    assert first_gap.active_tracks[0].center.shape == (3,)
    assert second_gap.active_tracks[0].center.shape == (3,)


def test_kalman_tracker_drops_predicted_active_track_after_max_missed() -> None:
    tracker = KalmanNNTracker(TrackingConfig(max_dist=2.0, max_missed=1, sticky_max_dist=3.0))
    tracker.step([_detection(0.0, detection_id=10)], 0, 1000)
    tracker.step([], 1, 2000)

    state = tracker.step([], 2, 3000)
    tracks = tracker.finalize()

    assert state.active_tracks == []
    assert tracks[1].ended_by_missed is True


def test_euclidean_tracker_emits_spawn_debug_state() -> None:
    tracker = EuclideanNNTracker(TrackingConfig(algorithm="euclidean_nn", max_dist=2.0, max_missed=1))

    state = tracker.step([_detection(0.0, detection_id=21)], 0, 1000)

    assert state.tracker_debug is not None
    assert state.tracker_metrics["spawned_count"] == 1
    assert state.tracker_debug.track_states[0].status == "spawned"
    assert state.tracker_debug.track_states[0].track_id == 1
    assert state.tracker_debug.detection_states[0].status == "spawned"
    assert state.tracker_debug.detection_states[0].spawned_track_id == 1


def test_tracker_suppresses_spawn_for_halo_detection_when_flagged() -> None:
    tracker = EuclideanNNTracker(TrackingConfig(algorithm="euclidean_nn", max_dist=2.0, max_missed=1))

    state = tracker.step([_detection(0.0, detection_id=31, tracking_halo_only=True, spawn_suppressed=True)], 0, 1000)
    tracks = tracker.finalize()

    assert state.tracker_debug is not None
    assert state.tracker_metrics["spawn_suppressed_count"] == 1
    assert state.tracker_metrics["halo_detection_count"] == 1
    assert state.tracker_debug.detection_states[0].status == "spawn_suppressed"
    assert state.tracker_debug.detection_states[0].spawn_suppressed is True
    assert state.active_tracks == []
    assert tracks == {}


def test_assignment_hungarian_beats_greedy_on_ambiguous_cost_matrix() -> None:
    cost = np.array(
        [
            [1.9, 3.0, 6.0],
            [0.1, 1.0, 4.0],
            [3.1, 2.0, 1.0],
        ],
        dtype=np.float32,
    )
    valid = np.ones_like(cost, dtype=bool)
    greedy_assignments, _, _ = assign_cost_matrix(cost, valid, method="greedy")
    hungarian_assignments, _, _ = assign_cost_matrix(cost, valid, method="hungarian")
    greedy_total = float(sum(cost[row, col] for row, col in greedy_assignments.items()))
    hungarian_total = float(sum(cost[row, col] for row, col in hungarian_assignments.items()))
    assert hungarian_total < greedy_total


def test_hungarian_kalman_reports_hungarian_assignment() -> None:
    tracker = HungarianKalmanTracker(TrackingConfig(max_dist=3.5, max_missed=1, sticky_max_dist=4.0))
    tracker.step([_detection(0.0), _detection(5.0)], 0, 1000)
    state = tracker.step([_detection(1.9), _detection(6.0)], 1, 2000)
    tracks = tracker.finalize()
    assert state.tracker_metrics["assignment_method"] == "hungarian"
    assert len(tracks) == 2
