from __future__ import annotations

import asyncio
import base64
import json
import threading
import types

import numpy as np
import pytest

from tracking_pipeline.config.models import QB2LiveInputConfig, QB2LiveMQTTConfig
from tracking_pipeline.domain.models import ObjectLabelData, SensorCalibrationData
from tracking_pipeline.infrastructure.readers import qb2_live_reader
from tracking_pipeline.infrastructure.readers.qb2_live_reader import QB2LiveReader, _PendingObjectSnapshot, _RigidTransform


def _live_config(
    *,
    mqtt_drain_tolerance_sec: float = 0.25,
    mqtt_max_pending_age_sec: float = 3.0,
) -> QB2LiveInputConfig:
    return QB2LiveInputConfig(
        sensor_name="class_qb2",
        ip="10.16.3.160",
        api_key="secret",
        mqtt=QB2LiveMQTTConfig(
            host="10.16.3.111",
            port=1883,
            topic="blickfeld/states_160",
            keepalive=60,
        ),
        max_frames=0,
        idle_timeout_sec=0.1,
        mqtt_drain_tolerance_sec=mqtt_drain_tolerance_sec,
        mqtt_max_pending_age_sec=mqtt_max_pending_age_sec,
    )


def _make_zone(uuid: str, *, center: tuple[float, float, float], dimensions: tuple[float, float, float]):
    return types.SimpleNamespace(
        uuid=uuid,
        shape=types.SimpleNamespace(
            pose=types.SimpleNamespace(
                position=types.SimpleNamespace(x=center[0], y=center[1], z=center[2]),
                orientation=types.SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
            box=types.SimpleNamespace(
                dimensions=types.SimpleNamespace(x=dimensions[0], y=dimensions[1], z=dimensions[2])
            ),
        ),
    )


def _make_raw_frame(
    *,
    timestamp_ns: int,
    points: np.ndarray,
    direction_id: np.ndarray,
    point_timestamps_ns: np.ndarray,
    photon_count: np.ndarray | None = None,
):
    binary = types.SimpleNamespace(
        cartesian=np.asarray(points, dtype=np.float32),
        direction_id=np.asarray(direction_id, dtype=np.uint32),
        timestamp=np.asarray(point_timestamps_ns, dtype=np.uint64),
        photon_count=None if photon_count is None else np.asarray(photon_count, dtype=np.uint16),
    )
    return types.SimpleNamespace(timestamp=int(timestamp_ns), binary=binary)


def test_qb2_live_reader_converts_raw_frame_to_frame_data() -> None:
    reader = QB2LiveReader(_live_config(), read_intensity=True)
    calibration = SensorCalibrationData(
        sensor_name="class_qb2",
        vertical_scanlines=4,
        horizontal_scanlines=4,
    )
    vertical_remap = qb2_live_reader._compute_vertical_remap(4)
    transform = _RigidTransform(
        rotation=np.eye(3, dtype=np.float32),
        translation=np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )
    raw_frame = _make_raw_frame(
        timestamp_ns=123,
        points=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        direction_id=np.array([0, 5], dtype=np.uint32),
        point_timestamps_ns=np.array([1000, 1001], dtype=np.uint64),
        photon_count=np.array([65535, 32767], dtype=np.uint16),
    )

    frame = reader._raw_frame_to_frame_data(
        raw_frame,
        calibration,
        4,
        vertical_remap,
        transform,
        7,
        "qb2_live://class_qb2@10.16.3.160",
    )

    assert frame.frame_index == 7
    assert frame.timestamp_ns == 123
    assert frame.source_path == "qb2_live://class_qb2@10.16.3.160"
    assert np.allclose(frame.points, np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32))
    assert np.array_equal(frame.point_timestamp_ns, np.array([1000, 1001], dtype=np.int64))
    assert np.array_equal(frame.scans[0].row_index, vertical_remap[np.array([0, 1], dtype=np.int32)])
    assert np.array_equal(frame.scans[0].col_index, np.array([0, 2], dtype=np.int32))
    assert np.allclose(frame.scans[0].ranges, np.array([1.0, 2.0], dtype=np.float32))
    expected_intensity = np.array([1.0, (32767.0 / 65535.0) * 4.0], dtype=np.float32)
    assert np.allclose(frame.point_intensity, expected_intensity, atol=1e-6)


def test_qb2_live_reader_parses_mqtt_snapshot_and_transforms_points() -> None:
    reader = QB2LiveReader(_live_config())
    calibration = SensorCalibrationData(sensor_name="class_qb2")
    lane_transform_map = {
        "lane-a": (
            np.eye(3, dtype=np.float32),
            np.array([10.0, -3.0, 0.0], dtype=np.float32),
        )
    }
    cartesian_b64 = base64.b64encode(np.asarray([[0.0, 0.0, 0.0], [1.0, 0.5, 0.0]], dtype=np.float32).tobytes()).decode(
        "ascii"
    )
    payload = {
        "states": {
            "timestamp": 555,
            "states": {
                "lane-a": {
                    "trafficLane": {
                        "vehicles": {
                            "7": {
                                "pointCloud": {"binary": {"cartesian": cartesian_b64}},
                                "modelClassification": {"predictedClass": {"best": "car"}},
                            }
                        }
                    }
                }
            },
        }
    }

    snapshot = reader._parse_mqtt_snapshot(
        json.dumps(payload).encode("utf-8"),
        calibration,
        lane_transform_map,
        "qb2_live://class_qb2@10.16.3.160",
    )

    assert snapshot is not None
    assert snapshot.timestamp_ns == 555
    assert len(snapshot.labels) == 1
    label = snapshot.labels[0]
    assert label.object_id == 7
    assert label.obj_class == "car"
    assert label.sensor_name == "class_qb2"
    assert label.source_path == "qb2_live://class_qb2@10.16.3.160"
    assert np.allclose(label.points, np.array([[10.0, -3.0, 0.0], [11.0, -2.5, 0.0]], dtype=np.float32))


def test_qb2_live_reader_drains_only_snapshots_up_to_frame_timestamp() -> None:
    reader = QB2LiveReader(_live_config(mqtt_drain_tolerance_sec=0.0))
    reader._pending_snapshots.extend(
        [
            _PendingObjectSnapshot(
                timestamp_ns=10,
                labels=[
                    ObjectLabelData(
                        object_id=1,
                        timestamp_ns=10,
                        points=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
                        obj_class="car",
                        sensor_name="class_qb2",
                        source_path="qb2_live://class_qb2@10.16.3.160",
                    )
                ],
            ),
            _PendingObjectSnapshot(
                timestamp_ns=20,
                labels=[
                    ObjectLabelData(
                        object_id=2,
                        timestamp_ns=20,
                        points=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
                        obj_class="van",
                        sensor_name="class_qb2",
                        source_path="qb2_live://class_qb2@10.16.3.160",
                    )
                ],
            ),
        ]
    )

    first_labels = reader._drain_object_labels_up_to(15, 3)
    remaining_labels = reader.drain_pending_object_labels(4)

    assert [label.object_id for label in first_labels] == [1]
    assert first_labels[0].frame_index == 3
    assert [label.object_id for label in remaining_labels] == [2]
    assert remaining_labels[0].frame_index == 4


def test_qb2_live_reader_drain_pending_object_labels_respects_max_timestamp() -> None:
    reader = QB2LiveReader(_live_config(mqtt_drain_tolerance_sec=0.0))
    reader._pending_snapshots.extend(
        [
            _PendingObjectSnapshot(
                timestamp_ns=10,
                labels=[
                    ObjectLabelData(
                        object_id=1,
                        timestamp_ns=10,
                        points=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
                        obj_class="car",
                        sensor_name="class_qb2",
                        source_path="qb2_live://class_qb2@10.16.3.160",
                    )
                ],
            ),
            _PendingObjectSnapshot(
                timestamp_ns=20,
                labels=[
                    ObjectLabelData(
                        object_id=2,
                        timestamp_ns=20,
                        points=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
                        obj_class="van",
                        sensor_name="class_qb2",
                        source_path="qb2_live://class_qb2@10.16.3.160",
                    )
                ],
            ),
        ]
    )

    drained_labels = reader.drain_pending_object_labels(4, max_timestamp_ns=15)

    assert [label.object_id for label in drained_labels] == [1]
    assert drained_labels[0].frame_index == 4
    assert len(reader._pending_snapshots) == 1
    assert reader._pending_snapshots[0].labels[0].object_id == 2


def test_qb2_live_reader_keeps_pending_snapshots_sorted_by_timestamp() -> None:
    reader = QB2LiveReader(_live_config(mqtt_drain_tolerance_sec=0.0))
    older = _PendingObjectSnapshot(
        timestamp_ns=10,
        labels=[
            ObjectLabelData(
                object_id=1,
                timestamp_ns=10,
                points=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
                obj_class="car",
                sensor_name="class_qb2",
                source_path="qb2_live://class_qb2@10.16.3.160",
            )
        ],
    )
    newer = _PendingObjectSnapshot(
        timestamp_ns=20,
        labels=[
            ObjectLabelData(
                object_id=2,
                timestamp_ns=20,
                points=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
                obj_class="van",
                sensor_name="class_qb2",
                source_path="qb2_live://class_qb2@10.16.3.160",
            )
        ],
    )

    reader._enqueue_pending_snapshot(newer)
    reader._enqueue_pending_snapshot(older)

    drained_labels = reader._drain_object_labels_up_to(10, 4)

    assert [label.object_id for label in drained_labels] == [1]
    assert len(reader._pending_snapshots) == 1
    assert reader._pending_snapshots[0].labels[0].object_id == 2


def test_qb2_live_reader_drops_pending_labels_older_than_three_seconds() -> None:
    reader = QB2LiveReader(_live_config(mqtt_drain_tolerance_sec=0.0, mqtt_max_pending_age_sec=3.0))
    reader._enqueue_pending_snapshot(
        _PendingObjectSnapshot(
            timestamp_ns=1_000_000_000,
            labels=[
                ObjectLabelData(
                    object_id=1,
                    timestamp_ns=1_000_000_000,
                    points=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
                    obj_class="car",
                    sensor_name="class_qb2",
                    source_path="qb2_live://class_qb2@10.16.3.160",
                )
            ],
            enqueued_monotonic=1.0,
        )
    )
    reader._enqueue_pending_snapshot(
        _PendingObjectSnapshot(
            timestamp_ns=4_500_000_000,
            labels=[
                ObjectLabelData(
                    object_id=2,
                    timestamp_ns=4_500_000_000,
                    points=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
                    obj_class="van",
                    sensor_name="class_qb2",
                    source_path="qb2_live://class_qb2@10.16.3.160",
                )
            ],
            enqueued_monotonic=4.2,
        )
    )

    original_monotonic = qb2_live_reader.time.monotonic
    qb2_live_reader.time.monotonic = lambda: 5.0
    try:
        dropped = reader._drop_expired_pending_snapshots()
        drained_labels = reader._drain_object_labels_up_to(5_000_000_000, 7)
        status = reader.status_snapshot()
    finally:
        qb2_live_reader.time.monotonic = original_monotonic

    assert dropped == 1
    assert [label.object_id for label in drained_labels] == [2]
    assert status["dropped_stale_snapshot_count"] == 1
    assert status["dropped_stale_label_count"] == 1
    assert status["pending_label_count"] == 0


def test_qb2_live_reader_caps_pending_mqtt_labels_at_ten() -> None:
    reader = QB2LiveReader(_live_config(mqtt_drain_tolerance_sec=0.0, mqtt_max_pending_age_sec=30.0))
    original_monotonic = qb2_live_reader.time.monotonic
    qb2_live_reader.time.monotonic = lambda: 10.0
    try:
        reader._enqueue_pending_snapshot(
            _PendingObjectSnapshot(
                timestamp_ns=10,
                labels=[
                    ObjectLabelData(
                        object_id=index,
                        timestamp_ns=10,
                        points=np.array([[float(index), 0.0, 0.0]], dtype=np.float32),
                        obj_class="car",
                        sensor_name="class_qb2",
                        source_path="qb2_live://class_qb2@10.16.3.160",
                    )
                    for index in range(1, 8)
                ],
                enqueued_monotonic=10.0,
            )
        )
        reader._enqueue_pending_snapshot(
            _PendingObjectSnapshot(
                timestamp_ns=20,
                labels=[
                    ObjectLabelData(
                        object_id=index,
                        timestamp_ns=20,
                        points=np.array([[float(index), 0.0, 0.0]], dtype=np.float32),
                        obj_class="van",
                        sensor_name="class_qb2",
                        source_path="qb2_live://class_qb2@10.16.3.160",
                    )
                    for index in range(8, 15)
                ],
                enqueued_monotonic=11.0,
            )
        )
        status = reader.status_snapshot()
        drained_labels = reader.drain_pending_object_labels(99, max_timestamp_ns=99)
    finally:
        qb2_live_reader.time.monotonic = original_monotonic

    assert status["pending_label_count"] == 10
    assert status["dropped_overflow_label_count"] == 4
    assert [label.object_id for label in drained_labels] == list(range(5, 15))


def test_qb2_live_reader_iter_frames_streams_raw_frames_and_attaches_mqtt_objects(monkeypatch) -> None:
    raw_frames = [
        _make_raw_frame(
            timestamp_ns=100,
            points=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float32),
            direction_id=np.array([0, 5], dtype=np.uint32),
            point_timestamps_ns=np.array([90, 91], dtype=np.uint64),
            photon_count=np.array([1000, 2000], dtype=np.uint16),
        ),
        _make_raw_frame(
            timestamp_ns=200,
            points=np.array([[1.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=np.float32),
            direction_id=np.array([1, 4], dtype=np.uint32),
            point_timestamps_ns=np.array([190, 191], dtype=np.uint64),
            photon_count=np.array([3000, 4000], dtype=np.uint16),
        ),
    ]
    mqtt_payload = json.dumps(
        {
            "states": {
                "timestamp": 150,
                "states": {
                    "lane-a": {
                        "trafficLane": {
                            "vehicles": {
                                "7": {
                                    "pointCloud": {
                                        "binary": {
                                            "cartesian": base64.b64encode(
                                                np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32).tobytes()
                                            ).decode("ascii")
                                        }
                                    },
                                    "modelClassification": {"predictedClass": {"best": "car"}},
                                }
                            }
                        }
                    }
                },
            }
        }
    ).encode("utf-8")

    class _FakeTokenFactory:
        def __init__(self, application_key_secret: str):
            self.application_key_secret = application_key_secret

    class _FakeChannel:
        def __init__(self, fqdn_or_ip: str, token: object):
            self.fqdn_or_ip = fqdn_or_ip
            self.token = token

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            _ = exc_type, exc, tb
            return False

    class _FakePointCloudService:
        def __init__(self, channel: object):
            _ = channel

        async def async_stream(self):
            for frame in raw_frames:
                yield types.SimpleNamespace(frame=frame)

    class _FakeScanPatternService:
        def __init__(self, channel: object):
            _ = channel

        def get(self):
            return types.SimpleNamespace(
                scan_pattern=types.SimpleNamespace(
                    vertical=types.SimpleNamespace(field_of_view=np.deg2rad(20.0), scanlines_up=2, scanlines_down=2),
                    horizontal=types.SimpleNamespace(field_of_view=np.deg2rad(40.0)),
                    pulse=types.SimpleNamespace(angle_spacing=np.deg2rad(10.0)),
                    frame_mode=1,
                )
            )

    class _FakeDataSourceService:
        def __init__(self, channel: object):
            _ = channel

        def get(self):
            transform = types.SimpleNamespace(
                translation=types.SimpleNamespace(x=1.0, y=0.0, z=0.0),
                rotation=types.SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            )
            lidar_cfg = types.SimpleNamespace(disabled=False, map_from_lidar=transform)
            return types.SimpleNamespace(data_source=types.SimpleNamespace(qb2_setup=types.SimpleNamespace(lidars=[lidar_cfg])))

    class _FakeZoneService:
        def __init__(self, channel: object):
            _ = channel

        def list(self):
            return types.SimpleNamespace(zones=[_make_zone("lane-a", center=(10.0, 0.0, 0.0), dimensions=(2.0, 6.0, 1.0))])

    class _FakeMQTTMessage:
        def __init__(self, payload: bytes):
            self.payload = payload

    class _FakeMQTTClient:
        pending_payloads = [mqtt_payload]
        instances: list["_FakeMQTTClient"] = []

        def __init__(self, protocol=None):
            self.protocol = protocol
            self.on_connect = None
            self.on_message = None
            self.connected = None
            self.subscriptions: list[str] = []
            self.loop_started = False
            self.loop_stopped = False
            self.disconnected = False
            _FakeMQTTClient.instances.append(self)

        def connect(self, host: str, port: int, keepalive: int) -> None:
            self.connected = (host, port, keepalive)

        def subscribe(self, topic: str) -> None:
            self.subscriptions.append(topic)

        def loop_start(self) -> None:
            self.loop_started = True
            if callable(self.on_connect):
                self.on_connect(self, None, None, 0, None)
            while _FakeMQTTClient.pending_payloads:
                payload = _FakeMQTTClient.pending_payloads.pop(0)
                if callable(self.on_message):
                    self.on_message(self, None, _FakeMQTTMessage(payload))

        def loop_stop(self) -> None:
            self.loop_stopped = True

        def disconnect(self) -> None:
            self.disconnected = True

    def _import_module(name: str):
        if name == "blickfeld_qb2":
            return types.SimpleNamespace(Channel=_FakeChannel, TokenFactory=_FakeTokenFactory)
        if name == "blickfeld_qb2.core_processing.services":
            return types.SimpleNamespace(PointCloud=_FakePointCloudService)
        if name == "blickfeld_qb2.system.services":
            return types.SimpleNamespace(ScanPattern=_FakeScanPatternService)
        if name == "blickfeld_qb2.percept_pipeline.services":
            return types.SimpleNamespace(DataSource=_FakeDataSourceService, Zone=_FakeZoneService)
        if name == "paho.mqtt.client":
            return types.SimpleNamespace(Client=_FakeMQTTClient, MQTTv5=5)
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(qb2_live_reader.importlib, "import_module", _import_module)
    reader = QB2LiveReader(_live_config(mqtt_drain_tolerance_sec=0.0), read_intensity=True)

    frames = list(reader.iter_frames(["qb2_live://class_qb2@10.16.3.160"]))

    assert len(frames) == 2
    assert frames[0].frame_index == 0
    assert frames[1].frame_index == 1
    assert frames[0].object_labels == []
    assert [label.object_id for label in frames[1].object_labels] == [7]
    assert frames[1].object_labels[0].frame_index == 1
    assert frames[1].object_labels[0].obj_class == "car"
    assert np.allclose(frames[0].points, np.array([[1.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=np.float32))
    assert np.allclose(frames[1].object_labels[0].points, np.array([[10.0, -3.0, 0.0]], dtype=np.float32))
    assert _FakeMQTTClient.instances
    assert _FakeMQTTClient.instances[0].connected == ("10.16.3.111", 1883, 60)
    assert "blickfeld/states_160" in _FakeMQTTClient.instances[0].subscriptions
    assert _FakeMQTTClient.instances[0].loop_started is True
    assert _FakeMQTTClient.instances[0].loop_stopped is True
    assert _FakeMQTTClient.instances[0].disconnected is True


def test_qb2_live_reader_streams_raw_frames_on_iterator_thread(monkeypatch) -> None:
    observed_thread_ids: list[int] = []
    raw_frames = [
        _make_raw_frame(
            timestamp_ns=100,
            points=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            direction_id=np.array([0], dtype=np.uint32),
            point_timestamps_ns=np.array([90], dtype=np.uint64),
        )
    ]

    class _FakeTokenFactory:
        def __init__(self, application_key_secret: str):
            self.application_key_secret = application_key_secret

    class _FakeChannel:
        def __init__(self, fqdn_or_ip: str, token: object):
            self.fqdn_or_ip = fqdn_or_ip
            self.token = token

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            _ = exc_type, exc, tb
            return False

    class _FakePointCloudService:
        def __init__(self, channel: object):
            _ = channel

        async def async_stream(self):
            observed_thread_ids.append(threading.get_ident())
            for frame in raw_frames:
                yield types.SimpleNamespace(frame=frame)

    class _FakeScanPatternService:
        def __init__(self, channel: object):
            _ = channel

        def get(self):
            return types.SimpleNamespace(
                scan_pattern=types.SimpleNamespace(
                    vertical=types.SimpleNamespace(field_of_view=np.deg2rad(20.0), scanlines_up=2, scanlines_down=2),
                    horizontal=types.SimpleNamespace(field_of_view=np.deg2rad(40.0)),
                    pulse=types.SimpleNamespace(angle_spacing=np.deg2rad(10.0)),
                    frame_mode=1,
                )
            )

    class _FakeDataSourceService:
        def __init__(self, channel: object):
            _ = channel

        def get(self):
            transform = types.SimpleNamespace(
                translation=types.SimpleNamespace(x=0.0, y=0.0, z=0.0),
                rotation=types.SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            )
            lidar_cfg = types.SimpleNamespace(disabled=False, map_from_lidar=transform)
            return types.SimpleNamespace(data_source=types.SimpleNamespace(qb2_setup=types.SimpleNamespace(lidars=[lidar_cfg])))

    class _FakeZoneService:
        def __init__(self, channel: object):
            _ = channel

        def list(self):
            return types.SimpleNamespace(zones=[_make_zone("lane-a", center=(10.0, 0.0, 0.0), dimensions=(2.0, 6.0, 1.0))])

    class _FakeMQTTClient:
        def __init__(self, protocol=None):
            self.protocol = protocol
            self.on_connect = None
            self.on_message = None

        def connect(self, host: str, port: int, keepalive: int) -> None:
            _ = host, port, keepalive

        def subscribe(self, topic: str) -> None:
            _ = topic

        def loop_start(self) -> None:
            return None

        def loop_stop(self) -> None:
            return None

        def disconnect(self) -> None:
            return None

    def _import_module(name: str):
        if name == "blickfeld_qb2":
            return types.SimpleNamespace(Channel=_FakeChannel, TokenFactory=_FakeTokenFactory)
        if name == "blickfeld_qb2.core_processing.services":
            return types.SimpleNamespace(PointCloud=_FakePointCloudService)
        if name == "blickfeld_qb2.system.services":
            return types.SimpleNamespace(ScanPattern=_FakeScanPatternService)
        if name == "blickfeld_qb2.percept_pipeline.services":
            return types.SimpleNamespace(DataSource=_FakeDataSourceService, Zone=_FakeZoneService)
        if name == "paho.mqtt.client":
            return types.SimpleNamespace(Client=_FakeMQTTClient, MQTTv5=5)
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(qb2_live_reader.importlib, "import_module", _import_module)
    reader = QB2LiveReader(_live_config())
    iterator_thread_id = threading.get_ident()

    frames = list(reader.iter_frames(["qb2_live://class_qb2@10.16.3.160"]))

    assert len(frames) == 1
    assert observed_thread_ids == [iterator_thread_id]
    assert reader._background_error is None


def test_qb2_live_reader_maps_stream_timeout_to_runtime_error(monkeypatch) -> None:
    class _FakeTokenFactory:
        def __init__(self, application_key_secret: str):
            self.application_key_secret = application_key_secret

    class _FakeChannel:
        def __init__(self, fqdn_or_ip: str, token: object):
            self.fqdn_or_ip = fqdn_or_ip
            self.token = token

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            _ = exc_type, exc, tb
            return False

    class _FakePointCloudService:
        def __init__(self, channel: object):
            _ = channel

        async def async_stream(self):
            yield types.SimpleNamespace(
                frame=_make_raw_frame(
                    timestamp_ns=100,
                    points=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
                    direction_id=np.array([0], dtype=np.uint32),
                    point_timestamps_ns=np.array([90], dtype=np.uint64),
                )
            )
            await asyncio.sleep(3600)

    class _FakeScanPatternService:
        def __init__(self, channel: object):
            _ = channel

        def get(self):
            return types.SimpleNamespace(
                scan_pattern=types.SimpleNamespace(
                    vertical=types.SimpleNamespace(field_of_view=np.deg2rad(20.0), scanlines_up=2, scanlines_down=2),
                    horizontal=types.SimpleNamespace(field_of_view=np.deg2rad(40.0)),
                    pulse=types.SimpleNamespace(angle_spacing=np.deg2rad(10.0)),
                    frame_mode=1,
                )
            )

    class _FakeDataSourceService:
        def __init__(self, channel: object):
            _ = channel

        def get(self):
            transform = types.SimpleNamespace(
                translation=types.SimpleNamespace(x=0.0, y=0.0, z=0.0),
                rotation=types.SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            )
            lidar_cfg = types.SimpleNamespace(disabled=False, map_from_lidar=transform)
            return types.SimpleNamespace(data_source=types.SimpleNamespace(qb2_setup=types.SimpleNamespace(lidars=[lidar_cfg])))

    class _FakeZoneService:
        def __init__(self, channel: object):
            _ = channel

        def list(self):
            return types.SimpleNamespace(zones=[_make_zone("lane-a", center=(10.0, 0.0, 0.0), dimensions=(2.0, 6.0, 1.0))])

    class _FakeMQTTClient:
        def __init__(self, protocol=None):
            self.protocol = protocol
            self.on_connect = None
            self.on_message = None

        def connect(self, host: str, port: int, keepalive: int) -> None:
            _ = host, port, keepalive

        def subscribe(self, topic: str) -> None:
            _ = topic

        def loop_start(self) -> None:
            return None

        def loop_stop(self) -> None:
            return None

        def disconnect(self) -> None:
            return None

    def _import_module(name: str):
        if name == "blickfeld_qb2":
            return types.SimpleNamespace(Channel=_FakeChannel, TokenFactory=_FakeTokenFactory)
        if name == "blickfeld_qb2.core_processing.services":
            return types.SimpleNamespace(PointCloud=_FakePointCloudService)
        if name == "blickfeld_qb2.system.services":
            return types.SimpleNamespace(ScanPattern=_FakeScanPatternService)
        if name == "blickfeld_qb2.percept_pipeline.services":
            return types.SimpleNamespace(DataSource=_FakeDataSourceService, Zone=_FakeZoneService)
        if name == "paho.mqtt.client":
            return types.SimpleNamespace(Client=_FakeMQTTClient, MQTTv5=5)
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(qb2_live_reader.importlib, "import_module", _import_module)
    reader = QB2LiveReader(_live_config())

    frame_iterator = reader.iter_frames(["qb2_live://class_qb2@10.16.3.160"])
    first_frame = next(frame_iterator)

    assert first_frame.frame_index == 0
    with pytest.raises(RuntimeError, match="No QB2 frames received within idle_timeout_sec=0.100"):
        next(frame_iterator)
