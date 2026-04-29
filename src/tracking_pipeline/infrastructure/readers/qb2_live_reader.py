from __future__ import annotations

import asyncio
import base64
import importlib
import json
import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from tracking_pipeline.config.models import QB2LiveInputConfig
from tracking_pipeline.domain.models import FrameData, LidarScanData, ObjectLabelData, SensorCalibrationData

LOGGER = logging.getLogger(__name__)
_END_OF_STREAM = object()


@dataclass(slots=True)
class _RigidTransform:
    rotation: np.ndarray
    translation: np.ndarray


@dataclass(slots=True)
class _PendingObjectSnapshot:
    timestamp_ns: int
    labels: list[ObjectLabelData]
    enqueued_monotonic: float = 0.0


def _rad2deg(radians: float) -> float:
    return float(radians) * 180.0 / math.pi


def _quat_to_rotmat(x: float, y: float, z: float, w: float) -> np.ndarray:
    q = np.array([x, y, z, w], dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm <= 0.0:
        return np.eye(3, dtype=np.float32)
    x, y, z, w = q / norm
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def _transform_points_xyz(xyz: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    return np.asarray(xyz, dtype=np.float32) @ np.asarray(rotation, dtype=np.float32).T + np.asarray(
        translation,
        dtype=np.float32,
    )[None, :]


def _compute_lane_transform_from_zone(zone: object) -> tuple[np.ndarray, np.ndarray]:
    pose = zone.shape.pose
    box = zone.shape.box
    center = np.array(
        [pose.position.x, pose.position.y, pose.position.z],
        dtype=np.float32,
    )
    extent = np.array(
        [box.dimensions.x, box.dimensions.y, box.dimensions.z],
        dtype=np.float32,
    )
    rotation = _quat_to_rotmat(
        getattr(pose.orientation, "x", 0.0),
        getattr(pose.orientation, "y", 0.0),
        getattr(pose.orientation, "z", 0.0),
        getattr(pose.orientation, "w", 1.0),
    )
    lane_front_local = np.array([0.0, -(extent[1] / 2.0), 0.0], dtype=np.float32)
    lane_front_origin = center + rotation @ lane_front_local
    lane_front_origin[2] = 0.0
    return rotation, lane_front_origin.astype(np.float32, copy=False)


def _transform_points_lane_to_world(points_lane: np.ndarray, rotation: np.ndarray, lane_front_origin: np.ndarray) -> np.ndarray:
    if points_lane.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if points_lane.ndim != 2 or points_lane.shape[1] != 3:
        points_lane = np.asarray(points_lane, dtype=np.float32).reshape(-1, 3)
    return np.asarray(points_lane, dtype=np.float32) @ np.asarray(rotation, dtype=np.float32).T + np.asarray(
        lane_front_origin,
        dtype=np.float32,
    )[None, :]


def _compute_horizontal_ids(direction_id: np.ndarray, horiz_bins: int) -> np.ndarray:
    width = max(1, int(horiz_bins))
    base = np.asarray(direction_id, dtype=np.int64) % (2 * width)
    horizontal = np.where(base < width, base, 2 * width - 1 - base)
    return horizontal.astype(np.int32, copy=False)


def _compute_vertical_remap(num_vertical: int) -> np.ndarray:
    count = max(1, int(num_vertical))
    if count == 1:
        return np.zeros((1,), dtype=np.int32)
    indices = np.arange(count, dtype=np.int32)
    physical = np.zeros((count,), dtype=np.int32)
    mask_positive = (indices > 0) & (indices % 2 == 1)
    physical[mask_positive] = (indices[mask_positive] + 1) // 2
    mask_negative = (indices > 0) & (indices % 2 == 0)
    physical[mask_negative] = -(indices[mask_negative] // 2)
    order = np.argsort(physical)
    remap = np.empty((count,), dtype=np.int32)
    remap[order] = np.arange(count, dtype=np.int32)
    return remap


def build_qb2_live_source_path(sensor_name: str, ip: str) -> str:
    return f"qb2_live://{sensor_name}@{ip}"


class QB2LiveReader:
    def __init__(self, config: QB2LiveInputConfig, read_intensity: bool = False):
        self.config = config
        self.read_intensity = bool(read_intensity)
        self._reset_runtime_state()

    def _reset_runtime_state(self) -> None:
        self._stop_event = threading.Event()
        self._pending_snapshots: deque[_PendingObjectSnapshot] = deque()
        self._pending_lock = threading.Lock()
        self._status_lock = threading.Lock()
        self._mqtt_client = None
        self._background_error: BaseException | None = None
        self._source_path = build_qb2_live_source_path(self.config.sensor_name, self.config.ip)
        self._status = {
            "reader_state": "idle",
            "source_path": self._source_path,
            "sensor_name": self.config.sensor_name,
            "ip": self.config.ip,
            "mqtt_host": self.config.mqtt.host,
            "mqtt_port": int(self.config.mqtt.port),
            "mqtt_topic": self.config.mqtt.topic,
            "mqtt_connected": False,
            "mqtt_messages_received": 0,
            "mqtt_snapshots_received": 0,
            "mqtt_labels_received": 0,
            "pending_snapshot_count": 0,
            "pending_label_count": 0,
            "dropped_stale_snapshot_count": 0,
            "dropped_stale_label_count": 0,
            "dropped_overflow_snapshot_count": 0,
            "dropped_overflow_label_count": 0,
            "raw_frames_received": 0,
            "raw_points_received": 0,
            "raw_stream_reconnect_count": 0,
            "last_raw_stream_error": None,
            "last_raw_stream_reconnect_unix_sec": None,
            "yielded_frame_count": 0,
            "last_raw_frame_index": -1,
            "last_raw_frame_timestamp_ns": None,
            "last_raw_point_count": 0,
            "last_mqtt_timestamp_ns": None,
            "waiting_for_first_raw_frame": True,
            "stop_requested": False,
            "background_error": None,
            "_last_raw_monotonic": None,
            "_last_mqtt_monotonic": None,
        }

    def close(self) -> None:
        self._stop_event.set()
        self._update_status(stop_requested=True)
        if self._mqtt_client is not None:
            try:
                self._mqtt_client.loop_stop()
            except Exception:
                pass
            try:
                self._mqtt_client.disconnect()
            except Exception:
                pass
            self._mqtt_client = None
        self._update_status(mqtt_connected=False, reader_state="stopped")

    def status_snapshot(self) -> dict[str, Any]:
        self._drop_expired_pending_snapshots()
        now = time.monotonic()
        with self._status_lock:
            snapshot = dict(self._status)
        last_raw_monotonic = snapshot.pop("_last_raw_monotonic", None)
        last_mqtt_monotonic = snapshot.pop("_last_mqtt_monotonic", None)
        snapshot["last_raw_age_sec"] = None if last_raw_monotonic is None else max(0.0, now - float(last_raw_monotonic))
        snapshot["last_mqtt_age_sec"] = (
            None if last_mqtt_monotonic is None else max(0.0, now - float(last_mqtt_monotonic))
        )
        snapshot["pending_data"] = bool(
            int(snapshot.get("pending_snapshot_count", 0)) > 0 or int(snapshot.get("pending_label_count", 0)) > 0
        )
        return snapshot

    def drain_pending_object_labels(
        self,
        frame_index: int,
        max_timestamp_ns: int | None = None,
    ) -> list[ObjectLabelData]:
        with self._pending_lock:
            if max_timestamp_ns is None:
                snapshots = list(self._pending_snapshots)
                self._pending_snapshots.clear()
            else:
                snapshots: list[_PendingObjectSnapshot] = []
                remaining: deque[_PendingObjectSnapshot] = deque()
                max_timestamp_ns = int(max_timestamp_ns)
                while self._pending_snapshots:
                    snapshot = self._pending_snapshots.popleft()
                    if int(snapshot.timestamp_ns) <= max_timestamp_ns:
                        snapshots.append(snapshot)
                    else:
                        remaining.append(snapshot)
                self._pending_snapshots = remaining
        self._sync_pending_status()
        return self._materialize_object_labels(snapshots, int(max(frame_index, 0)))

    def snapshot_pending_object_labels(self, frame_index: int) -> list[ObjectLabelData]:
        with self._pending_lock:
            snapshots = list(self._pending_snapshots)
        return self._materialize_latest_object_labels(snapshots, int(max(frame_index, 0)))

    def iter_frames(self, input_paths: list[str]):
        self.close()
        self._reset_runtime_state()
        if input_paths:
            self._source_path = str(input_paths[0])
        self._update_status(source_path=self._source_path, reader_state="initializing")

        modules = self._import_runtime_modules()
        token = modules["TokenFactory"](application_key_secret=self.config.api_key)
        channel_cls = modules["Channel"]
        loop, previous_loop = self._install_sync_event_loop()
        stream_iterator = None
        raw_point_count_total = 0
        try:
            self._update_status(reader_state="connecting_qb2")
            with channel_cls(fqdn_or_ip=self.config.ip, token=token) as channel:
                self._update_status(reader_state="loading_metadata")
                calibration, horiz_bins, vertical_remap = self._load_scan_pattern(modules["ScanPatternService"], channel)
                transform = self._load_sensor_transform(modules["DataSourceService"], channel)
                lane_transform_map = self._load_lane_transform_map(modules["ZoneService"], channel)
                self._update_status(reader_state="starting_mqtt")
                self._start_mqtt_client(modules["mqtt"], calibration, lane_transform_map)
                point_cloud_service = modules["QB2PointCloudService"](channel)

                frame_index = 0
                while True:
                    self._raise_background_error_if_any()
                    if self._stop_event.is_set():
                        break
                    if stream_iterator is None:
                        stream_iterator = point_cloud_service.async_stream().__aiter__()
                        self._update_status(
                            reader_state="waiting_for_raw" if frame_index == 0 else "reconnecting_raw",
                            waiting_for_first_raw_frame=frame_index == 0,
                        )
                    try:
                        timeout = float(self.config.idle_timeout_sec)
                        payload = loop.run_until_complete(self._next_stream_payload(stream_iterator, timeout))
                    except StopAsyncIteration:
                        break
                    except asyncio.TimeoutError as exc:
                        self._raise_background_error_if_any()
                        message = (
                            f"No QB2 frames received within idle_timeout_sec="
                            f"{float(self.config.idle_timeout_sec):.3f}"
                        )
                        if frame_index > 0:
                            self._record_raw_stream_reconnect(message, frame_index)
                            LOGGER.warning("QB2 raw stream timed out (%s); reconnecting raw stream", message)
                            self._close_async_stream(loop, stream_iterator)
                            stream_iterator = None
                            if self._stop_event.wait(self._raw_stream_reconnect_backoff_sec()):
                                break
                            continue
                        self._update_status(
                            reader_state="idle_timeout",
                            background_error=message,
                        )
                        raise RuntimeError(message) from exc
                    except Exception as exc:
                        self._raise_background_error_if_any()
                        if self._is_recoverable_raw_stream_error(exc):
                            self._record_raw_stream_reconnect(str(exc), frame_index)
                            LOGGER.warning("QB2 raw stream closed (%s); reconnecting raw stream", exc)
                            self._close_async_stream(loop, stream_iterator)
                            stream_iterator = None
                            if self._stop_event.wait(self._raw_stream_reconnect_backoff_sec()):
                                break
                            continue
                        self._update_status(reader_state="error", background_error=str(exc))
                        raise exc
                    self._raise_background_error_if_any()
                    frame = self._raw_frame_to_frame_data(
                        getattr(payload, "frame", payload),
                        calibration,
                        horiz_bins,
                        vertical_remap,
                        transform,
                        frame_index,
                        self._source_path,
                    )
                    stale_labels_dropped = self._drop_expired_pending_snapshots()
                    frame.object_labels = self._drain_object_labels_up_to(frame.timestamp_ns, frame.frame_index)
                    raw_point_count_total += int(len(frame.points))
                    self._update_status(
                        reader_state="streaming",
                        waiting_for_first_raw_frame=False,
                        raw_frames_received=int(frame_index + 1),
                        yielded_frame_count=int(frame_index + 1),
                        raw_points_received=int(raw_point_count_total),
                        last_raw_frame_index=int(frame.frame_index),
                        last_raw_frame_timestamp_ns=int(frame.timestamp_ns),
                        last_raw_point_count=int(len(frame.points)),
                        last_stale_labels_dropped=int(stale_labels_dropped),
                        _last_raw_monotonic=time.monotonic(),
                    )
                    yield frame
                    frame_index += 1
                    if int(self.config.max_frames) > 0 and frame_index >= int(self.config.max_frames):
                        break
        finally:
            self._close_async_stream(loop, stream_iterator)
            self._restore_sync_event_loop(loop, previous_loop)
            self.close()

    def _import_runtime_modules(self) -> dict[str, Any]:
        try:
            qb2_module = importlib.import_module("blickfeld_qb2")
            mqtt_module = importlib.import_module("paho.mqtt.client")
            point_cloud_module = importlib.import_module("blickfeld_qb2.core_processing.services")
            system_module = importlib.import_module("blickfeld_qb2.system.services")
            percept_module = importlib.import_module("blickfeld_qb2.percept_pipeline.services")
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "QB2 live input requires the optional 'live' dependencies: pip install -e '.[live]'"
            ) from exc
        return {
            "mqtt": mqtt_module,
            "Channel": getattr(qb2_module, "Channel"),
            "TokenFactory": getattr(qb2_module, "TokenFactory"),
            "QB2PointCloudService": getattr(point_cloud_module, "PointCloud"),
            "ScanPatternService": getattr(system_module, "ScanPattern"),
            "DataSourceService": getattr(percept_module, "DataSource"),
            "ZoneService": getattr(percept_module, "Zone"),
        }

    def _load_scan_pattern(self, service_cls, channel: object) -> tuple[SensorCalibrationData, int, np.ndarray]:
        scan_pattern = service_cls(channel).get().scan_pattern
        vertical_fov = _rad2deg(scan_pattern.vertical.field_of_view)
        horizontal_fov = _rad2deg(scan_pattern.horizontal.field_of_view)
        frame_mode = int(scan_pattern.frame_mode)
        scanlines_up = int(scan_pattern.vertical.scanlines_up)
        scanlines_down = int(scan_pattern.vertical.scanlines_down)
        if frame_mode == 1:
            vertical_scanlines = scanlines_up + scanlines_down
        elif frame_mode in {2, 3}:
            vertical_scanlines = scanlines_up
        else:
            vertical_scanlines = max(1, scanlines_up + scanlines_down)
        horizontal_spacing = _rad2deg(scan_pattern.pulse.angle_spacing)
        horiz_bins = max(
            1,
            int(round(float(horizontal_fov) / max(float(horizontal_spacing), 1e-6))),
        )
        calibration = SensorCalibrationData(
            sensor_name=self.config.sensor_name,
            vertical_fov=float(vertical_fov),
            horizontal_fov=float(horizontal_fov),
            vertical_scanlines=int(vertical_scanlines),
            horizontal_scanlines=int(horiz_bins),
            horizontal_angle_spacing=float(horizontal_spacing),
            frame_mode=str(frame_mode),
        )
        return calibration, horiz_bins, _compute_vertical_remap(vertical_scanlines)

    def _load_sensor_transform(self, service_cls, channel: object) -> _RigidTransform:
        data_source = service_cls(channel).get().data_source
        lidar_cfg = data_source.qb2_setup.lidars[0]
        if bool(getattr(lidar_cfg, "disabled", False)):
            return _RigidTransform(rotation=np.eye(3, dtype=np.float32), translation=np.zeros((3,), dtype=np.float32))
        transform = lidar_cfg.map_from_lidar
        translation = np.array(
            [transform.translation.x, transform.translation.y, transform.translation.z],
            dtype=np.float32,
        )
        rotation = _quat_to_rotmat(
            transform.rotation.x,
            transform.rotation.y,
            transform.rotation.z,
            transform.rotation.w,
        )
        return _RigidTransform(rotation=rotation, translation=translation)

    def _load_lane_transform_map(self, service_cls, channel: object) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        zones = list(service_cls(channel).list().zones)
        if not zones:
            raise RuntimeError("QB2 live input requires at least one zone to derive lane transforms")
        mapping: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for zone in zones:
            mapping[str(zone.uuid)] = _compute_lane_transform_from_zone(zone)
        return mapping

    def _start_mqtt_client(
        self,
        mqtt_module: Any,
        calibration: SensorCalibrationData,
        lane_transform_map: dict[str, tuple[np.ndarray, np.ndarray]],
    ) -> None:
        protocol = getattr(mqtt_module, "MQTTv5", None)
        client = mqtt_module.Client() if protocol is None else mqtt_module.Client(protocol=protocol)

        def _on_connect(client, userdata, flags, reason_code, properties=None):
            _ = userdata, flags, reason_code, properties
            self._update_status(mqtt_connected=True)
            client.subscribe(self.config.mqtt.topic)

        def _on_disconnect(client, userdata, disconnect_flags, reason_code, properties=None):
            _ = client, userdata, disconnect_flags, reason_code, properties
            self._update_status(mqtt_connected=False)

        def _on_message(client, userdata, message):
            _ = client, userdata
            self._increment_status_counter("mqtt_messages_received", 1)
            try:
                snapshot = self._parse_mqtt_snapshot(message.payload, calibration, lane_transform_map, self._source_path)
            except Exception as exc:  # pragma: no cover - exercised through mocked failure paths
                LOGGER.exception("Failed to parse QB2 MQTT payload")
                with self._pending_lock:
                    self._pending_snapshots.clear()
                self._sync_pending_status()
                self._set_background_error(exc)
                return
            if snapshot is None:
                return
            snapshot.enqueued_monotonic = float(time.monotonic())
            self._enqueue_pending_snapshot(snapshot)
            self._drop_expired_pending_snapshots()
            self._increment_status_counter("mqtt_snapshots_received", 1)
            self._increment_status_counter("mqtt_labels_received", len(snapshot.labels))
            self._update_status(
                last_mqtt_timestamp_ns=int(snapshot.timestamp_ns),
                _last_mqtt_monotonic=time.monotonic(),
            )

        client.on_connect = _on_connect
        if hasattr(client, "on_disconnect"):
            client.on_disconnect = _on_disconnect
        client.on_message = _on_message
        client.connect(self.config.mqtt.host, int(self.config.mqtt.port), keepalive=int(self.config.mqtt.keepalive))
        client.loop_start()
        self._mqtt_client = client

    def _raw_frame_to_frame_data(
        self,
        raw_frame: object,
        calibration: SensorCalibrationData,
        horiz_bins: int,
        vertical_remap: np.ndarray,
        transform: _RigidTransform,
        frame_index: int,
        source_path: str,
    ) -> FrameData:
        binary = raw_frame.binary
        xyz_sensor = np.asarray(binary.cartesian, dtype=np.float32)
        if xyz_sensor.ndim != 2 or xyz_sensor.shape[1] != 3:
            xyz_sensor = xyz_sensor.reshape(-1, 3)
        xyz_world = _transform_points_xyz(xyz_sensor, transform.rotation, transform.translation)
        finite_mask = np.isfinite(xyz_world).all(axis=1)
        xyz_world = np.asarray(xyz_world[finite_mask], dtype=np.float32)
        ranges = np.linalg.norm(xyz_world, axis=1).astype(np.float32)

        point_timestamp_ns = self._filter_optional_array(getattr(binary, "timestamp", None), finite_mask, np.int64)
        direction_id = self._filter_optional_array(getattr(binary, "direction_id", None), finite_mask, np.int64)
        if direction_id is None:
            direction_id = np.arange(len(xyz_world), dtype=np.int64)
        raw_row_index = np.asarray(direction_id // max(1, int(horiz_bins)), dtype=np.int32)
        raw_row_index = np.clip(raw_row_index, 0, len(vertical_remap) - 1)
        row_index = vertical_remap[raw_row_index].astype(np.int32, copy=False)
        col_index = _compute_horizontal_ids(direction_id, horiz_bins).astype(np.int32, copy=False)

        intensity = None
        if self.read_intensity:
            photon_count = self._filter_optional_array(getattr(binary, "photon_count", None), finite_mask, np.float32)
            if photon_count is not None:
                normalized = np.clip(photon_count / 65535.0, 0.0, 1.0).astype(np.float32, copy=False)
                intensity = (normalized * np.square(ranges)).astype(np.float32, copy=False)

        timestamp_ns = int(getattr(raw_frame, "timestamp", 0))
        scan = LidarScanData(
            sensor_name=calibration.sensor_name,
            timestamp_ns=timestamp_ns,
            xyz=xyz_world,
            intensity=intensity,
            point_timestamp_ns=point_timestamp_ns,
            ranges=ranges,
            row_index=row_index,
            col_index=col_index,
            calibration=calibration,
        )
        return FrameData(
            frame_index=int(frame_index),
            timestamp_ns=timestamp_ns,
            points=xyz_world,
            point_intensity=intensity,
            point_timestamp_ns=point_timestamp_ns,
            source_path=source_path,
            source_frame_index=int(frame_index),
            source_sequence_index=0,
            object_labels=[],
            scans=[scan],
        )

    def _parse_mqtt_snapshot(
        self,
        payload: bytes,
        calibration: SensorCalibrationData,
        lane_transform_map: dict[str, tuple[np.ndarray, np.ndarray]],
        source_path: str,
    ) -> _PendingObjectSnapshot | None:
        _ = calibration
        decoded = json.loads(payload.decode("utf-8"))
        payload_json = decoded.get("states", {}) if isinstance(decoded, dict) else {}
        states = payload_json.get("states", {}) if isinstance(payload_json, dict) else {}
        timestamp_ns = int(payload_json.get("timestamp", 0) or 0)
        labels: list[ObjectLabelData] = []
        for state_id, state_data in dict(states or {}).items():
            if str(state_id) not in lane_transform_map:
                raise RuntimeError(f"Unknown lane/state_id received from MQTT: {state_id}")
            rotation, lane_front_origin = lane_transform_map[str(state_id)]
            vehicles = (((state_data or {}).get("trafficLane", {}) or {}).get("vehicles", {}) or {})
            for vehicle_id, vehicle_data in dict(vehicles).items():
                labels.append(
                    self._vehicle_to_object_label(
                        vehicle_id=str(vehicle_id),
                        vehicle_data=vehicle_data,
                        timestamp_ns=timestamp_ns,
                        rotation=rotation,
                        lane_front_origin=lane_front_origin,
                        source_path=source_path,
                    )
                )
        if not labels:
            return None
        return _PendingObjectSnapshot(timestamp_ns=timestamp_ns, labels=labels)

    def _vehicle_to_object_label(
        self,
        vehicle_id: str,
        vehicle_data: dict[str, Any],
        timestamp_ns: int,
        rotation: np.ndarray,
        lane_front_origin: np.ndarray,
        source_path: str,
    ) -> ObjectLabelData:
        point_cloud = (((vehicle_data or {}).get("pointCloud", {}) or {}).get("binary", {}) or {})
        cartesian_b64 = str(point_cloud.get("cartesian", "") or "")
        if cartesian_b64:
            cartesian = np.frombuffer(base64.b64decode(cartesian_b64), dtype=np.float32)
            if cartesian.size % 3 != 0:
                raise RuntimeError(f"Invalid QB2 MQTT object point cloud length for object_id={vehicle_id}")
            points_lane = cartesian.reshape(-1, 3)
            points_world = _transform_points_lane_to_world(points_lane, rotation, lane_front_origin)
        else:
            points_world = np.zeros((0, 3), dtype=np.float32)

        classification = ((vehicle_data or {}).get("modelClassification", {}) or {}).get("predictedClass", {})
        if isinstance(classification, dict):
            object_class = str(next(iter(classification.values()), "") or "")
        else:
            object_class = str(classification or "")
        return ObjectLabelData(
            object_id=int(vehicle_id),
            timestamp_ns=int(timestamp_ns),
            points=np.asarray(points_world, dtype=np.float32),
            obj_class=object_class,
            obj_class_score=0.0,
            sensor_name=self.config.sensor_name,
            frame_index=-1,
            source_path=source_path,
        )

    def _drain_object_labels_up_to(self, timestamp_ns: int, frame_index: int) -> list[ObjectLabelData]:
        effective_timestamp_ns = int(timestamp_ns) + self._mqtt_drain_tolerance_ns()
        with self._pending_lock:
            snapshots: list[_PendingObjectSnapshot] = []
            while self._pending_snapshots and int(self._pending_snapshots[0].timestamp_ns) <= int(effective_timestamp_ns):
                snapshots.append(self._pending_snapshots.popleft())
        self._sync_pending_status()
        return self._materialize_object_labels(snapshots, frame_index)

    def _enqueue_pending_snapshot(self, snapshot: _PendingObjectSnapshot) -> None:
        dropped_snapshot_count = 0
        dropped_label_count = 0
        max_pending_labels = max(1, int(self.config.mqtt_max_pending_labels))
        with self._pending_lock:
            pending = list(self._pending_snapshots)
            pending.append(snapshot)
            pending.sort(key=lambda item: int(item.timestamp_ns))
            kept_label_count = 0
            kept_pending_reversed: list[_PendingObjectSnapshot] = []
            for item in reversed(pending):
                label_count = len(item.labels)
                if kept_label_count >= max_pending_labels:
                    dropped_snapshot_count += 1
                    dropped_label_count += label_count
                    continue
                remaining_capacity = max_pending_labels - kept_label_count
                if label_count <= remaining_capacity:
                    kept_pending_reversed.append(item)
                    kept_label_count += label_count
                    continue
                kept_pending_reversed.append(
                    _PendingObjectSnapshot(
                        timestamp_ns=int(item.timestamp_ns),
                        labels=list(item.labels[-remaining_capacity:]),
                        enqueued_monotonic=float(item.enqueued_monotonic),
                    )
                )
                kept_label_count += remaining_capacity
                dropped_label_count += label_count - remaining_capacity
            kept_pending_reversed.reverse()
            kept_pending = kept_pending_reversed
            self._pending_snapshots = deque(kept_pending)
        self._sync_pending_status()
        if dropped_snapshot_count > 0 or dropped_label_count > 0:
            self._increment_status_counter("dropped_overflow_snapshot_count", dropped_snapshot_count)
            self._increment_status_counter("dropped_overflow_label_count", dropped_label_count)

    def _drop_expired_pending_snapshots(self) -> int:
        max_pending_age_sec = float(self.config.mqtt_max_pending_age_sec)
        if max_pending_age_sec <= 0.0:
            return 0
        cutoff_monotonic = float(time.monotonic()) - max_pending_age_sec
        dropped_snapshot_count = 0
        dropped_label_count = 0
        with self._pending_lock:
            kept_snapshots: deque[_PendingObjectSnapshot] = deque()
            while self._pending_snapshots:
                snapshot = self._pending_snapshots.popleft()
                if float(snapshot.enqueued_monotonic or 0.0) < cutoff_monotonic:
                    dropped_snapshot_count += 1
                    dropped_label_count += len(snapshot.labels)
                    continue
                kept_snapshots.append(snapshot)
            self._pending_snapshots = kept_snapshots
        if dropped_snapshot_count > 0 or dropped_label_count > 0:
            self._sync_pending_status()
            self._increment_status_counter("dropped_stale_snapshot_count", dropped_snapshot_count)
            self._increment_status_counter("dropped_stale_label_count", dropped_label_count)
        return int(dropped_label_count)

    @staticmethod
    def _materialize_object_labels(snapshots: list[_PendingObjectSnapshot], frame_index: int) -> list[ObjectLabelData]:
        labels: list[ObjectLabelData] = []
        for snapshot in snapshots:
            for label in snapshot.labels:
                labels.append(
                    ObjectLabelData(
                        object_id=int(label.object_id),
                        timestamp_ns=int(label.timestamp_ns),
                        points=np.asarray(label.points, dtype=np.float32).copy(),
                        obj_class=str(label.obj_class),
                        obj_class_score=float(label.obj_class_score),
                        sensor_name=str(label.sensor_name),
                        frame_index=int(frame_index),
                        source_path=str(label.source_path),
                    )
                )
        return labels

    @staticmethod
    def _materialize_latest_object_labels(snapshots: list[_PendingObjectSnapshot], frame_index: int) -> list[ObjectLabelData]:
        latest_by_object_id: dict[int, ObjectLabelData] = {}
        for snapshot in snapshots:
            for label in snapshot.labels:
                object_id = int(label.object_id)
                current = latest_by_object_id.get(object_id)
                if current is not None and int(current.timestamp_ns) > int(label.timestamp_ns):
                    continue
                latest_by_object_id[object_id] = ObjectLabelData(
                    object_id=object_id,
                    timestamp_ns=int(label.timestamp_ns),
                    points=np.asarray(label.points, dtype=np.float32).copy(),
                    obj_class=str(label.obj_class),
                    obj_class_score=float(label.obj_class_score),
                    sensor_name=str(label.sensor_name),
                    frame_index=int(frame_index),
                    source_path=str(label.source_path),
                )
        return [latest_by_object_id[object_id] for object_id in sorted(latest_by_object_id)]

    @staticmethod
    def _filter_optional_array(values: Any, finite_mask: np.ndarray, dtype) -> np.ndarray | None:
        if values is None:
            return None
        arr = np.asarray(values, dtype=dtype).reshape(-1)
        if len(arr) != len(finite_mask):
            return None
        return np.asarray(arr[np.asarray(finite_mask, dtype=bool)], dtype=dtype)

    def _set_background_error(self, exc: BaseException) -> None:
        if self._background_error is None:
            self._background_error = exc
            self._update_status(reader_state="error", background_error=str(exc))
        self._stop_event.set()
        self._update_status(stop_requested=True)

    def _raise_background_error_if_any(self) -> None:
        if self._background_error is not None:
            raise self._background_error

    def _update_status(self, **updates: Any) -> None:
        with self._status_lock:
            self._status.update(updates)

    def _increment_status_counter(self, key: str, amount: int = 1) -> None:
        with self._status_lock:
            self._status[key] = int(self._status.get(key, 0)) + int(amount)

    def _mqtt_drain_tolerance_ns(self) -> int:
        return int(round(float(self.config.mqtt_drain_tolerance_sec) * 1_000_000_000))

    def _raw_stream_reconnect_backoff_sec(self) -> float:
        return min(2.0, max(0.25, float(self.config.idle_timeout_sec) * 0.1))

    def _record_raw_stream_reconnect(self, error_message: str, frame_index: int) -> None:
        self._increment_status_counter("raw_stream_reconnect_count", 1)
        self._update_status(
            reader_state="reconnecting_raw",
            waiting_for_first_raw_frame=frame_index == 0,
            last_raw_stream_error=str(error_message),
            last_raw_stream_reconnect_unix_sec=time.time(),
        )

    @staticmethod
    def _is_recoverable_raw_stream_error(exc: BaseException) -> bool:
        exc_name = type(exc).__name__
        message = str(exc)
        return (
            exc_name in {"StreamTerminatedError", "ConnectionResetError", "BrokenPipeError"}
            or "GOAWAY" in message
            or "closing connection" in message
        )

    def _sync_pending_status(self) -> None:
        with self._pending_lock:
            pending_snapshot_count = len(self._pending_snapshots)
            pending_label_count = sum(len(snapshot.labels) for snapshot in self._pending_snapshots)
        self._update_status(
            pending_snapshot_count=int(pending_snapshot_count),
            pending_label_count=int(pending_label_count),
        )

    @staticmethod
    async def _next_stream_payload(stream_iterator, timeout: float | None):
        if timeout is None:
            return await stream_iterator.__anext__()
        return await asyncio.wait_for(stream_iterator.__anext__(), timeout=timeout)

    @staticmethod
    def _install_sync_event_loop() -> tuple[asyncio.AbstractEventLoop, asyncio.AbstractEventLoop | None]:
        try:
            previous_loop = asyncio.get_event_loop()
        except RuntimeError:
            previous_loop = None
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop, previous_loop

    @staticmethod
    def _restore_sync_event_loop(loop: asyncio.AbstractEventLoop, previous_loop: asyncio.AbstractEventLoop | None) -> None:
        asyncio.set_event_loop(previous_loop)
        if not loop.is_closed():
            loop.close()

    @staticmethod
    def _close_async_stream(loop: asyncio.AbstractEventLoop, stream_iterator) -> None:
        if stream_iterator is None or loop.is_closed():
            return
        aclose = getattr(stream_iterator, "aclose", None)
        if callable(aclose):
            try:
                loop.run_until_complete(aclose())
            except Exception:
                pass
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            pass
