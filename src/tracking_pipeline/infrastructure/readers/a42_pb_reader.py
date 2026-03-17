from __future__ import annotations

import logging
import math
import struct
from pathlib import Path

import numpy as np

from tracking_pipeline.domain.models import FrameData, LidarScanData, ObjectLabelData, SensorCalibrationData
from tracking_pipeline.infrastructure.readers.a42_proto.frame import Frame

BYTES_PER_POINT = 12
BYTES_PER_CHANNEL = 4
LOGGER = logging.getLogger(__name__)


class A42PBReader:
    def __init__(self, read_intensity: bool = False):
        self.read_intensity = bool(read_intensity)

    def iter_frames(self, input_paths: list[str]) -> list[FrameData]:
        frames: list[FrameData] = []
        frame_index = 0
        last_timestamp_ns: int | None = None
        for sequence_index, input_path in enumerate(input_paths):
            path = Path(input_path).resolve()
            with path.open("rb") as handle:
                source_frame_index = 0
                while True:
                    header = handle.read(4)
                    if not header:
                        break
                    size = struct.unpack("<I", header)[0]
                    payload = handle.read(size)
                    frame = Frame().parse(payload)
                    timestamp_ns = int(frame.frame_timestamp_ns)
                    if last_timestamp_ns is not None and timestamp_ns < last_timestamp_ns:
                        LOGGER.warning(
                            "Input timestamp moved backwards across multi-file sequence: %s frame=%d ts=%d prev_ts=%d",
                            str(path),
                            source_frame_index,
                            timestamp_ns,
                            last_timestamp_ns,
                        )
                    scans: list[LidarScanData] = []
                    object_labels: list[ObjectLabelData] = []
                    for scan in frame.lidars:
                        object_labels.extend(self._object_labels_to_data(scan, frame_index, str(path)))
                        scan_data = self._scan_to_data(scan)
                        if scan_data is not None:
                            scans.append(scan_data)
                    points = np.concatenate([scan.xyz for scan in scans], axis=0) if scans else np.zeros((0, 3), dtype=np.float32)
                    point_intensity = None
                    if scans and all(scan.intensity is not None for scan in scans):
                        point_intensity = np.concatenate(
                            [np.asarray(scan.intensity, dtype=np.float32) for scan in scans if scan.intensity is not None],
                            axis=0,
                        ).astype(np.float32, copy=False)
                    point_timestamp_ns = None
                    if scans and all(scan.point_timestamp_ns is not None for scan in scans):
                        point_timestamp_ns = np.concatenate(
                            [np.asarray(scan.point_timestamp_ns, dtype=np.int64) for scan in scans if scan.point_timestamp_ns is not None],
                            axis=0,
                        ).astype(np.int64, copy=False)
                    frames.append(
                        FrameData(
                            frame_index=frame_index,
                            timestamp_ns=timestamp_ns,
                            points=points,
                            point_intensity=point_intensity,
                            point_timestamp_ns=point_timestamp_ns,
                            source_path=str(path),
                            source_frame_index=source_frame_index,
                            source_sequence_index=sequence_index,
                            object_labels=object_labels,
                            scans=scans,
                        )
                    )
                    last_timestamp_ns = timestamp_ns
                    frame_index += 1
                    source_frame_index += 1
        return frames

    def _scan_to_data(self, scan: object) -> LidarScanData | None:
        pointcloud = getattr(scan, "pointcloud", None)
        xyz_all = self._unpack_xyz_raw(pointcloud)
        if len(xyz_all) == 0:
            return None
        finite_mask = np.isfinite(xyz_all).all(axis=1)
        xyz = np.asarray(xyz_all[finite_mask], dtype=np.float32)
        if len(xyz) == 0:
            return None
        intensity = self._unpack_point_intensity(pointcloud, len(xyz_all))
        if intensity is not None:
            intensity = np.asarray(intensity[finite_mask], dtype=np.float32)
        point_timestamp_ns = self._decode_point_timestamp_ns(
            pointcloud,
            int(getattr(scan, "scan_timestamp_ns", 0)),
            len(xyz_all),
        )
        if point_timestamp_ns is not None:
            point_timestamp_ns = np.asarray(point_timestamp_ns[finite_mask], dtype=np.int64)
        calibration = self._calibration_to_data(getattr(scan, "calibration", None))
        ranges = np.linalg.norm(xyz, axis=1).astype(np.float32)
        row_index = self._compute_row_index(xyz, ranges, calibration, pointcloud)
        col_index = self._compute_col_index(xyz, calibration)
        return LidarScanData(
            sensor_name=calibration.sensor_name,
            timestamp_ns=int(getattr(scan, "scan_timestamp_ns", 0)),
            xyz=xyz,
            intensity=intensity,
            point_timestamp_ns=point_timestamp_ns,
            ranges=ranges,
            row_index=row_index,
            col_index=col_index,
            calibration=calibration,
        )

    def _object_labels_to_data(self, scan: object, frame_index: int, source_path: str) -> list[ObjectLabelData]:
        calibration = self._calibration_to_data(getattr(scan, "calibration", None))
        sensor_name = calibration.sensor_name
        object_labels = []
        for obj in getattr(scan, "object_list", []) or []:
            xyz = self._unpack_xyz(getattr(obj, "pointcloud", None))
            object_labels.append(
                ObjectLabelData(
                    object_id=int(getattr(obj, "id", 0)),
                    timestamp_ns=int(getattr(obj, "timestamp_ns", 0)),
                    points=xyz,
                    obj_class=str(getattr(obj, "obj_class", "") or ""),
                    obj_class_score=float(getattr(obj, "obj_class_score", 0.0) or 0.0),
                    sensor_name=sensor_name,
                    frame_index=int(frame_index),
                    source_path=source_path,
                )
            )
        return object_labels

    def _calibration_to_data(self, calibration: object) -> SensorCalibrationData:
        if calibration is None:
            return SensorCalibrationData(sensor_name="unknown")
        return SensorCalibrationData(
            sensor_name=str(getattr(calibration, "sensor_name", "unknown") or "unknown"),
            vertical_fov=float(getattr(calibration, "vertical_fov", 0.0) or 0.0),
            horizontal_fov=float(getattr(calibration, "horizontal_fov", 0.0) or 0.0),
            vertical_scanlines=int(getattr(calibration, "vertical_scanlines", 0) or 0),
            horizontal_scanlines=int(getattr(calibration, "horizontal_scanlines", 0) or 0),
            horizontal_angle_spacing=float(getattr(calibration, "horizontal_angle_spacing", 0.0) or 0.0),
            beam_altitude_angles=np.asarray(getattr(calibration, "beam_altitude_angles", []) or [], dtype=np.float32),
            beam_azimuth_angles=np.asarray(getattr(calibration, "beam_azimuth_angles", []) or [], dtype=np.float32),
            frame_mode=str(getattr(calibration, "frame_mode", "") or ""),
            scan_pattern=str(getattr(calibration, "scan_pattern", "") or ""),
        )

    def _unpack_xyz(self, pointcloud: object) -> np.ndarray:
        xyz = self._unpack_xyz_raw(pointcloud)
        if len(xyz) == 0:
            return xyz
        return xyz[np.isfinite(xyz).all(axis=1)]

    def _unpack_xyz_raw(self, pointcloud: object) -> np.ndarray:
        if not pointcloud or not getattr(pointcloud, "cartesian", None):
            return np.zeros((0, 3), dtype=np.float32)
        count = len(pointcloud.cartesian) // BYTES_PER_POINT
        if count <= 0:
            return np.zeros((0, 3), dtype=np.float32)
        return np.frombuffer(pointcloud.cartesian, dtype="<f4").reshape(count, 3)

    def _unpack_point_intensity(self, pointcloud: object, count: int) -> np.ndarray | None:
        if not self.read_intensity or not pointcloud or count <= 0:
            return None
        intensity = self._decode_intensity(getattr(pointcloud, "intensity", b""), count)
        if intensity is not None:
            return intensity
        return self._decode_reflectivity(getattr(pointcloud, "reflectivity", b""), count)

    def _decode_intensity(self, raw: bytes, count: int) -> np.ndarray | None:
        if not raw:
            return None
        if len(raw) == count:
            return (np.frombuffer(raw, dtype="<u1").astype(np.float32) / 255.0).astype(np.float32, copy=False)
        if len(raw) == count * 2:
            return (np.frombuffer(raw, dtype="<u2").astype(np.float32) / 65535.0).astype(np.float32, copy=False)
        if len(raw) == count * 4:
            values = np.frombuffer(raw, dtype="<f4").astype(np.float32)
            if not np.isfinite(values).all():
                values = values.copy()
                values[~np.isfinite(values)] = 0.0
            if float(np.min(values)) < 0.0 or float(np.max(values)) > 1.0:
                min_value = float(np.min(values))
                max_value = float(np.max(values))
                if max_value > min_value:
                    values = (values - min_value) / (max_value - min_value)
                else:
                    values = np.zeros((count,), dtype=np.float32)
            return np.clip(values, 0.0, 1.0).astype(np.float32, copy=False)
        return None

    def _decode_reflectivity(self, raw: bytes, count: int) -> np.ndarray | None:
        if not raw or len(raw) != count * 2:
            return None
        return (np.frombuffer(raw, dtype="<u2").astype(np.float32) / 65535.0).astype(np.float32, copy=False)

    def _decode_point_timestamp_ns(self, pointcloud: object, scan_timestamp_ns: int, count: int) -> np.ndarray | None:
        if not pointcloud or count <= 0:
            return None
        raw = getattr(pointcloud, "timestamp_offset", b"")
        if not raw or len(raw) != count * 8:
            return None
        offsets = np.frombuffer(raw, dtype="<u8").astype(np.int64, copy=False)
        return (offsets + np.int64(scan_timestamp_ns)).astype(np.int64, copy=False)

    def _compute_row_index(
        self,
        xyz: np.ndarray,
        ranges: np.ndarray,
        calibration: SensorCalibrationData,
        pointcloud: object,
    ) -> np.ndarray:
        channel_index = self._unpack_channel_id(pointcloud, len(xyz))
        if channel_index is not None:
            return channel_index

        vertical_beams = np.asarray(calibration.beam_altitude_angles, dtype=np.float32)
        elevation = np.arcsin(np.clip(xyz[:, 2] / np.maximum(ranges, 1e-6), -1.0, 1.0)).astype(np.float32)
        if len(vertical_beams) > 0:
            return np.argmin(np.abs(elevation[:, None] - vertical_beams[None, :]), axis=1).astype(np.int32)

        row_count = max(1, int(calibration.vertical_scanlines))
        if row_count == 1:
            return np.zeros((len(xyz),), dtype=np.int32)

        vertical_fov = float(calibration.vertical_fov)
        if vertical_fov > 0:
            half_fov = self._to_radians(vertical_fov) / 2.0
            normalized = (elevation + half_fov) / max(2.0 * half_fov, 1e-6)
        else:
            min_el = float(np.min(elevation))
            max_el = float(np.max(elevation))
            normalized = (elevation - min_el) / max(max_el - min_el, 1e-6)
        return np.clip(np.floor(normalized * row_count), 0, row_count - 1).astype(np.int32)

    def _compute_col_index(self, xyz: np.ndarray, calibration: SensorCalibrationData) -> np.ndarray:
        azimuth = np.arctan2(xyz[:, 1], xyz[:, 0]).astype(np.float32)
        normalized = (azimuth + math.pi) / (2.0 * math.pi)

        col_count = int(calibration.horizontal_scanlines)
        if col_count <= 0:
            spacing = float(calibration.horizontal_angle_spacing)
            if spacing > 0:
                spacing_rad = self._to_radians(spacing)
                col_count = max(16, int(round((2.0 * math.pi) / max(spacing_rad, 1e-6))))
            else:
                col_count = max(16, len(xyz))
        col = np.floor(normalized * col_count).astype(np.int32)
        col %= max(col_count, 1)
        return col

    def _unpack_channel_id(self, pointcloud: object, count: int) -> np.ndarray | None:
        if not pointcloud or not getattr(pointcloud, "channel_id", None):
            return None
        raw = pointcloud.channel_id
        if len(raw) != count * BYTES_PER_CHANNEL:
            return None
        channel = np.frombuffer(raw, dtype="<u4")
        if len(channel) != count:
            return None
        return channel.astype(np.int32)

    def _to_radians(self, value: float) -> float:
        return math.radians(value) if abs(value) > math.pi else value
