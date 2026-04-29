from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from tracking_pipeline.domain.models import FrameData, LidarScanData, ObjectLabelData, SensorCalibrationData


SEGMENT_FORMAT_VERSION = 1


class FrameSegmentWriter:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.frames_dir = self.root / "frames"
        self.manifest_path = self.root / "manifest.jsonl"
        self.metadata_path = self.root / "segment.json"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_handle = self.manifest_path.open("w", encoding="utf-8")
        self._frame_count = 0
        self._first_timestamp_ns: int | None = None
        self._last_timestamp_ns: int | None = None

    @property
    def frame_count(self) -> int:
        return int(self._frame_count)

    def write_frame(self, frame: FrameData) -> None:
        frame_index = int(frame.frame_index)
        file_name = f"frame_{self._frame_count:06d}.npz"
        arrays: dict[str, np.ndarray] = {
            "points": np.asarray(frame.points, dtype=np.float32),
        }
        row: dict[str, Any] = {
            "npz": str(Path("frames") / file_name),
            "frame_index": frame_index,
            "timestamp_ns": int(frame.timestamp_ns),
            "source_path": str(frame.source_path),
            "source_frame_index": int(frame.source_frame_index),
            "source_sequence_index": int(frame.source_sequence_index),
            "has_point_intensity": frame.point_intensity is not None,
            "has_point_timestamp_ns": frame.point_timestamp_ns is not None,
            "scans": [],
            "object_labels": [],
        }
        if frame.point_intensity is not None:
            arrays["point_intensity"] = np.asarray(frame.point_intensity, dtype=np.float32)
        if frame.point_timestamp_ns is not None:
            arrays["point_timestamp_ns"] = np.asarray(frame.point_timestamp_ns, dtype=np.int64)

        for scan_index, scan in enumerate(frame.scans):
            prefix = f"scan_{scan_index}"
            arrays[f"{prefix}_xyz"] = np.asarray(scan.xyz, dtype=np.float32)
            arrays[f"{prefix}_ranges"] = np.asarray(scan.ranges, dtype=np.float32)
            arrays[f"{prefix}_row_index"] = np.asarray(scan.row_index, dtype=np.int32)
            arrays[f"{prefix}_col_index"] = np.asarray(scan.col_index, dtype=np.int32)
            if scan.intensity is not None:
                arrays[f"{prefix}_intensity"] = np.asarray(scan.intensity, dtype=np.float32)
            if scan.point_timestamp_ns is not None:
                arrays[f"{prefix}_point_timestamp_ns"] = np.asarray(scan.point_timestamp_ns, dtype=np.int64)
            row["scans"].append(
                {
                    "sensor_name": str(scan.sensor_name),
                    "timestamp_ns": int(scan.timestamp_ns),
                    "calibration": _calibration_to_dict(scan.calibration),
                    "has_intensity": scan.intensity is not None,
                    "has_point_timestamp_ns": scan.point_timestamp_ns is not None,
                }
            )

        for label_index, label in enumerate(frame.object_labels):
            key = f"object_{label_index}_points"
            arrays[key] = np.asarray(label.points, dtype=np.float32)
            row["object_labels"].append(
                {
                    "object_id": int(label.object_id),
                    "timestamp_ns": int(label.timestamp_ns),
                    "obj_class": str(label.obj_class),
                    "obj_class_score": float(label.obj_class_score),
                    "sensor_name": str(label.sensor_name),
                    "frame_index": int(label.frame_index),
                    "source_path": str(label.source_path),
                    "points_key": key,
                }
            )

        np.savez(self.frames_dir / file_name, **arrays)
        self._manifest_handle.write(json.dumps(row, sort_keys=True) + "\n")
        self._manifest_handle.flush()
        self._frame_count += 1
        timestamp_ns = int(frame.timestamp_ns)
        if self._first_timestamp_ns is None:
            self._first_timestamp_ns = timestamp_ns
        self._last_timestamp_ns = timestamp_ns

    def close(self) -> None:
        if not self._manifest_handle.closed:
            self._manifest_handle.close()
        self.metadata_path.write_text(
            json.dumps(
                {
                    "format": "tracking_pipeline.frame_segment",
                    "version": SEGMENT_FORMAT_VERSION,
                    "frame_count": int(self._frame_count),
                    "first_timestamp_ns": self._first_timestamp_ns,
                    "last_timestamp_ns": self._last_timestamp_ns,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def __enter__(self) -> "FrameSegmentWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        _ = exc_type, exc, tb
        self.close()


class FrameSegmentReader:
    def iter_frames(self, input_paths: list[str]):
        frame_index_offset = 0
        for input_path in input_paths:
            root = Path(input_path)
            manifest_path = root / "manifest.jsonl"
            if not manifest_path.is_file():
                raise FileNotFoundError(f"Frame segment manifest does not exist: {manifest_path}")
            segment_frame_count = 0
            with manifest_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    frame = self._read_frame(root, row)
                    frame.frame_index = int(frame_index_offset + int(frame.frame_index))
                    segment_frame_count += 1
                    yield frame
            frame_index_offset += int(segment_frame_count)

    def close(self) -> None:
        return None

    def drain_pending_object_labels(self, frame_index: int, max_timestamp_ns: int | None = None) -> list[ObjectLabelData]:
        _ = frame_index, max_timestamp_ns
        return []

    def _read_frame(self, root: Path, row: dict[str, Any]) -> FrameData:
        npz_path = root / str(row["npz"])
        with np.load(npz_path, allow_pickle=False) as arrays:
            scans: list[LidarScanData] = []
            for scan_index, scan_row in enumerate(row.get("scans", []) or []):
                prefix = f"scan_{scan_index}"
                scans.append(
                    LidarScanData(
                        sensor_name=str(scan_row.get("sensor_name", "")),
                        timestamp_ns=int(scan_row.get("timestamp_ns", 0)),
                        xyz=np.asarray(arrays[f"{prefix}_xyz"], dtype=np.float32),
                        ranges=np.asarray(arrays[f"{prefix}_ranges"], dtype=np.float32),
                        row_index=np.asarray(arrays[f"{prefix}_row_index"], dtype=np.int32),
                        col_index=np.asarray(arrays[f"{prefix}_col_index"], dtype=np.int32),
                        calibration=_calibration_from_dict(dict(scan_row.get("calibration", {}) or {})),
                        intensity=(
                            np.asarray(arrays[f"{prefix}_intensity"], dtype=np.float32)
                            if bool(scan_row.get("has_intensity", False))
                            else None
                        ),
                        point_timestamp_ns=(
                            np.asarray(arrays[f"{prefix}_point_timestamp_ns"], dtype=np.int64)
                            if bool(scan_row.get("has_point_timestamp_ns", False))
                            else None
                        ),
                    )
                )
            object_labels = []
            for label_row in row.get("object_labels", []) or []:
                object_labels.append(
                    ObjectLabelData(
                        object_id=int(label_row.get("object_id", 0)),
                        timestamp_ns=int(label_row.get("timestamp_ns", 0)),
                        points=np.asarray(arrays[str(label_row["points_key"])], dtype=np.float32),
                        obj_class=str(label_row.get("obj_class", "")),
                        obj_class_score=float(label_row.get("obj_class_score", 0.0)),
                        sensor_name=str(label_row.get("sensor_name", "")),
                        frame_index=int(label_row.get("frame_index", -1)),
                        source_path=str(label_row.get("source_path", "")),
                    )
                )
            return FrameData(
                frame_index=int(row.get("frame_index", 0)),
                timestamp_ns=int(row.get("timestamp_ns", 0)),
                points=np.asarray(arrays["points"], dtype=np.float32),
                point_intensity=(
                    np.asarray(arrays["point_intensity"], dtype=np.float32)
                    if bool(row.get("has_point_intensity", False))
                    else None
                ),
                point_timestamp_ns=(
                    np.asarray(arrays["point_timestamp_ns"], dtype=np.int64)
                    if bool(row.get("has_point_timestamp_ns", False))
                    else None
                ),
                source_path=str(row.get("source_path", "")),
                source_frame_index=int(row.get("source_frame_index", -1)),
                source_sequence_index=int(row.get("source_sequence_index", 0)),
                object_labels=object_labels,
                scans=scans,
            )


def _calibration_to_dict(calibration: SensorCalibrationData) -> dict[str, Any]:
    return {
        "sensor_name": str(calibration.sensor_name),
        "vertical_fov": float(calibration.vertical_fov),
        "horizontal_fov": float(calibration.horizontal_fov),
        "vertical_scanlines": int(calibration.vertical_scanlines),
        "horizontal_scanlines": int(calibration.horizontal_scanlines),
        "horizontal_angle_spacing": float(calibration.horizontal_angle_spacing),
        "beam_altitude_angles": np.asarray(calibration.beam_altitude_angles, dtype=np.float32).tolist(),
        "beam_azimuth_angles": np.asarray(calibration.beam_azimuth_angles, dtype=np.float32).tolist(),
        "frame_mode": str(calibration.frame_mode),
        "scan_pattern": str(calibration.scan_pattern),
    }


def _calibration_from_dict(payload: dict[str, Any]) -> SensorCalibrationData:
    return SensorCalibrationData(
        sensor_name=str(payload.get("sensor_name", "unknown") or "unknown"),
        vertical_fov=float(payload.get("vertical_fov", 0.0) or 0.0),
        horizontal_fov=float(payload.get("horizontal_fov", 0.0) or 0.0),
        vertical_scanlines=int(payload.get("vertical_scanlines", 0) or 0),
        horizontal_scanlines=int(payload.get("horizontal_scanlines", 0) or 0),
        horizontal_angle_spacing=float(payload.get("horizontal_angle_spacing", 0.0) or 0.0),
        beam_altitude_angles=np.asarray(payload.get("beam_altitude_angles", []) or [], dtype=np.float32),
        beam_azimuth_angles=np.asarray(payload.get("beam_azimuth_angles", []) or [], dtype=np.float32),
        frame_mode=str(payload.get("frame_mode", "") or ""),
        scan_pattern=str(payload.get("scan_pattern", "") or ""),
    )
