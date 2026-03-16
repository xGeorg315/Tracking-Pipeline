from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class SensorCalibrationData:
    sensor_name: str
    vertical_fov: float = 0.0
    horizontal_fov: float = 0.0
    vertical_scanlines: int = 0
    horizontal_scanlines: int = 0
    horizontal_angle_spacing: float = 0.0
    beam_altitude_angles: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    beam_azimuth_angles: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    frame_mode: str = ""
    scan_pattern: str = ""


@dataclass(slots=True)
class LidarScanData:
    sensor_name: str
    timestamp_ns: int
    xyz: np.ndarray
    ranges: np.ndarray
    row_index: np.ndarray
    col_index: np.ndarray
    calibration: SensorCalibrationData
    intensity: np.ndarray | None = None


@dataclass(slots=True)
class ObjectLabelData:
    object_id: int
    timestamp_ns: int
    points: np.ndarray
    obj_class: str = ""
    obj_class_score: float = 0.0
    sensor_name: str = ""
    frame_index: int = -1
    source_path: str = ""


@dataclass(slots=True)
class FrameData:
    frame_index: int
    timestamp_ns: int
    points: np.ndarray
    point_intensity: np.ndarray | None = None
    source_path: str = ""
    source_frame_index: int = -1
    source_sequence_index: int = 0
    object_labels: list[ObjectLabelData] = field(default_factory=list)
    scans: list[LidarScanData] = field(default_factory=list)


@dataclass(slots=True)
class Detection:
    detection_id: int
    points: np.ndarray
    center: np.ndarray
    min_bound: np.ndarray
    max_bound: np.ndarray
    intensity: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def extent(self) -> np.ndarray:
        return np.asarray(self.max_bound - self.min_bound, dtype=np.float32)


@dataclass(slots=True)
class ClusterResult:
    lane_points: np.ndarray
    detections: list[Detection]
    metrics: dict[str, Any] = field(default_factory=dict)
    lane_intensity: np.ndarray | None = None


@dataclass(slots=True)
class ActiveTrackState:
    track_id: int
    points: np.ndarray
    center: np.ndarray
    intensity: np.ndarray | None = None


@dataclass(slots=True)
class FrameTrackingState:
    frame_index: int
    lane_points: np.ndarray
    detections: list[Detection]
    active_tracks: list[ActiveTrackState]
    lane_intensity: np.ndarray | None = None
    cluster_metrics: dict[str, Any] = field(default_factory=dict)
    tracker_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Track:
    track_id: int
    centers: list[np.ndarray] = field(default_factory=list)
    frame_ids: list[int] = field(default_factory=list)
    local_points: list[np.ndarray] = field(default_factory=list)
    world_points: list[np.ndarray] = field(default_factory=list)
    local_intensity: list[np.ndarray | None] = field(default_factory=list)
    world_intensity: list[np.ndarray | None] = field(default_factory=list)
    bbox_extents: list[np.ndarray] = field(default_factory=list)
    hit_count: int = 0
    age: int = 0
    missed: int = 0
    ended_by_missed: bool = False
    source_track_ids: list[int] = field(default_factory=list)
    quality_score: float | None = None
    quality_metrics: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)

    def current_center(self) -> np.ndarray:
        if not self.centers:
            return np.zeros((3,), dtype=np.float32)
        return np.asarray(self.centers[-1], dtype=np.float32)

    def current_extent(self) -> np.ndarray:
        if not self.bbox_extents:
            return np.zeros((3,), dtype=np.float32)
        return np.asarray(self.bbox_extents[-1], dtype=np.float32)

    def add_observation(
        self,
        center: np.ndarray,
        points_world: np.ndarray,
        frame_idx: int,
        extent: np.ndarray,
        intensity: np.ndarray | None = None,
    ) -> None:
        center = np.asarray(center, dtype=np.float32)
        extent = np.asarray(extent, dtype=np.float32)
        self.centers.append(center.copy())
        self.frame_ids.append(int(frame_idx))
        self.world_points.append(points_world.copy())
        self.local_points.append((points_world - center).copy())
        self.world_intensity.append(None if intensity is None else np.asarray(intensity, dtype=np.float32).copy())
        self.local_intensity.append(None if intensity is None else np.asarray(intensity, dtype=np.float32).copy())
        self.bbox_extents.append(extent.copy())
        self.hit_count = len(self.frame_ids)

    @property
    def first_frame(self) -> int:
        return -1 if not self.frame_ids else int(self.frame_ids[0])

    @property
    def last_frame(self) -> int:
        return -1 if not self.frame_ids else int(self.frame_ids[-1])


@dataclass(slots=True)
class AggregateResult:
    track_id: int
    points: np.ndarray
    selected_frame_ids: list[int]
    status: str
    metrics: dict[str, Any] = field(default_factory=dict)
    intensity: np.ndarray | None = None


@dataclass(slots=True)
class StagePerformance:
    wall_seconds: float = 0.0
    cpu_seconds: float = 0.0
    call_count: int = 0


@dataclass(slots=True)
class RunPerformance:
    total_wall_seconds: float = 0.0
    total_cpu_seconds: float = 0.0
    compute_wall_seconds: float = 0.0
    compute_cpu_seconds: float = 0.0
    io_wall_seconds: float = 0.0
    peak_rss_mb: float | None = None
    stages: dict[str, StagePerformance] = field(default_factory=dict)


@dataclass(slots=True)
class RunSummary:
    input_path: str
    input_paths: list[str]
    tracker_algorithm: str
    accumulator_algorithm: str
    clusterer_algorithm: str
    frame_count: int
    finished_track_count: int
    saved_aggregates: int
    registration_attempts: int
    registration_accepted: int
    registration_rejected: int
    output_dir: str
    postprocessing_methods: list[str] = field(default_factory=list)
    aggregate_status_counts: dict[str, int] = field(default_factory=dict)
    track_quality_mean: float = 0.0
    object_list_exported_count: int = 0
    object_list_seen_ids: int = 0
    object_list_skipped_empty: int = 0
    performance: RunPerformance | None = None
