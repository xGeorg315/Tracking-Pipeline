from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from tracking_pipeline.config.models import ClusteringConfig
from tracking_pipeline.domain.models import ClusterResult, Detection, FrameData
from tracking_pipeline.domain.value_objects import LaneBox
from tracking_pipeline.shared.geometry import apply_mask_optional, optional_concatenate


@dataclass(slots=True)
class SensorPointBatch:
    scan_index: int
    sensor_name: str
    points: np.ndarray
    ranges: np.ndarray
    row_index: np.ndarray
    col_index: np.ndarray
    intensity: np.ndarray | None = None


@dataclass(slots=True)
class SensorCell:
    scan_index: int
    row: int
    col: int
    points: np.ndarray
    center: np.ndarray
    mean_range: float
    intensity: np.ndarray | None = None


def crop_lane_points(frame: FrameData, lane_box: LaneBox) -> tuple[np.ndarray, np.ndarray | None]:
    xyz = np.asarray(frame.points, dtype=np.float32)
    if len(xyz) == 0:
        return np.zeros((0, 3), dtype=np.float32), None
    mask = lane_box.mask(xyz)
    return xyz[mask], apply_mask_optional(frame.point_intensity, mask)


def crop_lane_sensor_batches(
    frame: FrameData,
    lane_box: LaneBox,
    config: ClusteringConfig,
) -> tuple[np.ndarray, np.ndarray | None, list[SensorPointBatch]]:
    batches: list[SensorPointBatch] = []
    lane_parts: list[np.ndarray] = []
    lane_intensity_parts: list[np.ndarray | None] = []
    for scan_index, scan in enumerate(frame.scans):
        if len(scan.xyz) == 0:
            continue
        mask = lane_box.mask(scan.xyz)
        mask &= np.asarray(scan.ranges >= float(config.sensor_range_min))
        mask &= np.asarray(scan.ranges <= float(config.sensor_range_max))
        if config.sensor_ground_row_ignore > 0:
            mask &= np.asarray(scan.row_index >= int(config.sensor_ground_row_ignore))
        if not np.any(mask):
            continue
        points = np.asarray(scan.xyz[mask], dtype=np.float32)
        lane_parts.append(points)
        lane_intensity_parts.append(apply_mask_optional(scan.intensity, mask))
        batches.append(
            SensorPointBatch(
                scan_index=scan_index,
                sensor_name=scan.sensor_name,
                points=points,
                ranges=np.asarray(scan.ranges[mask], dtype=np.float32),
                row_index=np.asarray(scan.row_index[mask], dtype=np.int32),
                col_index=np.asarray(scan.col_index[mask], dtype=np.int32),
                intensity=apply_mask_optional(scan.intensity, mask),
            )
        )
    lane_points = np.concatenate(lane_parts, axis=0) if lane_parts else np.zeros((0, 3), dtype=np.float32)
    lane_intensity = optional_concatenate(lane_intensity_parts) if lane_parts else None
    return lane_points, lane_intensity, batches


def build_cluster_result(
    algorithm: str,
    lane_points: np.ndarray,
    lane_intensity: np.ndarray | None,
    cluster_points: np.ndarray,
    cluster_intensity: np.ndarray | None,
    labels: np.ndarray,
    config: ClusteringConfig,
    extra_metrics: dict[str, Any] | None = None,
) -> ClusterResult:
    metrics: dict[str, Any] = {
        "algorithm": algorithm,
        "lane_point_count": int(len(lane_points)),
        "cluster_input_point_count": int(len(cluster_points)),
        "raw_cluster_count": 0,
        "accepted_cluster_count": 0,
        "rejected_small_count": 0,
        "rejected_large_count": 0,
    }
    if extra_metrics:
        metrics.update(extra_metrics)
    if len(cluster_points) == 0 or len(labels) == 0:
        return ClusterResult(lane_points=lane_points, detections=[], metrics=metrics, lane_intensity=lane_intensity)

    detections: list[Detection] = []
    detection_id = 1
    unique_labels = [int(label) for label in np.unique(labels) if int(label) >= 0]
    metrics["raw_cluster_count"] = len(unique_labels)
    for label in unique_labels:
        mask = labels == label
        points = cluster_points[mask]
        if len(points) < int(config.vehicle_min_points):
            metrics["rejected_small_count"] += 1
            continue
        if len(points) > int(config.vehicle_max_points):
            metrics["rejected_large_count"] += 1
            continue
        detections.append(
            Detection(
                detection_id=detection_id,
                points=points,
                center=points.mean(axis=0).astype(np.float32),
                min_bound=points.min(axis=0).astype(np.float32),
                max_bound=points.max(axis=0).astype(np.float32),
                intensity=apply_mask_optional(cluster_intensity, mask),
                metadata={
                    "point_count": int(len(points)),
                    "algorithm": algorithm,
                },
            )
        )
        detection_id += 1
    metrics["accepted_cluster_count"] = len(detections)
    return ClusterResult(lane_points=lane_points, detections=detections, metrics=metrics, lane_intensity=lane_intensity)


def build_sensor_cluster_result(
    algorithm: str,
    lane_points: np.ndarray,
    lane_intensity: np.ndarray | None,
    batches: list[SensorPointBatch],
    config: ClusteringConfig,
    connect_fn: Callable[[SensorCell, SensorCell], bool],
    extra_metrics: dict[str, Any] | None = None,
) -> ClusterResult:
    cells = _build_sensor_cells(batches)
    components = _connected_components(
        cells,
        neighbor_rows=max(0, int(config.sensor_neighbor_rows)),
        neighbor_cols=max(0, int(config.sensor_neighbor_cols)),
        connect_fn=connect_fn,
    )
    filtered_components = [component for component in components if sum(len(cell.points) for cell in component) >= int(config.sensor_min_component_size)]
    cluster_points, cluster_intensity, labels = _components_to_points(filtered_components)
    metrics = {
        "sensor_scan_count": len(batches),
        "sensor_cell_count": len(cells),
        "sensor_component_count": len(components),
        "sensor_component_count_kept": len(filtered_components),
    }
    if extra_metrics:
        metrics.update(extra_metrics)
    return build_cluster_result(
        algorithm,
        lane_points,
        lane_intensity,
        cluster_points,
        cluster_intensity,
        labels,
        config,
        extra_metrics=metrics,
    )


def _build_sensor_cells(batches: list[SensorPointBatch]) -> dict[tuple[int, int, int], SensorCell]:
    cells: dict[tuple[int, int, int], SensorCell] = {}
    for batch in batches:
        if len(batch.points) == 0:
            continue
        rc = np.stack([batch.row_index, batch.col_index], axis=1)
        unique_rc, inverse = np.unique(rc, axis=0, return_inverse=True)
        for idx, values in enumerate(unique_rc):
            mask = inverse == idx
            points = np.asarray(batch.points[mask], dtype=np.float32)
            row = int(values[0])
            col = int(values[1])
            key = (batch.scan_index, row, col)
            cells[key] = SensorCell(
                scan_index=batch.scan_index,
                row=row,
                col=col,
                points=points,
                center=points.mean(axis=0).astype(np.float32),
                mean_range=float(np.mean(batch.ranges[mask])),
                intensity=apply_mask_optional(batch.intensity, mask),
            )
    return cells


def _connected_components(
    cells: dict[tuple[int, int, int], SensorCell],
    neighbor_rows: int,
    neighbor_cols: int,
    connect_fn: Callable[[SensorCell, SensorCell], bool],
) -> list[list[SensorCell]]:
    components: list[list[SensorCell]] = []
    visited: set[tuple[int, int, int]] = set()
    for key in cells:
        if key in visited:
            continue
        stack = [key]
        component: list[SensorCell] = []
        visited.add(key)
        while stack:
            current_key = stack.pop()
            current = cells[current_key]
            component.append(current)
            for d_row in range(-neighbor_rows, neighbor_rows + 1):
                for d_col in range(-neighbor_cols, neighbor_cols + 1):
                    if d_row == 0 and d_col == 0:
                        continue
                    neighbor_key = (current.scan_index, current.row + d_row, current.col + d_col)
                    if neighbor_key in visited or neighbor_key not in cells:
                        continue
                    neighbor = cells[neighbor_key]
                    if not connect_fn(current, neighbor):
                        continue
                    visited.add(neighbor_key)
                    stack.append(neighbor_key)
        components.append(component)
    return components


def _components_to_points(components: list[list[SensorCell]]) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    if not components:
        return np.zeros((0, 3), dtype=np.float32), None, np.zeros((0,), dtype=np.int32)
    point_parts: list[np.ndarray] = []
    intensity_parts: list[np.ndarray | None] = []
    label_parts: list[np.ndarray] = []
    for label, component in enumerate(components):
        for cell in component:
            point_parts.append(cell.points)
            intensity_parts.append(cell.intensity)
            label_parts.append(np.full((len(cell.points),), label, dtype=np.int32))
    return (
        np.concatenate(point_parts, axis=0),
        optional_concatenate(intensity_parts),
        np.concatenate(label_parts, axis=0),
    )
