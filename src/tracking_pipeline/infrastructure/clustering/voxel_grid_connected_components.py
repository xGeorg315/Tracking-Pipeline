from __future__ import annotations

import numpy as np

from tracking_pipeline.config.models import ClusteringConfig
from tracking_pipeline.domain.models import ClusterResult, FrameData
from tracking_pipeline.domain.value_objects import LaneBox
from tracking_pipeline.infrastructure.clustering.common import build_cluster_result, crop_lane_points


_FORWARD_NEIGHBOR_OFFSETS = tuple(
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if not (dx == 0 and dy == 0 and dz == 0)
    if dx > 0 or (dx == 0 and dy > 0) or (dx == 0 and dy == 0 and dz > 0)
)


class VoxelGridConnectedComponentsClusterer:
    def __init__(self, config: ClusteringConfig):
        self.config = config

    def cluster(self, frame: FrameData, lane_box: LaneBox) -> ClusterResult:
        lane_points, lane_intensity, lane_point_timestamp_ns = crop_lane_points(frame, lane_box)
        if len(lane_points) < self.config.vehicle_min_points:
            return ClusterResult(
                lane_points=lane_points,
                detections=[],
                metrics={"algorithm": "voxel_grid_connected_components", "lane_point_count": int(len(lane_points))},
                lane_intensity=lane_intensity,
            )

        labels, occupied_voxel_count, voxel_component_count = _point_labels_from_voxel_components(
            lane_points,
            voxel_size=float(self.config.voxel_size),
        )
        return build_cluster_result(
            "voxel_grid_connected_components",
            lane_points,
            lane_intensity,
            lane_points,
            lane_intensity,
            lane_point_timestamp_ns,
            labels,
            self.config,
            extra_metrics={
                "occupied_voxel_count": occupied_voxel_count,
                "voxel_component_count": voxel_component_count,
            },
        )


class _UnionFind:
    def __init__(self, size: int):
        self.parent = np.arange(size, dtype=np.int32)
        self.rank = np.zeros((size,), dtype=np.int8)

    def find(self, index: int) -> int:
        parent = self.parent
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        rank = self.rank
        parent = self.parent
        if rank[root_left] < rank[root_right]:
            parent[root_left] = root_right
            return
        if rank[root_left] > rank[root_right]:
            parent[root_right] = root_left
            return
        parent[root_right] = root_left
        rank[root_left] += 1

    def labels(self) -> np.ndarray:
        roots = np.fromiter((self.find(index) for index in range(len(self.parent))), dtype=np.int32, count=len(self.parent))
        _, labels = np.unique(roots, return_inverse=True)
        return labels.astype(np.int32, copy=False)


def _point_labels_from_voxel_components(points: np.ndarray, voxel_size: float) -> tuple[np.ndarray, int, int]:
    if len(points) == 0:
        return np.zeros((0,), dtype=np.int32), 0, 0

    voxels = np.floor(np.asarray(points, dtype=np.float32) / float(voxel_size)).astype(np.int32)
    occupied_voxels, inverse = np.unique(voxels, axis=0, return_inverse=True)
    voxel_count = int(len(occupied_voxels))
    if voxel_count == 0:
        return np.zeros((0,), dtype=np.int32), 0, 0

    voxel_lookup = {tuple(voxel.tolist()): index for index, voxel in enumerate(occupied_voxels)}
    components = _UnionFind(voxel_count)
    for index, voxel in enumerate(occupied_voxels):
        vx, vy, vz = int(voxel[0]), int(voxel[1]), int(voxel[2])
        for dx, dy, dz in _FORWARD_NEIGHBOR_OFFSETS:
            neighbor_index = voxel_lookup.get((vx + dx, vy + dy, vz + dz))
            if neighbor_index is not None:
                components.union(index, neighbor_index)

    voxel_labels = components.labels()
    point_labels = voxel_labels[inverse].astype(np.int32, copy=False)
    voxel_component_count = int(np.max(voxel_labels) + 1) if len(voxel_labels) > 0 else 0
    return point_labels, voxel_count, voxel_component_count
