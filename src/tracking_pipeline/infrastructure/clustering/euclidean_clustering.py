from __future__ import annotations

import numpy as np
import open3d as o3d

from tracking_pipeline.config.models import ClusteringConfig
from tracking_pipeline.domain.models import ClusterResult, FrameData
from tracking_pipeline.domain.value_objects import LaneBox
from tracking_pipeline.infrastructure.clustering.common import build_cluster_result, crop_lane_points


class EuclideanClusteringClusterer:
    def __init__(self, config: ClusteringConfig):
        self.config = config

    def cluster(self, frame: FrameData, lane_box: LaneBox) -> ClusterResult:
        lane_points, lane_intensity, lane_point_timestamp_ns = crop_lane_points(frame, lane_box)
        if len(lane_points) < self.config.vehicle_min_points:
            return ClusterResult(
                lane_points=lane_points,
                detections=[],
                metrics={"algorithm": "euclidean_clustering", "lane_point_count": int(len(lane_points))},
                lane_intensity=lane_intensity,
            )
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(lane_points.astype(np.float64))
        labels = np.asarray(pcd.cluster_dbscan(eps=float(self.config.eps), min_points=1, print_progress=False))
        return build_cluster_result(
            "euclidean_clustering",
            lane_points,
            lane_intensity,
            lane_points,
            lane_intensity,
            lane_point_timestamp_ns,
            labels,
            self.config,
        )
