from __future__ import annotations

import numpy as np
import open3d as o3d

from tracking_pipeline.config.models import ClusteringConfig
from tracking_pipeline.domain.models import ClusterResult, FrameData
from tracking_pipeline.domain.value_objects import LaneBox
from tracking_pipeline.infrastructure.clustering.common import build_cluster_result, crop_lane_points
from tracking_pipeline.shared.geometry import apply_mask_optional


class GroundRemovedDBSCANClusterer:
    def __init__(self, config: ClusteringConfig):
        self.config = config

    def cluster(self, frame: FrameData, lane_box: LaneBox) -> ClusterResult:
        lane_points, lane_intensity = crop_lane_points(frame, lane_box)
        if len(lane_points) < self.config.vehicle_min_points:
            return ClusterResult(
                lane_points=lane_points,
                detections=[],
                metrics={"algorithm": "ground_removed_dbscan", "lane_point_count": int(len(lane_points))},
                lane_intensity=lane_intensity,
            )
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(lane_points.astype(np.float64))
        extra_metrics = {"ground_removed_count": 0, "ground_model": None}
        cluster_input = lane_points
        cluster_intensity = lane_intensity
        if len(lane_points) >= max(self.config.plane_ransac_n, self.config.vehicle_min_points):
            plane_model, inliers = pcd.segment_plane(
                distance_threshold=float(self.config.plane_distance_threshold),
                ransac_n=int(self.config.plane_ransac_n),
                num_iterations=int(self.config.plane_num_iterations),
            )
            normal = np.asarray(plane_model[:3], dtype=np.float64)
            norm = float(np.linalg.norm(normal))
            if norm > 1e-6 and abs(normal[2] / norm) >= float(self.config.ground_normal_z_min):
                mask = np.ones((len(lane_points),), dtype=bool)
                mask[np.asarray(inliers, dtype=np.int32)] = False
                cluster_input = lane_points[mask]
                cluster_intensity = apply_mask_optional(lane_intensity, mask)
                extra_metrics["ground_removed_count"] = int(len(inliers))
                extra_metrics["ground_model"] = [float(value) for value in plane_model]
        if len(cluster_input) < self.config.vehicle_min_points:
            return ClusterResult(
                lane_points=lane_points,
                detections=[],
                metrics={"algorithm": "ground_removed_dbscan", **extra_metrics},
                lane_intensity=lane_intensity,
            )
        cluster_pcd = o3d.geometry.PointCloud()
        cluster_pcd.points = o3d.utility.Vector3dVector(cluster_input.astype(np.float64))
        labels = np.asarray(cluster_pcd.cluster_dbscan(eps=float(self.config.eps), min_points=int(self.config.min_points), print_progress=False))
        return build_cluster_result(
            "ground_removed_dbscan",
            lane_points,
            lane_intensity,
            cluster_input,
            cluster_intensity,
            labels,
            self.config,
            extra_metrics=extra_metrics,
        )
