from __future__ import annotations

import numpy as np
from dbscan import DBSCAN as fast_dbscan

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
        labels, _ = fast_dbscan(
            np.asarray(lane_points, dtype=np.float32),
            eps=float(self.config.eps),
            min_samples=1,
        )
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
