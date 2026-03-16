from __future__ import annotations

import numpy as np

from tracking_pipeline.config.models import ClusteringConfig
from tracking_pipeline.domain.models import ClusterResult, FrameData
from tracking_pipeline.domain.value_objects import LaneBox
from tracking_pipeline.infrastructure.clustering.common import build_cluster_result, crop_lane_points


class HDBSCANClusterer:
    def __init__(self, config: ClusteringConfig):
        self.config = config
        self._hdbscan = self._try_import()

    def cluster(self, frame: FrameData, lane_box: LaneBox) -> ClusterResult:
        lane_points, lane_intensity = crop_lane_points(frame, lane_box)
        if len(lane_points) < self.config.vehicle_min_points:
            return ClusterResult(
                lane_points=lane_points,
                detections=[],
                metrics={"algorithm": "hdbscan", "lane_point_count": int(len(lane_points))},
                lane_intensity=lane_intensity,
            )
        if self._hdbscan is None:
            raise RuntimeError("hdbscan is not installed. Install with: pip install -e .[benchmark]")
        clusterer = self._hdbscan.HDBSCAN(
            min_cluster_size=int(self.config.hdbscan_min_cluster_size),
            min_samples=int(self.config.hdbscan_min_samples),
        )
        labels = np.asarray(clusterer.fit_predict(lane_points), dtype=np.int32)
        return build_cluster_result("hdbscan", lane_points, lane_intensity, lane_points, lane_intensity, labels, self.config)

    def _try_import(self):
        try:
            import hdbscan  # type: ignore
        except Exception:
            return None
        return hdbscan
