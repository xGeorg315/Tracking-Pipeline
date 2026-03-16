from __future__ import annotations

from tracking_pipeline.config.models import ClusteringConfig
from tracking_pipeline.domain.models import ClusterResult, FrameData
from tracking_pipeline.domain.value_objects import LaneBox
from tracking_pipeline.infrastructure.clustering.common import build_sensor_cluster_result, crop_lane_sensor_batches
from tracking_pipeline.infrastructure.clustering.sensor_space import beam_neighbor_connectivity


class BeamNeighborRegionGrowingClusterer:
    def __init__(self, config: ClusteringConfig):
        self.config = config

    def cluster(self, frame: FrameData, lane_box: LaneBox) -> ClusterResult:
        lane_points, lane_intensity, batches = crop_lane_sensor_batches(frame, lane_box, self.config)
        if len(lane_points) < self.config.vehicle_min_points:
            return ClusterResult(
                lane_points=lane_points,
                detections=[],
                metrics={"algorithm": "beam_neighbor_region_growing", "lane_point_count": int(len(lane_points))},
                lane_intensity=lane_intensity,
            )
        return build_sensor_cluster_result(
            "beam_neighbor_region_growing",
            lane_points,
            lane_intensity,
            batches,
            self.config,
            connect_fn=beam_neighbor_connectivity(self.config),
        )
