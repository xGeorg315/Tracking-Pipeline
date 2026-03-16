from __future__ import annotations

import math

from tracking_pipeline.config.models import AggregationConfig, OutputConfig, TrackingConfig
from tracking_pipeline.infrastructure.aggregation.voxel_fusion import VoxelFusionAccumulator


class OccupancyConsensusFusionAccumulator(VoxelFusionAccumulator):
    fusion_method = "occupancy_consensus_fusion"

    def __init__(self, config: AggregationConfig, output_config: OutputConfig, tracking_config: TrackingConfig):
        super().__init__(config, output_config, tracking_config)

    def _required_observations(self, chunk_count: int) -> int:
        ratio_based = int(math.ceil(float(self.config.consensus_ratio) * float(max(1, chunk_count))))
        return max(int(self.config.fusion_min_observations), ratio_based)
