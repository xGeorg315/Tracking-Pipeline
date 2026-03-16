from __future__ import annotations

from tracking_pipeline.config.models import AggregationConfig, OutputConfig, TrackingConfig
from tracking_pipeline.infrastructure.aggregation.voxel_fusion import VoxelFusionAccumulator


class WeightedVoxelFusionAccumulator(VoxelFusionAccumulator):
    fusion_method = "weighted_voxel_fusion"

    def __init__(self, config: AggregationConfig, output_config: OutputConfig, tracking_config: TrackingConfig):
        super().__init__(config, output_config, tracking_config)
