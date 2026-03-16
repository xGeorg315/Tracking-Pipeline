from __future__ import annotations

from typing import Any

import numpy as np

from tracking_pipeline.config.models import AggregationConfig, OutputConfig, TrackingConfig
from tracking_pipeline.infrastructure.aggregation.registration_backends import build_registration_backend
from tracking_pipeline.infrastructure.aggregation.voxel_fusion import VoxelFusionAccumulator


class RegistrationVoxelFusionAccumulator(VoxelFusionAccumulator):
    fusion_method = "registration_voxel_fusion"

    def __init__(self, config: AggregationConfig, output_config: OutputConfig, tracking_config: TrackingConfig):
        super().__init__(config, output_config, tracking_config)
        self.backend = build_registration_backend(config)

    def _prepare_for_fusion(self, chunks: list[np.ndarray]) -> tuple[list[np.ndarray], dict[str, Any]]:
        return self.backend.align_chunks(chunks)
