from __future__ import annotations

import numpy as np

from tracking_pipeline.config.models import ClusteringConfig
from tracking_pipeline.infrastructure.clustering.common import SensorCell


def depth_continuity_limit(config: ClusteringConfig, scale: float = 1.0) -> tuple[float, float]:
    return float(config.sensor_depth_jump_abs) * scale, float(config.sensor_depth_jump_ratio) * scale


def range_jump_connectivity(config: ClusteringConfig, scale: float = 1.0):
    abs_limit, ratio_limit = depth_continuity_limit(config, scale)

    def _connect(left: SensorCell, right: SensorCell) -> bool:
        limit = max(abs_limit, min(left.mean_range, right.mean_range) * ratio_limit)
        return abs(left.mean_range - right.mean_range) <= limit

    return _connect


def beam_neighbor_connectivity(config: ClusteringConfig):
    euclidean_limit = float(config.eps)
    abs_limit = float(config.sensor_depth_jump_abs) * 1.5
    ratio_limit = float(config.sensor_depth_jump_ratio) * 1.5

    def _connect(left: SensorCell, right: SensorCell) -> bool:
        range_limit = max(abs_limit, min(left.mean_range, right.mean_range) * ratio_limit)
        if abs(left.mean_range - right.mean_range) > range_limit:
            return False
        return float(np.linalg.norm(left.center - right.center)) <= euclidean_limit

    return _connect
