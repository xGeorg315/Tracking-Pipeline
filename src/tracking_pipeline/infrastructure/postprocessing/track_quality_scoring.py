from __future__ import annotations

import numpy as np

from tracking_pipeline.config.models import PostprocessingConfig
from tracking_pipeline.domain.models import Track
from tracking_pipeline.domain.rules import axis_to_index, clamp01, orthogonal_axes


class TrackQualityScoringPostprocessor:
    name = "track_quality_scoring"

    def __init__(self, config: PostprocessingConfig, longitudinal_axis: str = "y", long_vehicle_length_threshold: float = 4.5):
        self.config = config
        self.axis_idx = axis_to_index(longitudinal_axis)
        self.lateral_idx, self.vertical_idx = orthogonal_axes(self.axis_idx)
        self.long_vehicle_length_threshold = float(long_vehicle_length_threshold)

    def process(self, tracks: dict[int, Track]) -> dict[int, Track]:
        _ = self.config
        for track in tracks.values():
            observation_count = len(track.frame_ids)
            if observation_count == 0:
                track.quality_score = 0.0
                track.quality_metrics = {"observation_count": 0, "is_long_vehicle": False}
                continue

            continuity = clamp01(observation_count / max(1, track.age))
            point_counts = np.asarray([len(points) for points in track.world_points], dtype=np.float64)
            point_count_cv = float(np.std(point_counts) / max(np.mean(point_counts), 1.0)) if len(point_counts) else 1.0
            extents = self._extents(track)
            length_values = extents[:, self.axis_idx] if len(extents) else np.zeros((0,), dtype=np.float64)
            width_values = extents[:, self.lateral_idx] if len(extents) else np.zeros((0,), dtype=np.float64)
            height_values = extents[:, self.vertical_idx] if len(extents) else np.zeros((0,), dtype=np.float64)
            length_cv = self._coefficient_of_variation(length_values)
            width_cv = self._coefficient_of_variation(width_values)
            height_cv = self._coefficient_of_variation(height_values)
            cross_section_cv = 0.5 * (width_cv + height_cv)
            longevity = clamp01(observation_count / 20.0)
            cross_section_stability = clamp01(1.0 / (1.0 + point_count_cv + cross_section_cv))
            length_stability = clamp01(1.0 / (1.0 + length_cv))
            quality = float(0.4 * continuity + 0.3 * longevity + 0.2 * cross_section_stability + 0.1 * length_stability)
            long_extent_p75 = float(np.percentile(length_values, 75)) if len(length_values) else 0.0
            is_long_vehicle = bool(long_extent_p75 >= self.long_vehicle_length_threshold)

            track.quality_score = quality
            track.quality_metrics = {
                "observation_count": observation_count,
                "continuity": float(continuity),
                "longevity": float(longevity),
                "point_count_cv": point_count_cv,
                "length_cv": float(length_cv),
                "width_cv": float(width_cv),
                "height_cv": float(height_cv),
                "cross_section_cv": float(cross_section_cv),
                "quality_score": quality,
                "is_long_vehicle": is_long_vehicle,
                "longitudinal_axis": ["x", "y", "z"][self.axis_idx],
                "longitudinal_extent_p75": long_extent_p75,
            }
        return tracks

    def _extents(self, track: Track) -> np.ndarray:
        if track.bbox_extents:
            return np.asarray(track.bbox_extents, dtype=np.float64)
        if not track.world_points:
            return np.zeros((0, 3), dtype=np.float64)
        extents = []
        for points in track.world_points:
            arr = np.asarray(points, dtype=np.float64)
            extents.append(np.max(arr, axis=0) - np.min(arr, axis=0))
        return np.asarray(extents, dtype=np.float64)

    def _coefficient_of_variation(self, values: np.ndarray) -> float:
        if values.size == 0:
            return 1.0
        return float(np.std(values) / max(np.mean(values), 1.0))
