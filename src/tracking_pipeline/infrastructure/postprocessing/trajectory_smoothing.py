from __future__ import annotations

from tracking_pipeline.config.models import PostprocessingConfig
from tracking_pipeline.domain.models import Track
from tracking_pipeline.domain.rules import moving_average_centers


class TrajectorySmoothingPostprocessor:
    name = "trajectory_smoothing"

    def __init__(self, config: PostprocessingConfig):
        self.config = config

    def process(self, tracks: dict[int, Track]) -> dict[int, Track]:
        for track in tracks.values():
            if len(track.centers) <= 2:
                continue
            smoothed = moving_average_centers(track.centers, int(self.config.smoothing_window))
            track.centers = smoothed
            if len(track.world_points) == len(track.centers):
                track.local_points = [points - center for points, center in zip(track.world_points, track.centers)]
            if len(track.world_intensity) != len(track.world_points):
                track.world_intensity = [None for _ in track.world_points]
            if len(track.local_intensity) != len(track.local_points):
                track.local_intensity = [None for _ in track.local_points]
        return tracks
