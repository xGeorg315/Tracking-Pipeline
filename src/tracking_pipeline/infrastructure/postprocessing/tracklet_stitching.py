from __future__ import annotations

from tracking_pipeline.config.models import PostprocessingConfig
from tracking_pipeline.domain.models import Track


class TrackletStitchingPostprocessor:
    name = "tracklet_stitching"

    def __init__(self, config: PostprocessingConfig):
        self.config = config

    def process(self, tracks: dict[int, Track]) -> dict[int, Track]:
        ordered = list(sorted(tracks.values(), key=lambda track: (track.first_frame, track.track_id)))
        changed = True
        while changed:
            changed = False
            merged: list[Track] = []
            skip_indices: set[int] = set()
            for index, track in enumerate(ordered):
                if index in skip_indices:
                    continue
                for next_index in range(index + 1, len(ordered)):
                    candidate = ordered[next_index]
                    if next_index in skip_indices:
                        continue
                    if self._can_merge(track, candidate):
                        track = self._merge_tracks(track, candidate)
                        skip_indices.add(next_index)
                        changed = True
                merged.append(track)
            ordered = merged
        return {track.track_id: track for track in ordered}

    def _can_merge(self, left: Track, right: Track) -> bool:
        if left.last_frame < 0 or right.first_frame < 0:
            return False
        gap = right.first_frame - left.last_frame
        if gap <= 0 or gap > int(self.config.stitching_max_gap):
            return False
        if len(left.centers) == 0 or len(right.centers) == 0:
            return False
        center_dist = float(((left.centers[-1] - right.centers[0]) ** 2).sum() ** 0.5)
        return center_dist <= float(self.config.stitching_max_center_dist)

    def _merge_tracks(self, left: Track, right: Track) -> Track:
        left.centers.extend(right.centers)
        left.frame_ids.extend(right.frame_ids)
        left.local_points.extend(right.local_points)
        left.world_points.extend(right.world_points)
        left.local_intensity.extend(right.local_intensity)
        left.world_intensity.extend(right.world_intensity)
        left.bbox_extents.extend(right.bbox_extents)
        left.hit_count = len(left.frame_ids)
        left.age = max(left.age, right.age, left.last_frame - left.first_frame + 1)
        left.missed = min(left.missed, right.missed)
        left.ended_by_missed = left.ended_by_missed or right.ended_by_missed
        merged_sources = list(dict.fromkeys((left.source_track_ids or [left.track_id]) + (right.source_track_ids or [right.track_id])))
        left.source_track_ids = merged_sources
        left.quality_score = None
        left.quality_metrics = {}
        return left
