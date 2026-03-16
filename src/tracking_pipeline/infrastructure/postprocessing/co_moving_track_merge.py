from __future__ import annotations

import numpy as np

from tracking_pipeline.config.models import PostprocessingConfig
from tracking_pipeline.domain.models import Track
from tracking_pipeline.domain.rules import axis_to_index, compute_extent, orthogonal_axes


class CoMovingTrackMergePostprocessor:
    name = "co_moving_track_merge"

    def __init__(self, config: PostprocessingConfig, longitudinal_axis: str = "y"):
        self.config = config
        self.axis_idx = axis_to_index(longitudinal_axis)
        self.lateral_idx, self.vertical_idx = orthogonal_axes(self.axis_idx)

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
                    if next_index in skip_indices:
                        continue
                    candidate = ordered[next_index]
                    if self._can_merge(track, candidate):
                        track = self._merge_tracks(track, candidate)
                        skip_indices.add(next_index)
                        changed = True
                merged.append(track)
            ordered = merged
        return {track.track_id: track for track in ordered}

    def _can_merge(self, left: Track, right: Track) -> bool:
        overlap = sorted(set(left.frame_ids).intersection(right.frame_ids))
        if len(overlap) < int(self.config.parallel_merge_min_overlap_frames):
            return False
        overlap_ratio = len(overlap) / float(max(1, min(len(left.frame_ids), len(right.frame_ids))))
        if overlap_ratio < float(self.config.parallel_merge_min_overlap_ratio):
            return False

        left_by_frame = self._by_frame(left)
        right_by_frame = self._by_frame(right)
        left_centers = np.asarray([left_by_frame[frame_id][0] for frame_id in overlap], dtype=np.float64)
        right_centers = np.asarray([right_by_frame[frame_id][0] for frame_id in overlap], dtype=np.float64)
        diffs = right_centers - left_centers
        lateral_offsets = np.abs(diffs[:, self.lateral_idx])
        vertical_offsets = np.abs(diffs[:, self.vertical_idx])
        longitudinal_gaps = np.abs(diffs[:, self.axis_idx])

        if float(np.mean(lateral_offsets)) > float(self.config.parallel_merge_max_lateral_offset):
            return False
        if float(np.max(longitudinal_gaps)) > float(self.config.parallel_merge_max_longitudinal_gap):
            return False
        if float(np.std(longitudinal_gaps)) > max(0.5, float(self.config.parallel_merge_max_longitudinal_gap) * 0.25):
            return False
        if float(np.mean(vertical_offsets)) > max(0.6, float(self.config.parallel_merge_max_lateral_offset)):
            return False
        return self._similar_motion(left_centers, right_centers)

    def _similar_motion(self, left_centers: np.ndarray, right_centers: np.ndarray) -> bool:
        if len(left_centers) < 2 or len(right_centers) < 2:
            return True
        left_disp = left_centers[-1] - left_centers[0]
        right_disp = right_centers[-1] - right_centers[0]
        left_norm = float(np.linalg.norm(left_disp))
        right_norm = float(np.linalg.norm(right_disp))
        if left_norm <= 1e-6 and right_norm <= 1e-6:
            return True
        if left_norm <= 1e-6 or right_norm <= 1e-6:
            return False
        cosine = float(np.dot(left_disp, right_disp) / max(left_norm * right_norm, 1e-6))
        if cosine < 0.85:
            return False
        left_velocity = float(left_disp[self.axis_idx] / max(1, len(left_centers) - 1))
        right_velocity = float(right_disp[self.axis_idx] / max(1, len(right_centers) - 1))
        if abs(left_velocity - right_velocity) > max(0.5, float(self.config.parallel_merge_max_longitudinal_gap) * 0.2):
            return False
        return True

    def _merge_tracks(self, left: Track, right: Track) -> Track:
        merged = Track(track_id=left.track_id)
        by_frame = {}
        for frame_id, values in self._by_frame(left).items():
            by_frame.setdefault(frame_id, []).append(values)
        for frame_id, values in self._by_frame(right).items():
            by_frame.setdefault(frame_id, []).append(values)

        for frame_id in sorted(by_frame):
            observations = by_frame[frame_id]
            world_points = np.concatenate([np.asarray(values[1], dtype=np.float32) for values in observations], axis=0)
            world_intensity = None
            if all(values[2] is not None for values in observations):
                world_intensity = np.concatenate(
                    [np.asarray(values[2], dtype=np.float32) for values in observations if values[2] is not None],
                    axis=0,
                ).astype(np.float32, copy=False)
            center = np.mean(world_points, axis=0).astype(np.float32)
            extent = compute_extent(world_points)
            merged.add_observation(center, world_points, int(frame_id), extent, intensity=world_intensity)

        merged.age = max(left.age, right.age, merged.last_frame - merged.first_frame + 1)
        merged.missed = min(left.missed, right.missed)
        merged.ended_by_missed = left.ended_by_missed or right.ended_by_missed
        merged.source_track_ids = list(dict.fromkeys((left.source_track_ids or [left.track_id]) + (right.source_track_ids or [right.track_id])))
        merged.quality_score = None
        merged.quality_metrics = {}
        return merged

    def _by_frame(
        self,
        track: Track,
    ) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None, np.ndarray]]:
        mapping: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None, np.ndarray]] = {}
        local_intensity = track.local_intensity if len(track.local_intensity) == len(track.local_points) else [None for _ in track.local_points]
        world_intensity = track.world_intensity if len(track.world_intensity) == len(track.world_points) else [None for _ in track.world_points]
        for center, frame_id, local_points, world_points, local_scalar, world_scalar, extent in zip(
            track.centers,
            track.frame_ids,
            track.local_points,
            track.world_points,
            local_intensity,
            world_intensity,
            track.bbox_extents,
        ):
            mapping[int(frame_id)] = (
                np.asarray(center, dtype=np.float32),
                np.asarray(world_points, dtype=np.float32),
                None if world_scalar is None else np.asarray(world_scalar, dtype=np.float32),
                np.asarray(local_points, dtype=np.float32),
                None if local_scalar is None else np.asarray(local_scalar, dtype=np.float32),
                np.asarray(extent, dtype=np.float32),
            )
        return mapping
