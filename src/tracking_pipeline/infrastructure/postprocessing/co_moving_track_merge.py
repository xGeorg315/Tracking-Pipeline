from __future__ import annotations

import numpy as np

from tracking_pipeline.config.models import PostprocessingConfig
from tracking_pipeline.domain.models import Track
from tracking_pipeline.domain.rules import axis_to_index, compute_extent, orthogonal_axes


class CoMovingTrackMergePostprocessor:
    name = "co_moving_track_merge"

    def __init__(self, config: PostprocessingConfig, longitudinal_axis: str = "y", long_vehicle_length_threshold: float = 4.5):
        self.config = config
        self.axis_idx = axis_to_index(longitudinal_axis)
        self.lateral_idx, self.vertical_idx = orthogonal_axes(self.axis_idx)
        self.long_vehicle_length_threshold = float(long_vehicle_length_threshold)

    def process(self, tracks: dict[int, Track]) -> dict[int, Track]:
        self._link_long_vehicle_fragments(tracks)
        remaining = {
            track_id: track
            for track_id, track in tracks.items()
            if not bool(track.state.get("long_vehicle_component_group_id"))
        }
        if not remaining:
            return tracks
        merged_remaining = self._merge_remaining_tracks(remaining)
        combined = {track_id: track for track_id, track in tracks.items() if bool(track.state.get("long_vehicle_component_group_id"))}
        combined.update(merged_remaining)
        return combined

    def _merge_remaining_tracks(self, tracks: dict[int, Track]) -> dict[int, Track]:
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

    def _link_long_vehicle_fragments(self, tracks: dict[int, Track]) -> None:
        ordered = list(sorted(tracks.values(), key=lambda track: (track.first_frame, track.track_id)))
        linked_track_ids: set[int] = set()
        for index, track in enumerate(ordered):
            if int(track.track_id) in linked_track_ids or bool(track.state.get("long_vehicle_component_group_id")):
                continue
            best_candidate: Track | None = None
            best_score: tuple[float, float, float, int] | None = None
            for next_index in range(index + 1, len(ordered)):
                candidate = ordered[next_index]
                if int(candidate.track_id) in linked_track_ids or bool(candidate.state.get("long_vehicle_component_group_id")):
                    continue
                if not self._can_merge(track, candidate):
                    continue
                if not self._is_long_vehicle_candidate(track, candidate):
                    continue
                score = self._fragment_link_score(track, candidate)
                if best_score is None or score > best_score:
                    best_candidate = candidate
                    best_score = score
            if best_candidate is not None:
                self._link_tracks(track, best_candidate)
                linked_track_ids.add(int(track.track_id))
                linked_track_ids.add(int(best_candidate.track_id))

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
            point_timestamps_ns = None
            if all(values[6] is not None for values in observations):
                point_timestamps_ns = np.concatenate(
                    [np.asarray(values[6], dtype=np.int64) for values in observations if values[6] is not None],
                    axis=0,
                ).astype(np.int64, copy=False)
            center = np.mean(world_points, axis=0).astype(np.float32)
            extent = compute_extent(world_points)
            frame_timestamp_ns = int(observations[0][7])
            merged.add_observation(
                center,
                world_points,
                int(frame_id),
                frame_timestamp_ns,
                extent,
                intensity=world_intensity,
                point_timestamp_ns=point_timestamps_ns,
            )

        merged.age = max(left.age, right.age, merged.last_frame - merged.first_frame + 1)
        merged.missed = min(left.missed, right.missed)
        merged.ended_by_missed = left.ended_by_missed or right.ended_by_missed
        merged.source_track_ids = list(dict.fromkeys((left.source_track_ids or [left.track_id]) + (right.source_track_ids or [right.track_id])))
        merged.quality_score = None
        merged.quality_metrics = {}
        merged.state = {**left.state, **right.state}
        if bool(left.state.get("articulated_vehicle")) or bool(right.state.get("articulated_vehicle")):
            merged.state["articulated_vehicle"] = True
            component_ids = []
            for track in (left, right):
                component_ids.extend(track.state.get("articulated_component_track_ids") or track.source_track_ids or [track.track_id])
            merged.state["articulated_component_track_ids"] = list(dict.fromkeys(component_ids))
            for key in ("articulated_rear_gap_mean", "articulated_rear_gap_std", "object_kind"):
                value = left.state.get(key)
                if value is None:
                    value = right.state.get(key)
                if value is not None:
                    merged.state[key] = value
        return merged

    def _is_long_vehicle_candidate(self, left: Track, right: Track) -> bool:
        left_extent = self._track_length_extent(left)
        right_extent = self._track_length_extent(right)
        if max(left_extent, right_extent) >= (0.85 * float(self.long_vehicle_length_threshold)):
            return True
        left_center = np.asarray(left.current_center(), dtype=np.float64)
        right_center = np.asarray(right.current_center(), dtype=np.float64)
        center_gap = abs(float(right_center[self.axis_idx] - left_center[self.axis_idx]))
        combined_extent = left_extent + right_extent + center_gap
        return combined_extent >= float(self.long_vehicle_length_threshold)

    def _fragment_link_score(self, left: Track, right: Track) -> tuple[float, float, float, int]:
        combined_extent = self._track_length_extent(left) + self._track_length_extent(right)
        point_support = float(sum(len(points) for points in left.world_points) + sum(len(points) for points in right.world_points))
        age_support = float(min(left.hit_count, right.hit_count))
        return (combined_extent, point_support, age_support, -abs(int(left.track_id) - int(right.track_id)))

    def _link_tracks(self, left: Track, right: Track) -> None:
        lead, fragment = self._lead_and_fragment(left, right)
        component_track_ids = list(dict.fromkeys((lead.source_track_ids or [lead.track_id]) + (fragment.source_track_ids or [fragment.track_id])))
        group_id = int(lead.track_id)
        component_metrics = {
            "lead_track_id": int(lead.track_id),
            "fragment_track_id": int(fragment.track_id),
        }
        lead.state.update(
            {
                "long_vehicle_component_group_id": group_id,
                "long_vehicle_component_group_kind": "fragment",
                "long_vehicle_component_role": "lead",
                "long_vehicle_component_track_ids": component_track_ids,
                "long_vehicle_component_metrics": component_metrics,
            }
        )
        fragment.state.update(
            {
                "long_vehicle_component_group_id": group_id,
                "long_vehicle_component_group_kind": "fragment",
                "long_vehicle_component_role": "fragment",
                "long_vehicle_component_track_ids": component_track_ids,
                "long_vehicle_component_metrics": component_metrics,
            }
        )

    def _lead_and_fragment(self, left: Track, right: Track) -> tuple[Track, Track]:
        overlap = sorted(set(left.frame_ids).intersection(right.frame_ids))
        if overlap:
            left_by_frame = self._by_frame(left)
            right_by_frame = self._by_frame(right)
            left_centers = np.asarray([left_by_frame[frame_id][0] for frame_id in overlap], dtype=np.float64)
            right_centers = np.asarray([right_by_frame[frame_id][0] for frame_id in overlap], dtype=np.float64)
            direction = self._pair_motion_direction(left_centers, right_centers)
            if direction != 0.0:
                left_position = float(np.median(left_centers[:, self.axis_idx] * direction))
                right_position = float(np.median(right_centers[:, self.axis_idx] * direction))
                if abs(left_position - right_position) > 0.25:
                    return (left, right) if left_position > right_position else (right, left)
        left_length = self._track_length_extent(left)
        right_length = self._track_length_extent(right)
        if left_length != right_length:
            return (left, right) if left_length > right_length else (right, left)
        left_points = sum(len(points) for points in left.world_points)
        right_points = sum(len(points) for points in right.world_points)
        if left_points != right_points:
            return (left, right) if left_points > right_points else (right, left)
        return (left, right) if int(left.track_id) <= int(right.track_id) else (right, left)

    def _pair_motion_direction(self, left_centers: np.ndarray, right_centers: np.ndarray) -> float:
        displacements = []
        if len(left_centers) >= 2:
            displacements.append(float(left_centers[-1, self.axis_idx] - left_centers[0, self.axis_idx]))
        if len(right_centers) >= 2:
            displacements.append(float(right_centers[-1, self.axis_idx] - right_centers[0, self.axis_idx]))
        if not displacements:
            return 0.0
        signed = float(np.median(np.asarray(displacements, dtype=np.float64)))
        if abs(signed) <= 1e-6:
            return 0.0
        return 1.0 if signed >= 0.0 else -1.0

    def _track_length_extent(self, track: Track) -> float:
        if track.bbox_extents:
            extents = np.asarray(track.bbox_extents, dtype=np.float64)
            if extents.ndim == 2 and extents.shape[1] > self.axis_idx:
                return float(np.percentile(extents[:, self.axis_idx], 75))
        if track.world_points:
            extents = np.asarray([compute_extent(points) for points in track.world_points], dtype=np.float64)
            if extents.ndim == 2 and extents.shape[1] > self.axis_idx:
                return float(np.percentile(extents[:, self.axis_idx], 75))
        return 0.0

    def _by_frame(
        self,
        track: Track,
    ) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None, int]]:
        mapping: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None, int]] = {}
        local_intensity = track.local_intensity if len(track.local_intensity) == len(track.local_points) else [None for _ in track.local_points]
        world_intensity = track.world_intensity if len(track.world_intensity) == len(track.world_points) else [None for _ in track.world_points]
        point_timestamps_ns = track.point_timestamps_ns if len(track.point_timestamps_ns) == len(track.world_points) else [None for _ in track.world_points]
        frame_timestamps_ns = track.frame_timestamps_ns if len(track.frame_timestamps_ns) == len(track.frame_ids) else [-1 for _ in track.frame_ids]
        for center, frame_id, local_points, world_points, local_scalar, world_scalar, extent, point_timestamp_ns, frame_timestamp_ns in zip(
            track.centers,
            track.frame_ids,
            track.local_points,
            track.world_points,
            local_intensity,
            world_intensity,
            track.bbox_extents,
            point_timestamps_ns,
            frame_timestamps_ns,
        ):
            mapping[int(frame_id)] = (
                np.asarray(center, dtype=np.float32),
                np.asarray(world_points, dtype=np.float32),
                None if world_scalar is None else np.asarray(world_scalar, dtype=np.float32),
                np.asarray(local_points, dtype=np.float32),
                None if local_scalar is None else np.asarray(local_scalar, dtype=np.float32),
                np.asarray(extent, dtype=np.float32),
                None if point_timestamp_ns is None else np.asarray(point_timestamp_ns, dtype=np.int64),
                int(frame_timestamp_ns),
            )
        return mapping
