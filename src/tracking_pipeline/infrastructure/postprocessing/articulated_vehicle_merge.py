from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tracking_pipeline.config.models import PostprocessingConfig
from tracking_pipeline.domain.models import Track
from tracking_pipeline.domain.rules import axis_to_index, orthogonal_axes


@dataclass(slots=True)
class _PairMetrics:
    lead_track_id: int
    rear_track_id: int
    gap_mean: float
    gap_std: float
    combined_length_p75: float
    overlap_count: int
    overlap_ratio: float


@dataclass(slots=True)
class _PairDebugRecord:
    lead_track_id: int
    rear_track_id: int
    accepted: bool
    rejection_reason: str
    overlap_start_frame_id: int
    overlap_end_frame_id: int
    full_gap_mean: float
    full_gap_std: float
    tail_gap_mean: float
    tail_gap_std: float
    tail_window_frame_count: int
    mean_lateral_offset: float
    mean_vertical_offset: float
    motion_signal_type: str = ""
    motion_window_frame_count: int = 0
    motion_cosine: float = float("nan")
    left_motion_speed: float = float("nan")
    right_motion_speed: float = float("nan")
    motion_speed_delta: float = float("nan")
    left_center_speed: float = float("nan")
    right_center_speed: float = float("nan")


@dataclass(slots=True)
class _PairEvaluation:
    metrics: _PairMetrics | None
    debug: _PairDebugRecord | None


@dataclass(slots=True)
class _MotionMetrics:
    accepted: bool
    signal_type: str
    window_frame_count: int
    cosine: float
    left_speed: float
    right_speed: float
    speed_delta: float
    left_center_speed: float
    right_center_speed: float

    def as_debug_kwargs(self) -> dict[str, float | int | str]:
        return {
            "motion_signal_type": str(self.signal_type),
            "motion_window_frame_count": int(self.window_frame_count),
            "motion_cosine": float(self.cosine),
            "left_motion_speed": float(self.left_speed),
            "right_motion_speed": float(self.right_speed),
            "motion_speed_delta": float(self.speed_delta),
            "left_center_speed": float(self.left_center_speed),
            "right_center_speed": float(self.right_center_speed),
        }


class ArticulatedVehicleMergePostprocessor:
    name = "articulated_vehicle_merge"

    def __init__(self, config: PostprocessingConfig, longitudinal_axis: str = "y"):
        self.config = config
        self.axis_idx = axis_to_index(longitudinal_axis)
        self.lateral_idx, self.vertical_idx = orthogonal_axes(self.axis_idx)
        self.debug_records: list[_PairDebugRecord] = []

    def process(self, tracks: dict[int, Track]) -> dict[int, Track]:
        self.debug_records = []
        ordered = list(sorted(tracks.values(), key=lambda track: (track.first_frame, track.track_id)))
        linked_track_ids: set[int] = set()
        for index, track in enumerate(ordered):
            if int(track.track_id) in linked_track_ids or bool(track.state.get("articulated_vehicle")):
                continue

            best_next_index = None
            best_pair_metrics = None
            for next_index in range(index + 1, len(ordered)):
                candidate = ordered[next_index]
                if int(candidate.track_id) in linked_track_ids or bool(candidate.state.get("articulated_vehicle")):
                    continue
                evaluation = self._pair_metrics(track, candidate)
                if evaluation.debug is not None:
                    self.debug_records.append(evaluation.debug)
                if evaluation.metrics is None:
                    continue
                pair_metrics = evaluation.metrics
                if best_pair_metrics is None or self._score(pair_metrics) > self._score(best_pair_metrics):
                    best_next_index = next_index
                    best_pair_metrics = pair_metrics

            if best_next_index is not None and best_pair_metrics is not None:
                self._link_tracks(track, ordered[best_next_index], best_pair_metrics)
                linked_track_ids.add(int(track.track_id))
                linked_track_ids.add(int(ordered[best_next_index].track_id))
        return {track.track_id: track for track in ordered}

    def _score(self, metrics: _PairMetrics) -> tuple[int, float, float, float]:
        return (
            int(metrics.overlap_count),
            float(metrics.overlap_ratio),
            -float(metrics.gap_std),
            -abs(float(metrics.gap_mean)),
        )

    def _pair_metrics(self, left: Track, right: Track) -> _PairEvaluation:
        overlap = sorted(set(left.frame_ids).intersection(right.frame_ids))
        if not overlap:
            return _PairEvaluation(metrics=None, debug=None)
        if len(overlap) < int(self.config.articulated_min_overlap_frames):
            return _PairEvaluation(metrics=None, debug=self._debug_record(left, right, overlap, False, "overlap_frames"))
        overlap_ratio = len(overlap) / float(max(1, min(len(left.frame_ids), len(right.frame_ids))))
        if overlap_ratio < float(self.config.articulated_min_overlap_ratio):
            return _PairEvaluation(metrics=None, debug=self._debug_record(left, right, overlap, False, "overlap_ratio"))

        left_by_frame = self._by_frame(left)
        right_by_frame = self._by_frame(right)
        left_centers = np.asarray([left_by_frame[frame_id][0] for frame_id in overlap], dtype=np.float64)
        right_centers = np.asarray([right_by_frame[frame_id][0] for frame_id in overlap], dtype=np.float64)
        diffs = right_centers - left_centers
        lateral_offsets = np.abs(diffs[:, self.lateral_idx])
        vertical_offsets = np.abs(diffs[:, self.vertical_idx])
        mean_lateral_offset = float(np.mean(lateral_offsets))
        mean_vertical_offset = float(np.mean(vertical_offsets))

        if mean_lateral_offset > float(self.config.articulated_max_lateral_offset):
            return _PairEvaluation(
                metrics=None,
                debug=self._debug_record(
                    left,
                    right,
                    overlap,
                    False,
                    "lateral_offset",
                    mean_lateral_offset=mean_lateral_offset,
                    mean_vertical_offset=mean_vertical_offset,
                ),
            )
        if mean_vertical_offset > float(self.config.articulated_max_vertical_offset):
            return _PairEvaluation(
                metrics=None,
                debug=self._debug_record(
                    left,
                    right,
                    overlap,
                    False,
                    "vertical_offset",
                    mean_lateral_offset=mean_lateral_offset,
                    mean_vertical_offset=mean_vertical_offset,
                ),
            )

        direction = self._movement_direction(left_centers, right_centers)
        if direction == 0.0:
            return _PairEvaluation(
                metrics=None,
                debug=self._debug_record(
                    left,
                    right,
                    overlap,
                    False,
                    "direction",
                    mean_lateral_offset=mean_lateral_offset,
                    mean_vertical_offset=mean_vertical_offset,
                ),
            )

        motion = self._similar_motion(
            overlap,
            left_centers,
            right_centers,
            left_by_frame,
            right_by_frame,
            direction,
        )
        motion_debug = motion.as_debug_kwargs()
        if not motion.accepted:
            return _PairEvaluation(
                metrics=None,
                debug=self._debug_record(
                    left,
                    right,
                    overlap,
                    False,
                    "motion",
                    mean_lateral_offset=mean_lateral_offset,
                    mean_vertical_offset=mean_vertical_offset,
                    **motion_debug,
                ),
            )

        signed_left = left_centers[:, self.axis_idx] * direction
        signed_right = right_centers[:, self.axis_idx] * direction
        signed_center_diff = signed_left - signed_right
        median_center_diff = float(np.median(signed_center_diff))
        if abs(median_center_diff) < 0.5:
            return _PairEvaluation(
                metrics=None,
                debug=self._debug_record(
                    left,
                    right,
                    overlap,
                    False,
                    "ordering",
                    mean_lateral_offset=mean_lateral_offset,
                    mean_vertical_offset=mean_vertical_offset,
                    **motion_debug,
                ),
            )

        left_ahead = median_center_diff > 0.0
        consistent_order = signed_center_diff > 0.0 if left_ahead else signed_center_diff < 0.0
        if float(np.mean(consistent_order.astype(np.float64))) < 0.9:
            return _PairEvaluation(
                metrics=None,
                debug=self._debug_record(
                    left,
                    right,
                    overlap,
                    False,
                    "ordering",
                    mean_lateral_offset=mean_lateral_offset,
                    mean_vertical_offset=mean_vertical_offset,
                    **motion_debug,
                ),
            )

        lead_track = left if left_ahead else right
        rear_track = right if left_ahead else left
        lead_by_frame = left_by_frame if left_ahead else right_by_frame
        rear_by_frame = right_by_frame if left_ahead else left_by_frame

        gaps = []
        combined_lengths = []
        for frame_id in overlap:
            lead_rear, lead_front = self._signed_edges(lead_by_frame[frame_id][1], direction)
            rear_rear, rear_front = self._signed_edges(rear_by_frame[frame_id][1], direction)
            if lead_rear is None or rear_front is None or rear_rear is None or lead_front is None:
                return _PairEvaluation(
                    metrics=None,
                    debug=self._debug_record(
                        lead_track,
                        rear_track,
                        overlap,
                        False,
                        "missing_points",
                        mean_lateral_offset=mean_lateral_offset,
                        mean_vertical_offset=mean_vertical_offset,
                        **motion_debug,
                    ),
                )
            gaps.append(lead_rear - rear_front)
            combined_lengths.append(max(lead_front, rear_front) - min(lead_rear, rear_rear))

        gaps_arr = np.asarray(gaps, dtype=np.float64)
        combined_arr = np.asarray(combined_lengths, dtype=np.float64)
        full_gap_mean = float(np.mean(gaps_arr))
        full_gap_std = float(np.std(gaps_arr))
        tail_count = min(len(overlap), int(self.config.articulated_gap_eval_window_frames))
        tail_gaps_arr = gaps_arr[-tail_count:]
        tail_gap_mean = float(np.mean(tail_gaps_arr))
        tail_gap_std = float(np.std(tail_gaps_arr))
        use_full_gap_fallback = tail_count < 3
        eval_gap_mean = full_gap_mean if use_full_gap_fallback else tail_gap_mean
        eval_gap_std = full_gap_std if use_full_gap_fallback else tail_gap_std
        gap_reason = "full_gap_fallback" if use_full_gap_fallback else "tail_gap"
        gap_mean_reason = "gap_full" if use_full_gap_fallback else "gap_tail"
        gap_std_reason = "gap_full_std" if use_full_gap_fallback else "gap_tail_std"
        if eval_gap_mean > float(self.config.articulated_max_hitch_gap):
            return _PairEvaluation(
                metrics=None,
                debug=self._debug_record(
                    lead_track,
                    rear_track,
                    overlap,
                    False,
                    gap_mean_reason,
                    full_gap_mean=full_gap_mean,
                    full_gap_std=full_gap_std,
                    tail_gap_mean=tail_gap_mean,
                    tail_gap_std=tail_gap_std,
                    tail_window_frame_count=tail_count,
                    mean_lateral_offset=mean_lateral_offset,
                    mean_vertical_offset=mean_vertical_offset,
                    **motion_debug,
                ),
            )
        if eval_gap_std > float(self.config.articulated_max_hitch_gap_std):
            return _PairEvaluation(
                metrics=None,
                debug=self._debug_record(
                    lead_track,
                    rear_track,
                    overlap,
                    False,
                    gap_std_reason,
                    full_gap_mean=full_gap_mean,
                    full_gap_std=full_gap_std,
                    tail_gap_mean=tail_gap_mean,
                    tail_gap_std=tail_gap_std,
                    tail_window_frame_count=tail_count,
                    mean_lateral_offset=mean_lateral_offset,
                    mean_vertical_offset=mean_vertical_offset,
                    **motion_debug,
                ),
            )
        if float(np.min(gaps_arr)) < -0.75:
            return _PairEvaluation(
                metrics=None,
                debug=self._debug_record(
                    lead_track,
                    rear_track,
                    overlap,
                    False,
                    "gap_negative",
                    full_gap_mean=full_gap_mean,
                    full_gap_std=full_gap_std,
                    tail_gap_mean=tail_gap_mean,
                    tail_gap_std=tail_gap_std,
                    tail_window_frame_count=tail_count,
                    mean_lateral_offset=mean_lateral_offset,
                    mean_vertical_offset=mean_vertical_offset,
                    **motion_debug,
                ),
            )

        combined_length_p75 = float(np.percentile(combined_arr, 75))
        if combined_length_p75 < float(self.config.articulated_min_combined_length):
            return _PairEvaluation(
                metrics=None,
                debug=self._debug_record(
                    lead_track,
                    rear_track,
                    overlap,
                    False,
                    "combined_length",
                    full_gap_mean=full_gap_mean,
                    full_gap_std=full_gap_std,
                    tail_gap_mean=tail_gap_mean,
                    tail_gap_std=tail_gap_std,
                    tail_window_frame_count=tail_count,
                    mean_lateral_offset=mean_lateral_offset,
                    mean_vertical_offset=mean_vertical_offset,
                    **motion_debug,
                ),
            )

        metrics = _PairMetrics(
            lead_track_id=int(lead_track.track_id),
            rear_track_id=int(rear_track.track_id),
            gap_mean=eval_gap_mean,
            gap_std=eval_gap_std,
            combined_length_p75=combined_length_p75,
            overlap_count=len(overlap),
            overlap_ratio=float(overlap_ratio),
        )
        debug = self._debug_record(
            lead_track,
            rear_track,
            overlap,
            True,
            gap_reason,
            full_gap_mean=full_gap_mean,
            full_gap_std=full_gap_std,
            tail_gap_mean=tail_gap_mean,
            tail_gap_std=tail_gap_std,
            tail_window_frame_count=tail_count,
            mean_lateral_offset=mean_lateral_offset,
            mean_vertical_offset=mean_vertical_offset,
            **motion_debug,
        )
        return _PairEvaluation(metrics=metrics, debug=debug)

    def _similar_motion(
        self,
        overlap: list[int],
        left_centers: np.ndarray,
        right_centers: np.ndarray,
        left_by_frame,
        right_by_frame,
        direction: float,
    ) -> _MotionMetrics:
        if len(left_centers) < 2 or len(right_centers) < 2:
            return _MotionMetrics(
                accepted=True,
                signal_type="insufficient_frames",
                window_frame_count=min(len(left_centers), len(right_centers)),
                cosine=float("nan"),
                left_speed=float("nan"),
                right_speed=float("nan"),
                speed_delta=float("nan"),
                left_center_speed=float("nan"),
                right_center_speed=float("nan"),
            )
        left_disp = left_centers[-1] - left_centers[0]
        right_disp = right_centers[-1] - right_centers[0]
        left_norm = float(np.linalg.norm(left_disp))
        right_norm = float(np.linalg.norm(right_disp))
        left_center_speed = float(left_disp[self.axis_idx] / max(1, len(left_centers) - 1))
        right_center_speed = float(right_disp[self.axis_idx] / max(1, len(right_centers) - 1))
        if left_norm <= 1e-6 and right_norm <= 1e-6:
            return _MotionMetrics(
                accepted=True,
                signal_type="rear_anchor",
                window_frame_count=0,
                cosine=1.0,
                left_speed=0.0,
                right_speed=0.0,
                speed_delta=0.0,
                left_center_speed=left_center_speed,
                right_center_speed=right_center_speed,
            )
        if left_norm <= 1e-6 or right_norm <= 1e-6:
            return _MotionMetrics(
                accepted=False,
                signal_type="rear_anchor",
                window_frame_count=0,
                cosine=float("nan"),
                left_speed=float("nan"),
                right_speed=float("nan"),
                speed_delta=float("nan"),
                left_center_speed=left_center_speed,
                right_center_speed=right_center_speed,
            )
        cosine = float(np.dot(left_disp, right_disp) / max(left_norm * right_norm, 1e-6))
        if cosine < 0.85:
            return _MotionMetrics(
                accepted=False,
                signal_type="rear_anchor",
                window_frame_count=0,
                cosine=cosine,
                left_speed=float("nan"),
                right_speed=float("nan"),
                speed_delta=float("nan"),
                left_center_speed=left_center_speed,
                right_center_speed=right_center_speed,
            )

        tail_count = min(len(overlap), int(self.config.articulated_gap_eval_window_frames))
        if tail_count < 2:
            return _MotionMetrics(
                accepted=True,
                signal_type="rear_anchor",
                window_frame_count=tail_count,
                cosine=cosine,
                left_speed=float("nan"),
                right_speed=float("nan"),
                speed_delta=float("nan"),
                left_center_speed=left_center_speed,
                right_center_speed=right_center_speed,
            )

        tail_frames = overlap[-tail_count:]
        left_anchor_positions = []
        right_anchor_positions = []
        for frame_id in tail_frames:
            left_rear, _ = self._signed_edges(left_by_frame[frame_id][1], direction)
            right_rear, _ = self._signed_edges(right_by_frame[frame_id][1], direction)
            if left_rear is None or right_rear is None:
                center_speed_delta = abs(left_center_speed - right_center_speed)
                return _MotionMetrics(
                    accepted=center_speed_delta <= float(self.config.articulated_max_speed_delta),
                    signal_type="center_fallback",
                    window_frame_count=len(overlap),
                    cosine=cosine,
                    left_speed=left_center_speed,
                    right_speed=right_center_speed,
                    speed_delta=center_speed_delta,
                    left_center_speed=left_center_speed,
                    right_center_speed=right_center_speed,
                )
            left_anchor_positions.append(left_rear)
            right_anchor_positions.append(right_rear)

        left_speed = self._fit_line_slope(tail_frames, left_anchor_positions)
        right_speed = self._fit_line_slope(tail_frames, right_anchor_positions)
        speed_delta = abs(left_speed - right_speed)
        return _MotionMetrics(
            accepted=speed_delta <= float(self.config.articulated_max_speed_delta),
            signal_type="rear_anchor",
            window_frame_count=tail_count,
            cosine=cosine,
            left_speed=left_speed,
            right_speed=right_speed,
            speed_delta=speed_delta,
            left_center_speed=left_center_speed,
            right_center_speed=right_center_speed,
        )

    def _movement_direction(self, left_centers: np.ndarray, right_centers: np.ndarray) -> float:
        pair_disp = float((left_centers[-1, self.axis_idx] - left_centers[0, self.axis_idx]) + (right_centers[-1, self.axis_idx] - right_centers[0, self.axis_idx]))
        if abs(pair_disp) <= 1e-6:
            pair_disp = float(left_centers[-1, self.axis_idx] - left_centers[0, self.axis_idx])
        if abs(pair_disp) <= 1e-6:
            pair_disp = float(right_centers[-1, self.axis_idx] - right_centers[0, self.axis_idx])
        if abs(pair_disp) <= 1e-6:
            return 0.0
        return 1.0 if pair_disp >= 0.0 else -1.0

    def _signed_edges(self, points: np.ndarray, direction: float) -> tuple[float | None, float | None]:
        arr = np.asarray(points, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] <= self.axis_idx or len(arr) == 0:
            return None, None
        values = arr[:, self.axis_idx]
        values = values[np.isfinite(values)]
        if len(values) == 0:
            return None, None
        signed = values * direction
        return float(np.min(signed)), float(np.max(signed))

    def _fit_line_slope(self, frame_ids: list[int], positions: list[float]) -> float:
        if len(frame_ids) < 2 or len(positions) < 2:
            return 0.0
        x = np.asarray(frame_ids, dtype=np.float64)
        y = np.asarray(positions, dtype=np.float64)
        x_centered = x - float(np.mean(x))
        denom = float(np.dot(x_centered, x_centered))
        if denom <= 1e-9:
            return 0.0
        y_centered = y - float(np.mean(y))
        return float(np.dot(x_centered, y_centered) / denom)

    def _debug_record(
        self,
        lead_track: Track,
        rear_track: Track,
        overlap: list[int],
        accepted: bool,
        rejection_reason: str,
        *,
        full_gap_mean: float = float("nan"),
        full_gap_std: float = float("nan"),
        tail_gap_mean: float = float("nan"),
        tail_gap_std: float = float("nan"),
        tail_window_frame_count: int = 0,
        mean_lateral_offset: float = float("nan"),
        mean_vertical_offset: float = float("nan"),
        motion_signal_type: str = "",
        motion_window_frame_count: int = 0,
        motion_cosine: float = float("nan"),
        left_motion_speed: float = float("nan"),
        right_motion_speed: float = float("nan"),
        motion_speed_delta: float = float("nan"),
        left_center_speed: float = float("nan"),
        right_center_speed: float = float("nan"),
    ) -> _PairDebugRecord:
        return _PairDebugRecord(
            lead_track_id=int(lead_track.track_id),
            rear_track_id=int(rear_track.track_id),
            accepted=bool(accepted),
            rejection_reason=str(rejection_reason),
            overlap_start_frame_id=-1 if not overlap else int(overlap[0]),
            overlap_end_frame_id=-1 if not overlap else int(overlap[-1]),
            full_gap_mean=float(full_gap_mean),
            full_gap_std=float(full_gap_std),
            tail_gap_mean=float(tail_gap_mean),
            tail_gap_std=float(tail_gap_std),
            tail_window_frame_count=int(tail_window_frame_count),
            mean_lateral_offset=float(mean_lateral_offset),
            mean_vertical_offset=float(mean_vertical_offset),
            motion_signal_type=str(motion_signal_type),
            motion_window_frame_count=int(motion_window_frame_count),
            motion_cosine=float(motion_cosine),
            left_motion_speed=float(left_motion_speed),
            right_motion_speed=float(right_motion_speed),
            motion_speed_delta=float(motion_speed_delta),
            left_center_speed=float(left_center_speed),
            right_center_speed=float(right_center_speed),
        )

    def _link_tracks(self, left: Track, right: Track, metrics: _PairMetrics) -> None:
        lead = left if left.track_id == metrics.lead_track_id else right
        rear = right if lead is left else left
        component_track_ids = list(dict.fromkeys((lead.source_track_ids or [lead.track_id]) + (rear.source_track_ids or [rear.track_id])))
        group_id = int(lead.track_id)

        lead.state.update(
            {
                "articulated_vehicle": True,
                "articulated_pair_id": group_id,
                "articulated_partner_track_id": int(rear.track_id),
                "articulated_role": "lead",
                "articulated_component_track_ids": component_track_ids,
                "articulated_pair_metrics": {
                    "lead_track_id": int(metrics.lead_track_id),
                    "rear_track_id": int(metrics.rear_track_id),
                    "gap_mean": float(metrics.gap_mean),
                    "gap_std": float(metrics.gap_std),
                    "combined_length_p75": float(metrics.combined_length_p75),
                    "overlap_count": int(metrics.overlap_count),
                    "overlap_ratio": float(metrics.overlap_ratio),
                },
                "articulated_rear_gap_mean": float(metrics.gap_mean),
                "articulated_rear_gap_std": float(metrics.gap_std),
                "object_kind": "truck_with_trailer",
                "long_vehicle_component_group_id": group_id,
                "long_vehicle_component_group_kind": "articulated",
                "long_vehicle_component_role": "lead",
                "long_vehicle_component_track_ids": component_track_ids,
            }
        )
        rear.state.update(
            {
                "articulated_vehicle": True,
                "articulated_pair_id": group_id,
                "articulated_partner_track_id": int(lead.track_id),
                "articulated_role": "rear",
                "articulated_component_track_ids": component_track_ids,
                "articulated_pair_metrics": {
                    "lead_track_id": int(metrics.lead_track_id),
                    "rear_track_id": int(metrics.rear_track_id),
                    "gap_mean": float(metrics.gap_mean),
                    "gap_std": float(metrics.gap_std),
                    "combined_length_p75": float(metrics.combined_length_p75),
                    "overlap_count": int(metrics.overlap_count),
                    "overlap_ratio": float(metrics.overlap_ratio),
                },
                "articulated_rear_gap_mean": float(metrics.gap_mean),
                "articulated_rear_gap_std": float(metrics.gap_std),
                "object_kind": "truck_with_trailer",
                "long_vehicle_component_group_id": group_id,
                "long_vehicle_component_group_kind": "articulated",
                "long_vehicle_component_role": "rear",
                "long_vehicle_component_track_ids": component_track_ids,
            }
        )

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
