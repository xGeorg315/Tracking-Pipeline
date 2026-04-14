from __future__ import annotations

import base64
from collections import deque
import copy
import math
import threading
import time
from typing import Any, Callable, Mapping

import numpy as np

from tracking_pipeline.domain.models import ClusterResult, FrameData, FrameTrackingState, RunSummary, TrackOutcomeDebug
from tracking_pipeline.domain.rules import axis_to_index, track_exit_line_value
from tracking_pipeline.domain.value_objects import LaneBox

_NSEC_PER_SEC = 1_000_000_000


class LiveFramePublisher:
    def __init__(
        self,
        *,
        lane_box: LaneBox,
        track_exit_line_axis: str | int,
        track_exit_edge_margin: float,
        max_points: int,
        history_sec: float,
        retain_all_frames: bool = True,
        point_source: str,
        color_by_intensity: bool,
        show_tracker_debug: bool,
        show_track_outcomes: bool,
        run_label: str = "",
        max_frames: int | None = None,
        reader_status_provider: Callable[[], dict[str, object]] | None = None,
    ) -> None:
        self._lane_box = lane_box
        self._track_exit_line_axis = str(track_exit_line_axis)
        self._track_exit_line_axis_index = axis_to_index(track_exit_line_axis)
        self._track_exit_edge_margin = float(track_exit_edge_margin)
        self._max_points = max(1, int(max_points))
        self._history_sec = max(0.05, float(history_sec))
        self._retain_all_frames = bool(retain_all_frames)
        normalized_point_source = str(point_source).strip().lower()
        self._point_source = "lane" if normalized_point_source == "lane" else "all"
        self._history_ns = int(round(self._history_sec * _NSEC_PER_SEC))
        self._max_frames = (
            max(1, int(max_frames))
            if max_frames is not None
            else max(8, int(math.ceil(self._history_sec * 20.0)) + 4)
        )
        self._color_by_intensity = bool(color_by_intensity)
        self._show_tracker_debug = bool(show_tracker_debug)
        self._show_track_outcomes = bool(show_track_outcomes)
        self._run_label = str(run_label)
        self._reader_status_provider = reader_status_provider
        self._lock = threading.Lock()
        self._frames: deque[dict[str, Any]] = deque()
        self._sequence_id = 0
        self._track_outcome_version = 0
        self._track_outcomes: list[dict[str, Any]] = []
        self._status: dict[str, Any] = {
            "pipeline_phase": "waiting_for_frames",
            "processed_frames": 0,
            "last_processed_frame_index": -1,
            "last_frame_timestamp_ns": None,
            "active_track_count": 0,
            "finished_track_count": 0,
            "saved_aggregates": 0,
            "interrupted": False,
            "updated_at_unix_sec": float(time.time()),
        }
        self._summary: dict[str, Any] = {}

    def update_status(self, **updates: object) -> None:
        if not updates:
            return
        with self._lock:
            for key, value in updates.items():
                self._status[str(key)] = _json_compatible_value(value)
            self._status["updated_at_unix_sec"] = float(time.time())

    def update_summary(self, summary: RunSummary | Mapping[str, Any] | None) -> None:
        if summary is None:
            return
        with self._lock:
            self._summary = _serialize_summary(summary)
            self._status["updated_at_unix_sec"] = float(time.time())

    def update_track_outcomes(self, track_outcomes: Mapping[int, TrackOutcomeDebug] | None) -> None:
        outcomes = {} if track_outcomes is None else dict(track_outcomes)
        with self._lock:
            previous_outcomes = {
                int(row.get("track_id", -1)): copy.deepcopy(row)
                for row in self._track_outcomes
            }
        now_unix_sec = float(time.time())
        serialized = []
        for track_id in sorted(outcomes):
            previous_row = previous_outcomes.get(int(track_id))
            updated_at_unix_sec = now_unix_sec
            if previous_row is not None:
                previous_updated_at = float(previous_row.get("updated_at_unix_sec", now_unix_sec))
                previous_comparable = dict(previous_row)
                previous_comparable.pop("updated_at_unix_sec", None)
                candidate_comparable = _serialize_track_outcome(
                    outcomes[track_id],
                    updated_at_unix_sec=previous_updated_at,
                )
                candidate_comparable.pop("updated_at_unix_sec", None)
                if candidate_comparable == previous_comparable:
                    updated_at_unix_sec = previous_updated_at
            serialized.append(
                _serialize_track_outcome(outcomes[track_id], updated_at_unix_sec=updated_at_unix_sec)
            )
        with self._lock:
            self._track_outcomes = serialized
            self._track_outcome_version += 1
            self._status["updated_at_unix_sec"] = now_unix_sec

    def publish_frame(
        self,
        frame: FrameData,
        cluster_result: ClusterResult,
        tracking_state: FrameTrackingState,
    ) -> int:
        points, _intensity = self._select_points(frame, cluster_result, tracking_state)
        capped_points, _capped_intensity = _cap_points(points, None, self._max_points)

        payload = {
            "sequence_id": -1,
            "frame_index": int(frame.frame_index),
            "timestamp_ns": int(frame.timestamp_ns),
            "point_count": int(len(capped_points)),
            "points_xyz_encoding": "f16",
            "points_xyz_b64": _serialize_float16_base64(capped_points),
            "detections": _serialize_detection_states(cluster_result, tracking_state),
            "track_states": _serialize_track_states(tracking_state),
        }
        with self._lock:
            self._sequence_id += 1
            payload["sequence_id"] = int(self._sequence_id)
            self._frames.append(payload)
            self._prune_frames_locked(int(frame.timestamp_ns))
            self._status["updated_at_unix_sec"] = float(time.time())
            return int(self._sequence_id)

    def get_frame(self, sequence_id: int) -> dict[str, Any] | None:
        requested = int(sequence_id)
        with self._lock:
            for payload in self._frames:
                if int(payload.get("sequence_id", -1)) == requested:
                    return payload
        return None

    def get_frames(self, start_sequence_id: int, *, limit: int) -> list[dict[str, Any]]:
        requested_start = int(start_sequence_id)
        requested_limit = max(1, int(limit))
        rows: list[dict[str, Any]] = []
        with self._lock:
            for payload in self._frames:
                sequence_id = int(payload.get("sequence_id", -1))
                if sequence_id < requested_start:
                    continue
                rows.append(payload)
                if len(rows) >= requested_limit:
                    break
        return rows

    def current_meta(self) -> dict[str, Any]:
        reader_status = self._read_reader_status()
        with self._lock:
            latest_sequence_id = -1 if not self._frames else int(self._frames[-1]["sequence_id"])
            oldest_sequence_id = -1 if not self._frames else int(self._frames[0]["sequence_id"])
            frame_count = len(self._frames)
            status = copy.deepcopy(self._status)
            summary = copy.deepcopy(self._summary)
            track_outcomes = copy.deepcopy(self._track_outcomes)
            track_outcome_version = int(self._track_outcome_version)
        return {
            "run_label": self._run_label,
            "lane_box": self._lane_box.to_list(),
            "exit_line": {
                "axis": self._track_exit_line_axis,
                "axis_index": int(self._track_exit_line_axis_index),
                "value": float(
                    track_exit_line_value(
                        self._lane_box,
                        self._track_exit_line_axis_index,
                        edge_margin=self._track_exit_edge_margin,
                    )
                ),
            },
            "history_sec": float(self._history_sec),
            "retain_all_frames": bool(self._retain_all_frames),
            "max_points": int(self._max_points),
            "point_source": str(self._point_source),
            "color_by_intensity": bool(self._color_by_intensity),
            "overlay_defaults": {
                "show_tracker_debug": bool(self._show_tracker_debug),
                "show_track_outcomes": bool(self._show_track_outcomes),
            },
            "sequence_window": {
                "oldest_sequence_id": int(oldest_sequence_id),
                "latest_sequence_id": int(latest_sequence_id),
                "frame_count": int(frame_count),
            },
            "status": status,
            "summary": summary,
            "track_outcomes": track_outcomes,
            "track_outcome_version": int(track_outcome_version),
            "reader": reader_status,
        }

    def mark_stopped(self, *, pipeline_phase: str = "stopped") -> None:
        self.update_status(pipeline_phase=pipeline_phase)

    def _prune_frames_locked(self, newest_timestamp_ns: int) -> None:
        if self._retain_all_frames:
            return
        cutoff_timestamp_ns = int(newest_timestamp_ns) - int(self._history_ns)
        while len(self._frames) > self._max_frames:
            self._frames.popleft()
        while len(self._frames) > 1 and int(self._frames[0].get("timestamp_ns", 0)) < cutoff_timestamp_ns:
            self._frames.popleft()

    def _read_reader_status(self) -> dict[str, Any]:
        provider = self._reader_status_provider
        if provider is None:
            return {}
        try:
            return {
                str(key): _json_compatible_value(value)
                for key, value in dict(provider() or {}).items()
            }
        except Exception as exc:  # pragma: no cover - defensive
            return {"reader_state": "status_error", "background_error": str(exc)}

    def _select_points(
        self,
        frame: FrameData,
        cluster_result: ClusterResult,
        tracking_state: FrameTrackingState,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if self._point_source == "lane":
            points = np.asarray(cluster_result.lane_points, dtype=np.float32)
            lane_intensity = tracking_state.lane_intensity
            if lane_intensity is None:
                lane_intensity = cluster_result.lane_intensity
            intensity = None if lane_intensity is None else np.asarray(lane_intensity, dtype=np.float32)
            return points, intensity
        points = np.asarray(frame.points, dtype=np.float32)
        intensity = None if frame.point_intensity is None else np.asarray(frame.point_intensity, dtype=np.float32)
        return points, intensity


def _cap_points(
    points: np.ndarray,
    intensity: np.ndarray | None,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    if len(points) <= int(max_points):
        return points.copy(), None if intensity is None else intensity.copy()
    indices = np.linspace(0, len(points) - 1, num=int(max_points), dtype=np.int64)
    capped_points = np.asarray(points[indices], dtype=np.float32)
    capped_intensity = None if intensity is None else np.asarray(intensity[indices], dtype=np.float32)
    return capped_points, capped_intensity


def _serialize_float16_base64(values: np.ndarray) -> str:
    arr = np.asarray(values, dtype=np.float16)
    if arr.size == 0:
        return ""
    return base64.b64encode(arr.reshape(-1).tobytes()).decode("ascii")


def _serialize_vector(vector: np.ndarray | None) -> list[float] | None:
    if vector is None:
        return None
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return None
    return [float(value) for value in arr[:3].tolist()]


def _serialize_detection_states(cluster_result: ClusterResult, tracking_state: FrameTrackingState) -> list[dict[str, Any]]:
    detection_lookup = {
        int(detection.detection_id): detection
        for detection in cluster_result.detections
    }
    debug = tracking_state.tracker_debug
    if debug is not None and debug.detection_states:
        rows = []
        for detection in debug.detection_states:
            source = detection_lookup.get(int(detection.detection_id))
            rows.append(
                {
                    "detection_id": int(detection.detection_id),
                    "center": _serialize_vector(detection.center),
                    "min_bound": None if source is None else _serialize_vector(source.min_bound),
                    "max_bound": None if source is None else _serialize_vector(source.max_bound),
                    "extent": None if source is None else _serialize_vector(source.extent),
                    "point_count": 0 if source is None else int(len(source.points)),
                    "status": str(detection.status),
                    "matched_track_id": None
                    if detection.matched_track_id is None
                    else int(detection.matched_track_id),
                    "spawned_track_id": None
                    if detection.spawned_track_id is None
                    else int(detection.spawned_track_id),
                    "spawn_suppressed": bool(detection.spawn_suppressed),
                    "tracking_halo_only": bool(detection.tracking_halo_only),
                }
            )
        return rows
    rows = []
    for detection in cluster_result.detections:
        rows.append(
            {
                "detection_id": int(detection.detection_id),
                "center": _serialize_vector(detection.center),
                "min_bound": _serialize_vector(detection.min_bound),
                "max_bound": _serialize_vector(detection.max_bound),
                "extent": _serialize_vector(detection.extent),
                "point_count": int(len(detection.points)),
                "status": "",
                "matched_track_id": None,
                "spawned_track_id": None,
                "spawn_suppressed": False,
                "tracking_halo_only": False,
            }
        )
    return rows


def _serialize_track_states(tracking_state: FrameTrackingState) -> list[dict[str, Any]]:
    debug = tracking_state.tracker_debug
    if debug is not None and debug.track_states:
        rows = []
        for track in debug.track_states:
            rows.append(
                {
                    "track_id": int(track.track_id),
                    "predicted_center": _serialize_vector(track.predicted_center),
                    "output_center": _serialize_vector(track.output_center),
                    "status": str(track.status),
                    "matched_detection_id": None
                    if track.matched_detection_id is None
                    else int(track.matched_detection_id),
                    "gate_radius": None if track.gate_radius is None else float(track.gate_radius),
                    "missed_before": int(track.missed_before),
                    "missed_after": int(track.missed_after),
                }
            )
        return rows
    rows = []
    for active_track in tracking_state.active_tracks:
        rows.append(
            {
                "track_id": int(active_track.track_id),
                "predicted_center": _serialize_vector(active_track.center),
                "output_center": _serialize_vector(active_track.center),
                "status": str(active_track.status),
                "matched_detection_id": None,
                "gate_radius": None,
                "missed_before": 0,
                "missed_after": 0,
            }
        )
    return rows


def _serialize_track_outcome(outcome: TrackOutcomeDebug, *, updated_at_unix_sec: float) -> dict[str, Any]:
    return {
        "track_id": int(outcome.track_id),
        "status": str(outcome.status),
        "decision_stage": str(outcome.decision_stage),
        "decision_reason_code": str(outcome.decision_reason_code),
        "decision_summary": str(outcome.decision_summary),
        "updated_at_unix_sec": float(updated_at_unix_sec),
        "last_frame_id": int(outcome.last_frame_id),
        "last_playback_index": int(outcome.last_playback_index),
        "last_center": _serialize_vector(outcome.last_center),
        "hit_count": int(outcome.hit_count),
        "age": int(outcome.age),
        "missed": int(outcome.missed),
        "ended_by_missed": bool(outcome.ended_by_missed),
        "quality_score": None if outcome.quality_score is None else float(outcome.quality_score),
        "selected_frame_ids": [int(frame_id) for frame_id in outcome.selected_frame_ids],
        "tracker_debug_summary": {
            str(key): int(value) for key, value in dict(outcome.tracker_debug_summary).items()
        },
        "predicted_class_id": None if outcome.predicted_class_id is None else int(outcome.predicted_class_id),
        "predicted_class_name": str(outcome.predicted_class_name),
        "predicted_class_score": None
        if outcome.predicted_class_score is None
        else float(outcome.predicted_class_score),
        "classification_backend": str(outcome.classification_backend),
        "classification_point_source": str(outcome.classification_point_source),
        "classification_input_point_count": int(outcome.classification_input_point_count),
        "gt_obj_class": str(outcome.gt_obj_class),
        "gt_obj_class_score": None if outcome.gt_obj_class_score is None else float(outcome.gt_obj_class_score),
    }


def _serialize_summary(summary: RunSummary | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(summary, RunSummary):
        aggregate_status_counts = {str(key): int(value) for key, value in summary.aggregate_status_counts.items()}
        return {
            "frame_count": int(summary.frame_count),
            "finished_track_count": int(summary.finished_track_count),
            "saved_aggregates": int(summary.saved_aggregates),
            "registration_attempts": int(summary.registration_attempts),
            "registration_accepted": int(summary.registration_accepted),
            "registration_rejected": int(summary.registration_rejected),
            "aggregate_status_counts": aggregate_status_counts,
            "object_list_exported_count": int(summary.object_list_exported_count),
            "object_list_seen_ids": int(summary.object_list_seen_ids),
            "gt_match_matched_count": int(summary.gt_match_matched_count),
            "gt_match_unmatched_saved_count": int(summary.gt_match_unmatched_saved_count),
            "gt_match_unmatched_gt_count": int(summary.gt_match_unmatched_gt_count),
        }
    return {
        str(key): _json_compatible_value(value)
        for key, value in dict(summary).items()
    }


def _json_compatible_value(value: object) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_compatible_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible_value(item) for item in value]
    return value
