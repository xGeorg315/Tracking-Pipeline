from __future__ import annotations

import base64
from collections import deque
import copy
from dataclasses import dataclass
import math
import threading
import time
from typing import Any, Callable, Mapping

import numpy as np

from tracking_pipeline.domain.models import ClusterResult, FrameData, FrameTrackingState, RunSummary, TrackOutcomeDebug
from tracking_pipeline.domain.rules import axis_to_index, track_exit_line_value
from tracking_pipeline.domain.value_objects import LaneBox

_NSEC_PER_SEC = 1_000_000_000


@dataclass(slots=True)
class _PendingFrame:
    sequence_id: int
    frame: FrameData
    cluster_result: ClusterResult
    tracking_state: FrameTrackingState


class LiveFramePublisher:
    def __init__(
        self,
        *,
        lane_box: LaneBox,
        track_exit_line_axis: str | int,
        track_exit_edge_margin: float,
        max_points: int,
        history_sec: float,
        point_source: str,
        color_by_intensity: bool,
        show_tracker_debug: bool,
        show_track_outcomes: bool,
        run_label: str = "",
        max_frames: int | None = None,
        reader_status_provider: Callable[[], dict[str, object]] | None = None,
        async_publish: bool = True,
    ) -> None:
        self._lane_box = lane_box
        self._track_exit_line_axis = str(track_exit_line_axis)
        self._track_exit_line_axis_index = axis_to_index(track_exit_line_axis)
        self._track_exit_edge_margin = float(track_exit_edge_margin)
        self._max_points = max(1, int(max_points))
        self._history_sec = max(0.05, float(history_sec))
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
        self._async_publish = bool(async_publish)
        self._lock = threading.Lock()
        self._publish_condition = threading.Condition(self._lock)
        self._frames: deque[dict[str, Any]] = deque()
        self._frames_by_sequence: dict[int, dict[str, Any]] = {}
        self._pruned_frame_count = 0
        self._sequence_id = 0
        self._track_outcome_version = 0
        self._track_outcomes: list[dict[str, Any]] = []
        self._pending_frame: _PendingFrame | None = None
        self._worker_busy = False
        self._closed = False
        self._publish_thread: threading.Thread | None = None
        self._status: dict[str, Any] = {
            "pipeline_phase": "waiting_for_frames",
            "processed_frames": 0,
            "last_processed_frame_index": -1,
            "last_frame_timestamp_ns": None,
            "active_track_count": 0,
            "finished_track_count": 0,
            "live_finished_track_processed_count": 0,
            "live_finished_track_queue_count": 0,
            "live_snapshot_track_count": 0,
            "live_snapshot_aggregate_count": 0,
            "saved_aggregates": 0,
            "interrupted": False,
            "live_web_pending_frame": False,
            "live_web_worker_busy": False,
            "live_web_dropped_frame_count": 0,
            "live_web_pruned_frame_count": 0,
            "live_web_retained_frame_count": 0,
            "live_web_oldest_sequence_id": -1,
            "live_web_latest_sequence_id": -1,
            "updated_at_unix_sec": float(time.time()),
        }
        self._summary: dict[str, Any] = {}
        if self._async_publish:
            self._publish_thread = threading.Thread(
                target=self._publish_worker_main,
                name="tracking_pipeline_live_frame_publisher",
                daemon=True,
            )
            self._publish_thread.start()

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
        if not self._async_publish:
            sequence_id = self._reserve_sequence_id()
            payload = self._build_frame_payload(sequence_id, frame, cluster_result, tracking_state)
            self._store_frame_payload(payload, int(frame.timestamp_ns))
            return int(sequence_id)
        with self._publish_condition:
            sequence_id = self._reserve_sequence_id_locked()
            if self._pending_frame is not None:
                self._status["live_web_dropped_frame_count"] = int(self._status.get("live_web_dropped_frame_count", 0)) + 1
            self._pending_frame = _PendingFrame(
                sequence_id=int(sequence_id),
                frame=frame,
                cluster_result=cluster_result,
                tracking_state=tracking_state,
            )
            self._status["live_web_pending_frame"] = True
            self._status["updated_at_unix_sec"] = float(time.time())
            self._publish_condition.notify_all()
            return int(sequence_id)

    def get_frame(self, sequence_id: int) -> dict[str, Any] | None:
        requested = int(sequence_id)
        with self._lock:
            return self._frames_by_sequence.get(requested)

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
                "pruned_frame_count": int(status.get("live_web_pruned_frame_count", 0)),
            },
            "status": status,
            "summary": summary,
            "track_outcomes": track_outcomes,
            "track_outcome_version": int(track_outcome_version),
            "reader": reader_status,
            "monitoring": {
                "status_line": _format_monitoring_status_line(status, reader_status),
                "updated_at_unix_sec": float(status.get("updated_at_unix_sec", time.time()) or time.time()),
            },
        }

    def mark_stopped(self, *, pipeline_phase: str = "stopped") -> None:
        self.update_status(pipeline_phase=pipeline_phase)

    def flush_pending(self, timeout: float | None = None) -> None:
        if not self._async_publish:
            return
        deadline = None if timeout is None else float(time.monotonic()) + max(0.0, float(timeout))
        with self._publish_condition:
            while self._worker_busy or self._pending_frame is not None:
                if deadline is None:
                    self._publish_condition.wait()
                    continue
                remaining = deadline - float(time.monotonic())
                if remaining <= 0.0:
                    break
                self._publish_condition.wait(timeout=remaining)

    def close(self, timeout: float | None = 2.0) -> None:
        if not self._async_publish:
            return
        self.flush_pending(timeout=timeout)
        with self._publish_condition:
            self._closed = True
            self._publish_condition.notify_all()
        thread = self._publish_thread
        if isinstance(thread, threading.Thread):
            thread.join(timeout=timeout)
        self._publish_thread = None

    def _prune_frames_locked(self, newest_timestamp_ns: int) -> None:
        cutoff_timestamp_ns = int(newest_timestamp_ns) - int(self._history_ns)
        while len(self._frames) > self._max_frames:
            self._drop_oldest_frame_locked()
        while len(self._frames) > 1 and int(self._frames[0].get("timestamp_ns", 0)) < cutoff_timestamp_ns:
            self._drop_oldest_frame_locked()
        oldest_sequence_id = -1 if not self._frames else int(self._frames[0].get("sequence_id", -1))
        latest_sequence_id = -1 if not self._frames else int(self._frames[-1].get("sequence_id", -1))
        self._status["live_web_pruned_frame_count"] = int(self._pruned_frame_count)
        self._status["live_web_retained_frame_count"] = int(len(self._frames))
        self._status["live_web_oldest_sequence_id"] = int(oldest_sequence_id)
        self._status["live_web_latest_sequence_id"] = int(latest_sequence_id)

    def _drop_oldest_frame_locked(self) -> None:
        if not self._frames:
            return
        payload = self._frames.popleft()
        sequence_id = int(payload.get("sequence_id", -1))
        self._frames_by_sequence.pop(sequence_id, None)
        self._pruned_frame_count += 1

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

    def _reserve_sequence_id(self) -> int:
        with self._lock:
            return self._reserve_sequence_id_locked()

    def _reserve_sequence_id_locked(self) -> int:
        self._sequence_id += 1
        return int(self._sequence_id)

    def _build_frame_payload(
        self,
        sequence_id: int,
        frame: FrameData,
        cluster_result: ClusterResult,
        tracking_state: FrameTrackingState,
    ) -> dict[str, Any]:
        points, _intensity = self._select_points(frame, cluster_result, tracking_state)
        capped_points, _capped_intensity = _cap_points(points, None, self._max_points)
        return {
            "sequence_id": int(sequence_id),
            "frame_index": int(frame.frame_index),
            "timestamp_ns": int(frame.timestamp_ns),
            "point_count": int(len(capped_points)),
            "points_xyz_encoding": "f16",
            "points_xyz_b64": _serialize_float16_base64(capped_points),
            "detections": _serialize_detection_states(cluster_result, tracking_state),
            "track_states": _serialize_track_states(tracking_state),
        }

    def _store_frame_payload(self, payload: dict[str, Any], timestamp_ns: int) -> None:
        with self._lock:
            self._frames.append(payload)
            self._frames_by_sequence[int(payload.get("sequence_id", -1))] = payload
            self._prune_frames_locked(int(timestamp_ns))
            self._status["updated_at_unix_sec"] = float(time.time())

    def _publish_worker_main(self) -> None:
        while True:
            with self._publish_condition:
                while not self._closed and self._pending_frame is None:
                    self._publish_condition.wait()
                if self._closed and self._pending_frame is None:
                    self._worker_busy = False
                    self._status["live_web_worker_busy"] = False
                    self._status["live_web_pending_frame"] = False
                    self._status["updated_at_unix_sec"] = float(time.time())
                    self._publish_condition.notify_all()
                    return
                pending = self._pending_frame
                self._pending_frame = None
                self._worker_busy = pending is not None
                self._status["live_web_pending_frame"] = self._pending_frame is not None
                self._status["live_web_worker_busy"] = self._worker_busy
                self._status["updated_at_unix_sec"] = float(time.time())
            if pending is None:
                continue
            payload = self._build_frame_payload(
                pending.sequence_id,
                pending.frame,
                pending.cluster_result,
                pending.tracking_state,
            )
            self._store_frame_payload(payload, int(pending.frame.timestamp_ns))
            with self._publish_condition:
                self._worker_busy = False
                self._status["live_web_pending_frame"] = self._pending_frame is not None
                self._status["live_web_worker_busy"] = False
                self._status["updated_at_unix_sec"] = float(time.time())
                self._publish_condition.notify_all()


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


def _format_monitoring_status_line(status: Mapping[str, Any], reader: Mapping[str, Any]) -> str:
    return (
        "live phase={phase} f={processed} hz={recent_hz:.2f}/{total_hz:.2f} tr={active_tracks} "
        "finq={finished_queue} snaptr={snapshot_tracks} saved={saved_vehicles} "
        "aw={artifact_writes} ow={object_writes} raw={raw} mqtt={mqtt_msgs} snap={mqtt_snapshots} "
        "q={pending_labels}/{pending_snapshots} drop={dropped_overflow_labels}/{dropped_stale_labels} "
        "conn={mqtt_connected} wait={waiting_first_raw} reconn={raw_reconnects} "
        "raw_age={last_raw_age} mqtt_age={last_mqtt_age} state={reader_state} "
        "step={pipeline_step} step_age={pipeline_step_age}"
    ).format(
        phase=str(status.get("pipeline_phase", "unknown")),
        processed=int(status.get("processed_frames", 0) or 0),
        recent_hz=float(status.get("processing_recent_hz", 0.0) or 0.0),
        total_hz=float(status.get("processing_total_hz", 0.0) or 0.0),
        active_tracks=int(status.get("active_track_count", 0) or 0),
        finished_queue=int(status.get("live_finished_track_queue_count", 0) or 0),
        snapshot_tracks=int(status.get("live_snapshot_track_count", 0) or 0),
        saved_vehicles=int(status.get("saved_aggregates", 0) or 0),
        artifact_writes=int(status.get("live_artifact_write_count", 0) or 0),
        object_writes=int(status.get("live_object_list_write_count", 0) or 0),
        raw=int(reader.get("raw_frames_received", 0) or 0),
        mqtt_msgs=int(reader.get("mqtt_messages_received", 0) or 0),
        mqtt_snapshots=int(reader.get("mqtt_snapshots_received", 0) or 0),
        pending_labels=int(reader.get("pending_label_count", 0) or 0),
        pending_snapshots=int(reader.get("pending_snapshot_count", 0) or 0),
        dropped_overflow_labels=int(reader.get("dropped_overflow_label_count", 0) or 0),
        dropped_stale_labels=int(reader.get("dropped_stale_label_count", 0) or 0),
        mqtt_connected="yes" if bool(reader.get("mqtt_connected", False)) else "no",
        waiting_first_raw="yes" if bool(reader.get("waiting_for_first_raw_frame", False)) else "no",
        raw_reconnects=int(reader.get("raw_stream_reconnect_count", 0) or 0),
        last_raw_age=_format_monitoring_age(reader.get("last_raw_age_sec")),
        last_mqtt_age=_format_monitoring_age(reader.get("last_mqtt_age_sec")),
        reader_state=str(reader.get("reader_state", "unknown")),
        pipeline_step=str(status.get("current_pipeline_step", "unknown")),
        pipeline_step_age=_format_monitoring_age(status.get("current_pipeline_step_age_sec")),
    )


def _format_monitoring_age(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.1f}s"


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
