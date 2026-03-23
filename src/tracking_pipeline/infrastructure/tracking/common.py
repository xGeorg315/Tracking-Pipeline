from __future__ import annotations

import numpy as np

from tracking_pipeline.domain.models import Detection, DetectionDebugState, FrameTrackerDebug, Track


class Kalman3D:
    def __init__(self, initial_pos: np.ndarray, init_var: float, process_var: float, meas_var: float):
        self.x = np.zeros((6, 1), dtype=np.float64)
        self.x[:3, 0] = np.asarray(initial_pos, dtype=np.float64)
        self.f = np.eye(6, dtype=np.float64)
        self.f[0, 3] = 1.0
        self.f[1, 4] = 1.0
        self.f[2, 5] = 1.0
        self.h = np.zeros((3, 6), dtype=np.float64)
        self.h[0, 0] = 1.0
        self.h[1, 1] = 1.0
        self.h[2, 2] = 1.0
        self.p = np.eye(6, dtype=np.float64) * float(init_var)
        q = float(process_var)
        self.q = np.diag([q, q, q, q, q, q]).astype(np.float64)
        self.r = np.eye(3, dtype=np.float64) * float(meas_var)
        self.i = np.eye(6, dtype=np.float64)

    def predict(self) -> np.ndarray:
        self.x = self.f @ self.x
        self.p = self.f @ self.p @ self.f.T + self.q
        return self.x[:3, 0].astype(np.float32)

    def update(self, measurement: np.ndarray) -> np.ndarray:
        z = np.asarray(measurement, dtype=np.float64).reshape(3, 1)
        innovation = z - self.h @ self.x
        s = self.h @ self.p @ self.h.T + self.r
        k = self.p @ self.h.T @ np.linalg.pinv(s)
        self.x = self.x + k @ innovation
        self.p = (self.i - k @ self.h) @ self.p
        return self.x[:3, 0].astype(np.float32)


def initialize_track(
    track_id: int,
    detection: Detection,
    frame_idx: int,
    frame_timestamp_ns: int,
    kalman: Kalman3D | None = None,
) -> Track:
    track = Track(track_id=track_id, hit_count=1, age=1, source_track_ids=[track_id])
    if kalman is not None:
        track.state["kf"] = kalman
    track.state["tracker_debug_summary"] = {
        "spawned_count": 1,
        "matched_count": 0,
        "missed_count": 0,
        "suppressed_spawn_count": 0,
    }
    track.add_observation(
        detection.center,
        detection.points,
        frame_idx,
        frame_timestamp_ns,
        detection.extent,
        intensity=detection.intensity,
        point_timestamp_ns=detection.point_timestamp_ns,
    )
    return track


def append_detection(track: Track, detection: Detection, frame_idx: int, frame_timestamp_ns: int, center: np.ndarray) -> None:
    track.missed = 0
    track.age += 1
    _increment_track_debug_summary(track, "matched_count")
    track.add_observation(
        center,
        detection.points,
        frame_idx,
        frame_timestamp_ns,
        detection.extent,
        intensity=detection.intensity,
        point_timestamp_ns=detection.point_timestamp_ns,
    )


def record_missed_track(track: Track) -> tuple[int, int]:
    missed_before = int(track.missed)
    track.missed += 1
    track.age += 1
    _increment_track_debug_summary(track, "missed_count")
    return missed_before, int(track.missed)


def make_detection_debug_states(detections: list[Detection]) -> list[DetectionDebugState]:
    return [
        DetectionDebugState(
            detection_id=int(detection.detection_id),
            center=np.asarray(detection.center, dtype=np.float32).copy(),
            status="unmatched",
            tracking_halo_only=bool(detection.metadata.get("tracking_halo_only", False)),
        )
        for detection in detections
    ]


def build_tracker_metrics(debug: FrameTrackerDebug) -> dict[str, int | str]:
    return {
        "assignment_method": debug.assignment_method,
        "matched_count": int(debug.matched_count),
        "unmatched_track_count": int(debug.missed_count),
        "unmatched_detection_count": int(
            sum(1 for state in debug.detection_states if state.status != "matched")
        ),
        "spawned_count": int(debug.spawned_count),
        "spawn_suppressed_count": int(debug.suppressed_count),
        "halo_detection_count": int(debug.halo_detection_count),
    }


def track_debug_summary(track: Track) -> dict[str, int]:
    summary = track.state.get("tracker_debug_summary")
    if not isinstance(summary, dict):
        return {
            "spawned_count": 0,
            "matched_count": 0,
            "missed_count": 0,
            "suppressed_spawn_count": 0,
        }
    return {
        "spawned_count": int(summary.get("spawned_count", 0)),
        "matched_count": int(summary.get("matched_count", 0)),
        "missed_count": int(summary.get("missed_count", 0)),
        "suppressed_spawn_count": int(summary.get("suppressed_spawn_count", 0)),
    }


def _increment_track_debug_summary(track: Track, key: str) -> None:
    summary = track.state.get("tracker_debug_summary")
    if not isinstance(summary, dict):
        summary = {
            "spawned_count": 0,
            "matched_count": 0,
            "missed_count": 0,
            "suppressed_spawn_count": 0,
        }
        track.state["tracker_debug_summary"] = summary
    summary[key] = int(summary.get(key, 0)) + 1
