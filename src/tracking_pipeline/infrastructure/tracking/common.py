from __future__ import annotations

import numpy as np

from tracking_pipeline.domain.models import Detection, Track


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


def initialize_track(track_id: int, detection: Detection, frame_idx: int, kalman: Kalman3D | None = None) -> Track:
    track = Track(track_id=track_id, hit_count=1, age=1, source_track_ids=[track_id])
    if kalman is not None:
        track.state["kf"] = kalman
    track.add_observation(detection.center, detection.points, frame_idx, detection.extent, intensity=detection.intensity)
    return track


def append_detection(track: Track, detection: Detection, frame_idx: int, center: np.ndarray) -> None:
    track.missed = 0
    track.age += 1
    track.add_observation(center, detection.points, frame_idx, detection.extent, intensity=detection.intensity)
