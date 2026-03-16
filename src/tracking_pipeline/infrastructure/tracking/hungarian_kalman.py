from __future__ import annotations

from tracking_pipeline.infrastructure.tracking.kalman_nn import KalmanNNTracker


class HungarianKalmanTracker(KalmanNNTracker):
    assignment_method = "hungarian"
