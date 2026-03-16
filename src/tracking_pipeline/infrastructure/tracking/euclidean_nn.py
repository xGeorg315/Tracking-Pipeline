from __future__ import annotations

import numpy as np

from tracking_pipeline.config.models import TrackingConfig
from tracking_pipeline.domain.models import ActiveTrackState, Detection, FrameTrackingState, Track
from tracking_pipeline.infrastructure.tracking.assignment import assign_cost_matrix, build_cost_matrix
from tracking_pipeline.infrastructure.tracking.common import append_detection, initialize_track


class EuclideanNNTracker:
    def __init__(self, config: TrackingConfig):
        self.config = config
        self.next_id = 1
        self.tracks: dict[int, Track] = {}
        self.finished_tracks: dict[int, Track] = {}

    def _spawn_track(self, detection: Detection, frame_idx: int) -> ActiveTrackState:
        track_id = self.next_id
        self.next_id += 1
        self.tracks[track_id] = initialize_track(track_id, detection, frame_idx)
        return ActiveTrackState(track_id=track_id, points=detection.points, center=detection.center, intensity=detection.intensity)

    def _associate(self, detections: list[Detection]) -> tuple[dict[int, int], list[int], list[int]]:
        track_ids = list(self.tracks.keys())
        if not track_ids and not detections:
            return {}, [], []
        if not track_ids:
            return {}, [], list(range(len(detections)))
        if not detections:
            return {}, track_ids, []

        predicted = np.array([self.tracks[track_id].current_center() for track_id in track_ids], dtype=np.float32)
        predicted_extents = np.array([self.tracks[track_id].current_extent() for track_id in track_ids], dtype=np.float32)
        detection_centers = np.array([detection.center for detection in detections], dtype=np.float32)
        detection_extents = np.array([detection.extent for detection in detections], dtype=np.float32)
        cost = build_cost_matrix(predicted, predicted_extents, detection_centers, detection_extents, self.config.association_size_weight)
        valid = cost <= float(self.config.max_dist)
        assignment_rows, unmatched_rows, unmatched_detections = assign_cost_matrix(cost, valid, method="greedy")
        assignments = {track_ids[row]: detection_idx for row, detection_idx in assignment_rows.items()}
        unmatched_tracks = [track_ids[row] for row in unmatched_rows]
        return assignments, unmatched_tracks, unmatched_detections

    def step(self, detections: list[Detection], frame_idx: int) -> FrameTrackingState:
        assignments, unmatched_tracks, unmatched_detections = self._associate(detections)
        active_tracks: list[ActiveTrackState] = []

        for track_id, detection_idx in assignments.items():
            detection = detections[detection_idx]
            append_detection(self.tracks[track_id], detection, frame_idx, np.asarray(detection.center, dtype=np.float32))
            active_tracks.append(
                ActiveTrackState(
                    track_id=track_id,
                    points=detection.points,
                    center=detection.center,
                    intensity=detection.intensity,
                )
            )

        to_delete = []
        for track_id in unmatched_tracks:
            track = self.tracks[track_id]
            track.missed += 1
            track.age += 1
            if track.missed > int(self.config.max_missed):
                track.ended_by_missed = True
                self.finished_tracks[track_id] = track
                to_delete.append(track_id)
        for track_id in to_delete:
            del self.tracks[track_id]

        for detection_idx in unmatched_detections:
            active_tracks.append(self._spawn_track(detections[detection_idx], frame_idx))

        return FrameTrackingState(
            frame_index=frame_idx,
            lane_points=np.zeros((0, 3), dtype=np.float32),
            detections=detections,
            active_tracks=active_tracks,
            tracker_metrics={
                "assignment_method": "greedy",
                "matched_count": len(assignments),
                "unmatched_track_count": len(unmatched_tracks),
                "unmatched_detection_count": len(unmatched_detections),
            },
        )

    def finalize(self) -> dict[int, Track]:
        for track_id, track in self.tracks.items():
            self.finished_tracks[track_id] = track
        self.tracks = {}
        return dict(sorted(self.finished_tracks.items()))
