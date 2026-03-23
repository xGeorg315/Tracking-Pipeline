from __future__ import annotations

import numpy as np

from tracking_pipeline.config.models import TrackingConfig
from tracking_pipeline.domain.models import ActiveTrackState, Detection, FrameTrackerDebug, FrameTrackingState, Track, TrackDebugState
from tracking_pipeline.infrastructure.tracking.assignment import assign_cost_matrix, build_cost_matrix
from tracking_pipeline.infrastructure.tracking.common import (
    Kalman3D,
    append_detection,
    build_tracker_metrics,
    initialize_track,
    make_detection_debug_states,
    record_missed_track,
)


class KalmanNNTracker:
    assignment_method = "greedy"

    def __init__(self, config: TrackingConfig):
        self.config = config
        self.next_id = 1
        self.tracks: dict[int, Track] = {}
        self.finished_tracks: dict[int, Track] = {}

    def _spawn_track(self, detection: Detection, frame_idx: int, frame_timestamp_ns: int) -> ActiveTrackState:
        track_id = self.next_id
        self.next_id += 1
        kalman = Kalman3D(detection.center, self.config.kf_init_var, self.config.kf_process_var, self.config.kf_meas_var)
        self.tracks[track_id] = initialize_track(track_id, detection, frame_idx, frame_timestamp_ns, kalman=kalman)
        return ActiveTrackState(
            track_id=track_id,
            points=detection.points,
            center=detection.center,
            intensity=detection.intensity,
            status="spawned",
        )

    def _predict(self, track: Track) -> np.ndarray:
        return track.state["kf"].predict()

    def _update(self, track: Track, center: np.ndarray) -> np.ndarray:
        return track.state["kf"].update(center)

    @staticmethod
    def _predicted_active_track(track_id: int, track: Track, predicted_center: np.ndarray | None) -> ActiveTrackState | None:
        if predicted_center is None:
            return None
        center = np.asarray(predicted_center, dtype=np.float32).copy()
        if track.local_points:
            points = np.asarray(track.local_points[-1], dtype=np.float32) + center
        elif track.world_points:
            points = np.asarray(track.world_points[-1], dtype=np.float32).copy()
            points += center - track.current_center()
        else:
            points = np.zeros((0, 3), dtype=np.float32)
        intensity = None
        if track.world_intensity:
            latest_intensity = track.world_intensity[-1]
            intensity = None if latest_intensity is None else np.asarray(latest_intensity, dtype=np.float32).copy()
        return ActiveTrackState(
            track_id=int(track_id),
            points=points,
            center=center,
            intensity=intensity,
            status="predicted",
        )

    def _associate(
        self,
        detections: list[Detection],
    ) -> tuple[dict[int, int], list[int], list[int], dict[int, np.ndarray], dict[int, float]]:
        track_ids = list(self.tracks.keys())
        if not track_ids and not detections:
            return {}, [], [], {}, {}
        if not track_ids:
            return {}, [], list(range(len(detections))), {}, {}
        predicted = np.array([self._predict(self.tracks[track_id]) for track_id in track_ids], dtype=np.float32)
        if not detections:
            return {}, track_ids, [], {track_id: predicted[idx] for idx, track_id in enumerate(track_ids)}, {
                track_id: min(
                    float(self.config.sticky_max_dist),
                    float(self.config.max_dist) + float(self.config.sticky_extra_dist_per_missed) * float(self.tracks[track_id].missed),
                )
                for track_id in track_ids
            }

        predicted_extents = np.array([self.tracks[track_id].current_extent() for track_id in track_ids], dtype=np.float32)
        detection_centers = np.array([detection.center for detection in detections], dtype=np.float32)
        detection_extents = np.array([detection.extent for detection in detections], dtype=np.float32)
        cost = build_cost_matrix(predicted, predicted_extents, detection_centers, detection_extents, self.config.association_size_weight)
        gates = np.array(
            [
                min(
                    float(self.config.sticky_max_dist),
                    float(self.config.max_dist)
                    + float(self.config.sticky_extra_dist_per_missed) * float(self.tracks[track_id].missed),
                )
                for track_id in track_ids
            ],
            dtype=np.float32,
        )
        valid = cost <= gates[:, None]
        assignment_rows, unmatched_rows, unmatched_detections = assign_cost_matrix(cost, valid, method=self.assignment_method)
        assignments = {track_ids[row]: detection_idx for row, detection_idx in assignment_rows.items()}
        unmatched_tracks = [track_ids[row] for row in unmatched_rows]
        return (
            assignments,
            unmatched_tracks,
            unmatched_detections,
            {track_id: predicted[idx] for idx, track_id in enumerate(track_ids)},
            {track_id: float(gates[idx]) for idx, track_id in enumerate(track_ids)},
        )

    def step(self, detections: list[Detection], frame_idx: int, frame_timestamp_ns: int) -> FrameTrackingState:
        assignments, unmatched_tracks, unmatched_detections, predicted_by_track, gate_by_track = self._associate(detections)
        active_tracks: list[ActiveTrackState] = []
        detection_debug_states = make_detection_debug_states(detections)
        track_debug_states: list[TrackDebugState] = []

        for track_id, detection_idx in assignments.items():
            detection = detections[detection_idx]
            missed_before = int(self.tracks[track_id].missed)
            filtered_center = self._update(self.tracks[track_id], detection.center)
            append_detection(self.tracks[track_id], detection, frame_idx, frame_timestamp_ns, filtered_center)
            detection_debug_states[detection_idx].status = "matched"
            detection_debug_states[detection_idx].matched_track_id = int(track_id)
            track_debug_states.append(
                TrackDebugState(
                    track_id=int(track_id),
                    predicted_center=np.asarray(predicted_by_track.get(track_id), dtype=np.float32).copy(),
                    output_center=np.asarray(filtered_center, dtype=np.float32).copy(),
                    status="matched",
                    matched_detection_id=int(detection.detection_id),
                    gate_radius=float(gate_by_track.get(track_id, 0.0)),
                    missed_before=missed_before,
                    missed_after=int(self.tracks[track_id].missed),
                )
            )
            active_tracks.append(
                ActiveTrackState(
                    track_id=track_id,
                    points=detection.points,
                    center=filtered_center,
                    intensity=detection.intensity,
                    status="matched",
                )
            )

        to_delete = []
        for track_id in unmatched_tracks:
            track = self.tracks[track_id]
            missed_before, missed_after = record_missed_track(track)
            predicted_center = predicted_by_track.get(track_id)
            track_debug_states.append(
                TrackDebugState(
                    track_id=int(track_id),
                    predicted_center=None if predicted_center is None else np.asarray(predicted_center, dtype=np.float32).copy(),
                    output_center=None if predicted_center is None else np.asarray(predicted_center, dtype=np.float32).copy(),
                    status="missed",
                    gate_radius=float(gate_by_track.get(track_id, 0.0)),
                    missed_before=missed_before,
                    missed_after=missed_after,
                )
            )
            if track.missed > int(self.config.max_missed):
                track.ended_by_missed = True
                self.finished_tracks[track_id] = track
                to_delete.append(track_id)
                continue
            predicted_active_track = self._predicted_active_track(track_id, track, predicted_center)
            if predicted_active_track is not None:
                active_tracks.append(predicted_active_track)
        for track_id in to_delete:
            del self.tracks[track_id]

        for detection_idx in unmatched_detections:
            detection = detections[detection_idx]
            if self._should_suppress_spawn(detection):
                detection_debug_states[detection_idx].status = "spawn_suppressed"
                detection_debug_states[detection_idx].spawn_suppressed = True
                continue
            spawned_state = self._spawn_track(detection, frame_idx, frame_timestamp_ns)
            active_tracks.append(spawned_state)
            detection_debug_states[detection_idx].status = "spawned"
            detection_debug_states[detection_idx].spawned_track_id = int(spawned_state.track_id)
            track_debug_states.append(
                TrackDebugState(
                    track_id=int(spawned_state.track_id),
                    output_center=np.asarray(spawned_state.center, dtype=np.float32).copy(),
                    status="spawned",
                    matched_detection_id=int(detection.detection_id),
                )
            )

        tracker_debug = FrameTrackerDebug(
            assignment_method=self.assignment_method,
            track_states=track_debug_states,
            detection_states=detection_debug_states,
            matched_count=sum(1 for state in track_debug_states if state.status == "matched"),
            missed_count=sum(1 for state in track_debug_states if state.status == "missed"),
            spawned_count=sum(1 for state in track_debug_states if state.status == "spawned"),
            suppressed_count=sum(1 for state in detection_debug_states if state.status == "spawn_suppressed"),
            halo_detection_count=sum(1 for state in detection_debug_states if state.tracking_halo_only),
        )

        return FrameTrackingState(
            frame_index=frame_idx,
            lane_points=np.zeros((0, 3), dtype=np.float32),
            detections=detections,
            active_tracks=active_tracks,
            tracker_metrics=build_tracker_metrics(tracker_debug),
            tracker_debug=tracker_debug,
        )

    @staticmethod
    def _should_suppress_spawn(detection: Detection) -> bool:
        return bool(detection.metadata.get("spawn_suppressed", False))

    def finalize(self) -> dict[int, Track]:
        for track_id, track in self.tracks.items():
            self.finished_tracks[track_id] = track
        self.tracks = {}
        return dict(sorted(self.finished_tracks.items()))
