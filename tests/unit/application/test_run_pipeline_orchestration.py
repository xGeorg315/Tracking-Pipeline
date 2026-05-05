from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pytest

from tracking_pipeline.application.class_normalization import ClassNormalizer
from tracking_pipeline.application.performance import PerformanceProfiler, derive_hz
from tracking_pipeline.application.replay_run import replay_run
from tracking_pipeline.application.run_pipeline import (
    _live_gt_match_label_history,
    _live_snapshot_tracker_states,
    _maybe_write_incremental_live_artifact_snapshot,
    _maybe_write_live_object_list_snapshot,
    _snapshot_tracker_tracks,
    _update_live_web_status,
    run_pipeline,
)
from tracking_pipeline.config.models import (
    AggregationConfig,
    ClusteringConfig,
    ClassNormalizationConfig,
    InputConfig,
    OutputConfig,
    PipelineConfig,
    PostprocessingConfig,
    PreprocessingConfig,
    QB2LiveInputConfig,
    QB2LiveMQTTConfig,
    TrackingConfig,
    VisualizationConfig,
)
from tracking_pipeline.domain.models import (
    ActiveTrackState,
    AggregateResult,
    ClusterResult,
    ClassificationPrediction,
    Detection,
    FrameData,
    FrameTrackerDebug,
    FrameTrackingState,
    GTMatchResult,
    ObjectLabelData,
    Track,
    TrackOutcomeDebug,
)
from tracking_pipeline.infrastructure.io.dataset_artifact_writer import DatasetArtifactWriter
from tracking_pipeline.infrastructure.io.frame_segment import FrameSegmentReader
from tracking_pipeline.infrastructure.postprocessing.articulated_vehicle_merge import ArticulatedVehicleMergePostprocessor


class _FakeReader:
    def iter_frames(self, input_paths: list[str]) -> list[FrameData]:
        return [
            FrameData(
                frame_index=index,
                timestamp_ns=index + 1,
                points=np.array([[float(index), 0.0, 0.0]], dtype=np.float32),
                point_intensity=np.array([0.25 + 0.25 * float(index)], dtype=np.float32),
                source_path=input_path,
                source_frame_index=0,
                source_sequence_index=index,
            )
            for index, input_path in enumerate(input_paths)
        ]


class _FakeClusterer:
    def cluster(self, frame: FrameData, lane_box):
        _ = lane_box
        detection = Detection(
            detection_id=1,
            points=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            center=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            min_bound=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            max_bound=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )
        return ClusterResult(lane_points=frame.points, detections=[detection], metrics={"algorithm": "fake"})


class _FakeTracker:
    def __init__(self):
        self.track = Track(track_id=1, hit_count=5, age=5)
        self.seen_frame_ids: list[int] = []
        self.track.centers.append(np.array([0.95, 0.0, 0.0], dtype=np.float32))
        self.track.frame_ids.append(0)
        self.track.frame_timestamps_ns.append(200)
        self.track.local_points.append(np.array([[0.0, 0.0, 0.0]], dtype=np.float32))
        self.track.world_points.append(np.array([[0.95, 0.0, 0.0]], dtype=np.float32))
        self.track.bbox_extents.append(np.array([0.1, 0.1, 0.1], dtype=np.float32))

    def step(self, detections, frame_idx, frame_timestamp_ns):
        _ = detections
        _ = frame_timestamp_ns
        self.seen_frame_ids.append(int(frame_idx))
        return FrameTrackingState(
            frame_index=int(frame_idx),
            lane_points=np.zeros((0, 3), dtype=np.float32),
            detections=[],
            active_tracks=[],
            tracker_metrics={"assignment_method": "fake", "matched_count": 0},
            tracker_debug=FrameTrackerDebug(assignment_method="fake"),
        )

    def finalize(self):
        return {1: self.track}


class _FakeAccumulator:
    def accumulate(self, track: Track, lane_box):
        _ = track, lane_box
        return AggregateResult(
            track_id=1,
            points=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            selected_frame_ids=[0],
            status="saved",
            metrics={
                "registration_pairs": 1,
                "registration_accepted": 1,
                "registration_rejected": 0,
                "prepared_chunk_count": 1,
                "registration_wall_seconds": 0.0,
                "registration_cpu_seconds": 0.0,
                "fusion_core_wall_seconds": 0.2,
                "fusion_core_cpu_seconds": 0.1,
                "fusion_post_wall_seconds": 0.3,
                "fusion_post_cpu_seconds": 0.15,
                "fusion_total_wall_seconds": 0.5,
                "fusion_total_cpu_seconds": 0.25,
            },
        )


class _FakeStateAwareAccumulator:
    def accumulate(self, track: Track, lane_box):
        _ = lane_box
        metrics = {}
        if bool(track.state.get("articulated_vehicle")):
            metrics["articulated_vehicle"] = True
            metrics["articulated_component_track_ids"] = list(track.state.get("articulated_component_track_ids") or [])
            metrics["articulated_rear_gap_mean"] = float(track.state.get("articulated_rear_gap_mean", 0.0))
            metrics["articulated_rear_gap_std"] = float(track.state.get("articulated_rear_gap_std", 0.0))
            metrics["object_kind"] = str(track.state.get("object_kind") or "truck_with_trailer")
        return AggregateResult(
            track_id=track.track_id,
            points=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            selected_frame_ids=list(track.frame_ids),
            status="saved",
            metrics=metrics,
        )

    def merge_long_vehicle_aggregates(self, tracks: dict[int, Track], aggregate_results: list[AggregateResult], lane_box):
        _ = lane_box
        by_track_id = {int(result.track_id): result for result in aggregate_results}
        for track_id, track in tracks.items():
            if str(track.state.get("articulated_role") or "") != "lead":
                continue
            component_ids = [int(component_id) for component_id in track.state.get("articulated_component_track_ids", [])]
            if len(component_ids) < 2:
                continue
            lead_result = by_track_id[int(track_id)]
            lead_result.metrics["merged_post_aggregation"] = True
            lead_result.metrics["post_merge_component_ids"] = list(component_ids)
            lead_result.metrics["long_vehicle_component_count"] = len(component_ids)
            lead_result.metrics["long_vehicle_component_roles"] = ["lead", "rear"]
            for component_id in component_ids:
                if int(component_id) == int(track_id):
                    continue
                component_result = by_track_id[int(component_id)]
                component_result.status = "merged_into_long_vehicle_group"
                component_result.metrics["merged_post_aggregation"] = True
                component_result.metrics["merged_target_track_id"] = int(track_id)
                component_result.metrics["post_merge_component_ids"] = list(component_ids)
        return [by_track_id[track_id] for track_id in sorted(by_track_id)]


class _FakeWriter:
    def __init__(self, base: Path):
        self.base = base
        self.object_labels = None
        self.object_list_write_count = 0
        self.summary_write_count = 0
        self.track_write_count = 0
        self.tracker_debug_write_count = 0
        self.track_outcomes_write_count = 0
        self.class_stats_write_count = 0
        self.gt_matching_write_count = 0
        self.aggregate_write_intensity_flags: list[bool] = []
        self.aggregate_write_metrics: list[dict[str, object]] = []
        self.tracker_debug_states = None
        self.track_outcomes = None
        self.class_stats = None
        self.written_tracks = None
        self.written_aggregate_results = None
        self.gt_matches = None
        self.gt_unmatched_saved = None
        self.gt_unmatched_objects = None
        self.gt_summary = None

    def prepare_run_dir(self, config):
        _ = config
        path = self.base / "run"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_config_snapshot(self, run_dir, config):
        _ = config
        (run_dir / "config.snapshot.yaml").write_text("ok\n", encoding="utf-8")

    def write_aggregate(self, run_dir, result, save_intensity=False):
        self.aggregate_write_intensity_flags.append(bool(save_intensity))
        self.aggregate_write_metrics.append(dict(result.metrics))
        (run_dir / "aggregate.txt").write_text("saved\n", encoding="utf-8")

    def write_summary(self, run_dir, summary):
        self.summary_write_count += 1
        (run_dir / "summary.txt").write_text(str(summary.saved_aggregates), encoding="utf-8")

    def write_tracks(self, run_dir, tracks, aggregate_results):
        self.track_write_count += 1
        self.written_tracks = tracks
        self.written_aggregate_results = aggregate_results
        (run_dir / "tracks.txt").write_text("tracks\n", encoding="utf-8")

    def write_tracker_debug(self, run_dir, states):
        self.tracker_debug_write_count += 1
        self.tracker_debug_states = states
        (run_dir / "tracker_debug.txt").write_text(str(len(states)), encoding="utf-8")

    def write_track_outcomes(self, run_dir, track_outcomes):
        self.track_outcomes_write_count += 1
        self.track_outcomes = track_outcomes
        (run_dir / "track_outcomes.txt").write_text(str(len(track_outcomes)), encoding="utf-8")

    def write_class_stats(self, run_dir, class_stats):
        self.class_stats_write_count += 1
        self.class_stats = class_stats
        (run_dir / "class_stats.txt").write_text(str(class_stats), encoding="utf-8")

    def write_object_list(self, run_dir, object_labels):
        self.object_labels = object_labels
        self.object_list_write_count += 1
        (run_dir / "object_list_manifest.txt").write_text(str(sorted(object_labels)), encoding="utf-8")

    def write_gt_matching(self, run_dir, matches, unmatched_saved_tracks, unmatched_gt_objects, summary):
        self.gt_matching_write_count += 1
        self.gt_matches = matches
        self.gt_unmatched_saved = unmatched_saved_tracks
        self.gt_unmatched_objects = unmatched_gt_objects
        self.gt_summary = summary
        (run_dir / "gt_matching.txt").write_text(
            f"{len(matches)}/{len(unmatched_saved_tracks)}/{len(unmatched_gt_objects)}",
            encoding="utf-8",
        )


class _FakeViewer:
    def __init__(self):
        self.states = None
        self.aggregate_results = None
        self.track_outcomes = None
        self.articulated_merge_debug_events = None

    def replay(self, states, lane_box, aggregate_results, track_outcomes, articulated_merge_debug_events):
        _ = lane_box
        self.states = states
        self.aggregate_results = aggregate_results
        self.track_outcomes = track_outcomes
        self.articulated_merge_debug_events = articulated_merge_debug_events


class _FakeClassifier:
    backend = "pointnext"

    def __init__(self):
        self.seen_points: list[np.ndarray] = []

    def classify_points(self, points: np.ndarray) -> ClassificationPrediction:
        arr = np.asarray(points, dtype=np.float32)
        self.seen_points.append(arr.copy())
        return ClassificationPrediction(class_id=4, class_name="trailer", score=0.88)


class _FakeObjectReader:
    def iter_frames(self, input_paths: list[str]) -> list[FrameData]:
        _ = input_paths
        return [
            FrameData(
                frame_index=0,
                timestamp_ns=10,
                points=np.zeros((1, 3), dtype=np.float32),
                source_path="a.pb",
                object_labels=[
                    ObjectLabelData(
                        object_id=7,
                        timestamp_ns=100,
                        points=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
                        obj_class="car",
                        obj_class_score=0.9,
                        sensor_name="sensor_a",
                        frame_index=0,
                        source_path="a.pb",
                    ),
                    ObjectLabelData(
                        object_id=8,
                        timestamp_ns=150,
                        points=np.array([[0.2, 0.0, 0.0]], dtype=np.float32),
                        obj_class="van",
                        obj_class_score=0.7,
                        sensor_name="sensor_a",
                        frame_index=0,
                        source_path="a.pb",
                    ),
                    ObjectLabelData(
                        object_id=9,
                        timestamp_ns=101,
                        points=np.zeros((0, 3), dtype=np.float32),
                        obj_class="truck",
                        obj_class_score=0.8,
                        sensor_name="sensor_a",
                        frame_index=0,
                        source_path="a.pb",
                    ),
                ],
            ),
            FrameData(
                frame_index=1,
                timestamp_ns=11,
                points=np.zeros((1, 3), dtype=np.float32),
                source_path="b.pb",
                object_labels=[
                    ObjectLabelData(
                        object_id=7,
                        timestamp_ns=200,
                        points=np.array([[1.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=np.float32),
                        obj_class="car",
                        obj_class_score=0.95,
                        sensor_name="sensor_a",
                        frame_index=1,
                        source_path="b.pb",
                    ),
                    ObjectLabelData(
                        object_id=8,
                        timestamp_ns=150,
                        points=np.array([[2.0, 0.0, 0.0], [2.5, 0.0, 0.0]], dtype=np.float32),
                        obj_class="van",
                        obj_class_score=0.75,
                        sensor_name="sensor_a",
                        frame_index=1,
                        source_path="b.pb",
                    ),
                ],
            ),
        ]


def test_live_snapshot_tracker_states_keeps_only_latest_frame() -> None:
    states = [
        FrameTrackingState(frame_index=index, lane_points=np.zeros((0, 3), dtype=np.float32), detections=[], active_tracks=[])
        for index in range(5)
    ]

    snapshot_states = _live_snapshot_tracker_states(states)

    assert len(snapshot_states) == 1
    assert snapshot_states[0].frame_index == 4


def test_live_gt_match_label_history_scopes_labels_to_track_time_window() -> None:
    track = Track(
        track_id=1,
        centers=[np.array([0.0, 0.0, 0.0], dtype=np.float32)],
        frame_ids=[10],
        frame_timestamps_ns=[100_000_000_000],
    )
    old_label = ObjectLabelData(
        object_id=1,
        timestamp_ns=80_000_000_000,
        points=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        obj_class="car",
        sensor_name="sensor_a",
        source_path="live",
    )
    nearby_label = ObjectLabelData(
        object_id=1,
        timestamp_ns=104_000_000_000,
        points=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        obj_class="car",
        sensor_name="sensor_a",
        source_path="live",
    )
    future_label = ObjectLabelData(
        object_id=2,
        timestamp_ns=120_000_000_000,
        points=np.array([[2.0, 0.0, 0.0]], dtype=np.float32),
        obj_class="van",
        sensor_name="sensor_a",
        source_path="live",
    )

    scoped = _live_gt_match_label_history(
        {1: track},
        {1: [old_label, nearby_label], 2: [future_label]},
        margin_sec=5.0,
    )

    assert scoped == {1: [nearby_label]}


class _InterruptingLiveReader:
    def __init__(self, frames: list[FrameData], pending_object_labels: list[ObjectLabelData] | None = None):
        self._frames = list(frames)
        self._pending_object_labels = list(pending_object_labels or [])
        self.close_calls = 0
        self.drain_calls: list[int] = []
        self.drain_max_timestamp_calls: list[int | None] = []
        self._status = {
            "reader_state": "waiting_for_raw",
            "mqtt_connected": True,
            "mqtt_messages_received": 1 if self._pending_object_labels else 0,
            "mqtt_snapshots_received": 1 if self._pending_object_labels else 0,
            "pending_snapshot_count": 1 if self._pending_object_labels else 0,
            "pending_label_count": len(self._pending_object_labels),
            "raw_frames_received": 0,
            "waiting_for_first_raw_frame": not bool(self._frames),
        }

    def iter_frames(self, input_paths: list[str]):
        _ = input_paths
        for index, frame in enumerate(self._frames, start=1):
            self._status.update(
                reader_state="streaming",
                raw_frames_received=int(index),
                waiting_for_first_raw_frame=False,
                last_raw_frame_index=int(frame.frame_index),
                last_raw_frame_timestamp_ns=int(frame.timestamp_ns),
            )
            yield frame
        raise KeyboardInterrupt()

    def close(self) -> None:
        self.close_calls += 1
        self._status["reader_state"] = "stopped"
        self._status["mqtt_connected"] = False

    def drain_pending_object_labels(self, frame_index: int, max_timestamp_ns: int | None = None) -> list[ObjectLabelData]:
        self.drain_calls.append(int(frame_index))
        self.drain_max_timestamp_calls.append(None if max_timestamp_ns is None else int(max_timestamp_ns))
        self._status["pending_snapshot_count"] = 0
        self._status["pending_label_count"] = 0
        return [
            ObjectLabelData(
                object_id=int(label.object_id),
                timestamp_ns=int(label.timestamp_ns),
                points=np.asarray(label.points, dtype=np.float32).copy(),
                obj_class=str(label.obj_class),
                obj_class_score=float(label.obj_class_score),
                sensor_name=str(label.sensor_name),
                frame_index=int(frame_index),
                source_path=str(label.source_path),
            )
            for label in self._pending_object_labels
            if max_timestamp_ns is None or int(label.timestamp_ns) <= int(max_timestamp_ns)
        ]

    def snapshot_pending_object_labels(self, frame_index: int) -> list[ObjectLabelData]:
        return [
            ObjectLabelData(
                object_id=int(label.object_id),
                timestamp_ns=int(label.timestamp_ns),
                points=np.asarray(label.points, dtype=np.float32).copy(),
                obj_class=str(label.obj_class),
                obj_class_score=float(label.obj_class_score),
                sensor_name=str(label.sensor_name),
                frame_index=int(frame_index),
                source_path=str(label.source_path),
            )
            for label in self._pending_object_labels
        ]

    def status_snapshot(self) -> dict[str, object]:
        return dict(self._status)


def _box_points(center: np.ndarray, *, width: float, length: float, height: float) -> np.ndarray:
    signs = np.asarray(
        [
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    scale = np.asarray([width * 0.5, length * 0.5, height * 0.5], dtype=np.float32)
    return center.astype(np.float32) + signs * scale


def _articulated_track(
    track_id: int,
    frame_ids: list[int],
    longitudinal_centers: list[float],
    *,
    lateral_center: float = 0.0,
    vertical_center: float = 1.0,
    width: float = 0.9,
    length: float = 3.4,
    height: float = 1.4,
) -> Track:
    track = Track(track_id=track_id, age=len(frame_ids), hit_count=len(frame_ids), ended_by_missed=True, source_track_ids=[track_id])
    extent = np.asarray([width, length, height], dtype=np.float32)
    for frame_id, longitudinal_center in zip(frame_ids, longitudinal_centers):
        center = np.asarray([lateral_center, longitudinal_center, vertical_center], dtype=np.float32)
        points = _box_points(center, width=width, length=length, height=height)
        track.add_observation(center, points, frame_id, frame_id * 1_000_000, extent)
    track.age = max(track.age, track.last_frame - track.first_frame + 1)
    track.hit_count = len(track.frame_ids)
    return track


class _FakeArticulatedTracker:
    def __init__(self):
        self.seen_frame_ids: list[int] = []
        self.front = _articulated_track(11, [0, 1, 2, 3], [10.0, 11.0, 12.0, 13.0], lateral_center=0.0)
        self.rear = _articulated_track(12, [0, 1, 2, 3], [6.8, 7.8, 8.8, 9.8], lateral_center=0.1)

    def step(self, detections, frame_idx, frame_timestamp_ns):
        _ = detections
        _ = frame_timestamp_ns
        self.seen_frame_ids.append(int(frame_idx))
        active_tracks: list[ActiveTrackState] = []
        for track in (self.front, self.rear):
            if int(frame_idx) not in track.frame_ids:
                continue
            observation_index = track.frame_ids.index(int(frame_idx))
            active_tracks.append(
                ActiveTrackState(
                    track_id=int(track.track_id),
                    points=np.asarray(track.world_points[observation_index], dtype=np.float32).copy(),
                    center=np.asarray(track.centers[observation_index], dtype=np.float32).copy(),
                    intensity=None if observation_index >= len(track.world_intensity) else track.world_intensity[observation_index],
                )
            )
        return FrameTrackingState(
            frame_index=int(frame_idx),
            lane_points=np.zeros((0, 3), dtype=np.float32),
            detections=[],
            active_tracks=active_tracks,
            tracker_metrics={"assignment_method": "fake", "matched_count": 0},
            tracker_debug=FrameTrackerDebug(assignment_method="fake"),
        )

    def finalize(self):
        return {self.front.track_id: self.front, self.rear.track_id: self.rear}


def test_run_pipeline_orchestrates_dependencies(monkeypatch, tmp_path: Path) -> None:
    config = PipelineConfig(
        input=InputConfig(paths=["ignored_a.pb", "ignored_b.pb"]),
        preprocessing=PreprocessingConfig(lane_box=[-1, 1, -1, 1, -1, 1]),
        clustering=ClusteringConfig(),
        tracking=TrackingConfig(),
        aggregation=AggregationConfig(),
        output=OutputConfig(root_dir=str(tmp_path)),
        visualization=VisualizationConfig(),
    )

    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_reader", lambda cfg: _FakeReader())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_clusterer", lambda cfg: _FakeClusterer())
    fake_tracker = _FakeTracker()
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_tracker", lambda cfg: fake_tracker)
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_track_postprocessors", lambda cfg: [])
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_accumulator", lambda cfg: _FakeAccumulator())
    fake_writer = _FakeWriter(tmp_path)
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_artifact_writer", lambda cfg, root: fake_writer)

    summary = run_pipeline(config, tmp_path)

    assert summary.frame_count == 2
    assert summary.saved_aggregates == 1
    assert summary.input_paths == ["ignored_a.pb", "ignored_b.pb"]
    assert fake_tracker.seen_frame_ids == [0, 1]
    assert fake_writer.aggregate_write_intensity_flags == [False]
    assert (tmp_path / "run" / "summary.txt").exists()
    assert (tmp_path / "run" / "tracker_debug.txt").exists()
    assert (tmp_path / "run" / "track_outcomes.txt").exists()
    assert (tmp_path / "run" / "class_stats.txt").exists()
    assert len(fake_writer.tracker_debug_states) == 2
    assert isinstance(fake_writer.track_outcomes[1], TrackOutcomeDebug)
    assert fake_writer.track_outcomes[1].status == "saved"
    assert summary.predicted_class_counts == {}
    assert summary.gt_class_counts == {}
    assert summary.matched_gt_class_counts == {}
    assert summary.class_comparison_count == 0
    assert summary.class_match_count == 0
    assert summary.class_mismatch_count == 0
    assert summary.class_count_rows == []
    assert summary.performance is not None
    assert summary.performance.aggregation_components["registration"].wall_seconds == 0.0
    assert summary.performance.aggregation_components["fusion_core"].wall_seconds == 0.2
    assert summary.performance.aggregation_components["fusion_post"].wall_seconds == 0.3
    assert summary.performance.aggregation_components["fusion_total"].wall_seconds == 0.5
    assert summary.performance.aggregation_components["fusion_total"].call_count == 1
    assert summary.performance.total_hz == derive_hz(summary.frame_count, summary.performance.total_wall_seconds)
    assert summary.performance.compute_hz == derive_hz(summary.frame_count, summary.performance.compute_wall_seconds)


def test_run_pipeline_exports_latest_object_list_observation(monkeypatch, tmp_path: Path) -> None:
    config = PipelineConfig(
        input=InputConfig(paths=["ignored_a.pb"]),
        preprocessing=PreprocessingConfig(lane_box=[-1, 1, -1, 1, -1, 1]),
        clustering=ClusteringConfig(),
        tracking=TrackingConfig(),
        aggregation=AggregationConfig(),
        output=OutputConfig(root_dir=str(tmp_path)),
        visualization=VisualizationConfig(),
    )

    fake_writer = _FakeWriter(tmp_path)
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_reader", lambda cfg: _FakeObjectReader())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_clusterer", lambda cfg: _FakeClusterer())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_tracker", lambda cfg: _FakeTracker())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_track_postprocessors", lambda cfg: [])
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_accumulator", lambda cfg: _FakeAccumulator())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_artifact_writer", lambda cfg, root: fake_writer)

    summary = run_pipeline(config, tmp_path)

    assert summary.object_list_exported_count == 2
    assert summary.object_list_seen_ids == 3
    assert summary.object_list_skipped_empty == 1
    assert summary.gt_match_saved_track_count == 1
    assert summary.gt_match_matched_count == 1
    assert summary.gt_match_unmatched_gt_count == 1
    assert set(fake_writer.object_labels.keys()) == {7, 8}
    assert fake_writer.object_labels[7].timestamp_ns == 200
    assert fake_writer.object_labels[7].frame_index == 1
    assert fake_writer.object_labels[8].timestamp_ns == 150
    assert fake_writer.object_labels[8].frame_index == 1
    assert len(fake_writer.gt_matches) == 1
    assert fake_writer.gt_matches[0].track_id == 1
    assert fake_writer.gt_matches[0].gt_object_id == 7
    assert fake_writer.gt_matches[0].gt_obj_class == "car"
    assert len(fake_writer.gt_unmatched_objects) == 1
    assert fake_writer.gt_unmatched_objects[0].gt_object_id == 8
    assert fake_writer.gt_unmatched_objects[0].gt_obj_class == "van"
    assert fake_writer.aggregate_write_metrics[0]["gt_obj_class"] == "car"
    assert fake_writer.aggregate_write_metrics[0]["gt_obj_class_score"] == 0.95
    assert summary.gt_class_counts == {"car": 1, "van": 1}
    assert summary.matched_gt_class_counts == {"car": 1}
    assert summary.class_comparison_count == 0
    assert summary.class_match_count == 0
    assert summary.class_mismatch_count == 0
    assert summary.class_count_rows == [
        {"class_name": "car", "predicted_count": 0, "gt_match_count": 1},
        {"class_name": "TOTAL", "predicted_count": 0, "gt_match_count": 1},
    ]
    assert fake_writer.class_stats["gt_class_counts"] == {"car": 1, "van": 1}
    assert fake_writer.class_stats["matched_gt_class_counts"] == {"car": 1}
    assert fake_writer.class_stats["class_comparison_count"] == 0
    assert fake_writer.class_stats["class_match_count"] == 0
    assert fake_writer.class_stats["class_mismatch_count"] == 0


def test_run_pipeline_passes_aggregate_intensity_flag_to_writer(monkeypatch, tmp_path: Path) -> None:
    config = PipelineConfig(
        input=InputConfig(paths=["ignored_a.pb"]),
        preprocessing=PreprocessingConfig(lane_box=[-1, 1, -1, 1, -1, 1]),
        clustering=ClusteringConfig(),
        tracking=TrackingConfig(),
        aggregation=AggregationConfig(),
        output=OutputConfig(root_dir=str(tmp_path), save_aggregate_intensity=True),
        visualization=VisualizationConfig(),
    )

    fake_writer = _FakeWriter(tmp_path)
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_reader", lambda cfg: _FakeReader())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_clusterer", lambda cfg: _FakeClusterer())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_tracker", lambda cfg: _FakeTracker())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_track_postprocessors", lambda cfg: [])
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_accumulator", lambda cfg: _FakeAccumulator())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_artifact_writer", lambda cfg, root: fake_writer)

    summary = run_pipeline(config, tmp_path)

    assert fake_writer.aggregate_write_intensity_flags == [True]


def test_run_pipeline_records_raw_frames_inside_dataset_root(monkeypatch, tmp_path: Path) -> None:
    config = PipelineConfig(
        input=InputConfig(paths=["ignored_a.pb", "ignored_b.pb"]),
        preprocessing=PreprocessingConfig(lane_box=[-1, 1, -1, 1, -1, 1]),
        clustering=ClusteringConfig(),
        tracking=TrackingConfig(),
        aggregation=AggregationConfig(),
        output=OutputConfig(mode="dataset", dataset_root_dir=str(tmp_path / "new-config-dataset"), raw_frames_enabled=True),
        visualization=VisualizationConfig(),
    )

    class _DatasetLikeFakeWriter(_FakeWriter):
        def prepare_run_dir(self, config):
            path = Path(config.output.dataset_root_dir)
            path.mkdir(parents=True, exist_ok=True)
            self._run_id = "test_run"
            return path

    fake_writer = _DatasetLikeFakeWriter(tmp_path)
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_reader", lambda cfg: _FakeReader())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_clusterer", lambda cfg: _FakeClusterer())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_tracker", lambda cfg: _FakeTracker())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_track_postprocessors", lambda cfg: [])
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_accumulator", lambda cfg: _FakeAccumulator())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_artifact_writer", lambda cfg, root: fake_writer)

    summary = run_pipeline(config, tmp_path)

    segment_dir = tmp_path / "new-config-dataset" / "_raw_frames" / "test_run"
    loaded = list(FrameSegmentReader().iter_frames([str(segment_dir)]))
    assert summary.output_dir == str(tmp_path / "new-config-dataset")
    assert (segment_dir / "manifest.jsonl").exists()
    assert (segment_dir / "segment.json").exists()
    assert len(loaded) == 1
    assert loaded[0].frame_index == 0
    assert loaded[0].timestamp_ns == 200
    assert loaded[0].source_path == "track://1/chunk_quality_kept"
    assert np.array_equal(loaded[0].points, np.array([[0.95, 0.0, 0.0]], dtype=np.float32))


def test_run_pipeline_continues_when_raw_frame_writer_cannot_start(monkeypatch, tmp_path: Path) -> None:
    config = PipelineConfig(
        input=InputConfig(paths=["ignored_a.pb", "ignored_b.pb"]),
        preprocessing=PreprocessingConfig(lane_box=[-1, 1, -1, 1, -1, 1]),
        clustering=ClusteringConfig(),
        tracking=TrackingConfig(),
        aggregation=AggregationConfig(),
        output=OutputConfig(mode="dataset", dataset_root_dir=str(tmp_path / "new-config-dataset"), raw_frames_enabled=True),
        visualization=VisualizationConfig(),
    )

    class _DatasetLikeFakeWriter(_FakeWriter):
        def prepare_run_dir(self, config):
            path = Path(config.output.dataset_root_dir)
            path.mkdir(parents=True, exist_ok=True)
            self._run_id = "test_run"
            return path

    class _PermissionDeniedFrameSegmentWriter:
        def __init__(self, root):
            raise PermissionError(13, "Permission denied", str(root))

    fake_writer = _DatasetLikeFakeWriter(tmp_path)
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.FrameSegmentWriter", _PermissionDeniedFrameSegmentWriter)
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_reader", lambda cfg: _FakeReader())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_clusterer", lambda cfg: _FakeClusterer())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_tracker", lambda cfg: _FakeTracker())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_track_postprocessors", lambda cfg: [])
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_accumulator", lambda cfg: _FakeAccumulator())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_artifact_writer", lambda cfg, root: fake_writer)

    summary = run_pipeline(config, tmp_path)

    assert summary.output_dir == str(tmp_path / "new-config-dataset")
    assert summary.saved_aggregates == 1
    assert fake_writer.summary_write_count > 0
    assert not (tmp_path / "new-config-dataset" / "_raw_frames").exists()


def test_run_pipeline_merges_articulated_vehicle_tracks(monkeypatch, tmp_path: Path) -> None:
    config = PipelineConfig(
        input=InputConfig(paths=["ignored_a.pb", "ignored_b.pb", "ignored_c.pb", "ignored_d.pb"]),
        preprocessing=PreprocessingConfig(lane_box=[-1, 1, -1, 1, -1, 1]),
        clustering=ClusteringConfig(),
        tracking=TrackingConfig(),
        aggregation=AggregationConfig(frame_selection_line_axis="y"),
        postprocessing=PostprocessingConfig(enable_articulated_vehicle_merge=True, enable_track_quality_scoring=True),
        output=OutputConfig(root_dir=str(tmp_path)),
        visualization=VisualizationConfig(),
    )

    fake_writer = _FakeWriter(tmp_path)
    fake_tracker = _FakeArticulatedTracker()
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_reader", lambda cfg: _FakeReader())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_clusterer", lambda cfg: _FakeClusterer())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_tracker", lambda cfg: fake_tracker)
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_accumulator", lambda cfg: _FakeStateAwareAccumulator())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_artifact_writer", lambda cfg, root: fake_writer)

    summary = run_pipeline(config, tmp_path)

    assert summary.finished_track_count == 2
    assert summary.saved_aggregates == 1
    assert summary.postprocessing_methods == ["articulated_vehicle_merge", "track_quality_scoring"]
    assert summary.articulated_vehicle_pair_count == 1
    assert summary.articulated_vehicle_track_count == 2
    assert summary.articulated_vehicle_merged_component_count == 1
    assert summary.articulated_vehicle_saved_count == 1
    assert fake_tracker.seen_frame_ids == [0, 1, 2, 3]
    assert set(fake_writer.written_tracks.keys()) == {11, 12}
    lead_track = fake_writer.written_tracks[11]
    rear_track = fake_writer.written_tracks[12]
    assert lead_track.state["articulated_vehicle"] is True
    assert rear_track.state["articulated_vehicle"] is True
    assert lead_track.state["articulated_role"] == "lead"
    assert rear_track.state["articulated_role"] == "rear"
    assert lead_track.state["articulated_component_track_ids"] == [11, 12]
    assert rear_track.state["articulated_component_track_ids"] == [11, 12]
    assert lead_track.quality_metrics["is_articulated_vehicle"] is True
    assert rear_track.quality_metrics["is_articulated_vehicle"] is True
    assert lead_track.quality_metrics["object_kind"] == "truck_with_trailer"
    assert rear_track.quality_metrics["object_kind"] == "truck_with_trailer"
    assert len(fake_writer.written_aggregate_results) == 2
    lead_result = next(result for result in fake_writer.written_aggregate_results if int(result.track_id) == 11)
    rear_result = next(result for result in fake_writer.written_aggregate_results if int(result.track_id) == 12)
    assert lead_result.status == "saved"
    assert rear_result.status == "merged_into_long_vehicle_group"


def test_incremental_live_pipeline_merges_articulated_vehicle_tracks(monkeypatch, tmp_path: Path) -> None:
    config = PipelineConfig(
        input=InputConfig(
            paths=["qb2_live://class_qb2@10.16.3.160"],
            format="qb2_live",
            qb2_live=QB2LiveInputConfig(
                sensor_name="class_qb2",
                ip="10.16.3.160",
                api_key="secret",
                mqtt=QB2LiveMQTTConfig(host="10.16.3.111", topic="blickfeld/states_160"),
            ),
        ),
        preprocessing=PreprocessingConfig(lane_box=[-1, 1, -1, 1, -1, 1]),
        clustering=ClusteringConfig(),
        tracking=TrackingConfig(),
        aggregation=AggregationConfig(frame_selection_line_axis="y"),
        postprocessing=PostprocessingConfig(enable_articulated_vehicle_merge=True, enable_track_quality_scoring=True),
        output=OutputConfig(root_dir=str(tmp_path), final_full_recompute=False, live_artifact_flush_interval_sec=0.0),
        visualization=VisualizationConfig(),
    )

    class _IncrementalArticulatedTracker(_FakeArticulatedTracker):
        def __init__(self):
            super().__init__()
            self.finished_tracks: dict[int, Track] = {}

        def step(self, detections, frame_idx, frame_timestamp_ns):
            state = super().step(detections, frame_idx, frame_timestamp_ns)
            if int(frame_idx) >= 0:
                self.finished_tracks[int(self.front.track_id)] = self.front
            if int(frame_idx) >= 1:
                self.finished_tracks[int(self.rear.track_id)] = self.rear
            return state

    reader = _InterruptingLiveReader(
        frames=[
            FrameData(frame_index=index, timestamp_ns=index + 1, points=np.zeros((1, 3), dtype=np.float32))
            for index in range(4)
        ]
    )
    fake_writer = _FakeWriter(tmp_path)
    fake_tracker = _IncrementalArticulatedTracker()
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_reader", lambda cfg: reader)
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_clusterer", lambda cfg: _FakeClusterer())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_tracker", lambda cfg: fake_tracker)
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_accumulator", lambda cfg: _FakeStateAwareAccumulator())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_artifact_writer", lambda cfg, root: fake_writer)

    summary = run_pipeline(config, tmp_path)

    assert summary.saved_aggregates == 1
    assert summary.articulated_vehicle_pair_count == 1
    assert summary.articulated_vehicle_merged_component_count == 1
    assert fake_tracker.finished_tracks == {}
    assert set(fake_writer.written_tracks.keys()) == {11, 12}
    assert fake_writer.written_tracks[11].state["articulated_role"] == "lead"
    assert fake_writer.written_tracks[12].state["articulated_role"] == "rear"
    lead_result = next(result for result in fake_writer.written_aggregate_results if int(result.track_id) == 11)
    rear_result = next(result for result in fake_writer.written_aggregate_results if int(result.track_id) == 12)
    assert lead_result.status == "saved"
    assert rear_result.status == "merged_into_long_vehicle_group"
    assert lead_result.metrics["post_merge_component_ids"] == [11, 12]
    assert rear_result.metrics["merged_target_track_id"] == 11


def test_run_pipeline_propagates_classification_to_results_and_track_outcomes(monkeypatch, tmp_path: Path) -> None:
    config = PipelineConfig(
        input=InputConfig(paths=["ignored_a.pb"]),
        preprocessing=PreprocessingConfig(lane_box=[-1, 1, -1, 1, -1, 1]),
        clustering=ClusteringConfig(),
        tracking=TrackingConfig(),
        aggregation=AggregationConfig(),
        output=OutputConfig(root_dir=str(tmp_path)),
        visualization=VisualizationConfig(),
    )

    fake_writer = _FakeWriter(tmp_path)
    fake_classifier = _FakeClassifier()
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_reader", lambda cfg: _FakeReader())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_clusterer", lambda cfg: _FakeClusterer())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_tracker", lambda cfg: _FakeTracker())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_track_postprocessors", lambda cfg: [])
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_accumulator", lambda cfg: _FakeAccumulator())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_classifier", lambda cfg: fake_classifier)
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_artifact_writer", lambda cfg, root: fake_writer)

    summary = run_pipeline(config, tmp_path)

    assert len(fake_classifier.seen_points) == 1
    result = fake_writer.written_aggregate_results[0]
    assert result.metrics["predicted_class_id"] == 4
    assert result.metrics["predicted_class_name"] == "trailer"
    assert result.metrics["predicted_class_score"] == 0.88
    assert result.metrics["classification_backend"] == "pointnext"
    assert result.metrics["classification_point_source"] == "result_points"
    assert result.metrics["classification_input_point_count"] == 1
    outcome = fake_writer.track_outcomes[1]
    assert outcome.predicted_class_id == 4
    assert outcome.predicted_class_name == "trailer"
    assert outcome.predicted_class_score == 0.88
    assert outcome.classification_backend == "pointnext"
    assert outcome.classification_point_source == "result_points"
    assert outcome.classification_input_point_count == 1
    assert summary.predicted_class_counts == {"trailer": 1}
    assert summary.matched_gt_class_counts == {}
    assert summary.class_comparison_count == 0
    assert summary.class_match_count == 0
    assert summary.class_mismatch_count == 0
    assert summary.class_count_rows == [
        {"class_name": "trailer", "predicted_count": 1, "gt_match_count": 0},
        {"class_name": "TOTAL", "predicted_count": 1, "gt_match_count": 0},
    ]
    assert fake_writer.class_stats["predicted_class_counts"] == {"trailer": 1}
    assert fake_writer.class_stats["class_comparison_count"] == 0
    assert fake_writer.class_stats["class_match_count"] == 0
    assert fake_writer.class_stats["class_mismatch_count"] == 0
    assert fake_writer.class_stats["class_count_rows"] == [
        {"class_name": "trailer", "predicted_count": 1, "gt_match_count": 0},
        {"class_name": "TOTAL", "predicted_count": 1, "gt_match_count": 0},
    ]


def test_run_pipeline_finalizes_qb2_live_run_after_keyboard_interrupt(monkeypatch, tmp_path: Path) -> None:
    config = PipelineConfig(
        input=InputConfig(
            paths=["qb2_live://class_qb2@10.16.3.160"],
            format="qb2_live",
            qb2_live=QB2LiveInputConfig(
                sensor_name="class_qb2",
                ip="10.16.3.160",
                api_key="secret",
                mqtt=QB2LiveMQTTConfig(host="10.16.3.111", topic="blickfeld/states_160"),
            ),
        ),
        preprocessing=PreprocessingConfig(lane_box=[-1, 1, -1, 1, -1, 1]),
        clustering=ClusteringConfig(),
        tracking=TrackingConfig(),
        aggregation=AggregationConfig(),
        output=OutputConfig(root_dir=str(tmp_path)),
        visualization=VisualizationConfig(),
    )

    pending_object = ObjectLabelData(
        object_id=77,
        timestamp_ns=150,
        points=np.array([[2.0, 0.0, 0.0]], dtype=np.float32),
        obj_class="car",
        obj_class_score=0.9,
        sensor_name="class_qb2",
        frame_index=-1,
        source_path=config.input.paths[0],
    )
    future_pending_object = ObjectLabelData(
        object_id=78,
        timestamp_ns=250,
        points=np.array([[3.0, 0.0, 0.0]], dtype=np.float32),
        obj_class="van",
        obj_class_score=0.8,
        sensor_name="class_qb2",
        frame_index=-1,
        source_path=config.input.paths[0],
    )
    reader = _InterruptingLiveReader(
        frames=[
            FrameData(
                frame_index=0,
                timestamp_ns=200,
                points=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
                source_path=config.input.paths[0],
            )
        ],
        pending_object_labels=[pending_object, future_pending_object],
    )
    fake_writer = _FakeWriter(tmp_path)
    fake_tracker = _FakeTracker()
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_reader", lambda cfg: reader)
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_clusterer", lambda cfg: _FakeClusterer())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_tracker", lambda cfg: fake_tracker)
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_track_postprocessors", lambda cfg: [])
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_accumulator", lambda cfg: _FakeAccumulator())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_artifact_writer", lambda cfg, root: fake_writer)

    summary = run_pipeline(config, tmp_path)

    assert summary.frame_count == 1
    assert summary.input_path == "qb2_live://class_qb2@10.16.3.160"
    assert summary.object_list_exported_count == 2
    assert fake_tracker.seen_frame_ids == [0]
    assert reader.close_calls >= 1
    assert reader.drain_calls == [0]
    assert reader.drain_max_timestamp_calls == [200]
    assert set(fake_writer.object_labels) == {77, 78}
    assert fake_writer.object_labels[77].frame_index == 0
    assert fake_writer.object_labels[77].source_path == config.input.paths[0]
    assert fake_writer.object_labels[78].frame_index == 0
    live_status_path = tmp_path / "run" / "live_status.json"
    assert live_status_path.exists()
    live_status = json.loads(live_status_path.read_text(encoding="utf-8"))
    assert live_status["pipeline_phase"] == "completed"
    assert live_status["processed_frames"] == 1
    assert live_status["object_list_exported_count"] == 2
    assert "processing_total_hz" in live_status
    assert float(live_status["processing_total_hz"]) >= 0.0
    assert live_status["output_dir"] == str(tmp_path / "run")
    assert live_status["status_file"] == str(live_status_path)
    assert live_status["reader"]["raw_frames_received"] == 1
    assert live_status["reader"]["mqtt_messages_received"] == 1
    assert live_status["reader"]["reader_state"] == "stopped"


def test_run_pipeline_writes_object_list_live_for_qb2_live(monkeypatch, tmp_path: Path) -> None:
    config = PipelineConfig(
        input=InputConfig(
            paths=["qb2_live://class_qb2@10.16.3.160"],
            format="qb2_live",
            qb2_live=QB2LiveInputConfig(
                sensor_name="class_qb2",
                ip="10.16.3.160",
                api_key="secret",
                mqtt=QB2LiveMQTTConfig(host="10.16.3.111", topic="blickfeld/states_160"),
            ),
        ),
        preprocessing=PreprocessingConfig(lane_box=[-1, 1, -1, 1, -1, 1]),
        clustering=ClusteringConfig(),
        tracking=TrackingConfig(),
        aggregation=AggregationConfig(),
        output=OutputConfig(root_dir=str(tmp_path)),
        visualization=VisualizationConfig(),
    )

    live_object = ObjectLabelData(
        object_id=42,
        timestamp_ns=200,
        points=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        obj_class="car",
        obj_class_score=0.95,
        sensor_name="class_qb2",
        frame_index=0,
        source_path=config.input.paths[0],
    )
    reader = _InterruptingLiveReader(
        frames=[
            FrameData(
                frame_index=0,
                timestamp_ns=200,
                points=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
                source_path=config.input.paths[0],
                object_labels=[live_object],
            )
        ]
    )
    fake_writer = _FakeWriter(tmp_path)
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_reader", lambda cfg: reader)
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_clusterer", lambda cfg: _FakeClusterer())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_tracker", lambda cfg: _FakeTracker())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_track_postprocessors", lambda cfg: [])
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_accumulator", lambda cfg: _FakeAccumulator())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_artifact_writer", lambda cfg, root: fake_writer)

    summary = run_pipeline(config, tmp_path)

    assert summary.object_list_exported_count == 1
    assert fake_writer.object_list_write_count >= 2
    assert set(fake_writer.object_labels) == {42}
    live_status = json.loads((tmp_path / "run" / "live_status.json").read_text(encoding="utf-8"))
    assert live_status["live_object_list_write_count"] >= 2
    assert live_status["object_list_manifest_path"] == str(tmp_path / "run" / "object_list" / "manifest.jsonl")
    assert live_status["last_live_object_list_write_unix_sec"] is not None


def test_run_pipeline_writes_live_snapshot_artifacts_for_qb2_live(monkeypatch, tmp_path: Path) -> None:
    config = PipelineConfig(
        input=InputConfig(
            paths=["qb2_live://class_qb2@10.16.3.160"],
            format="qb2_live",
            qb2_live=QB2LiveInputConfig(
                sensor_name="class_qb2",
                ip="10.16.3.160",
                api_key="secret",
                mqtt=QB2LiveMQTTConfig(host="10.16.3.111", topic="blickfeld/states_160"),
            ),
        ),
        preprocessing=PreprocessingConfig(lane_box=[-1, 1, -1, 1, -1, 1]),
        clustering=ClusteringConfig(),
        tracking=TrackingConfig(),
        aggregation=AggregationConfig(),
        output=OutputConfig(root_dir=str(tmp_path)),
        visualization=VisualizationConfig(),
    )

    reader = _InterruptingLiveReader(
        frames=[
            FrameData(
                frame_index=0,
                timestamp_ns=200,
                points=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
                source_path=config.input.paths[0],
            )
        ]
    )
    fake_writer = _FakeWriter(tmp_path)
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_reader", lambda cfg: reader)
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_clusterer", lambda cfg: _FakeClusterer())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_tracker", lambda cfg: _FakeTracker())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_track_postprocessors", lambda cfg: [])
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_accumulator", lambda cfg: _FakeAccumulator())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_artifact_writer", lambda cfg, root: fake_writer)

    run_pipeline(config, tmp_path)

    assert fake_writer.aggregate_write_intensity_flags
    assert len(fake_writer.aggregate_write_intensity_flags) >= 2
    assert fake_writer.summary_write_count >= 2
    assert fake_writer.track_write_count >= 2
    assert fake_writer.tracker_debug_write_count >= 2
    assert fake_writer.track_outcomes_write_count >= 2
    assert fake_writer.class_stats_write_count >= 2
    assert fake_writer.gt_matching_write_count >= 2
    assert (tmp_path / "run" / "aggregate.txt").exists()
    assert (tmp_path / "run" / "summary.txt").exists()
    assert (tmp_path / "run" / "tracks.txt").exists()
    assert (tmp_path / "run" / "tracker_debug.txt").exists()
    assert (tmp_path / "run" / "track_outcomes.txt").exists()
    assert (tmp_path / "run" / "class_stats.txt").exists()
    assert (tmp_path / "run" / "gt_matching.txt").exists()
    live_status = json.loads((tmp_path / "run" / "live_status.json").read_text(encoding="utf-8"))
    assert live_status["live_artifact_dir"] == str(tmp_path / "run")
    assert live_status["live_artifact_write_count"] >= 1
    assert live_status["last_live_artifact_write_unix_sec"] is not None


def test_run_pipeline_writes_live_dataset_artifacts_for_qb2_live(monkeypatch, tmp_path: Path) -> None:
    config = PipelineConfig(
        input=InputConfig(
            paths=["qb2_live://class_qb2@10.16.3.160"],
            format="qb2_live",
            qb2_live=QB2LiveInputConfig(
                sensor_name="class_qb2",
                ip="10.16.3.160",
                api_key="secret",
                mqtt=QB2LiveMQTTConfig(host="10.16.3.111", topic="blickfeld/states_160"),
            ),
        ),
        preprocessing=PreprocessingConfig(lane_box=[-1, 1, -1, 1, -1, 1]),
        clustering=ClusteringConfig(),
        tracking=TrackingConfig(),
        aggregation=AggregationConfig(),
        output=OutputConfig(mode="dataset", dataset_root_dir=str(tmp_path / "dataset"), root_dir=str(tmp_path / "runs_unused")),
        visualization=VisualizationConfig(),
    )

    live_object = ObjectLabelData(
        object_id=42,
        timestamp_ns=200,
        points=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        obj_class="car",
        obj_class_score=0.95,
        sensor_name="class_qb2",
        frame_index=0,
        source_path=config.input.paths[0],
    )
    reader = _InterruptingLiveReader(
        frames=[
            FrameData(
                frame_index=0,
                timestamp_ns=200,
                points=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
                source_path=config.input.paths[0],
                object_labels=[live_object],
            )
        ]
    )
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_reader", lambda cfg: reader)
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_clusterer", lambda cfg: _FakeClusterer())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_tracker", lambda cfg: _FakeTracker())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_track_postprocessors", lambda cfg: [])
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_accumulator", lambda cfg: _FakeAccumulator())

    summary = run_pipeline(config, tmp_path)

    dataset_root = tmp_path / "dataset"
    assert summary.output_mode == "dataset"
    assert summary.output_dir == str(dataset_root)
    assert not (tmp_path / "runs_unused").exists()
    assert (dataset_root / "car" / "1970-01-01" / "gt-pred-different" / "gt").exists()
    assert (dataset_root / "car" / "1970-01-01" / "gt-pred-different" / "pred").exists()
    active_status_dirs = sorted((dataset_root / "_stats" / "_active").iterdir())
    assert len(active_status_dirs) == 1
    active_status_path = active_status_dirs[0] / "live_status.json"
    assert active_status_path.exists()
    live_status = json.loads(active_status_path.read_text(encoding="utf-8"))
    assert live_status["output_dir"] == str(dataset_root)
    day_stats_dirs = sorted((dataset_root / "_stats" / "1970-01-01").iterdir())
    assert len(day_stats_dirs) == 1
    assert (day_stats_dirs[0] / "live_status.json").exists()


def test_run_pipeline_writes_live_dataset_artifacts_for_qb2_live_without_final_recompute(monkeypatch, tmp_path: Path) -> None:
    config = PipelineConfig(
        input=InputConfig(
            paths=["qb2_live://class_qb2@10.16.3.160"],
            format="qb2_live",
            qb2_live=QB2LiveInputConfig(
                sensor_name="class_qb2",
                ip="10.16.3.160",
                api_key="secret",
                mqtt=QB2LiveMQTTConfig(host="10.16.3.111", topic="blickfeld/states_160"),
            ),
        ),
        preprocessing=PreprocessingConfig(lane_box=[-1, 1, -1, 1, -1, 1]),
        clustering=ClusteringConfig(),
        tracking=TrackingConfig(),
        aggregation=AggregationConfig(),
        output=OutputConfig(
            mode="dataset",
            dataset_root_dir=str(tmp_path / "dataset"),
            root_dir=str(tmp_path / "runs_unused"),
            final_full_recompute=False,
        ),
        visualization=VisualizationConfig(),
    )

    live_object = ObjectLabelData(
        object_id=42,
        timestamp_ns=200,
        points=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        obj_class="car",
        obj_class_score=0.95,
        sensor_name="class_qb2",
        frame_index=0,
        source_path=config.input.paths[0],
    )
    reader = _InterruptingLiveReader(
        frames=[
            FrameData(
                frame_index=0,
                timestamp_ns=200,
                points=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
                source_path=config.input.paths[0],
                object_labels=[live_object],
            )
        ]
    )
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_reader", lambda cfg: reader)
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_clusterer", lambda cfg: _FakeClusterer())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_tracker", lambda cfg: _FakeTracker())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_track_postprocessors", lambda cfg: [])
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_accumulator", lambda cfg: _FakeAccumulator())

    summary = run_pipeline(config, tmp_path)

    dataset_root = tmp_path / "dataset"
    assert summary.output_mode == "dataset"
    assert summary.output_dir == str(dataset_root)
    assert summary.saved_aggregates == 1
    assert summary.gt_match_matched_count == 1
    assert not (tmp_path / "runs_unused").exists()
    assert (dataset_root / "car" / "1970-01-01" / "gt-pred-different" / "gt").exists()
    assert (dataset_root / "car" / "1970-01-01" / "gt-pred-different" / "pred").exists()
    day_stats_dirs = sorted((dataset_root / "_stats" / "1970-01-01").iterdir())
    assert len(day_stats_dirs) == 1
    assert (day_stats_dirs[0] / "gt_matching" / "summary.json").exists()


def test_run_pipeline_statistics_disabled_keeps_live_aggregates_and_gt_without_stats(monkeypatch, tmp_path: Path) -> None:
    config = PipelineConfig(
        input=InputConfig(
            paths=["qb2_live://class_qb2@10.16.3.160"],
            format="qb2_live",
            qb2_live=QB2LiveInputConfig(
                sensor_name="class_qb2",
                ip="10.16.3.160",
                api_key="secret",
                mqtt=QB2LiveMQTTConfig(host="10.16.3.111", topic="blickfeld/states_160"),
            ),
        ),
        preprocessing=PreprocessingConfig(lane_box=[-1, 1, -1, 1, -1, 1]),
        clustering=ClusteringConfig(),
        tracking=TrackingConfig(),
        aggregation=AggregationConfig(),
        output=OutputConfig(
            root_dir=str(tmp_path),
            final_full_recompute=False,
            statistics_enabled=False,
            live_artifact_flush_interval_sec=0.0,
        ),
        visualization=VisualizationConfig(),
    )
    live_object = ObjectLabelData(
        object_id=42,
        timestamp_ns=200,
        points=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        obj_class="car",
        obj_class_score=0.95,
        sensor_name="class_qb2",
        frame_index=0,
        source_path=config.input.paths[0],
    )
    reader = _InterruptingLiveReader(
        frames=[
            FrameData(
                frame_index=0,
                timestamp_ns=200,
                points=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
                source_path=config.input.paths[0],
                object_labels=[live_object],
            )
        ]
    )
    fake_writer = _FakeWriter(tmp_path)
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_reader", lambda cfg: reader)
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_clusterer", lambda cfg: _FakeClusterer())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_tracker", lambda cfg: _FakeTracker())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_track_postprocessors", lambda cfg: [])
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_accumulator", lambda cfg: _FakeAccumulator())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_artifact_writer", lambda cfg, root: fake_writer)

    summary = run_pipeline(config, tmp_path)

    assert summary.saved_aggregates == 1
    assert summary.gt_match_saved_track_count == 1
    assert summary.performance is None
    assert fake_writer.aggregate_write_intensity_flags
    assert fake_writer.gt_matching_write_count >= 1
    assert fake_writer.summary_write_count == 0
    assert fake_writer.track_write_count == 0
    assert fake_writer.tracker_debug_write_count == 0
    assert fake_writer.track_outcomes_write_count == 0
    assert fake_writer.class_stats_write_count == 0
    assert (tmp_path / "run" / "aggregate.txt").exists()
    assert (tmp_path / "run" / "gt_matching.txt").exists()
    assert not (tmp_path / "run" / "summary.txt").exists()
    assert not (tmp_path / "run" / "tracks.txt").exists()
    assert not (tmp_path / "run" / "tracker_debug.txt").exists()
    assert not (tmp_path / "run" / "track_outcomes.txt").exists()
    assert not (tmp_path / "run" / "class_stats.txt").exists()
    assert not (tmp_path / "run" / "live_status.json").exists()


def test_run_pipeline_writes_live_gt_matching_for_pending_qb2_labels_without_final_recompute(monkeypatch, tmp_path: Path) -> None:
    config = PipelineConfig(
        input=InputConfig(
            paths=["qb2_live://class_qb2@10.16.3.160"],
            format="qb2_live",
            qb2_live=QB2LiveInputConfig(
                sensor_name="class_qb2",
                ip="10.16.3.160",
                api_key="secret",
                mqtt=QB2LiveMQTTConfig(host="10.16.3.111", topic="blickfeld/states_160"),
            ),
        ),
        preprocessing=PreprocessingConfig(lane_box=[-1, 1, -1, 1, -1, 1]),
        clustering=ClusteringConfig(),
        tracking=TrackingConfig(),
        aggregation=AggregationConfig(),
        output=OutputConfig(root_dir=str(tmp_path), final_full_recompute=False),
        visualization=VisualizationConfig(),
    )

    pending_object = ObjectLabelData(
        object_id=77,
        timestamp_ns=250,
        points=np.array([[2.0, 0.0, 0.0]], dtype=np.float32),
        obj_class="car",
        obj_class_score=0.9,
        sensor_name="class_qb2",
        frame_index=-1,
        source_path=config.input.paths[0],
    )
    reader = _InterruptingLiveReader(
        frames=[
            FrameData(
                frame_index=0,
                timestamp_ns=200,
                points=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
                source_path=config.input.paths[0],
            )
        ],
        pending_object_labels=[pending_object],
    )

    class _FakeFinishingTracker(_FakeTracker):
        def __init__(self):
            super().__init__()
            self.finished_tracks: dict[int, Track] = {}

        def step(self, detections, frame_idx, frame_timestamp_ns):
            state = super().step(detections, frame_idx, frame_timestamp_ns)
            self.finished_tracks[int(self.track.track_id)] = self.track
            return state

    fake_writer = _FakeWriter(tmp_path)
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_reader", lambda cfg: reader)
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_clusterer", lambda cfg: _FakeClusterer())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_tracker", lambda cfg: _FakeFinishingTracker())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_track_postprocessors", lambda cfg: [])
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_accumulator", lambda cfg: _FakeAccumulator())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_artifact_writer", lambda cfg, root: fake_writer)

    summary = run_pipeline(config, tmp_path)

    assert summary.saved_aggregates == 1
    assert summary.gt_match_saved_track_count == 1
    assert fake_writer.gt_matching_write_count >= 2
    assert fake_writer.gt_summary["gt_match_saved_track_count"] == 1
    assert fake_writer.gt_summary["gt_match_unmatched_saved_count"] == 0


def test_incremental_live_dataset_flush_keeps_existing_pred_without_new_saved_tracks(tmp_path: Path) -> None:
    config = PipelineConfig(
        input=InputConfig(
            paths=["qb2_live://class_qb2@10.16.3.160"],
            format="qb2_live",
            qb2_live=QB2LiveInputConfig(
                sensor_name="class_qb2",
                ip="10.16.3.160",
                api_key="secret",
                mqtt=QB2LiveMQTTConfig(host="10.16.3.111", topic="blickfeld/states_160"),
            ),
        ),
        preprocessing=PreprocessingConfig(lane_box=[-1, 1, -1, 1, -1, 1]),
        clustering=ClusteringConfig(),
        tracking=TrackingConfig(),
        aggregation=AggregationConfig(),
        output=OutputConfig(
            mode="dataset",
            dataset_root_dir=str(tmp_path / "dataset"),
            root_dir=str(tmp_path / "runs_unused"),
            final_full_recompute=False,
            live_artifact_flush_interval_sec=0.0,
        ),
        visualization=VisualizationConfig(),
    )

    class_normalizer = ClassNormalizer.from_config(config.class_normalization)
    profiler = PerformanceProfiler()
    writer = DatasetArtifactWriter(tmp_path)
    run_dir = writer.prepare_run_dir(config)
    writer.write_config_snapshot(run_dir, config)

    fake_tracker = _FakeTracker()
    track = fake_tracker.track
    aggregate_result = AggregateResult(
        track_id=1,
        points=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        selected_frame_ids=[0],
        status="saved",
        metrics={
            "predicted_class_id": 4,
            "predicted_class_name": "car",
            "predicted_class_score": 0.95,
            "registration_pairs": 1,
            "registration_accepted": 1,
            "registration_rejected": 0,
        },
    )
    object_label = ObjectLabelData(
        object_id=42,
        timestamp_ns=200,
        points=np.array([[0.95, 0.0, 0.0]], dtype=np.float32),
        obj_class="car",
        obj_class_score=0.9,
        sensor_name="class_qb2",
        frame_index=0,
        source_path=config.input.paths[0],
    )

    class _SnapshotTracker:
        finished_tracks: dict[int, Track] = {}

        def __init__(self, track: Track):
            self._track = track

        def snapshot_tracks(self) -> dict[int, Track]:
            return {int(self._track.track_id): self._track}

    tracker = _SnapshotTracker(track)
    live_artifact_state = {
        "last_flush_monotonic": None,
        "last_tracker_debug_flush_monotonic": None,
        "pending_flush": True,
        "pending_saved_results": {1: aggregate_result},
    }
    live_snapshot_tracks = {1: track}
    live_snapshot_aggregate_results = {1: aggregate_result}
    live_snapshot_track_outcomes = {
        1: TrackOutcomeDebug(
            track_id=1,
            status="saved",
            decision_stage="saved",
            decision_reason_code="saved",
            decision_summary="saved points=1",
            predicted_class_id=4,
            predicted_class_name="car",
            predicted_class_score=0.95,
            gt_obj_class="car",
            gt_obj_class_score=0.9,
        )
    }
    tracker_states = [
        FrameTrackingState(
            frame_index=0,
            lane_points=np.zeros((0, 3), dtype=np.float32),
            detections=[],
            active_tracks=[],
        )
    ]
    frame_to_playback = {1: 0}
    last_active_by_track: dict[int, dict[str, object]] = {}
    object_label_history_by_id = {42: [object_label]}
    latest_object_labels = {42: object_label}
    object_list_seen_ids = {42}
    announced_finished_track_ids: set[int] = set()

    _maybe_write_incremental_live_artifact_snapshot(
        config=config,
        profiler=profiler,
        writer=writer,
        run_dir=run_dir,
        lane_box=None,
        tracker=tracker,
        postprocessors=[],
        accumulator=None,
        classifier=None,
        class_normalizer=class_normalizer,
        latest_object_labels=latest_object_labels,
        object_label_history_by_id=object_label_history_by_id,
        object_list_seen_ids=object_list_seen_ids,
        object_list_skipped_empty=0,
        tracker_states=tracker_states,
        frame_to_playback=frame_to_playback,
        last_active_by_track=last_active_by_track,
        frame_count=1,
        live_status_reporter=None,
        live_web_runtime=None,
        live_artifact_state=live_artifact_state,
        live_snapshot_tracks=live_snapshot_tracks,
        live_snapshot_aggregate_results=live_snapshot_aggregate_results,
        live_snapshot_track_outcomes=live_snapshot_track_outcomes,
        live_snapshot_announced_finished_track_ids=announced_finished_track_ids,
        save_aggregate_intensity=False,
    )

    pred_dir = run_dir / "car" / "1970-01-01" / "gt-pred-same" / "pred"
    first_pred_json_paths = sorted(pred_dir.glob("*.json"))
    first_pred_pcd_paths = sorted(pred_dir.glob("*.pcd"))
    assert len(first_pred_json_paths) == 1
    assert len(first_pred_pcd_paths) == 1

    _maybe_write_incremental_live_artifact_snapshot(
        config=config,
        profiler=profiler,
        writer=writer,
        run_dir=run_dir,
        lane_box=None,
        tracker=tracker,
        postprocessors=[],
        accumulator=None,
        classifier=None,
        class_normalizer=class_normalizer,
        latest_object_labels=latest_object_labels,
        object_label_history_by_id=object_label_history_by_id,
        object_list_seen_ids=object_list_seen_ids,
        object_list_skipped_empty=0,
        tracker_states=tracker_states,
        frame_to_playback=frame_to_playback,
        last_active_by_track=last_active_by_track,
        frame_count=2,
        live_status_reporter=None,
        live_web_runtime=None,
        live_artifact_state=live_artifact_state,
        live_snapshot_tracks=live_snapshot_tracks,
        live_snapshot_aggregate_results=live_snapshot_aggregate_results,
        live_snapshot_track_outcomes=live_snapshot_track_outcomes,
        live_snapshot_announced_finished_track_ids=announced_finished_track_ids,
        save_aggregate_intensity=False,
    )

    second_pred_json_paths = sorted(pred_dir.glob("*.json"))
    second_pred_pcd_paths = sorted(pred_dir.glob("*.pcd"))
    assert second_pred_json_paths == first_pred_json_paths
    assert second_pred_pcd_paths == first_pred_pcd_paths


def test_incremental_live_artifact_snapshot_skips_track_snapshot_before_flush_interval(tmp_path: Path) -> None:
    config = PipelineConfig(
        input=InputConfig(
            paths=["qb2_live://class_qb2@10.16.3.160"],
            format="qb2_live",
            qb2_live=QB2LiveInputConfig(
                sensor_name="class_qb2",
                ip="10.16.3.160",
                api_key="secret",
                mqtt=QB2LiveMQTTConfig(host="10.16.3.111", topic="blickfeld/states_160"),
            ),
        ),
        preprocessing=PreprocessingConfig(lane_box=[-1, 1, -1, 1, -1, 1]),
        clustering=ClusteringConfig(),
        tracking=TrackingConfig(),
        aggregation=AggregationConfig(),
        output=OutputConfig(
            mode="dataset",
            dataset_root_dir=str(tmp_path / "dataset"),
            final_full_recompute=False,
            live_artifact_flush_interval_sec=60.0,
        ),
        visualization=VisualizationConfig(),
    )

    class _SnapshotCountingTracker:
        finished_tracks: dict[int, Track] = {}

        def __init__(self):
            self.snapshot_calls = 0

        def snapshot_tracks(self) -> dict[int, Track]:
            self.snapshot_calls += 1
            return {}

    tracker = _SnapshotCountingTracker()

    _maybe_write_incremental_live_artifact_snapshot(
        config=config,
        profiler=PerformanceProfiler(),
        writer=_FakeWriter(tmp_path),
        run_dir=tmp_path,
        lane_box=None,
        tracker=tracker,
        postprocessors=[],
        accumulator=None,
        classifier=None,
        class_normalizer=ClassNormalizer.from_config(config.class_normalization),
        latest_object_labels={},
        object_label_history_by_id={},
        object_list_seen_ids=set(),
        object_list_skipped_empty=0,
        tracker_states=[],
        frame_to_playback={},
        last_active_by_track={},
        frame_count=1,
        live_status_reporter=None,
        live_web_runtime=None,
        live_artifact_state={
            "last_flush_monotonic": time.monotonic(),
            "last_tracker_debug_flush_monotonic": None,
            "pending_flush": False,
            "pending_saved_results": {},
            "labels_dirty": False,
            "cached_matches": [],
            "cached_unmatched_saved_tracks": [],
            "cached_unmatched_gt_objects": [],
            "cached_gt_match_summary": {},
            "cached_class_stats": {},
        },
        live_snapshot_tracks={},
        live_snapshot_aggregate_results={},
        live_snapshot_track_outcomes={},
        live_snapshot_announced_finished_track_ids=set(),
        save_aggregate_intensity=False,
    )

    assert tracker.snapshot_calls == 0


def test_incremental_live_artifact_snapshot_skips_active_only_snapshot_after_flush_interval(tmp_path: Path) -> None:
    config = PipelineConfig(
        input=InputConfig(
            paths=["qb2_live://class_qb2@10.16.3.160"],
            format="qb2_live",
            qb2_live=QB2LiveInputConfig(
                sensor_name="class_qb2",
                ip="10.16.3.160",
                api_key="secret",
                mqtt=QB2LiveMQTTConfig(host="10.16.3.111", topic="blickfeld/states_160"),
            ),
        ),
        preprocessing=PreprocessingConfig(lane_box=[-1, 1, -1, 1, -1, 1]),
        clustering=ClusteringConfig(),
        tracking=TrackingConfig(),
        aggregation=AggregationConfig(),
        output=OutputConfig(
            mode="dataset",
            dataset_root_dir=str(tmp_path / "dataset"),
            final_full_recompute=False,
            live_artifact_flush_interval_sec=0.0,
        ),
        visualization=VisualizationConfig(),
    )

    class _SnapshotCountingTracker:
        finished_tracks: dict[int, Track] = {}

        def __init__(self):
            self.snapshot_calls = 0

        def snapshot_tracks(self) -> dict[int, Track]:
            self.snapshot_calls += 1
            return {}

    tracker = _SnapshotCountingTracker()

    _maybe_write_incremental_live_artifact_snapshot(
        config=config,
        profiler=PerformanceProfiler(),
        writer=_FakeWriter(tmp_path),
        run_dir=tmp_path,
        lane_box=None,
        tracker=tracker,
        postprocessors=[],
        accumulator=None,
        classifier=None,
        class_normalizer=ClassNormalizer.from_config(config.class_normalization),
        latest_object_labels={},
        object_label_history_by_id={},
        object_list_seen_ids=set(),
        object_list_skipped_empty=0,
        tracker_states=[],
        frame_to_playback={},
        last_active_by_track={},
        frame_count=5,
        live_status_reporter=None,
        live_web_runtime=None,
        live_artifact_state={
            "last_flush_monotonic": time.monotonic() - 120.0,
            "last_tracker_debug_flush_monotonic": None,
            "pending_flush": False,
            "pending_saved_results": {},
            "labels_dirty": False,
            "cached_matches": [],
            "cached_unmatched_saved_tracks": [],
            "cached_unmatched_gt_objects": [],
            "cached_gt_match_summary": {},
            "cached_class_stats": {},
        },
        live_snapshot_tracks={},
        live_snapshot_aggregate_results={},
        live_snapshot_track_outcomes={},
        live_snapshot_announced_finished_track_ids=set(),
        save_aggregate_intensity=False,
    )

    assert tracker.snapshot_calls == 0


def test_snapshot_tracker_tracks_keeps_only_lightweight_track_metadata() -> None:
    track = Track(
        track_id=7,
        centers=[
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
        ],
        frame_ids=[10, 11],
        frame_timestamps_ns=[100, 200],
        local_points=[
            np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            np.array([[1.0, 1.0, 1.0]], dtype=np.float32),
        ],
        world_points=[
            np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            np.array([[2.0, 2.0, 2.0]], dtype=np.float32),
        ],
        local_intensity=[np.array([0.1], dtype=np.float32), np.array([0.2], dtype=np.float32)],
        world_intensity=[np.array([0.3], dtype=np.float32), np.array([0.4], dtype=np.float32)],
        point_timestamps_ns=[np.array([100], dtype=np.int64), np.array([200], dtype=np.int64)],
        bbox_extents=[
            np.array([1.0, 1.0, 1.0], dtype=np.float32),
            np.array([2.0, 2.0, 2.0], dtype=np.float32),
        ],
        hit_count=2,
        age=3,
        missed=1,
        ended_by_missed=True,
        source_track_ids=[7],
        quality_score=0.8,
        quality_metrics={"continuity": 1.0},
        state={"tracker_debug_summary": {"matched_count": 2}},
    )

    class _SnapshotTracker:
        def snapshot_tracks(self) -> dict[int, Track]:
            return {int(track.track_id): track}

    snapshot = _snapshot_tracker_tracks(_SnapshotTracker())

    assert set(snapshot) == {7}
    lightweight_track = snapshot[7]
    assert lightweight_track.frame_ids == [10, 11]
    assert lightweight_track.frame_timestamps_ns == [100, 200]
    assert len(lightweight_track.centers) == 1
    assert np.allclose(lightweight_track.centers[0], np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert len(lightweight_track.bbox_extents) == 1
    assert np.allclose(lightweight_track.bbox_extents[0], np.array([2.0, 2.0, 2.0], dtype=np.float32))
    assert lightweight_track.local_points == []
    assert lightweight_track.world_points == []
    assert lightweight_track.local_intensity == []
    assert lightweight_track.world_intensity == []
    assert lightweight_track.point_timestamps_ns == []
    assert lightweight_track.hit_count == 2
    assert lightweight_track.age == 3
    assert lightweight_track.missed == 1
    assert lightweight_track.ended_by_missed is True
    assert lightweight_track.quality_metrics == {"continuity": 1.0}
    assert lightweight_track.state["tracker_debug_summary"] == {"matched_count": 2}


def test_live_object_list_snapshot_heartbeat_does_not_write_without_dirty_state(tmp_path: Path) -> None:
    class _CountingObjectListWriter:
        def __init__(self):
            self.calls = 0

        def write_live_object_list_snapshot(self, run_dir, object_labels):
            _ = run_dir, object_labels
            self.calls += 1

    writer = _CountingObjectListWriter()
    state = {
        "dirty": False,
        "last_flush_monotonic": time.monotonic(),
        "flush_interval_sec": 0.0,
    }
    object_label = ObjectLabelData(
        object_id=1,
        timestamp_ns=100,
        points=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        obj_class="car",
        obj_class_score=0.9,
        sensor_name="sensor_a",
        frame_index=0,
        source_path="dummy.pb",
    )

    wrote = _maybe_write_live_object_list_snapshot(
        writer,
        tmp_path,
        {1: object_label},
        reporter=None,
        state=state,
        mark_dirty=False,
    )

    assert wrote is False
    assert writer.calls == 0


def test_run_pipeline_rejects_qb2_live_interrupt_before_first_frame(monkeypatch, tmp_path: Path) -> None:
    config = PipelineConfig(
        input=InputConfig(
            paths=["qb2_live://class_qb2@10.16.3.160"],
            format="qb2_live",
            qb2_live=QB2LiveInputConfig(
                sensor_name="class_qb2",
                ip="10.16.3.160",
                api_key="secret",
                mqtt=QB2LiveMQTTConfig(host="10.16.3.111", topic="blickfeld/states_160"),
            ),
        ),
        preprocessing=PreprocessingConfig(lane_box=[-1, 1, -1, 1, -1, 1]),
        clustering=ClusteringConfig(),
        tracking=TrackingConfig(),
        aggregation=AggregationConfig(),
        output=OutputConfig(root_dir=str(tmp_path)),
        visualization=VisualizationConfig(),
    )

    reader = _InterruptingLiveReader(frames=[])
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_reader", lambda cfg: reader)
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_clusterer", lambda cfg: _FakeClusterer())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_tracker", lambda cfg: _FakeTracker())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_track_postprocessors", lambda cfg: [])
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_accumulator", lambda cfg: _FakeAccumulator())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_artifact_writer", lambda cfg, root: _FakeWriter(tmp_path))

    with pytest.raises(RuntimeError, match="Run interrupted before any frames were received"):
        run_pipeline(config, tmp_path)


def test_replay_run_uses_multi_file_input_without_tracker_reset(monkeypatch, tmp_path: Path) -> None:
    config = PipelineConfig(
        input=InputConfig(paths=["ignored_a.pb", "ignored_b.pb"]),
        preprocessing=PreprocessingConfig(lane_box=[-1, 1, -1, 1, -1, 1]),
        clustering=ClusteringConfig(),
        tracking=TrackingConfig(),
        aggregation=AggregationConfig(),
        output=OutputConfig(root_dir=str(tmp_path)),
        visualization=VisualizationConfig(),
    )

    fake_tracker = _FakeTracker()
    fake_viewer = _FakeViewer()
    monkeypatch.setattr("tracking_pipeline.application.replay_run.build_reader", lambda cfg: _FakeReader())
    monkeypatch.setattr("tracking_pipeline.application.replay_run.build_clusterer", lambda cfg: _FakeClusterer())
    monkeypatch.setattr("tracking_pipeline.application.replay_run.build_tracker", lambda cfg: fake_tracker)
    monkeypatch.setattr("tracking_pipeline.application.replay_run.build_track_postprocessors", lambda cfg: [])
    monkeypatch.setattr("tracking_pipeline.application.replay_run.build_accumulator", lambda cfg: _FakeAccumulator())
    monkeypatch.setattr("tracking_pipeline.application.replay_run.build_viewer", lambda cfg: fake_viewer)

    replay_run(config, tmp_path)

    assert fake_tracker.seen_frame_ids == [0, 1]
    assert [state.frame_index for state in fake_viewer.states] == [0, 1]
    assert np.allclose(fake_viewer.states[0].full_frame_points, np.array([[0.0, 0.0, 0.0]], dtype=np.float32))
    assert np.allclose(fake_viewer.states[1].full_frame_points, np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
    assert np.allclose(fake_viewer.states[0].full_frame_intensity, np.array([0.25], dtype=np.float32))
    assert np.allclose(fake_viewer.states[1].full_frame_intensity, np.array([0.5], dtype=np.float32))
    assert set(fake_viewer.aggregate_results.keys()) == {1}
    assert set(fake_viewer.track_outcomes.keys()) == {1}
    assert fake_viewer.articulated_merge_debug_events == []


def test_replay_run_propagates_classification_to_viewer_data(monkeypatch, tmp_path: Path) -> None:
    config = PipelineConfig(
        input=InputConfig(paths=["ignored_a.pb"]),
        preprocessing=PreprocessingConfig(lane_box=[-1, 1, -1, 1, -1, 1]),
        clustering=ClusteringConfig(),
        tracking=TrackingConfig(),
        aggregation=AggregationConfig(),
        output=OutputConfig(root_dir=str(tmp_path)),
        visualization=VisualizationConfig(),
    )

    fake_classifier = _FakeClassifier()
    fake_viewer = _FakeViewer()
    monkeypatch.setattr("tracking_pipeline.application.replay_run.build_reader", lambda cfg: _FakeReader())
    monkeypatch.setattr("tracking_pipeline.application.replay_run.build_clusterer", lambda cfg: _FakeClusterer())
    monkeypatch.setattr("tracking_pipeline.application.replay_run.build_tracker", lambda cfg: _FakeTracker())
    monkeypatch.setattr("tracking_pipeline.application.replay_run.build_track_postprocessors", lambda cfg: [])
    monkeypatch.setattr("tracking_pipeline.application.replay_run.build_accumulator", lambda cfg: _FakeAccumulator())
    monkeypatch.setattr("tracking_pipeline.application.replay_run.build_classifier", lambda cfg: fake_classifier)
    monkeypatch.setattr("tracking_pipeline.application.replay_run.build_viewer", lambda cfg: fake_viewer)

    replay_run(config, tmp_path)

    assert len(fake_classifier.seen_points) == 1
    result = fake_viewer.aggregate_results[1]
    assert result.metrics["predicted_class_name"] == "trailer"
    assert result.metrics["predicted_class_score"] == 0.88
    outcome = fake_viewer.track_outcomes[1]
    assert outcome.predicted_class_id == 4
    assert outcome.predicted_class_name == "trailer"
    assert outcome.predicted_class_score == 0.88


def test_replay_run_rejects_qb2_live_input(tmp_path: Path) -> None:
    config = PipelineConfig(
        input=InputConfig(
            paths=["qb2_live://class_qb2@10.16.3.160"],
            format="qb2_live",
            qb2_live=QB2LiveInputConfig(
                sensor_name="class_qb2",
                ip="10.16.3.160",
                api_key="secret",
                mqtt=QB2LiveMQTTConfig(host="10.16.3.111", topic="blickfeld/states_160"),
            ),
        ),
        preprocessing=PreprocessingConfig(lane_box=[-1, 1, -1, 1, -1, 1]),
        clustering=ClusteringConfig(),
        tracking=TrackingConfig(),
        aggregation=AggregationConfig(),
        output=OutputConfig(root_dir=str(tmp_path)),
        visualization=VisualizationConfig(),
    )

    with pytest.raises(ValueError, match="Live input is only supported for `run`"):
        replay_run(config, tmp_path)


def test_replay_run_propagates_gt_class_to_viewer_data(monkeypatch, tmp_path: Path) -> None:
    config = PipelineConfig(
        input=InputConfig(paths=["ignored_a.pb"]),
        preprocessing=PreprocessingConfig(lane_box=[-1, 1, -1, 1, -1, 1]),
        clustering=ClusteringConfig(),
        tracking=TrackingConfig(),
        aggregation=AggregationConfig(),
        output=OutputConfig(root_dir=str(tmp_path)),
        visualization=VisualizationConfig(),
    )

    fake_viewer = _FakeViewer()
    monkeypatch.setattr("tracking_pipeline.application.replay_run.build_reader", lambda cfg: _FakeObjectReader())
    monkeypatch.setattr("tracking_pipeline.application.replay_run.build_clusterer", lambda cfg: _FakeClusterer())
    monkeypatch.setattr("tracking_pipeline.application.replay_run.build_tracker", lambda cfg: _FakeTracker())
    monkeypatch.setattr("tracking_pipeline.application.replay_run.build_track_postprocessors", lambda cfg: [])
    monkeypatch.setattr("tracking_pipeline.application.replay_run.build_accumulator", lambda cfg: _FakeAccumulator())
    monkeypatch.setattr("tracking_pipeline.application.replay_run.build_viewer", lambda cfg: fake_viewer)

    replay_run(config, tmp_path)

    result = fake_viewer.aggregate_results[1]
    assert result.metrics["gt_obj_class"] == "car"
    assert result.metrics["gt_obj_class_score"] == 0.95
    outcome = fake_viewer.track_outcomes[1]
    assert outcome.gt_obj_class == "car"
    assert outcome.gt_obj_class_score == 0.95


def test_run_pipeline_normalizes_class_names_in_results_and_stats(monkeypatch, tmp_path: Path) -> None:
    config = PipelineConfig(
        input=InputConfig(paths=["ignored_a.pb"]),
        preprocessing=PreprocessingConfig(lane_box=[-1, 1, -1, 1, -1, 1]),
        clustering=ClusteringConfig(),
        tracking=TrackingConfig(),
        aggregation=AggregationConfig(),
        class_normalization=ClassNormalizationConfig(
            enabled=True,
            aliases={
                "trailer": "TLS_VEHICLE_TRUCK_WITH_TRAILER",
                "car": "TLS_VEHICLE_CAR",
                "van": "TLS_VEHICLE_VAN",
            },
        ),
        output=OutputConfig(root_dir=str(tmp_path)),
        visualization=VisualizationConfig(),
    )

    fake_writer = _FakeWriter(tmp_path)
    fake_classifier = _FakeClassifier()
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_reader", lambda cfg: _FakeObjectReader())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_clusterer", lambda cfg: _FakeClusterer())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_tracker", lambda cfg: _FakeTracker())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_track_postprocessors", lambda cfg: [])
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_accumulator", lambda cfg: _FakeAccumulator())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_classifier", lambda cfg: fake_classifier)
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_artifact_writer", lambda cfg, root: fake_writer)

    summary = run_pipeline(config, tmp_path)

    result = fake_writer.written_aggregate_results[0]
    assert result.metrics["predicted_class_name"] == "TLS_VEHICLE_TRUCK_WITH_TRAILER"
    assert result.metrics["gt_obj_class"] == "TLS_VEHICLE_CAR"
    assert fake_writer.gt_matches[0].gt_obj_class == "TLS_VEHICLE_CAR"
    assert fake_writer.gt_unmatched_objects[0].gt_obj_class == "TLS_VEHICLE_VAN"
    assert fake_writer.object_labels[7].obj_class == "TLS_VEHICLE_CAR"
    assert fake_writer.object_labels[8].obj_class == "TLS_VEHICLE_VAN"
    assert summary.predicted_class_counts == {"TLS_VEHICLE_TRUCK_WITH_TRAILER": 1}
    assert summary.gt_class_counts == {"TLS_VEHICLE_CAR": 1, "TLS_VEHICLE_VAN": 1}
    assert summary.matched_gt_class_counts == {"TLS_VEHICLE_CAR": 1}
    assert summary.class_comparison_count == 1
    assert summary.class_match_count == 0
    assert summary.class_mismatch_count == 1
    assert summary.class_count_rows == [
        {"class_name": "TLS_VEHICLE_CAR", "predicted_count": 0, "gt_match_count": 1},
        {"class_name": "TLS_VEHICLE_TRUCK_WITH_TRAILER", "predicted_count": 1, "gt_match_count": 0},
        {"class_name": "TOTAL", "predicted_count": 1, "gt_match_count": 1},
    ]


def test_run_pipeline_starts_embedded_live_web_viewer_for_qb2_live(monkeypatch, tmp_path: Path) -> None:
    config = PipelineConfig(
        input=InputConfig(
            paths=["qb2_live://class_qb2@10.16.3.160"],
            format="qb2_live",
            qb2_live=QB2LiveInputConfig(
                sensor_name="class_qb2",
                ip="10.16.3.160",
                api_key="secret",
                mqtt=QB2LiveMQTTConfig(host="10.16.3.111", topic="blickfeld/states_160"),
            ),
        ),
        preprocessing=PreprocessingConfig(lane_box=[-1, 1, 0, 8, 0, 2]),
        clustering=ClusteringConfig(),
        tracking=TrackingConfig(),
        aggregation=AggregationConfig(),
        output=OutputConfig(root_dir=str(tmp_path)),
        visualization=VisualizationConfig(
            live_web_enabled=True,
            live_web_host="0.0.0.0",
            live_web_port=8765,
            live_web_history_sec=0.8,
        ),
    )
    reader = _InterruptingLiveReader(
        frames=[
            FrameData(
                frame_index=0,
                timestamp_ns=100,
                points=np.array([[0.0, 0.0, 0.0], [0.2, 0.1, 0.0]], dtype=np.float32),
                point_intensity=np.array([0.2, 0.4], dtype=np.float32),
                source_path="qb2_live://class_qb2@10.16.3.160",
            )
        ]
    )
    fake_writer = _FakeWriter(tmp_path)
    fake_tracker = _FakeTracker()
    seen: dict[str, object] = {}

    class _FakeLiveFramePublisher:
        def __init__(self, **kwargs):
            seen["publisher_kwargs"] = dict(kwargs)
            seen["published_frames"] = []
            seen["status_updates"] = []
            seen["track_outcomes"] = []
            seen["summaries"] = []
            seen["stop_phases"] = []

        def update_status(self, **updates):
            seen["status_updates"].append(dict(updates))

        def publish_frame(self, frame, cluster_result, tracking_state):
            seen["published_frames"].append(
                {
                    "frame_index": int(frame.frame_index),
                    "point_count": int(len(frame.points)),
                    "detection_count": int(len(cluster_result.detections)),
                    "tracking_frame_index": int(tracking_state.frame_index),
                }
            )
            return len(seen["published_frames"])

        def update_track_outcomes(self, track_outcomes):
            seen["track_outcomes"].append(sorted(int(track_id) for track_id in dict(track_outcomes)))

        def update_summary(self, summary):
            seen["summaries"].append(int(summary.saved_aggregates))

        def mark_stopped(self, *, pipeline_phase: str = "stopped"):
            seen["stop_phases"].append(str(pipeline_phase))

    class _FakeLivePCDWebServer:
        def __init__(self, publisher, *, host: str, port: int):
            seen["server_publisher"] = publisher
            seen["server_host"] = host
            seen["server_port"] = port
            seen["server_started"] = False
            seen["server_stopped"] = False
            self.port = 9876

        def start(self):
            seen["server_started"] = True

        def stop(self):
            seen["server_stopped"] = True

    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_reader", lambda cfg: reader)
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_clusterer", lambda cfg: _FakeClusterer())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_tracker", lambda cfg: fake_tracker)
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_track_postprocessors", lambda cfg: [])
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_accumulator", lambda cfg: _FakeAccumulator())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_artifact_writer", lambda cfg, root: fake_writer)
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.LiveFramePublisher", _FakeLiveFramePublisher)
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.LivePCDWebServer", _FakeLivePCDWebServer)

    summary = run_pipeline(config, tmp_path)

    assert summary.frame_count == 1
    assert seen["server_host"] == "0.0.0.0"
    assert seen["server_port"] == 8765
    assert "retain_all_frames" not in seen["publisher_kwargs"]
    assert seen["server_started"] is True
    assert seen["server_stopped"] is True
    assert seen["published_frames"] == [
        {
            "frame_index": 0,
            "point_count": 2,
            "detection_count": 1,
            "tracking_frame_index": 0,
        }
    ]
    assert any(update.get("pipeline_phase") == "processing_frames" for update in seen["status_updates"])
    assert any(update.get("pipeline_phase") == "completed" for update in seen["status_updates"])
    assert seen["track_outcomes"][-1] == [1]
    assert seen["summaries"][-1] == 1
    assert seen["stop_phases"] == ["stopped"]


def test_live_web_status_updates_include_processing_hz(monkeypatch) -> None:
    seen: list[dict[str, object]] = []

    class _FakeLiveFramePublisher:
        def update_status(self, **updates):
            seen.append(dict(updates))

    monotonic_values = iter([101.0, 103.0])

    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.LiveFramePublisher", _FakeLiveFramePublisher)
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.time.monotonic", lambda: next(monotonic_values))

    runtime = {
        "publisher": _FakeLiveFramePublisher(),
        "_started_monotonic": 100.0,
        "_last_processed_monotonic": None,
        "_last_processed_frame_count": 0,
    }

    _update_live_web_status(runtime, processed_frames=1)
    _update_live_web_status(runtime, processed_frames=5)

    assert seen[0]["processing_total_hz"] == pytest.approx(1.0)
    assert seen[0]["processing_recent_hz"] == pytest.approx(0.0)
    assert seen[1]["processing_total_hz"] == pytest.approx(5.0 / 3.0)
    assert seen[1]["processing_recent_hz"] == pytest.approx(2.0)


def test_run_pipeline_restores_live_saved_track_outcomes_without_live_snapshot_writes(monkeypatch, tmp_path: Path) -> None:
    config = PipelineConfig(
        input=InputConfig(
            paths=["qb2_live://class_qb2@10.16.3.160"],
            format="qb2_live",
            qb2_live=QB2LiveInputConfig(
                sensor_name="class_qb2",
                ip="10.16.3.160",
                api_key="secret",
                mqtt=QB2LiveMQTTConfig(host="10.16.3.111", topic="blickfeld/states_160"),
            ),
        ),
        preprocessing=PreprocessingConfig(lane_box=[-1, 1, 0, 8, 0, 2]),
        clustering=ClusteringConfig(),
        tracking=TrackingConfig(),
        aggregation=AggregationConfig(),
        output=OutputConfig(root_dir=str(tmp_path)),
        visualization=VisualizationConfig(
            live_web_enabled=True,
            live_web_host="0.0.0.0",
            live_web_port=8765,
            live_web_history_sec=0.8,
        ),
    )

    class _FakeFinishingTracker(_FakeTracker):
        def __init__(self):
            super().__init__()
            self.finished_tracks: dict[int, Track] = {}

        def step(self, detections, frame_idx, frame_timestamp_ns):
            state = super().step(detections, frame_idx, frame_timestamp_ns)
            self.finished_tracks[int(self.track.track_id)] = self.track
            return state

    reader = _InterruptingLiveReader(
        frames=[
            FrameData(
                frame_index=0,
                timestamp_ns=100,
                points=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
                point_intensity=np.array([0.2], dtype=np.float32),
                source_path="qb2_live://class_qb2@10.16.3.160",
            )
        ]
    )
    fake_writer = _FakeWriter(tmp_path)
    fake_tracker = _FakeFinishingTracker()
    seen: dict[str, object] = {}

    class _FakeLiveFramePublisher:
        def __init__(self, **kwargs):
            seen["publisher_kwargs"] = dict(kwargs)
            seen["track_outcomes"] = []
            seen["status_updates"] = []

        def update_status(self, **updates):
            seen["status_updates"].append(dict(updates))

        def publish_frame(self, frame, cluster_result, tracking_state):
            _ = frame, cluster_result, tracking_state
            return 1

        def update_track_outcomes(self, track_outcomes):
            seen["track_outcomes"].append(sorted(int(track_id) for track_id in dict(track_outcomes)))

        def update_summary(self, summary):
            _ = summary

        def mark_stopped(self, *, pipeline_phase: str = "stopped"):
            _ = pipeline_phase

    class _FakeLivePCDWebServer:
        def __init__(self, publisher, *, host: str, port: int):
            _ = publisher, host, port
            self.port = 9876

        def start(self):
            return None

        def stop(self):
            return None

    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_reader", lambda cfg: reader)
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_clusterer", lambda cfg: _FakeClusterer())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_tracker", lambda cfg: fake_tracker)
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_track_postprocessors", lambda cfg: [])
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_accumulator", lambda cfg: _FakeAccumulator())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_artifact_writer", lambda cfg, root: fake_writer)
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.LiveFramePublisher", _FakeLiveFramePublisher)
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.LivePCDWebServer", _FakeLivePCDWebServer)

    run_pipeline(config, tmp_path)

    assert len(seen["track_outcomes"]) >= 2
    assert all(track_outcome_ids == [1] for track_outcome_ids in seen["track_outcomes"])
    assert fake_writer.summary_write_count >= 2


def test_run_pipeline_skips_embedded_live_web_viewer_when_disabled(monkeypatch, tmp_path: Path) -> None:
    config = PipelineConfig(
        input=InputConfig(paths=["ignored_a.pb"]),
        preprocessing=PreprocessingConfig(lane_box=[-1, 1, -1, 1, -1, 1]),
        clustering=ClusteringConfig(),
        tracking=TrackingConfig(),
        aggregation=AggregationConfig(),
        output=OutputConfig(root_dir=str(tmp_path)),
        visualization=VisualizationConfig(live_web_enabled=True),
    )

    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_reader", lambda cfg: _FakeReader())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_clusterer", lambda cfg: _FakeClusterer())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_tracker", lambda cfg: _FakeTracker())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_track_postprocessors", lambda cfg: [])
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_accumulator", lambda cfg: _FakeAccumulator())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_artifact_writer", lambda cfg, root: _FakeWriter(tmp_path))
    monkeypatch.setattr(
        "tracking_pipeline.application.run_pipeline.LiveFramePublisher",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("live web publisher should not start for non-qb2 input")),
    )

    summary = run_pipeline(config, tmp_path)

    assert summary.frame_count == 1


def test_replay_run_normalizes_class_names_before_viewer_receives_data(monkeypatch, tmp_path: Path) -> None:
    config = PipelineConfig(
        input=InputConfig(paths=["ignored_a.pb"]),
        preprocessing=PreprocessingConfig(lane_box=[-1, 1, -1, 1, -1, 1]),
        clustering=ClusteringConfig(),
        tracking=TrackingConfig(),
        aggregation=AggregationConfig(),
        class_normalization=ClassNormalizationConfig(
            enabled=True,
            aliases={
                "trailer": "TLS_VEHICLE_TRUCK_WITH_TRAILER",
                "car": "TLS_VEHICLE_CAR",
            },
        ),
        output=OutputConfig(root_dir=str(tmp_path)),
        visualization=VisualizationConfig(),
    )

    fake_classifier = _FakeClassifier()
    fake_viewer = _FakeViewer()
    monkeypatch.setattr("tracking_pipeline.application.replay_run.build_reader", lambda cfg: _FakeObjectReader())
    monkeypatch.setattr("tracking_pipeline.application.replay_run.build_clusterer", lambda cfg: _FakeClusterer())
    monkeypatch.setattr("tracking_pipeline.application.replay_run.build_tracker", lambda cfg: _FakeTracker())
    monkeypatch.setattr("tracking_pipeline.application.replay_run.build_track_postprocessors", lambda cfg: [])
    monkeypatch.setattr("tracking_pipeline.application.replay_run.build_accumulator", lambda cfg: _FakeAccumulator())
    monkeypatch.setattr("tracking_pipeline.application.replay_run.build_classifier", lambda cfg: fake_classifier)
    monkeypatch.setattr("tracking_pipeline.application.replay_run.build_viewer", lambda cfg: fake_viewer)

    replay_run(config, tmp_path)

    result = fake_viewer.aggregate_results[1]
    assert result.metrics["predicted_class_name"] == "TLS_VEHICLE_TRUCK_WITH_TRAILER"
    assert result.metrics["gt_obj_class"] == "TLS_VEHICLE_CAR"
    outcome = fake_viewer.track_outcomes[1]
    assert outcome.predicted_class_name == "TLS_VEHICLE_TRUCK_WITH_TRAILER"
    assert outcome.gt_obj_class == "TLS_VEHICLE_CAR"


def test_replay_run_passes_articulated_merge_debug_events_to_viewer(monkeypatch, tmp_path: Path) -> None:
    config = PipelineConfig(
        input=InputConfig(paths=["ignored_a.pb", "ignored_b.pb", "ignored_c.pb", "ignored_d.pb"]),
        preprocessing=PreprocessingConfig(lane_box=[-1, 1, -1, 1, -1, 1]),
        clustering=ClusteringConfig(),
        tracking=TrackingConfig(),
        aggregation=AggregationConfig(frame_selection_line_axis="y"),
        postprocessing=PostprocessingConfig(enable_articulated_vehicle_merge=True),
        output=OutputConfig(root_dir=str(tmp_path)),
        visualization=VisualizationConfig(show_articulated_merge_debug=True),
    )

    fake_tracker = _FakeArticulatedTracker()
    fake_viewer = _FakeViewer()
    monkeypatch.setattr("tracking_pipeline.application.replay_run.build_reader", lambda cfg: _FakeReader())
    monkeypatch.setattr("tracking_pipeline.application.replay_run.build_clusterer", lambda cfg: _FakeClusterer())
    monkeypatch.setattr("tracking_pipeline.application.replay_run.build_tracker", lambda cfg: fake_tracker)
    monkeypatch.setattr(
        "tracking_pipeline.application.replay_run.build_track_postprocessors",
        lambda cfg: [ArticulatedVehicleMergePostprocessor(cfg.postprocessing, longitudinal_axis=cfg.aggregation.frame_selection_line_axis)],
    )
    monkeypatch.setattr("tracking_pipeline.application.replay_run.build_accumulator", lambda cfg: _FakeAccumulator())
    monkeypatch.setattr("tracking_pipeline.application.replay_run.build_viewer", lambda cfg: fake_viewer)

    replay_run(config, tmp_path)

    assert len(fake_viewer.articulated_merge_debug_events) == 1
    event = fake_viewer.articulated_merge_debug_events[0]
    assert event.accepted is True
    assert (event.lead_track_id, event.rear_track_id) == (11, 12)
    assert event.rejection_reason == "tail_gap"
    assert event.center is not None
    assert np.all(np.isfinite(event.center))
