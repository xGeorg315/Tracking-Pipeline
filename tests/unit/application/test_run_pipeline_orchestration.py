from __future__ import annotations

from pathlib import Path

import numpy as np

from tracking_pipeline.application.replay_run import replay_run
from tracking_pipeline.application.run_pipeline import run_pipeline
from tracking_pipeline.config.models import (
    AggregationConfig,
    ClusteringConfig,
    InputConfig,
    OutputConfig,
    PipelineConfig,
    PostprocessingConfig,
    PreprocessingConfig,
    TrackingConfig,
    VisualizationConfig,
)
from tracking_pipeline.domain.models import (
    ActiveTrackState,
    AggregateResult,
    ClusterResult,
    Detection,
    FrameData,
    FrameTrackerDebug,
    FrameTrackingState,
    GTMatchResult,
    ObjectLabelData,
    Track,
    TrackOutcomeDebug,
)
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
        self.aggregate_write_intensity_flags: list[bool] = []
        self.tracker_debug_states = None
        self.track_outcomes = None
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
        _ = result
        self.aggregate_write_intensity_flags.append(bool(save_intensity))
        (run_dir / "aggregate.txt").write_text("saved\n", encoding="utf-8")

    def write_summary(self, run_dir, summary):
        (run_dir / "summary.txt").write_text(str(summary.saved_aggregates), encoding="utf-8")

    def write_tracks(self, run_dir, tracks, aggregate_results):
        self.written_tracks = tracks
        self.written_aggregate_results = aggregate_results
        (run_dir / "tracks.txt").write_text("tracks\n", encoding="utf-8")

    def write_tracker_debug(self, run_dir, states):
        self.tracker_debug_states = states
        (run_dir / "tracker_debug.txt").write_text(str(len(states)), encoding="utf-8")

    def write_track_outcomes(self, run_dir, track_outcomes):
        self.track_outcomes = track_outcomes
        (run_dir / "track_outcomes.txt").write_text(str(len(track_outcomes)), encoding="utf-8")

    def write_object_list(self, run_dir, object_labels):
        self.object_labels = object_labels
        (run_dir / "object_list_manifest.txt").write_text(str(sorted(object_labels)), encoding="utf-8")

    def write_gt_matching(self, run_dir, matches, unmatched_saved_tracks, unmatched_gt_objects, summary):
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
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_artifact_writer", lambda root: fake_writer)

    summary = run_pipeline(config, tmp_path)

    assert summary.frame_count == 2
    assert summary.saved_aggregates == 1
    assert summary.input_paths == ["ignored_a.pb", "ignored_b.pb"]
    assert fake_tracker.seen_frame_ids == [0, 1]
    assert fake_writer.aggregate_write_intensity_flags == [False]
    assert (tmp_path / "run" / "summary.txt").exists()
    assert (tmp_path / "run" / "tracker_debug.txt").exists()
    assert (tmp_path / "run" / "track_outcomes.txt").exists()
    assert len(fake_writer.tracker_debug_states) == 2
    assert isinstance(fake_writer.track_outcomes[1], TrackOutcomeDebug)
    assert fake_writer.track_outcomes[1].status == "saved"
    assert summary.performance is not None
    assert summary.performance.aggregation_components["registration"].wall_seconds == 0.0
    assert summary.performance.aggregation_components["fusion_core"].wall_seconds == 0.2
    assert summary.performance.aggregation_components["fusion_post"].wall_seconds == 0.3
    assert summary.performance.aggregation_components["fusion_total"].wall_seconds == 0.5
    assert summary.performance.aggregation_components["fusion_total"].call_count == 1


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
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_artifact_writer", lambda root: fake_writer)

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
    assert len(fake_writer.gt_unmatched_objects) == 1
    assert fake_writer.gt_unmatched_objects[0].gt_object_id == 8


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
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_artifact_writer", lambda root: fake_writer)

    run_pipeline(config, tmp_path)

    assert fake_writer.aggregate_write_intensity_flags == [True]


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
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_artifact_writer", lambda root: fake_writer)

    summary = run_pipeline(config, tmp_path)

    assert summary.finished_track_count == 2
    assert summary.saved_aggregates == 1
    assert summary.postprocessing_methods == ["articulated_vehicle_merge", "track_quality_scoring"]
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
    assert lead_result.metrics["post_merge_component_ids"] == [11, 12]
    assert rear_result.metrics["merged_target_track_id"] == 11


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
