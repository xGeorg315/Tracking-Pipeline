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
    PreprocessingConfig,
    TrackingConfig,
    VisualizationConfig,
)
from tracking_pipeline.domain.models import AggregateResult, ClusterResult, Detection, FrameData, FrameTrackingState, ObjectLabelData, Track


class _FakeReader:
    def iter_frames(self, input_paths: list[str]) -> list[FrameData]:
        return [
            FrameData(
                frame_index=index,
                timestamp_ns=index + 1,
                points=np.zeros((1, 3), dtype=np.float32),
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
        self.track.local_points.append(np.array([[0.0, 0.0, 0.0]], dtype=np.float32))
        self.track.world_points.append(np.array([[0.95, 0.0, 0.0]], dtype=np.float32))
        self.track.bbox_extents.append(np.array([0.1, 0.1, 0.1], dtype=np.float32))

    def step(self, detections, frame_idx):
        _ = detections
        self.seen_frame_ids.append(int(frame_idx))
        return FrameTrackingState(frame_index=int(frame_idx), lane_points=np.zeros((0, 3), dtype=np.float32), detections=[], active_tracks=[])

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
            metrics={"registration_pairs": 1, "registration_accepted": 1, "registration_rejected": 0},
        )


class _FakeWriter:
    def __init__(self, base: Path):
        self.base = base
        self.object_labels = None
        self.aggregate_write_intensity_flags: list[bool] = []

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
        _ = tracks, aggregate_results
        (run_dir / "tracks.txt").write_text("tracks\n", encoding="utf-8")

    def write_object_list(self, run_dir, object_labels):
        self.object_labels = object_labels
        (run_dir / "object_list_manifest.txt").write_text(str(sorted(object_labels)), encoding="utf-8")


class _FakeViewer:
    def __init__(self):
        self.states = None
        self.aggregate_results = None

    def replay(self, states, lane_box, aggregate_results):
        _ = lane_box
        self.states = states
        self.aggregate_results = aggregate_results


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
    assert set(fake_writer.object_labels.keys()) == {7, 8}
    assert fake_writer.object_labels[7].timestamp_ns == 200
    assert fake_writer.object_labels[7].frame_index == 1
    assert fake_writer.object_labels[8].timestamp_ns == 150
    assert fake_writer.object_labels[8].frame_index == 1


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
    assert set(fake_viewer.aggregate_results.keys()) == {1}
