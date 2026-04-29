from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from tracking_pipeline.application.live_local_test import (
    AggregatePointCloudEntry,
    compare_aggregate_pointclouds,
    live_local_test,
)
from tracking_pipeline.config.models import (
    AggregationConfig,
    ClusteringConfig,
    InputConfig,
    OutputConfig,
    PipelineConfig,
    PreprocessingConfig,
    QB2LiveInputConfig,
    QB2LiveMQTTConfig,
    TrackingConfig,
)
from tracking_pipeline.domain.models import AggregateResult, ClusterResult, Detection, FrameData, FrameTrackingState, Track
from tracking_pipeline.infrastructure.io.pcd_writer import PCDWriter


def _entry(
    track_id: int,
    points: list[list[float]],
    *,
    frame_ids: list[int] | None = None,
    selected_frame_ids: list[int] | None = None,
    metrics: dict[str, object] | None = None,
) -> AggregatePointCloudEntry:
    return AggregatePointCloudEntry(
        track_id=track_id,
        points=np.asarray(points, dtype=np.float32),
        selected_frame_ids=list(selected_frame_ids or frame_ids or [0]),
        frame_ids=list(frame_ids or selected_frame_ids or [0]),
        metrics={"status": "saved", **dict(metrics or {})},
    )


def test_aggregate_compare_passes_for_same_points_in_different_order() -> None:
    report = compare_aggregate_pointclouds(
        [_entry(1, [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])],
        [_entry(1, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])],
        window_start_frame=0,
        window_end_frame=10,
        incomplete_margin_frames=0,
        tolerance=1e-5,
    )

    assert report["passed"] is True
    assert report["summary"]["compared"] == 1
    assert report["comparisons"][0]["max_abs_diff"] == 0.0


def test_aggregate_compare_fails_for_coordinate_difference_above_tolerance() -> None:
    report = compare_aggregate_pointclouds(
        [_entry(1, [[0.0, 0.0, 0.0]])],
        [_entry(1, [[0.0, 0.0, 0.001]])],
        window_start_frame=0,
        window_end_frame=10,
        incomplete_margin_frames=0,
        tolerance=1e-5,
    )

    assert report["passed"] is False
    assert report["summary"]["failed"] == 1
    assert report["comparisons"][0]["reason"] == "coordinate_mismatch"


def test_aggregate_compare_skips_tracks_near_segment_end() -> None:
    report = compare_aggregate_pointclouds(
        [_entry(1, [[0.0, 0.0, 0.0]], frame_ids=[9])],
        [_entry(1, [[0.0, 0.0, 0.0]], frame_ids=[9])],
        window_start_frame=0,
        window_end_frame=10,
        incomplete_margin_frames=2,
        tolerance=1e-5,
    )

    assert report["status"] == "no_comparable_aggregates"
    assert report["summary"]["skipped"] == 2
    assert {row["reason"] for row in report["skipped"]} == {"window_incomplete"}


class _FiniteLiveReader:
    def iter_frames(self, input_paths: list[str]):
        _ = input_paths
        for index in range(3):
            yield FrameData(
                frame_index=index,
                timestamp_ns=index * 1_000_000,
                points=np.array([[float(index), 0.0, 0.0]], dtype=np.float32),
            )

    def close(self) -> None:
        return None


class _OneDetectionClusterer:
    def cluster(self, frame: FrameData, lane_box):
        _ = lane_box
        detection = Detection(
            detection_id=1,
            points=frame.points,
            center=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            min_bound=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            max_bound=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )
        return ClusterResult(lane_points=frame.points, detections=[detection])


class _FinishedTracker:
    def __init__(self) -> None:
        self.track = Track(track_id=1)
        self.track.frame_ids = [0]
        self.track.frame_timestamps_ns = [0]
        self.track.centers = [np.array([0.0, 0.0, 0.0], dtype=np.float32)]
        self.track.world_points = [np.array([[0.0, 0.0, 0.0]], dtype=np.float32)]
        self.finished_tracks: dict[int, Track] = {}

    def step(self, detections, frame_idx: int, frame_timestamp_ns: int):
        _ = detections, frame_timestamp_ns
        if int(frame_idx) == 1:
            self.finished_tracks[1] = self.track
        return FrameTrackingState(
            frame_index=int(frame_idx),
            lane_points=np.zeros((0, 3), dtype=np.float32),
            detections=[],
            active_tracks=[],
        )

    def finalize(self):
        self.finished_tracks[1] = self.track
        return {1: self.track}


class _AggregateAccumulator:
    def accumulate(self, track: Track, lane_box):
        _ = track, lane_box
        return AggregateResult(
            track_id=1,
            points=np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32),
            selected_frame_ids=[0],
            status="saved",
            metrics={},
        )


def test_live_local_test_records_segment_and_writes_report(monkeypatch, tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
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
        tracking=TrackingConfig(max_missed=0),
        aggregation=AggregationConfig(),
        output=OutputConfig(root_dir=str(tmp_path / "runs"), statistics_enabled=True, final_full_recompute=False),
    )

    def _fake_subprocess(command, cwd):
        _ = cwd
        config_path = Path(command[-1])
        raw_config = config_path.read_text(encoding="utf-8")
        root_line = next(line for line in raw_config.splitlines() if line.startswith("  root_dir:"))
        local_root = Path(root_line.split(":", 1)[1].strip())
        output_dir = local_root / "local_result"
        aggregate_dir = output_dir / "aggregates"
        aggregate_dir.mkdir(parents=True, exist_ok=True)
        PCDWriter().write(aggregate_dir / "track_0001.pcd", np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32))
        (aggregate_dir / "track_0001.json").write_text(
            json.dumps({"track_id": 1, "status": "saved", "selected_frame_ids": [0], "metrics": {}}) + "\n",
            encoding="utf-8",
        )
        (output_dir / "tracks.jsonl").write_text(json.dumps({"track_id": 1, "frame_ids": [0]}) + "\n", encoding="utf-8")
        (output_dir / "summary.json").write_text("{}\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_reader", lambda cfg: _FiniteLiveReader())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_clusterer", lambda cfg: _OneDetectionClusterer())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_tracker", lambda cfg: _FinishedTracker())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_track_postprocessors", lambda cfg: [])
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_accumulator", lambda cfg: _AggregateAccumulator())
    monkeypatch.setattr("tracking_pipeline.application.run_pipeline.build_classifier", lambda cfg: None)

    report = live_local_test(
        config,
        project_root,
        duration_sec=0.001,
        compare_tolerance=1e-5,
        live_aggregate_timeout_sec=0.0,
        local_cpu_cores=1,
        subprocess_runner=_fake_subprocess,
    )

    assert report is not None
    assert report["passed"] is True
    assert report["summary"]["compared"] == 1
    assert Path(report["segment_dir"]).exists()
    assert Path(report["report_path"]).is_file()
