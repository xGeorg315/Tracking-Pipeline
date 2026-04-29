from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from tracking_pipeline.config.models import InputConfig, OutputConfig, PipelineConfig, PreprocessingConfig
from tracking_pipeline.domain.models import AggregateResult, GTMatchResult, ObjectLabelData, RunSummary, Track, TrackOutcomeDebug
from tracking_pipeline.infrastructure.io.dataset_artifact_writer import DatasetArtifactWriter


def _basic_config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        input=InputConfig(paths=["dummy.pb"]),
        preprocessing=PreprocessingConfig(lane_box=[-1.0, 1.0, 0.0, 10.0, 0.0, 2.0]),
        output=OutputConfig(mode="dataset", dataset_root_dir=str(tmp_path / "dataset")),
    )


def _track(track_id: int, timestamp_ns: int) -> Track:
    track = Track(track_id=track_id, hit_count=1, age=1, ended_by_missed=True)
    track.frame_ids = [0]
    track.frame_timestamps_ns = [int(timestamp_ns)]
    track.centers = [np.array([0.0, 0.0, 0.0], dtype=np.float32)]
    return track


def _summary(output_dir: Path) -> RunSummary:
    return RunSummary(
        input_path="dummy.pb",
        input_paths=["dummy.pb"],
        output_mode="dataset",
        tracker_algorithm="kalman_nn",
        accumulator_algorithm="voxel_fusion",
        clusterer_algorithm="dbscan",
        frame_count=1,
        finished_track_count=1,
        saved_aggregates=1,
        registration_attempts=0,
        registration_accepted=0,
        registration_rejected=0,
        output_dir=str(output_dir),
    )


def test_dataset_artifact_writer_writes_dataset_tree_and_stats(tmp_path: Path) -> None:
    config = _basic_config(tmp_path)
    writer = DatasetArtifactWriter(tmp_path)
    output_dir = writer.prepare_run_dir(config)
    writer.write_config_snapshot(output_dir, config)
    writer.begin_snapshot(output_dir)

    gt_car = ObjectLabelData(
        object_id=7,
        timestamp_ns=1000,
        points=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        obj_class="car",
        obj_class_score=0.9,
        sensor_name="sensor_a",
        frame_index=0,
        source_path="dummy.pb",
    )
    gt_van = ObjectLabelData(
        object_id=8,
        timestamp_ns=1001,
        points=np.array([[2.0, 0.0, 0.0]], dtype=np.float32),
        obj_class="van",
        obj_class_score=0.8,
        sensor_name="sensor_a",
        frame_index=0,
        source_path="dummy.pb",
    )
    result = AggregateResult(
        track_id=5,
        points=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        selected_frame_ids=[0],
        status="saved",
        metrics={"predicted_class_name": "car"},
    )
    match = GTMatchResult(
        track_id=5,
        gt_object_id=7,
        our_last_timestamp_ns=1000,
        gt_timestamp_ns=1000,
        timestamp_delta_ns=0,
        our_last_frame_id=0,
        gt_frame_index=0,
        assignment_cost=0.0,
        matched=True,
        gt_obj_class="car",
        gt_obj_class_score=0.9,
    )
    unmatched_gt = GTMatchResult(
        track_id=-1,
        gt_object_id=8,
        our_last_timestamp_ns=-1,
        gt_timestamp_ns=1001,
        timestamp_delta_ns=None,
        our_last_frame_id=-1,
        gt_frame_index=0,
        assignment_cost=None,
        matched=False,
        unmatched_reason="unmatched_gt",
        gt_obj_class="van",
        gt_obj_class_score=0.8,
    )

    writer.write_aggregate(output_dir, result)
    writer.write_object_list(output_dir, {7: gt_car, 8: gt_van})
    writer.write_gt_matching(
        output_dir,
        [match],
        [],
        [unmatched_gt],
        {
            "gt_match_saved_track_count": 1,
            "gt_match_matched_count": 1,
            "gt_match_unmatched_saved_count": 0,
            "gt_match_unmatched_gt_count": 1,
            "gt_match_mode": "timestamp_only",
            "gt_match_assignment": "one_to_one",
            "gt_match_mean_timestamp_delta_ns": 0.0,
            "gt_match_max_timestamp_delta_ns": 0,
        },
    )
    writer.write_tracks(output_dir, {5: _track(5, 1000)}, [result])
    writer.write_track_outcomes(
        output_dir,
        {
            5: TrackOutcomeDebug(
                track_id=5,
                status="saved",
                decision_stage="saved",
                decision_reason_code="saved",
                decision_summary="saved",
                last_frame_id=0,
            )
        },
    )
    writer.write_class_stats(output_dir, {"predicted_class_counts": {"car": 1}, "gt_class_counts": {"car": 1, "van": 1}})
    writer.write_summary(output_dir, _summary(output_dir))

    same_bucket = output_dir / "car" / "1970-01-01" / "gt-pred-same"
    unmatched_gt_bucket = output_dir / "van" / "1970-01-01" / "unmatched_gt"
    stats_root = output_dir / "_stats" / "1970-01-01"

    assert same_bucket.exists()
    assert list((same_bucket / "gt").glob("*.pcd"))
    assert list((same_bucket / "pred").glob("*.pcd"))
    assert list((same_bucket / "gt_matching").glob("*.json"))
    assert unmatched_gt_bucket.exists()
    assert list((unmatched_gt_bucket / "gt").glob("*.pcd"))
    stats_dirs = [path for path in stats_root.iterdir() if path.is_dir()]
    assert len(stats_dirs) == 1
    assert (stats_dirs[0] / "summary.json").exists()
    assert (stats_dirs[0] / "config.snapshot.yaml").exists()
    assert (stats_dirs[0] / "gt_matching" / "summary.json").exists()


def test_dataset_artifact_writer_reconciles_unmatched_gt_to_matched_bucket(tmp_path: Path) -> None:
    config = _basic_config(tmp_path)
    writer = DatasetArtifactWriter(tmp_path)
    output_dir = writer.prepare_run_dir(config)
    writer.write_config_snapshot(output_dir, config)

    gt_car = ObjectLabelData(
        object_id=7,
        timestamp_ns=1000,
        points=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        obj_class="car",
        obj_class_score=0.9,
        sensor_name="sensor_a",
        frame_index=0,
        source_path="dummy.pb",
    )

    writer.write_object_list(output_dir, {7: gt_car})
    unmatched_gt_bucket = output_dir / "car" / "1970-01-01" / "unmatched_gt"
    assert list(unmatched_gt_bucket.rglob("*.pcd"))

    writer.begin_snapshot(output_dir)
    result = AggregateResult(
        track_id=5,
        points=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        selected_frame_ids=[0],
        status="saved",
        metrics={"predicted_class_name": "car"},
    )
    match = GTMatchResult(
        track_id=5,
        gt_object_id=7,
        our_last_timestamp_ns=1000,
        gt_timestamp_ns=1000,
        timestamp_delta_ns=0,
        our_last_frame_id=0,
        gt_frame_index=0,
        assignment_cost=0.0,
        matched=True,
        gt_obj_class="car",
        gt_obj_class_score=0.9,
    )
    writer.write_aggregate(output_dir, result)
    writer.write_object_list(output_dir, {7: gt_car})
    writer.write_gt_matching(
        output_dir,
        [match],
        [],
        [],
        {
            "gt_match_saved_track_count": 1,
            "gt_match_matched_count": 1,
            "gt_match_unmatched_saved_count": 0,
            "gt_match_unmatched_gt_count": 0,
            "gt_match_mode": "timestamp_only",
            "gt_match_assignment": "one_to_one",
            "gt_match_mean_timestamp_delta_ns": 0.0,
            "gt_match_max_timestamp_delta_ns": 0,
        },
    )

    same_bucket = output_dir / "car" / "1970-01-01" / "gt-pred-same"
    manifest_rows = [
        json.loads(line)
        for line in (same_bucket / "gt_matching" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert not unmatched_gt_bucket.exists()
    assert list((same_bucket / "gt").glob("*.pcd"))
    assert list((same_bucket / "pred").glob("*.pcd"))
    assert len(manifest_rows) == 1
    assert manifest_rows[0]["bucket"] == "gt-pred-same"


def test_dataset_artifact_writer_keeps_existing_pred_files_unchanged_for_identical_snapshot(tmp_path: Path) -> None:
    config = _basic_config(tmp_path)
    writer = DatasetArtifactWriter(tmp_path)
    output_dir = writer.prepare_run_dir(config)
    writer.write_config_snapshot(output_dir, config)

    def build_gt_car() -> ObjectLabelData:
        return ObjectLabelData(
            object_id=7,
            timestamp_ns=1000,
            points=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            obj_class="car",
            obj_class_score=0.9,
            sensor_name="sensor_a",
            frame_index=0,
            source_path="dummy.pb",
        )

    def build_result() -> AggregateResult:
        return AggregateResult(
            track_id=5,
            points=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            selected_frame_ids=[0],
            status="saved",
            metrics={"predicted_class_name": "car", "predicted_class_score": 0.95},
        )

    match = GTMatchResult(
        track_id=5,
        gt_object_id=7,
        our_last_timestamp_ns=1000,
        gt_timestamp_ns=1000,
        timestamp_delta_ns=0,
        our_last_frame_id=0,
        gt_frame_index=0,
        assignment_cost=0.0,
        matched=True,
        gt_obj_class="car",
        gt_obj_class_score=0.9,
    )
    gt_summary = {
        "gt_match_saved_track_count": 1,
        "gt_match_matched_count": 1,
        "gt_match_unmatched_saved_count": 0,
        "gt_match_unmatched_gt_count": 0,
        "gt_match_mode": "timestamp_only",
        "gt_match_assignment": "one_to_one",
        "gt_match_mean_timestamp_delta_ns": 0.0,
        "gt_match_max_timestamp_delta_ns": 0,
    }

    writer.begin_snapshot(output_dir)
    writer.write_aggregate(output_dir, build_result())
    writer.begin_sample_batch()
    writer.write_object_list(output_dir, {7: build_gt_car()})
    writer.write_gt_matching(output_dir, [match], [], [], gt_summary)
    writer.end_sample_batch()

    pred_dir = output_dir / "car" / "1970-01-01" / "gt-pred-same" / "pred"
    pred_json_path = next(pred_dir.glob("*.json"))
    pred_pcd_path = next(pred_dir.glob("*.pcd"))
    first_json_mtime = pred_json_path.stat().st_mtime_ns
    first_pcd_mtime = pred_pcd_path.stat().st_mtime_ns

    time.sleep(0.02)

    writer.begin_snapshot(output_dir)
    writer.write_aggregate(output_dir, build_result())
    writer.begin_sample_batch()
    writer.write_object_list(output_dir, {7: build_gt_car()})
    writer.write_gt_matching(output_dir, [match], [], [], gt_summary)
    writer.end_sample_batch()

    assert pred_json_path.stat().st_mtime_ns == first_json_mtime
    assert pred_pcd_path.stat().st_mtime_ns == first_pcd_mtime


def test_dataset_artifact_writer_batches_stats_flushes(tmp_path: Path) -> None:
    class _CountingDatasetArtifactWriter(DatasetArtifactWriter):
        def __init__(self, project_root: Path):
            super().__init__(project_root)
            self.flush_calls = 0

        def _flush_stats_dirs(self) -> None:
            self.flush_calls += 1

    config = _basic_config(tmp_path)
    writer = _CountingDatasetArtifactWriter(tmp_path)
    output_dir = writer.prepare_run_dir(config)
    writer.write_config_snapshot(output_dir, config)
    writer.flush_calls = 0

    writer.begin_stats_batch()
    writer.write_tracks(output_dir, {5: _track(5, 1000)}, [])
    writer.write_track_outcomes(
        output_dir,
        {
            5: TrackOutcomeDebug(
                track_id=5,
                status="saved",
                decision_stage="saved",
                decision_reason_code="saved",
                decision_summary="saved",
                last_frame_id=0,
            )
        },
    )
    writer.write_class_stats(output_dir, {"predicted_class_counts": {"car": 1}})
    writer.write_summary(output_dir, _summary(output_dir))

    assert writer.flush_calls == 0

    writer.end_stats_batch()

    assert writer.flush_calls == 1


def test_dataset_artifact_writer_disables_stats_tree_but_keeps_dataset_samples(tmp_path: Path) -> None:
    config = PipelineConfig(
        input=InputConfig(paths=["dummy.pb"]),
        preprocessing=PreprocessingConfig(lane_box=[-1.0, 1.0, 0.0, 10.0, 0.0, 2.0]),
        output=OutputConfig(
            mode="dataset",
            dataset_root_dir=str(tmp_path / "dataset"),
            statistics_enabled=False,
        ),
    )
    writer = DatasetArtifactWriter(tmp_path)
    output_dir = writer.prepare_run_dir(config)
    writer.write_config_snapshot(output_dir, config)
    writer.begin_snapshot(output_dir)

    gt_car = ObjectLabelData(
        object_id=7,
        timestamp_ns=1000,
        points=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        obj_class="car",
        obj_class_score=0.9,
        sensor_name="sensor_a",
        frame_index=0,
        source_path="dummy.pb",
    )
    result = AggregateResult(
        track_id=5,
        points=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        selected_frame_ids=[0],
        status="saved",
        metrics={"predicted_class_name": "car"},
    )
    match = GTMatchResult(
        track_id=5,
        gt_object_id=7,
        our_last_timestamp_ns=1000,
        gt_timestamp_ns=1000,
        timestamp_delta_ns=0,
        our_last_frame_id=0,
        gt_frame_index=0,
        assignment_cost=0.0,
        matched=True,
        gt_obj_class="car",
        gt_obj_class_score=0.9,
    )

    writer.write_aggregate(output_dir, result)
    writer.begin_sample_batch()
    writer.write_object_list(output_dir, {7: gt_car})
    writer.write_gt_matching(
        output_dir,
        [match],
        [],
        [],
        {
            "gt_match_saved_track_count": 1,
            "gt_match_matched_count": 1,
            "gt_match_unmatched_saved_count": 0,
            "gt_match_unmatched_gt_count": 0,
            "gt_match_mode": "timestamp_only",
            "gt_match_assignment": "one_to_one",
            "gt_match_mean_timestamp_delta_ns": 0.0,
            "gt_match_max_timestamp_delta_ns": 0,
        },
    )
    writer.end_sample_batch()
    writer.write_tracks(output_dir, {5: _track(5, 1000)}, [result])
    writer.write_track_outcomes(
        output_dir,
        {
            5: TrackOutcomeDebug(
                track_id=5,
                status="saved",
                decision_stage="saved",
                decision_reason_code="saved",
                decision_summary="saved",
                last_frame_id=0,
            )
        },
    )
    writer.write_class_stats(output_dir, {"predicted_class_counts": {"car": 1}})
    writer.write_summary(output_dir, _summary(output_dir))

    same_bucket = output_dir / "car" / "1970-01-01" / "gt-pred-same"
    unmatched_gt_bucket = output_dir / "van" / "1970-01-01" / "unmatched_gt"
    assert list((same_bucket / "gt").glob("*.pcd"))
    assert list((same_bucket / "pred").glob("*.pcd"))
    assert list((same_bucket / "gt_matching").glob("*.json"))
    assert not unmatched_gt_bucket.exists()
    assert not (output_dir / "_stats").exists()


def test_dataset_artifact_writer_statistics_disabled_does_not_write_unmatched_gt_or_active_manifest(tmp_path: Path) -> None:
    config = PipelineConfig(
        input=InputConfig(paths=["dummy.pb"]),
        preprocessing=PreprocessingConfig(lane_box=[-1.0, 1.0, 0.0, 10.0, 0.0, 2.0]),
        output=OutputConfig(
            mode="dataset",
            dataset_root_dir=str(tmp_path / "dataset"),
            statistics_enabled=False,
        ),
    )
    writer = DatasetArtifactWriter(tmp_path)
    output_dir = writer.prepare_run_dir(config)
    writer.write_config_snapshot(output_dir, config)

    gt_car = ObjectLabelData(
        object_id=7,
        timestamp_ns=1000,
        points=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        obj_class="car",
        obj_class_score=0.9,
        sensor_name="sensor_a",
        frame_index=0,
        source_path="dummy.pb",
    )
    gt_van = ObjectLabelData(
        object_id=8,
        timestamp_ns=1001,
        points=np.array([[2.0, 0.0, 0.0]], dtype=np.float32),
        obj_class="van",
        obj_class_score=0.8,
        sensor_name="sensor_a",
        frame_index=0,
        source_path="dummy.pb",
    )
    result = AggregateResult(
        track_id=5,
        points=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        selected_frame_ids=[0],
        status="saved",
        metrics={"predicted_class_name": "car"},
    )
    match = GTMatchResult(
        track_id=5,
        gt_object_id=7,
        our_last_timestamp_ns=1000,
        gt_timestamp_ns=1000,
        timestamp_delta_ns=0,
        our_last_frame_id=0,
        gt_frame_index=0,
        assignment_cost=0.0,
        matched=True,
        gt_obj_class="car",
        gt_obj_class_score=0.9,
    )
    unmatched_gt = GTMatchResult(
        track_id=-1,
        gt_object_id=8,
        our_last_timestamp_ns=-1,
        gt_timestamp_ns=1001,
        timestamp_delta_ns=None,
        our_last_frame_id=-1,
        gt_frame_index=0,
        assignment_cost=None,
        matched=False,
        unmatched_reason="unmatched_gt",
        gt_obj_class="van",
        gt_obj_class_score=0.8,
    )

    writer.write_aggregate(output_dir, result)
    writer.begin_sample_batch()
    writer.write_object_list(output_dir, {7: gt_car, 8: gt_van})
    writer.write_gt_matching(
        output_dir,
        [match],
        [],
        [unmatched_gt],
        {
            "gt_match_saved_track_count": 1,
            "gt_match_matched_count": 1,
            "gt_match_unmatched_saved_count": 0,
            "gt_match_unmatched_gt_count": 1,
            "gt_match_mode": "timestamp_only",
            "gt_match_assignment": "one_to_one",
            "gt_match_mean_timestamp_delta_ns": 0.0,
            "gt_match_max_timestamp_delta_ns": 0,
        },
    )
    writer.end_sample_batch()

    assert (output_dir / "car" / "1970-01-01" / "gt-pred-same").exists()
    assert not (output_dir / "van" / "1970-01-01" / "unmatched_gt").exists()
    assert not (output_dir / "_stats").exists()
