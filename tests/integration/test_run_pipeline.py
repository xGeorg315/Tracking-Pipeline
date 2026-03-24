from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from tracking_pipeline.application.run_pipeline import run_pipeline
from tracking_pipeline.config.loader import load_config
from tracking_pipeline.infrastructure.readers.a42_proto import common, data, frame, label, sensors


def _pack_xyz(points: list[list[float]]) -> bytes:
    return np.asarray(points, dtype="<f4").reshape(-1, 3).tobytes()


def _copy_sample_pb(path: Path) -> Path:
    fixture = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "sample_a42.pb"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(fixture.read_bytes())
    return path


def _write_object_list_fixture(path: Path) -> Path:
    calibration = common.SensorCalibration(sensor_name="sensor_a")
    frames = [
        frame.Frame(
            frame_timestamp_ns=1000,
            frame_id=1,
            lidars=[
                sensors.LidarScan(
                    scan_timestamp_ns=1000,
                    pointcloud=data.PointCloud(cartesian=_pack_xyz([[0.0, 0.0, 0.0]])),
                    calibration=calibration,
                    object_list=[
                        label.ObjectBBox(
                            id=7,
                            timestamp_ns=1000,
                            pointcloud=data.PointCloud(cartesian=_pack_xyz([[1.0, 0.0, 0.0]])),
                            obj_class="car",
                            obj_class_score=0.9,
                        ),
                        label.ObjectBBox(
                            id=9,
                            timestamp_ns=1001,
                            pointcloud=data.PointCloud(cartesian=b""),
                            obj_class="truck",
                            obj_class_score=0.7,
                        ),
                    ],
                )
            ],
        ),
        frame.Frame(
            frame_timestamp_ns=1001,
            frame_id=2,
            lidars=[
                sensors.LidarScan(
                    scan_timestamp_ns=1001,
                    pointcloud=data.PointCloud(cartesian=_pack_xyz([[0.1, 0.0, 0.0]])),
                    calibration=calibration,
                    object_list=[
                        label.ObjectBBox(
                            id=7,
                            timestamp_ns=1002,
                            pointcloud=data.PointCloud(cartesian=_pack_xyz([[2.0, 0.0, 0.0], [2.5, 0.0, 0.0]])),
                            obj_class="car",
                            obj_class_score=0.95,
                        )
                    ],
                )
            ],
        ),
    ]
    with path.open("wb") as handle:
        for item in frames:
            payload = bytes(item)
            handle.write(struct.pack("<I", len(payload)))
            handle.write(payload)
    return path


def test_run_pipeline_creates_run_artifacts(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    config = load_config(project_root / "configs" / "euclidean_voxel.yaml")
    config.aggregation.enable_tail_bridge = False
    config.output.root_dir = str(tmp_path)

    summary = run_pipeline(config, project_root)
    run_dir = Path(summary.output_dir)

    assert run_dir.exists()
    assert (run_dir / "config.snapshot.yaml").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "tracks.jsonl").exists()
    assert (run_dir / "gt_matching").exists()
    assert (run_dir / "gt_matching" / "matches.jsonl").exists()
    assert (run_dir / "gt_matching" / "unmatched_saved_tracks.jsonl").exists()
    assert (run_dir / "gt_matching" / "unmatched_gt_objects.jsonl").exists()
    assert (run_dir / "gt_matching" / "summary.json").exists()
    payload = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    track_rows = [
        json.loads(line)
        for line in (run_dir / "tracks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert payload["tracker_algorithm"] == "euclidean_nn"
    assert payload["accumulator_algorithm"] == "voxel_fusion"
    assert payload["input_paths"] == config.input.paths
    assert payload["gt_match_mode"] == "timestamp_only"
    assert payload["gt_match_assignment"] == "one_to_one"
    assert "performance" in payload
    assert "read_frames" in payload["performance"]["stages"]
    assert "cluster_frames" in payload["performance"]["stages"]
    assert "aggregation_components" in payload["performance"]
    assert set(payload["performance"]["aggregation_components"]) == {
        "registration",
        "fusion_core",
        "post_filter",
        "tail_bridge",
        "confidence_cap",
        "symmetry_completion",
        "fusion_post",
        "fusion_total",
    }
    assert payload["performance"]["aggregation_components"]["registration"]["wall_seconds"] == 0.0
    assert payload["performance"]["aggregation_components"]["tail_bridge"]["wall_seconds"] == 0.0
    assert payload["performance"]["aggregation_components"]["tail_bridge"]["call_count"] == 0
    assert payload["performance"]["total_wall_seconds"] >= payload["performance"]["compute_wall_seconds"]
    assert track_rows
    assert "registration_wall_seconds" in track_rows[0]["aggregation_metrics"]
    assert "post_filter_wall_seconds" in track_rows[0]["aggregation_metrics"]
    assert "tail_bridge_wall_seconds" in track_rows[0]["aggregation_metrics"]
    assert track_rows[0]["aggregation_metrics"]["tail_bridge_wall_seconds"] == 0.0
    assert "confidence_cap_wall_seconds" in track_rows[0]["aggregation_metrics"]
    assert "symmetry_completion_wall_seconds" in track_rows[0]["aggregation_metrics"]
    assert "fusion_total_wall_seconds" in track_rows[0]["aggregation_metrics"]


def test_run_pipeline_accepts_multiple_input_files_as_one_sequence(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    fixture = project_root / "tests" / "fixtures" / "sample_a42.pb"
    config = load_config(project_root / "configs" / "euclidean_voxel.yaml")
    config.input.paths = [str(fixture), str(fixture)]
    config.output.root_dir = str(tmp_path / "multi")

    summary = run_pipeline(config, project_root)
    payload = json.loads((Path(summary.output_dir) / "summary.json").read_text(encoding="utf-8"))

    assert summary.frame_count == 20
    assert payload["frame_count"] == 20
    assert payload["input_paths"] == [str(fixture), str(fixture)]


def test_run_pipeline_expands_directory_input_to_sorted_pb_sequence(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    sequence_dir = tmp_path / "sequence"
    first = _copy_sample_pb(sequence_dir / "02_second.pb")
    second = _copy_sample_pb(sequence_dir / "01_first.pb")
    (sequence_dir / "ignore.txt").write_text("not a sequence file", encoding="utf-8")

    config_path = tmp_path / "dir_input.yaml"
    config_path.write_text(
        "\n".join(
            [
                "input:",
                "  paths:",
                "    - sequence",
                "  format: a42_pb",
                "preprocessing:",
                "  lane_box: [-2.10, 1.80, 4.0, 35.30, 0.12, 5.15]",
                "tracking:",
                "  algorithm: euclidean_nn",
                "output:",
                f"  root_dir: {tmp_path / 'dir_runs'}",
                "  require_track_exit: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config = load_config(config_path)

    summary = run_pipeline(config, project_root)
    payload = json.loads((Path(summary.output_dir) / "summary.json").read_text(encoding="utf-8"))

    assert summary.frame_count == 20
    assert payload["frame_count"] == 20
    assert payload["input_paths"] == [str(second.resolve()), str(first.resolve())]
    assert payload["input_path"] == str(second.resolve())


def test_run_pipeline_exports_latest_object_list_artifacts(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    fixture = _write_object_list_fixture(tmp_path / "object_list.pb")
    config = load_config(project_root / "configs" / "euclidean_voxel.yaml")
    config.input.paths = [str(fixture)]
    config.output.root_dir = str(tmp_path / "object_export")

    summary = run_pipeline(config, project_root)
    run_dir = Path(summary.output_dir)
    payload = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    manifest_rows = [
        json.loads(line)
        for line in (run_dir / "object_list" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    gt_match_rows = [
        json.loads(line)
        for line in (run_dir / "gt_matching" / "matches.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    track_rows = [
        json.loads(line)
        for line in (run_dir / "tracks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    unmatched_gt_rows = [
        json.loads(line)
        for line in (run_dir / "gt_matching" / "unmatched_gt_objects.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert (run_dir / "object_list").exists()
    assert (run_dir / "gt_matching").exists()
    assert (run_dir / "object_list" / "object_0007.pcd").exists()
    assert not (run_dir / "object_list" / "object_0009.pcd").exists()
    assert payload["object_list_exported_count"] == 1
    assert payload["object_list_seen_ids"] == 2
    assert payload["object_list_skipped_empty"] == 1
    assert len(manifest_rows) == 1
    assert manifest_rows[0]["object_id"] == 7
    assert manifest_rows[0]["timestamp_ns"] == 1002
    assert manifest_rows[0]["frame_index"] == 1
    assert manifest_rows[0]["point_count"] == 2
    assert payload["gt_match_saved_track_count"] >= 0
    assert payload["gt_match_mode"] == "timestamp_only"
    assert payload["gt_match_assignment"] == "one_to_one"
    assert len(gt_match_rows) + len(unmatched_gt_rows) >= 1
    matched_track_rows = [row for row in track_rows if row.get("gt_matched") is True]
    if matched_track_rows:
        assert matched_track_rows[0]["matched_gt_pcd_path"] == "object_list/object_0007.pcd"
