from __future__ import annotations

import json
from pathlib import Path

from tracking_pipeline.application.run_pipeline import run_pipeline
from tracking_pipeline.config.loader import load_config

def test_algorithm_switching_produces_distinct_runs(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]

    euclidean_cfg = load_config(project_root / "configs" / "euclidean_voxel.yaml")
    euclidean_cfg.output.root_dir = str(tmp_path / "euclidean")
    euclidean_summary = run_pipeline(euclidean_cfg, project_root)

    registration_cfg = load_config(project_root / "configs" / "kalman_registration.yaml")
    registration_cfg.output.root_dir = str(tmp_path / "registration")
    registration_summary = run_pipeline(registration_cfg, project_root)

    euclidean_payload = json.loads((Path(euclidean_summary.output_dir) / "summary.json").read_text(encoding="utf-8"))
    registration_payload = json.loads((Path(registration_summary.output_dir) / "summary.json").read_text(encoding="utf-8"))

    assert euclidean_payload["tracker_algorithm"] == "euclidean_nn"
    assert euclidean_payload["accumulator_algorithm"] == "voxel_fusion"
    assert registration_payload["tracker_algorithm"] == "kalman_nn"
    assert registration_payload["accumulator_algorithm"] == "registration_voxel_fusion"


def test_hungarian_weighted_preset_runs(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    config = load_config(project_root / "configs" / "hungarian_weighted.yaml")
    config.output.root_dir = str(tmp_path / "hungarian_weighted")

    summary = run_pipeline(config, project_root)
    payload = json.loads((Path(summary.output_dir) / "summary.json").read_text(encoding="utf-8"))

    assert payload["tracker_algorithm"] == "hungarian_kalman"
    assert payload["clusterer_algorithm"] == "euclidean_clustering"
    assert payload["accumulator_algorithm"] == "weighted_voxel_fusion"


def test_new_registration_presets_run(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    fixture = project_root / "tests" / "fixtures" / "sample_a42.pb"

    generalized_cfg = load_config(project_root / "configs" / "kalman_generalized_icp.yaml")
    generalized_cfg.input.paths = [str(fixture)]
    generalized_cfg.output.root_dir = str(tmp_path / "generalized")

    feature_cfg = load_config(project_root / "configs" / "kalman_feature_global_then_local.yaml")
    feature_cfg.input.paths = [str(fixture)]
    feature_cfg.output.root_dir = str(tmp_path / "feature")

    generalized_summary = run_pipeline(generalized_cfg, project_root)
    feature_summary = run_pipeline(feature_cfg, project_root)

    generalized_payload = json.loads((Path(generalized_summary.output_dir) / "summary.json").read_text(encoding="utf-8"))
    feature_payload = json.loads((Path(feature_summary.output_dir) / "summary.json").read_text(encoding="utf-8"))
    assert generalized_payload["accumulator_algorithm"] == "registration_voxel_fusion"
    assert feature_payload["accumulator_algorithm"] == "registration_voxel_fusion"


def test_sensor_space_preset_runs(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    fixture = project_root / "tests" / "fixtures" / "sample_a42.pb"
    config = load_config(project_root / "configs" / "range_image_hungarian_weighted.yaml")
    config.input.paths = [str(fixture)]
    config.output.root_dir = str(tmp_path / "sensor")

    summary = run_pipeline(config, project_root)
    payload = json.loads((Path(summary.output_dir) / "summary.json").read_text(encoding="utf-8"))

    assert payload["tracker_algorithm"] == "hungarian_kalman"
    assert payload["clusterer_algorithm"] == "range_image_connected_components"
    assert payload["accumulator_algorithm"] == "weighted_voxel_fusion"


def test_long_vehicle_presets_run(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    fixture = project_root / "tests" / "fixtures" / "sample_a42.pb"
    preset_names = [
        "long_vehicle_hungarian_weighted.yaml",
        "long_vehicle_range_image_hungarian_weighted.yaml",
        "long_vehicle_beam_neighbor_hungarian_weighted.yaml",
        "long_vehicle_kalman_weighted.yaml",
        "long_vehicle_range_image_kalman_weighted.yaml",
        "long_vehicle_range_image_depth_jump_kalman_weighted.yaml",
        "long_vehicle_beam_neighbor_kalman_weighted.yaml",
    ]
    for preset_name in preset_names:
        config = load_config(project_root / "configs" / preset_name)
        config.input.paths = [str(fixture)]
        config.output.root_dir = str(tmp_path / preset_name.replace(".yaml", ""))
        summary = run_pipeline(config, project_root)
        payload = json.loads((Path(summary.output_dir) / "summary.json").read_text(encoding="utf-8"))
        assert payload["accumulator_algorithm"] == "weighted_voxel_fusion"
        if "kalman" in preset_name and "hungarian" not in preset_name:
            assert payload["tracker_algorithm"] == "kalman_nn"
        else:
            assert payload["tracker_algorithm"] == "hungarian_kalman"
