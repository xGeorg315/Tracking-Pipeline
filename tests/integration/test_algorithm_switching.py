from __future__ import annotations

import json
from pathlib import Path

import yaml

from tracking_pipeline.application.run_pipeline import run_pipeline
from tracking_pipeline.config.loader import load_config


def _load_preset_with_fixture(project_root: Path, preset_name: str, fixture: Path, tmp_path: Path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    for source_name in ("base.yaml", preset_name):
        source_path = project_root / "configs" / source_name
        payload = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        payload.setdefault("input", {})["paths"] = [str(fixture)]
        (config_dir / source_name).write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    return load_config(config_dir / preset_name)


def test_algorithm_switching_produces_distinct_runs(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    fixture = project_root / "tests" / "fixtures" / "sample_a42.pb"

    euclidean_cfg = _load_preset_with_fixture(project_root, "euclidean_voxel.yaml", fixture, tmp_path / "euclidean_cfg")
    euclidean_cfg.output.root_dir = str(tmp_path / "euclidean")
    euclidean_summary = run_pipeline(euclidean_cfg, project_root)

    registration_cfg = _load_preset_with_fixture(project_root, "kalman_registration.yaml", fixture, tmp_path / "registration_cfg")
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
    fixture = project_root / "tests" / "fixtures" / "sample_a42.pb"
    config = _load_preset_with_fixture(project_root, "hungarian_weighted.yaml", fixture, tmp_path / "hungarian_cfg")
    config.output.root_dir = str(tmp_path / "hungarian_weighted")

    summary = run_pipeline(config, project_root)
    payload = json.loads((Path(summary.output_dir) / "summary.json").read_text(encoding="utf-8"))

    assert payload["tracker_algorithm"] == "hungarian_kalman"
    assert payload["clusterer_algorithm"] == "euclidean_clustering"
    assert payload["accumulator_algorithm"] == "weighted_voxel_fusion"


def test_new_registration_presets_run(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    fixture = project_root / "tests" / "fixtures" / "sample_a42.pb"

    generalized_cfg = _load_preset_with_fixture(project_root, "kalman_generalized_icp.yaml", fixture, tmp_path / "generalized_cfg")
    generalized_cfg.output.root_dir = str(tmp_path / "generalized")

    feature_cfg = _load_preset_with_fixture(
        project_root,
        "kalman_feature_global_then_local.yaml",
        fixture,
        tmp_path / "feature_cfg",
    )
    feature_cfg.output.root_dir = str(tmp_path / "feature")

    kiss_cfg = _load_preset_with_fixture(
        project_root,
        "kalman_kiss_matcher_icp.yaml",
        fixture,
        tmp_path / "kiss_cfg",
    )
    kiss_cfg.output.root_dir = str(tmp_path / "kiss")

    kiss_only_cfg = _load_preset_with_fixture(
        project_root,
        "kalman_kiss_matcher.yaml",
        fixture,
        tmp_path / "kiss_only_cfg",
    )
    kiss_only_cfg.output.root_dir = str(tmp_path / "kiss_only")

    generalized_summary = run_pipeline(generalized_cfg, project_root)
    feature_summary = run_pipeline(feature_cfg, project_root)
    kiss_summary = run_pipeline(kiss_cfg, project_root)
    kiss_only_summary = run_pipeline(kiss_only_cfg, project_root)

    generalized_payload = json.loads((Path(generalized_summary.output_dir) / "summary.json").read_text(encoding="utf-8"))
    feature_payload = json.loads((Path(feature_summary.output_dir) / "summary.json").read_text(encoding="utf-8"))
    kiss_payload = json.loads((Path(kiss_summary.output_dir) / "summary.json").read_text(encoding="utf-8"))
    kiss_only_payload = json.loads((Path(kiss_only_summary.output_dir) / "summary.json").read_text(encoding="utf-8"))
    assert generalized_payload["accumulator_algorithm"] == "registration_voxel_fusion"
    assert feature_payload["accumulator_algorithm"] == "registration_voxel_fusion"
    assert kiss_payload["accumulator_algorithm"] == "registration_voxel_fusion"
    assert kiss_only_payload["accumulator_algorithm"] == "registration_voxel_fusion"


def test_sensor_space_preset_runs(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    fixture = project_root / "tests" / "fixtures" / "sample_a42.pb"
    config = _load_preset_with_fixture(project_root, "range_image_hungarian_weighted.yaml", fixture, tmp_path / "sensor_cfg")
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
        config = _load_preset_with_fixture(project_root, preset_name, fixture, tmp_path / preset_name.replace(".yaml", "_cfg"))
        config.output.root_dir = str(tmp_path / preset_name.replace(".yaml", ""))
        summary = run_pipeline(config, project_root)
        payload = json.loads((Path(summary.output_dir) / "summary.json").read_text(encoding="utf-8"))
        assert payload["accumulator_algorithm"] == "weighted_voxel_fusion"
        if "kalman" in preset_name and "hungarian" not in preset_name:
            assert payload["tracker_algorithm"] == "kalman_nn"
        else:
            assert payload["tracker_algorithm"] == "hungarian_kalman"
