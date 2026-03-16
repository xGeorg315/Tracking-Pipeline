from __future__ import annotations

from pathlib import Path

import pytest

from tracking_pipeline.config.loader import load_benchmark_config, load_config
from tracking_pipeline.config.validation import ConfigError


def test_load_config_resolves_relative_input_paths(tmp_path: Path) -> None:
    fixture_src = Path(__file__).resolve().parents[2] / "fixtures" / "sample_a42.pb"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    fixture_dst = data_dir / "sample_a42.pb"
    fixture_dst.write_bytes(fixture_src.read_bytes())

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "input:",
                "  paths:",
                "    - data/sample_a42.pb",
                "  format: a42_pb",
                "preprocessing:",
                "  lane_box: [-1.0, 1.0, 0.0, 10.0, 0.0, 2.0]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.input.paths == [str(fixture_dst.resolve())]


def test_load_config_rejects_empty_input_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "input:",
                "  paths: []",
                "  format: a42_pb",
                "preprocessing:",
                "  lane_box: [-1.0, 1.0, 0.0, 10.0, 0.0, 2.0]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="input.paths must not be empty"):
        load_config(config_path)


def test_load_benchmark_config_reads_performance_defaults(tmp_path: Path) -> None:
    fixture_src = Path(__file__).resolve().parents[2] / "fixtures" / "sample_a42.pb"
    preset_src = Path(__file__).resolve().parents[3] / "configs" / "kalman_voxel.yaml"
    data_dir = tmp_path / "data"
    configs_dir = tmp_path / "configs"
    data_dir.mkdir()
    configs_dir.mkdir()
    fixture_dst = data_dir / "sample_a42.pb"
    fixture_dst.write_bytes(fixture_src.read_bytes())
    preset_dst = configs_dir / "kalman_voxel.yaml"
    preset_dst.write_text(preset_src.read_text(encoding="utf-8"), encoding="utf-8")
    manifest = tmp_path / "benchmark.yaml"
    manifest.write_text(
        "\n".join(
            [
                "name: perf_defaults",
                "sequences:",
                "  - data/sample_a42.pb",
                "presets:",
                "  - configs/kalman_voxel.yaml",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = load_benchmark_config(manifest)

    assert config.warmup_runs == 1
    assert config.measure_runs == 3
    assert config.sequences == [str(fixture_dst.resolve())]
    assert config.presets == [str(preset_dst.resolve())]


def test_load_benchmark_config_rejects_invalid_repeat_counts(tmp_path: Path) -> None:
    fixture_src = Path(__file__).resolve().parents[2] / "fixtures" / "sample_a42.pb"
    preset_src = Path(__file__).resolve().parents[3] / "configs" / "kalman_voxel.yaml"
    data_dir = tmp_path / "data"
    configs_dir = tmp_path / "configs"
    data_dir.mkdir()
    configs_dir.mkdir()
    fixture_dst = data_dir / "sample_a42.pb"
    fixture_dst.write_bytes(fixture_src.read_bytes())
    preset_dst = configs_dir / "kalman_voxel.yaml"
    preset_dst.write_text(preset_src.read_text(encoding="utf-8"), encoding="utf-8")
    manifest = tmp_path / "benchmark_invalid.yaml"
    manifest.write_text(
        "\n".join(
            [
                "name: perf_invalid",
                "warmup_runs: -1",
                "measure_runs: 0",
                "sequences:",
                "  - data/sample_a42.pb",
                "presets:",
                "  - configs/kalman_voxel.yaml",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="benchmark.warmup_runs must be >= 0"):
        load_benchmark_config(manifest)
