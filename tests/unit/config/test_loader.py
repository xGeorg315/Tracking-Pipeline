from __future__ import annotations

from pathlib import Path

import pytest

from tracking_pipeline.config.loader import load_benchmark_config, load_config
from tracking_pipeline.config.validation import ConfigError


def _copy_sample_pb(path: Path) -> Path:
    fixture_src = Path(__file__).resolve().parents[2] / "fixtures" / "sample_a42.pb"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(fixture_src.read_bytes())
    return path


def test_load_config_resolves_relative_input_paths(tmp_path: Path) -> None:
    fixture_dst = _copy_sample_pb(tmp_path / "data" / "sample_a42.pb")

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


def test_load_config_reads_show_full_frame_pcd_flag(tmp_path: Path) -> None:
    fixture_dst = _copy_sample_pb(tmp_path / "data" / "sample_a42.pb")

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
                "visualization:",
                "  show_full_frame_pcd: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.input.paths == [str(fixture_dst.resolve())]
    assert config.visualization.show_full_frame_pcd is True


def test_load_config_reads_confidence_point_cap_settings(tmp_path: Path) -> None:
    fixture_dst = _copy_sample_pb(tmp_path / "data" / "sample_a42.pb")

    config_path = tmp_path / "config_confidence_cap.yaml"
    config_path.write_text(
        "\n".join(
            [
                "input:",
                "  paths:",
                "    - data/sample_a42.pb",
                "  format: a42_pb",
                "preprocessing:",
                "  lane_box: [-1.0, 1.0, 0.0, 10.0, 0.0, 2.0]",
                "aggregation:",
                "  enable_confidence_point_cap: true",
                "  confidence_point_cap_max_points: 1024",
                "  confidence_point_cap_bins: 8",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.input.paths == [str(fixture_dst.resolve())]
    assert config.aggregation.enable_confidence_point_cap is True
    assert config.aggregation.confidence_point_cap_max_points == 1024
    assert config.aggregation.confidence_point_cap_bins == 8


def test_load_config_reads_registration_underfill_fallback_settings(tmp_path: Path) -> None:
    fixture_dst = _copy_sample_pb(tmp_path / "data" / "sample_a42.pb")

    config_path = tmp_path / "config_registration_fallback.yaml"
    config_path.write_text(
        "\n".join(
            [
                "input:",
                "  paths:",
                "    - data/sample_a42.pb",
                "  format: a42_pb",
                "preprocessing:",
                "  lane_box: [-1.0, 1.0, 0.0, 10.0, 0.0, 2.0]",
                "aggregation:",
                "  enable_registration_underfill_fallback: true",
                "  registration_min_kept_chunks: 5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.input.paths == [str(fixture_dst.resolve())]
    assert config.aggregation.enable_registration_underfill_fallback is True
    assert config.aggregation.registration_min_kept_chunks == 5


@pytest.mark.parametrize("method", ["quality_coverage", "tail_coverage", "center_diversity"])
def test_load_config_accepts_new_frame_selection_methods(tmp_path: Path, method: str) -> None:
    fixture_dst = _copy_sample_pb(tmp_path / "data" / "sample_a42.pb")

    config_path = tmp_path / f"config_{method}.yaml"
    config_path.write_text(
        "\n".join(
            [
                "input:",
                "  paths:",
                "    - data/sample_a42.pb",
                "  format: a42_pb",
                "preprocessing:",
                "  lane_box: [-1.0, 1.0, 0.0, 10.0, 0.0, 2.0]",
                "aggregation:",
                f"  frame_selection_method: {method}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.input.paths == [str(fixture_dst.resolve())]
    assert config.aggregation.frame_selection_method == method


def test_load_config_rejects_invalid_confidence_point_cap_settings(tmp_path: Path) -> None:
    fixture_dst = _copy_sample_pb(tmp_path / "data" / "sample_a42.pb")
    _ = fixture_dst

    config_path = tmp_path / "config_invalid_confidence_cap.yaml"
    config_path.write_text(
        "\n".join(
            [
                "input:",
                "  paths:",
                "    - data/sample_a42.pb",
                "  format: a42_pb",
                "preprocessing:",
                "  lane_box: [-1.0, 1.0, 0.0, 10.0, 0.0, 2.0]",
                "aggregation:",
                "  enable_confidence_point_cap: true",
                "  confidence_point_cap_max_points: 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="aggregation.confidence_point_cap_max_points must be >= 1"):
        load_config(config_path)

    config_path.write_text(
        "\n".join(
            [
                "input:",
                "  paths:",
                "    - data/sample_a42.pb",
                "  format: a42_pb",
                "preprocessing:",
                "  lane_box: [-1.0, 1.0, 0.0, 10.0, 0.0, 2.0]",
                "aggregation:",
                "  enable_confidence_point_cap: true",
                "  confidence_point_cap_max_points: 32",
                "  confidence_point_cap_bins: 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="aggregation.confidence_point_cap_bins must be >= 1"):
        load_config(config_path)

    config_path.write_text(
        "\n".join(
            [
                "input:",
                "  paths:",
                "    - data/sample_a42.pb",
                "  format: a42_pb",
                "preprocessing:",
                "  lane_box: [-1.0, 1.0, 0.0, 10.0, 0.0, 2.0]",
                "aggregation:",
                "  enable_registration_underfill_fallback: true",
                "  registration_min_kept_chunks: 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="aggregation.registration_min_kept_chunks must be >= 1"):
        load_config(config_path)


def test_load_config_expands_directory_inputs_to_sorted_pb_files(tmp_path: Path) -> None:
    sequence_dir = tmp_path / "seq"
    first = _copy_sample_pb(sequence_dir / "02_middle.pb")
    second = _copy_sample_pb(sequence_dir / "01_first.pb")
    (sequence_dir / "notes.txt").write_text("ignore me", encoding="utf-8")

    config_path = tmp_path / "config_dir.yaml"
    config_path.write_text(
        "\n".join(
            [
                "input:",
                "  paths:",
                "    - seq",
                "  format: a42_pb",
                "preprocessing:",
                "  lane_box: [-1.0, 1.0, 0.0, 10.0, 0.0, 2.0]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.input.paths == [str(second.resolve()), str(first.resolve())]


def test_load_config_preserves_path_order_and_duplicate_files_when_mixing_files_and_directories(tmp_path: Path) -> None:
    standalone = _copy_sample_pb(tmp_path / "data" / "standalone.pb")
    seq_dir = tmp_path / "seq"
    seq_a = _copy_sample_pb(seq_dir / "a.pb")
    seq_b = _copy_sample_pb(seq_dir / "b.pb")

    config_path = tmp_path / "config_mixed.yaml"
    config_path.write_text(
        "\n".join(
            [
                "input:",
                "  paths:",
                "    - data/standalone.pb",
                "    - seq",
                "    - data/standalone.pb",
                "  format: a42_pb",
                "preprocessing:",
                "  lane_box: [-1.0, 1.0, 0.0, 10.0, 0.0, 2.0]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.input.paths == [
        str(standalone.resolve()),
        str(seq_a.resolve()),
        str(seq_b.resolve()),
        str(standalone.resolve()),
    ]


def test_load_config_rejects_directories_without_pb_files(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty_seq"
    empty_dir.mkdir()
    (empty_dir / "notes.txt").write_text("no pb here", encoding="utf-8")

    config_path = tmp_path / "config_empty_dir.yaml"
    config_path.write_text(
        "\n".join(
            [
                "input:",
                "  paths:",
                "    - empty_seq",
                "  format: a42_pb",
                "preprocessing:",
                "  lane_box: [-1.0, 1.0, 0.0, 10.0, 0.0, 2.0]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match=r"input\.paths directory does not contain any \.pb files"):
        load_config(config_path)


def test_load_benchmark_config_reads_performance_defaults(tmp_path: Path) -> None:
    preset_src = Path(__file__).resolve().parents[3] / "configs" / "kalman_voxel.yaml"
    data_dir = tmp_path / "data"
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    fixture_dst = _copy_sample_pb(data_dir / "sample_a42.pb")
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
    assert config.measure_runs == 1
    assert config.sequences == [str(fixture_dst.resolve())]
    assert config.presets == [str(preset_dst.resolve())]


def test_load_benchmark_config_preserves_explicit_measure_runs(tmp_path: Path) -> None:
    preset_src = Path(__file__).resolve().parents[3] / "configs" / "kalman_voxel.yaml"
    data_dir = tmp_path / "data"
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    fixture_dst = _copy_sample_pb(data_dir / "sample_a42.pb")
    preset_dst = configs_dir / "kalman_voxel.yaml"
    preset_dst.write_text(preset_src.read_text(encoding="utf-8"), encoding="utf-8")
    manifest = tmp_path / "benchmark_explicit_measure_runs.yaml"
    manifest.write_text(
        "\n".join(
            [
                "name: perf_explicit_measure_runs",
                "measure_runs: 3",
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

    assert config.measure_runs == 3
    assert config.sequences == [str(fixture_dst.resolve())]
    assert config.presets == [str(preset_dst.resolve())]


def test_load_benchmark_config_rejects_invalid_repeat_counts(tmp_path: Path) -> None:
    preset_src = Path(__file__).resolve().parents[3] / "configs" / "kalman_voxel.yaml"
    data_dir = tmp_path / "data"
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    fixture_dst = _copy_sample_pb(data_dir / "sample_a42.pb")
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


def test_load_benchmark_config_keeps_directory_sequences_unexpanded(tmp_path: Path) -> None:
    preset_src = Path(__file__).resolve().parents[3] / "configs" / "kalman_voxel.yaml"
    sequence_dir = tmp_path / "seq"
    configs_dir = tmp_path / "configs"
    _copy_sample_pb(sequence_dir / "a.pb")
    _copy_sample_pb(sequence_dir / "b.pb")
    configs_dir.mkdir()
    preset_dst = configs_dir / "kalman_voxel.yaml"
    preset_dst.write_text(preset_src.read_text(encoding="utf-8"), encoding="utf-8")
    manifest = tmp_path / "benchmark_dir.yaml"
    manifest.write_text(
        "\n".join(
            [
                "name: perf_dir",
                "sequences:",
                "  - seq",
                "presets:",
                "  - configs/kalman_voxel.yaml",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = load_benchmark_config(manifest)

    assert config.sequences == [str(sequence_dir.resolve())]
    assert config.presets == [str(preset_dst.resolve())]
