from __future__ import annotations

import json
from pathlib import Path

import yaml

from tracking_pipeline.application.benchmark_run import BenchmarkRunner
from tracking_pipeline.config.loader import load_benchmark_config


def _copy_sample_pb(path: Path) -> Path:
    fixture = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "sample_a42.pb"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(fixture.read_bytes())
    return path


def test_benchmark_runner_writes_proxy_reports(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    fixture = project_root / "tests" / "fixtures" / "sample_a42.pb"
    manifest = tmp_path / "benchmark.yaml"
    manifest.write_text(
        "\n".join(
            [
                "name: smoke_proxy",
                f"output_root: {tmp_path / 'benchmarks'}",
                "warmup_runs: 0",
                "measure_runs: 2",
                "sequences:",
                f"  - {fixture}",
                "presets:",
                f"  - {project_root / 'configs' / 'euclidean_voxel.yaml'}",
                f"  - {project_root / 'configs' / 'hungarian_weighted.yaml'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = load_benchmark_config(manifest)
    output_dir = BenchmarkRunner(project_root).run(config)

    assert (output_dir / "results.csv").exists()
    assert (output_dir / "results.json").exists()
    assert (output_dir / "leaderboard.md").exists()
    assert (output_dir / "leaderboard_long_vehicle.md").exists()
    assert (output_dir / "performance_runs.jsonl").exists()
    assert (output_dir / "performance_leaderboard.md").exists()
    assert (output_dir / "performance_components.md").exists()

    payload = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    assert len(payload["results"]) == 2
    assert {row["preset_name"] for row in payload["results"]} == {"euclidean_voxel", "hungarian_weighted"}
    for row in payload["results"]:
        assert "mean_longitudinal_extent_saved" in row
        assert "mean_component_count_saved" in row
        assert "mean_tail_bridge_count_saved" in row
        assert "long_vehicle_saved_count" in row
        assert row["performance_samples"] == 2
        assert "total_wall_seconds_median" in row
        assert "total_hz_median" in row
        assert "compute_hz_median" in row
        assert "aggregation_registration_wall_seconds_median" in row
        assert "aggregation_post_filter_wall_seconds_median" in row
        assert "aggregation_tail_bridge_wall_seconds_median" in row
        assert "aggregation_confidence_cap_wall_seconds_median" in row
        assert "aggregation_symmetry_completion_wall_seconds_median" in row
        assert "aggregation_fusion_total_wall_seconds_median" in row
        assert "stage_read_frames_wall_seconds_median" in row
        assert "representative_run_dir" in row

    perf_rows = [
        json.loads(line)
        for line in (output_dir / "performance_runs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(perf_rows) == 4
    assert {row["phase"] for row in perf_rows} == {"measure"}
    assert "aggregation_registration_wall_seconds" in perf_rows[0]
    assert "aggregation_post_filter_wall_seconds" in perf_rows[0]
    assert "aggregation_tail_bridge_wall_seconds" in perf_rows[0]
    assert "aggregation_confidence_cap_wall_seconds" in perf_rows[0]
    assert "aggregation_symmetry_completion_wall_seconds" in perf_rows[0]
    assert "aggregation_fusion_total_wall_seconds" in perf_rows[0]
    assert "total_hz" in perf_rows[0]
    assert "compute_hz" in perf_rows[0]

    performance_markdown = (output_dir / "performance_leaderboard.md").read_text(encoding="utf-8")
    component_markdown = (output_dir / "performance_components.md").read_text(encoding="utf-8")
    assert "Total Hz Median" in performance_markdown
    assert "Compute Hz Median" in performance_markdown
    assert "Total Hz Median" in component_markdown
    assert "Compute Hz Median" in component_markdown


def test_benchmark_runner_records_registration_time_only_for_registration_presets(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    fixture = project_root / "tests" / "fixtures" / "sample_a42.pb"
    preset_dir = tmp_path / "presets"
    preset_dir.mkdir(parents=True, exist_ok=True)
    (preset_dir / "base.yaml").write_text((project_root / "configs" / "base.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    (preset_dir / "euclidean_voxel.yaml").write_text(
        "\n".join(
            [
                "input:",
                f"  paths:",
                f"    - {fixture}",
                "tracking:",
                "  algorithm: euclidean_nn",
                "aggregation:",
                "  algorithm: voxel_fusion",
                "  min_saved_aggregate_points: 0",
                "output:",
                "  require_track_exit: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (preset_dir / "euclidean_icp.yaml").write_text(
        "\n".join(
            [
                "input:",
                "  paths:",
                f"    - {fixture}",
                "tracking:",
                "  algorithm: euclidean_nn",
                "  min_track_hits: 1",
                "aggregation:",
                "  algorithm: registration_voxel_fusion",
                "  registration_backend: icp_point_to_plane",
                "  frame_selection_method: all_track_frames",
                "  frame_downsample_voxel: 0.0",
                "  registration_max_iter: 10",
                "  registration_min_fitness: 0.0",
                "  min_saved_aggregate_points: 0",
                "output:",
                "  require_track_exit: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "benchmark_registration.yaml"
    manifest.write_text(
        "\n".join(
            [
                "name: registration_components",
                f"output_root: {tmp_path / 'benchmarks'}",
                "warmup_runs: 0",
                "measure_runs: 1",
                "sequences:",
                f"  - {fixture}",
                "presets:",
                f"  - {preset_dir / 'euclidean_voxel.yaml'}",
                f"  - {preset_dir / 'euclidean_icp.yaml'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = load_benchmark_config(manifest)
    output_dir = BenchmarkRunner(project_root).run(config)
    perf_rows = [
        json.loads(line)
        for line in (output_dir / "performance_runs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_preset = {row["preset_name"]: row for row in perf_rows}

    assert by_preset["euclidean_voxel"]["aggregation_registration_wall_seconds"] == 0.0
    assert by_preset["euclidean_voxel"]["aggregation_fusion_total_wall_seconds"] >= 0.0
    assert by_preset["euclidean_icp"]["aggregation_registration_call_count"] > 0
    assert by_preset["euclidean_icp"]["aggregation_fusion_total_wall_seconds"] >= 0.0


def test_benchmark_runner_expands_directory_sequence_into_sorted_input_paths(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    sequence_dir = tmp_path / "sequence_dir"
    first = _copy_sample_pb(sequence_dir / "02_second.pb")
    second = _copy_sample_pb(sequence_dir / "01_first.pb")
    manifest = tmp_path / "benchmark_dir.yaml"
    manifest.write_text(
        "\n".join(
            [
                "name: dir_sequence",
                f"output_root: {tmp_path / 'benchmarks'}",
                "warmup_runs: 0",
                "measure_runs: 1",
                "sequences:",
                f"  - {sequence_dir}",
                "presets:",
                f"  - {project_root / 'configs' / 'euclidean_voxel.yaml'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = load_benchmark_config(manifest)
    output_dir = BenchmarkRunner(project_root).run(config)
    resolved_configs = sorted((output_dir / "resolved_configs").rglob("*.yaml"))

    assert len(resolved_configs) == 1
    resolved_payload = yaml.safe_load(resolved_configs[0].read_text(encoding="utf-8"))
    assert resolved_payload["input"]["paths"] == [str(second.resolve()), str(first.resolve())]

    results_payload = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    assert len(results_payload["results"]) == 1
    assert results_payload["results"][0]["sequence_name"] == sequence_dir.name
