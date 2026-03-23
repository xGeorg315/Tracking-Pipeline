from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from tracking_pipeline.application.performance import STAGE_NAMES
from tracking_pipeline.application.services import build_benchmark_name, resolve_benchmark_root
from tracking_pipeline.config.loader import load_config, resolve_input_paths
from tracking_pipeline.config.models import BenchmarkConfig, PipelineConfig
from tracking_pipeline.infrastructure.io.manifest_writer import ManifestWriter


PERFORMANCE_TOTAL_FIELDS = (
    "total_wall_seconds",
    "total_cpu_seconds",
    "compute_wall_seconds",
    "compute_cpu_seconds",
    "io_wall_seconds",
    "peak_rss_mb",
    "wall_ms_per_frame",
    "cpu_ms_per_frame",
    "accumulate_wall_ms_per_track",
)
COMPONENT_FIELDS = (
    "tracker_algorithm",
    "clusterer_algorithm",
    "accumulator_algorithm",
    "registration_backend",
)
ROW_CONTEXT_FIELDS = (
    "sequence_name",
    "tracker_algorithm",
    "clusterer_algorithm",
    "accumulator_algorithm",
    "registration_backend",
    "frame_selection_method",
    "long_vehicle_mode",
)


class BenchmarkRunner:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.manifest_writer = ManifestWriter()

    def run(self, config: BenchmarkConfig) -> Path:
        benchmark_root = resolve_benchmark_root(config, self.project_root)
        benchmark_dir = benchmark_root / build_benchmark_name(config)
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        runs_root = benchmark_dir / "runs"
        config_root = benchmark_dir / "resolved_configs"
        rows: list[dict[str, Any]] = []
        performance_runs: list[dict[str, Any]] = []

        for sequence_path in config.sequences:
            sequence_name = Path(sequence_path).stem
            for preset_path in config.presets:
                preset_name = Path(preset_path).stem
                measured_rows: list[dict[str, Any]] = []
                for phase, count in (("warmup", config.warmup_runs), ("measure", config.measure_runs)):
                    for run_index in range(1, count + 1):
                        pipeline_config = load_config(preset_path)
                        pipeline_config.input.paths = resolve_input_paths(
                            [sequence_path],
                            config.config_path.parent if config.config_path is not None else self.project_root,
                            field_name="benchmark.sequences",
                        )
                        run_root = (runs_root / sequence_name / preset_name / f"{phase}_{run_index:02d}").resolve()
                        pipeline_config.output.root_dir = str(run_root)
                        resolved_config_path = self._write_resolved_config(
                            config_root,
                            pipeline_config,
                            sequence_name,
                            preset_name,
                            phase,
                            run_index,
                        )
                        summary_payload = self._execute_run(resolved_config_path, run_root)
                        row = self._build_run_row(
                            summary_payload=summary_payload,
                            sequence_name=sequence_name,
                            preset_name=preset_name,
                            run_index=run_index,
                            phase=phase,
                            pipeline_config=pipeline_config,
                        )
                        if phase == "measure":
                            measured_rows.append(row)
                            performance_runs.append(row)

                rows.append(self._aggregate_rows(measured_rows))

        rows.sort(key=self._sort_key)
        self._write_outputs(benchmark_dir, config, rows, performance_runs)
        return benchmark_dir

    def _write_resolved_config(
        self,
        config_root: Path,
        pipeline_config: PipelineConfig,
        sequence_name: str,
        preset_name: str,
        phase: str,
        run_index: int,
    ) -> Path:
        path = config_root / sequence_name / preset_name / f"{phase}_{run_index:02d}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(pipeline_config.to_dict(), handle, sort_keys=False)
        return path

    def _execute_run(self, config_path: Path, run_root: Path) -> dict[str, Any]:
        env = os.environ.copy()
        src_path = str((self.project_root / "src").resolve())
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = src_path if not existing_pythonpath else f"{src_path}{os.pathsep}{existing_pythonpath}"
        command = [sys.executable, "-m", "tracking_pipeline.cli", "run", "-c", str(config_path)]
        try:
            subprocess.run(
                command,
                cwd=self.project_root,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:  # pragma: no cover - exercised through integration failure
            raise RuntimeError(exc.stderr.strip() or exc.stdout.strip() or f"Run failed for {config_path}") from exc

        summary_paths = sorted(run_root.rglob("summary.json"))
        if len(summary_paths) != 1:
            raise RuntimeError(f"Expected exactly one summary.json under {run_root}, found {len(summary_paths)}")
        return json.loads(summary_paths[0].read_text(encoding="utf-8"))

    def _build_run_row(
        self,
        summary_payload: dict[str, Any],
        sequence_name: str,
        preset_name: str,
        run_index: int,
        phase: str,
        pipeline_config: PipelineConfig,
    ) -> dict[str, Any]:
        run_dir = Path(str(summary_payload["output_dir"]))
        tracks_path = run_dir / "tracks.jsonl"
        track_rows = [json.loads(line) for line in tracks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        saved_rows = [row for row in track_rows if row.get("aggregate_status") == "saved"]
        mean_selected_frames = (
            sum(len(row.get("selected_frame_ids", [])) for row in saved_rows) / len(saved_rows) if saved_rows else 0.0
        )
        mean_points = (
            sum(float(row.get("aggregation_metrics", {}).get("point_count_after_downsample", 0)) for row in saved_rows) / len(saved_rows)
            if saved_rows
            else 0.0
        )
        longitudinal_extents = [
            float(row.get("aggregation_metrics", {}).get("longitudinal_extent", 0.0))
            for row in saved_rows
            if float(row.get("aggregation_metrics", {}).get("longitudinal_extent", 0.0)) > 0.0
        ]
        component_counts = [
            float(row.get("aggregation_metrics", {}).get("component_count_post_fusion", 0.0))
            for row in saved_rows
        ]
        tail_bridge_counts = [
            float(row.get("aggregation_metrics", {}).get("tail_bridge_count", 0.0))
            for row in saved_rows
        ]
        long_vehicle_saved_count = sum(1 for row in saved_rows if self._is_long_vehicle_row(row))
        registration_attempts = int(summary_payload.get("registration_attempts", 0))
        registration_accepted = int(summary_payload.get("registration_accepted", 0))
        finished_track_count = int(summary_payload.get("finished_track_count", 0))
        frame_count = int(summary_payload.get("frame_count", 0))
        aggregate_save_rate = float(summary_payload.get("saved_aggregates", 0)) / float(finished_track_count) if finished_track_count else 0.0
        registration_accept_rate = float(registration_accepted) / float(registration_attempts) if registration_attempts else 0.0
        performance = dict(summary_payload.get("performance") or {})
        stage_data = dict(performance.get("stages") or {})
        stage_metrics = self._stage_metrics_from_summary(stage_data)
        total_wall_seconds = float(performance.get("total_wall_seconds", 0.0) or 0.0)
        total_cpu_seconds = float(performance.get("total_cpu_seconds", 0.0) or 0.0)
        track_count_for_accumulate = max(1, finished_track_count)
        return {
            "sequence_name": sequence_name,
            "preset_name": preset_name,
            "phase": phase,
            "run_index": run_index,
            "tracker_algorithm": str(summary_payload.get("tracker_algorithm", "")),
            "clusterer_algorithm": str(summary_payload.get("clusterer_algorithm", "")),
            "accumulator_algorithm": str(summary_payload.get("accumulator_algorithm", "")),
            "registration_backend": str(pipeline_config.aggregation.registration_backend),
            "frame_selection_method": str(pipeline_config.aggregation.frame_selection_method),
            "long_vehicle_mode": bool(pipeline_config.aggregation.long_vehicle_mode),
            "frame_count": frame_count,
            "finished_track_count": finished_track_count,
            "saved_aggregates": int(summary_payload.get("saved_aggregates", 0)),
            "aggregate_save_rate": round(aggregate_save_rate, 6),
            "registration_attempts": registration_attempts,
            "registration_accept_rate": round(registration_accept_rate, 6),
            "track_quality_mean": round(float(summary_payload.get("track_quality_mean", 0.0) or 0.0), 6),
            "mean_selected_frames_per_saved_aggregate": round(float(mean_selected_frames), 6),
            "mean_points_per_saved_aggregate": round(float(mean_points), 6),
            "mean_longitudinal_extent_saved": round(self._mean(longitudinal_extents), 6),
            "p90_longitudinal_extent_saved": round(self._percentile(longitudinal_extents, 90.0), 6),
            "mean_component_count_saved": round(self._mean(component_counts), 6),
            "mean_tail_bridge_count_saved": round(self._mean(tail_bridge_counts), 6),
            "long_vehicle_saved_count": int(long_vehicle_saved_count),
            "aggregate_status_counts": summary_payload.get("aggregate_status_counts", {}),
            "run_dir": str(run_dir),
            "runtime_seconds": round(total_wall_seconds, 6),
            "total_wall_seconds": round(total_wall_seconds, 6),
            "total_cpu_seconds": round(total_cpu_seconds, 6),
            "compute_wall_seconds": round(float(performance.get("compute_wall_seconds", 0.0) or 0.0), 6),
            "compute_cpu_seconds": round(float(performance.get("compute_cpu_seconds", 0.0) or 0.0), 6),
            "io_wall_seconds": round(float(performance.get("io_wall_seconds", 0.0) or 0.0), 6),
            "peak_rss_mb": self._round_optional(performance.get("peak_rss_mb")),
            "wall_ms_per_frame": round((total_wall_seconds * 1000.0) / float(max(1, frame_count)), 6),
            "cpu_ms_per_frame": round((total_cpu_seconds * 1000.0) / float(max(1, frame_count)), 6),
            "accumulate_wall_ms_per_track": round(
                (float(stage_metrics["stage_accumulate_tracks_wall_seconds"]) * 1000.0) / float(track_count_for_accumulate),
                6,
            ),
            **stage_metrics,
        }

    def _stage_metrics_from_summary(self, stage_data: dict[str, Any]) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        for stage_name in STAGE_NAMES:
            payload = stage_data.get(stage_name) or {}
            metrics[f"stage_{stage_name}_wall_seconds"] = round(float(payload.get("wall_seconds", 0.0) or 0.0), 6)
            metrics[f"stage_{stage_name}_cpu_seconds"] = round(float(payload.get("cpu_seconds", 0.0) or 0.0), 6)
            metrics[f"stage_{stage_name}_call_count"] = int(payload.get("call_count", 0) or 0)
        return metrics

    def _aggregate_rows(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            raise RuntimeError("Expected at least one measured run for aggregation")

        representative = self._representative_row(rows)
        aggregated = {
            key: representative[key]
            for key in (
                "sequence_name",
                "preset_name",
                "tracker_algorithm",
                "clusterer_algorithm",
                "accumulator_algorithm",
                "registration_backend",
                "frame_selection_method",
                "long_vehicle_mode",
                "frame_count",
                "finished_track_count",
                "saved_aggregates",
                "aggregate_save_rate",
                "registration_attempts",
                "registration_accept_rate",
                "track_quality_mean",
                "mean_selected_frames_per_saved_aggregate",
                "mean_points_per_saved_aggregate",
                "mean_longitudinal_extent_saved",
                "p90_longitudinal_extent_saved",
                "mean_component_count_saved",
                "mean_tail_bridge_count_saved",
                "long_vehicle_saved_count",
                "aggregate_status_counts",
            )
        }
        aggregated["performance_samples"] = len(rows)
        aggregated["representative_run_dir"] = representative["run_dir"]
        aggregated["run_dir"] = representative["run_dir"]
        for field in PERFORMANCE_TOTAL_FIELDS:
            values = [self._float_or_zero(row.get(field)) for row in rows]
            aggregated[f"{field}_median"] = round(self._median(values), 6)
            aggregated[f"{field}_min"] = round(min(values), 6)
            aggregated[f"{field}_max"] = round(max(values), 6)
        for stage_name in STAGE_NAMES:
            for suffix in ("wall_seconds", "cpu_seconds"):
                field = f"stage_{stage_name}_{suffix}"
                values = [self._float_or_zero(row.get(field)) for row in rows]
                aggregated[f"{field}_median"] = round(self._median(values), 6)
                aggregated[f"{field}_min"] = round(min(values), 6)
                aggregated[f"{field}_max"] = round(max(values), 6)
            call_field = f"stage_{stage_name}_call_count"
            aggregated[call_field] = int(representative.get(call_field, 0))
        aggregated["runtime_seconds"] = aggregated["total_wall_seconds_median"]
        return aggregated

    def _representative_row(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        median_wall = self._median([self._float_or_zero(row.get("total_wall_seconds")) for row in rows])
        return min(
            rows,
            key=lambda row: (
                abs(self._float_or_zero(row.get("total_wall_seconds")) - median_wall),
                int(row.get("run_index", 0)),
            ),
        )

    def _write_outputs(
        self,
        benchmark_dir: Path,
        config: BenchmarkConfig,
        rows: list[dict[str, Any]],
        performance_runs: list[dict[str, Any]],
    ) -> None:
        payload = {
            "config": config.to_dict(),
            "results": rows,
        }
        self.manifest_writer.write_json(benchmark_dir / "results.json", payload)
        self._write_csv(benchmark_dir / "results.csv", rows)
        self.manifest_writer.write_jsonl(benchmark_dir / "performance_runs.jsonl", performance_runs)
        (benchmark_dir / "leaderboard.md").write_text(self._render_markdown(rows), encoding="utf-8")
        (benchmark_dir / "leaderboard_long_vehicle.md").write_text(self._render_long_vehicle_markdown(rows), encoding="utf-8")
        (benchmark_dir / "performance_leaderboard.md").write_text(self._render_performance_markdown(rows), encoding="utf-8")
        (benchmark_dir / "performance_components.md").write_text(self._render_component_markdown(rows), encoding="utf-8")

    def _write_csv(self, path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        flat_rows = []
        for row in rows:
            flat = dict(row)
            flat["aggregate_status_counts"] = json.dumps(row.get("aggregate_status_counts", {}), sort_keys=True)
            flat_rows.append(flat)
        fieldnames = self._csv_fieldnames()
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_rows)

    def _csv_fieldnames(self) -> list[str]:
        fieldnames = [
            "sequence_name",
            "preset_name",
            "tracker_algorithm",
            "clusterer_algorithm",
            "accumulator_algorithm",
            "registration_backend",
            "frame_selection_method",
            "long_vehicle_mode",
            "frame_count",
            "finished_track_count",
            "saved_aggregates",
            "aggregate_save_rate",
            "registration_attempts",
            "registration_accept_rate",
            "track_quality_mean",
            "mean_selected_frames_per_saved_aggregate",
            "mean_points_per_saved_aggregate",
            "mean_longitudinal_extent_saved",
            "p90_longitudinal_extent_saved",
            "mean_component_count_saved",
            "mean_tail_bridge_count_saved",
            "long_vehicle_saved_count",
            "performance_samples",
            "runtime_seconds",
            "representative_run_dir",
            "aggregate_status_counts",
            "run_dir",
        ]
        for field in PERFORMANCE_TOTAL_FIELDS:
            for suffix in ("median", "min", "max"):
                fieldnames.append(f"{field}_{suffix}")
        for stage_name in STAGE_NAMES:
            for suffix in ("wall_seconds", "cpu_seconds"):
                for stat in ("median", "min", "max"):
                    fieldnames.append(f"stage_{stage_name}_{suffix}_{stat}")
            fieldnames.append(f"stage_{stage_name}_call_count")
        return fieldnames

    def _render_markdown(self, rows: list[dict[str, Any]]) -> str:
        lines = [
            "# Benchmark Leaderboard",
            "",
            "| Sequence | Preset | Saved Aggregates | Quality Mean | Registration Accept | Runtime (s) |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for row in rows:
            lines.append(
                "| {sequence_name} | {preset_name} | {saved_aggregates} | {track_quality_mean:.3f} | {registration_accept_rate:.3f} | {runtime_seconds:.3f} |".format(
                    **row
                )
            )
        lines.append("")
        return "\n".join(lines)

    def _render_long_vehicle_markdown(self, rows: list[dict[str, Any]]) -> str:
        ordered = sorted(rows, key=self._long_vehicle_sort_key)
        lines = [
            "# Long-Vehicle Leaderboard",
            "",
            "| Sequence | Preset | Mean Longitudinal Extent | Mean Components | Tail Bridges | Saved Aggregates | Runtime (s) |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in ordered:
            lines.append(
                "| {sequence_name} | {preset_name} | {mean_longitudinal_extent_saved:.3f} | {mean_component_count_saved:.3f} | {mean_tail_bridge_count_saved:.3f} | {saved_aggregates} | {runtime_seconds:.3f} |".format(
                    **row
                )
            )
        lines.append("")
        return "\n".join(lines)

    def _render_performance_markdown(self, rows: list[dict[str, Any]]) -> str:
        sections = [
            ("Schnellste Presets", sorted(rows, key=lambda row: self._float_or_zero(row.get("total_wall_seconds_median")))),
            (
                "Rechenintensivste Presets",
                sorted(
                    rows,
                    key=lambda row: (
                        -self._float_or_zero(row.get("compute_wall_seconds_median")),
                        -self._float_or_zero(row.get("total_cpu_seconds_median")),
                    ),
                ),
            ),
            ("Speicherintensivste Presets", sorted(rows, key=lambda row: -self._float_or_zero(row.get("peak_rss_mb_max")))),
        ]
        lines = ["# Performance Leaderboard", ""]
        for title, ordered in sections:
            lines.append(f"## {title}")
            lines.append("")
            lines.append(
                "| Sequence | Preset | Tracker | Clusterer | Registration | Total Wall Median (s) | Compute Wall Median (s) | CPU Median (s) | Peak RSS Max (MB) | Wall / Frame (ms) |"
            )
            lines.append("| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
            for row in ordered:
                lines.append(
                    "| {sequence_name} | {preset_name} | {tracker_algorithm} | {clusterer_algorithm} | {registration_backend} | {total_wall_seconds_median:.3f} | {compute_wall_seconds_median:.3f} | {total_cpu_seconds_median:.3f} | {peak_rss_mb_max:.3f} | {wall_ms_per_frame_median:.3f} |".format(
                        **row
                    )
                )
            lines.append("")
        return "\n".join(lines)

    def _render_component_markdown(self, rows: list[dict[str, Any]]) -> str:
        lines = ["# Performance Components", ""]
        for component in COMPONENT_FIELDS:
            lines.append(f"## {component}")
            lines.append("")
            lines.append("### Deskriptive Gruppierung")
            lines.append("")
            lines.append("| Value | Cases | Total Wall Median (s) | Compute Wall Median (s) | CPU Median (s) | Peak RSS Max (MB) |")
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
            for group_row in self._summarize_component_rows(rows, component):
                lines.append(
                    "| {value} | {cases} | {total_wall_seconds_median:.3f} | {compute_wall_seconds_median:.3f} | {total_cpu_seconds_median:.3f} | {peak_rss_mb_max:.3f} |".format(
                        **group_row
                    )
                )
            lines.append("")
            lines.append("### Gematchte Vergleiche")
            lines.append("")
            matched_groups = self._matched_component_groups(rows, component)
            if not matched_groups:
                lines.append("Keine gematchten Vergleichsgruppen vorhanden.")
                lines.append("")
                continue
            for context, group_rows in matched_groups:
                lines.append(f"#### {context}")
                lines.append("")
                lines.append("| Preset | Value | Total Wall Median (s) | Compute Wall Median (s) | CPU Median (s) | Peak RSS Max (MB) |")
                lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
                for row in group_rows:
                    lines.append(
                        "| {preset_name} | {value} | {total_wall_seconds_median:.3f} | {compute_wall_seconds_median:.3f} | {total_cpu_seconds_median:.3f} | {peak_rss_mb_max:.3f} |".format(
                            preset_name=row["preset_name"],
                            value=row[component],
                            total_wall_seconds_median=row["total_wall_seconds_median"],
                            compute_wall_seconds_median=row["compute_wall_seconds_median"],
                            total_cpu_seconds_median=row["total_cpu_seconds_median"],
                            peak_rss_mb_max=row["peak_rss_mb_max"],
                        )
                    )
                lines.append("")
        return "\n".join(lines)

    def _summarize_component_rows(self, rows: list[dict[str, Any]], component: str) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            key = str(row.get(component, ""))
            grouped.setdefault(key, []).append(row)
        summary_rows = []
        for value, group_rows in sorted(grouped.items()):
            summary_rows.append(
                {
                    "value": value,
                    "cases": len(group_rows),
                    "total_wall_seconds_median": self._median(
                        [self._float_or_zero(row.get("total_wall_seconds_median")) for row in group_rows]
                    ),
                    "compute_wall_seconds_median": self._median(
                        [self._float_or_zero(row.get("compute_wall_seconds_median")) for row in group_rows]
                    ),
                    "total_cpu_seconds_median": self._median(
                        [self._float_or_zero(row.get("total_cpu_seconds_median")) for row in group_rows]
                    ),
                    "peak_rss_mb_max": max(self._float_or_zero(row.get("peak_rss_mb_max")) for row in group_rows),
                }
            )
        return sorted(summary_rows, key=lambda row: row["total_wall_seconds_median"])

    def _matched_component_groups(self, rows: list[dict[str, Any]], component: str) -> list[tuple[str, list[dict[str, Any]]]]:
        grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for row in rows:
            key = tuple(row.get(field) for field in ROW_CONTEXT_FIELDS if field != component)
            grouped.setdefault(key, []).append(row)
        result = []
        for key, group_rows in grouped.items():
            distinct_values = {str(row.get(component, "")) for row in group_rows}
            if len(distinct_values) < 2:
                continue
            context_parts = []
            remaining_fields = [field for field in ROW_CONTEXT_FIELDS if field != component]
            for field, value in zip(remaining_fields, key, strict=True):
                context_parts.append(f"{field}={value}")
            result.append(
                (
                    ", ".join(context_parts),
                    sorted(group_rows, key=lambda row: self._float_or_zero(row.get("total_wall_seconds_median"))),
                )
            )
        return sorted(result, key=lambda item: item[0])

    def _sort_key(self, row: dict[str, Any]) -> tuple[float, float, float, float]:
        return (
            -float(row.get("saved_aggregates", 0)),
            -float(row.get("track_quality_mean", 0.0)),
            -float(row.get("registration_accept_rate", 0.0)),
            float(row.get("runtime_seconds", 0.0)),
        )

    def _long_vehicle_sort_key(self, row: dict[str, Any]) -> tuple[float, float, float, float]:
        return (
            -float(row.get("mean_longitudinal_extent_saved", 0.0)),
            float(row.get("mean_component_count_saved", 0.0)),
            -float(row.get("saved_aggregates", 0)),
            float(row.get("runtime_seconds", 0.0)),
        )

    def _mean(self, values: list[float]) -> float:
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    def _median(self, values: list[float]) -> float:
        if not values:
            return 0.0
        ordered = sorted(float(value) for value in values)
        mid = len(ordered) // 2
        if len(ordered) % 2 == 1:
            return ordered[mid]
        return (ordered[mid - 1] + ordered[mid]) / 2.0

    def _percentile(self, values: list[float], q: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(float(value) for value in values)
        if len(ordered) == 1:
            return ordered[0]
        index = (len(ordered) - 1) * (float(q) / 100.0)
        lo = int(index)
        hi = min(lo + 1, len(ordered) - 1)
        blend = index - lo
        return float((1.0 - blend) * ordered[lo] + blend * ordered[hi])

    def _is_long_vehicle_row(self, row: dict[str, Any]) -> bool:
        metrics = row.get("aggregation_metrics", {})
        quality_metrics = row.get("quality_metrics") or row.get("aggregation_metrics", {}).get("quality_metrics", {})
        return bool(
            quality_metrics.get("is_long_vehicle")
            or metrics.get("long_vehicle_mode_applied")
            or float(metrics.get("longitudinal_extent", 0.0)) >= 4.5
        )

    def _float_or_zero(self, value: Any) -> float:
        if value is None:
            return 0.0
        return float(value)

    def _round_optional(self, value: Any) -> float | None:
        if value is None:
            return None
        return round(float(value), 6)
