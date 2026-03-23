from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from tracking_pipeline.config.models import (
    AggregationConfig,
    BenchmarkConfig,
    ClusteringConfig,
    InputConfig,
    OutputConfig,
    PipelineConfig,
    PostprocessingConfig,
    PreprocessingConfig,
    TrackingConfig,
    VisualizationConfig,
)
from tracking_pipeline.config.validation import ConfigError, validate_benchmark_config, validate_config


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def resolve_input_paths(values: list[str | Path], base_dir: Path, *, field_name: str = "input.paths") -> list[str]:
    resolved_paths: list[str] = []
    for value in values:
        input_path = _resolve_path(value, base_dir)
        if not input_path.is_dir():
            resolved_paths.append(str(input_path))
            continue
        pb_files = sorted(
            (
                child.resolve()
                for child in input_path.iterdir()
                if child.is_file() and child.suffix == ".pb"
            ),
            key=lambda path: path.name,
        )
        if not pb_files:
            raise ConfigError(f"{field_name} directory does not contain any .pb files: {input_path}")
        resolved_paths.extend(str(path) for path in pb_files)
    return resolved_paths


def load_config(path: str | Path) -> PipelineConfig:
    cfg_path = Path(path).resolve()
    raw = _read_yaml(cfg_path)

    base_path = cfg_path.with_name("base.yaml")
    if cfg_path.name != "base.yaml" and base_path.exists():
        raw = _deep_merge(_read_yaml(base_path), raw)

    input_cfg = dict(raw["input"])
    input_cfg["paths"] = resolve_input_paths(input_cfg.get("paths", []), cfg_path.parent, field_name="input.paths")

    config = PipelineConfig(
        input=InputConfig(**input_cfg),
        preprocessing=PreprocessingConfig(**raw["preprocessing"]),
        clustering=ClusteringConfig(**raw.get("clustering", {})),
        tracking=TrackingConfig(**raw.get("tracking", {})),
        aggregation=AggregationConfig(**raw.get("aggregation", {})),
        postprocessing=PostprocessingConfig(**raw.get("postprocessing", {})),
        output=OutputConfig(**raw.get("output", {})),
        visualization=VisualizationConfig(**raw.get("visualization", {})),
        config_path=cfg_path,
    )
    validate_config(config)
    return config


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    cfg_path = Path(path).resolve()
    raw = _read_yaml(cfg_path)

    sequences = []
    for value in raw.get("sequences", []):
        sequences.append(str(_resolve_path(value, cfg_path.parent)))

    presets = []
    for value in raw.get("presets", []):
        presets.append(str(_resolve_path(value, cfg_path.parent)))

    config = BenchmarkConfig(
        sequences=sequences,
        presets=presets,
        output_root=str(raw.get("output_root", "benchmarks")),
        name=str(raw.get("name", "curated_proxy")),
        warmup_runs=int(raw.get("warmup_runs", 1)),
        measure_runs=int(raw.get("measure_runs", 3)),
        config_path=cfg_path,
    )
    validate_benchmark_config(config)
    return config
