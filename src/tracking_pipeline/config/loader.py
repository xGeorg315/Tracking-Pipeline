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
from tracking_pipeline.config.validation import validate_benchmark_config, validate_config


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


def load_config(path: str | Path) -> PipelineConfig:
    cfg_path = Path(path).resolve()
    raw = _read_yaml(cfg_path)

    base_path = cfg_path.with_name("base.yaml")
    if cfg_path.name != "base.yaml" and base_path.exists():
        raw = _deep_merge(_read_yaml(base_path), raw)

    input_cfg = dict(raw["input"])
    input_paths = []
    for value in input_cfg.get("paths", []):
        input_path = Path(value)
        if not input_path.is_absolute():
            input_path = (cfg_path.parent / input_path).resolve()
        input_paths.append(str(input_path))
    input_cfg["paths"] = input_paths

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
        sequence_path = Path(value)
        if not sequence_path.is_absolute():
            sequence_path = (cfg_path.parent / sequence_path).resolve()
        sequences.append(str(sequence_path))

    presets = []
    for value in raw.get("presets", []):
        preset_path = Path(value)
        if not preset_path.is_absolute():
            preset_path = (cfg_path.parent / preset_path).resolve()
        presets.append(str(preset_path))

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
