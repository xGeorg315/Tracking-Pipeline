from __future__ import annotations

from datetime import datetime
from pathlib import Path

from tracking_pipeline.config.models import BenchmarkConfig, PipelineConfig


def build_run_name(config: PipelineConfig) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{config.tracking.algorithm}_{config.aggregation.algorithm}"


def build_benchmark_name(config: BenchmarkConfig) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{config.name}"


def resolve_output_root(config: PipelineConfig, project_root: Path) -> Path:
    root = Path(config.output.root_dir)
    if root.is_absolute():
        return root
    return (project_root / root).resolve()


def resolve_benchmark_root(config: BenchmarkConfig, project_root: Path) -> Path:
    root = Path(config.output_root)
    if root.is_absolute():
        return root
    return (project_root / root).resolve()
