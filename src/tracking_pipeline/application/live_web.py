from __future__ import annotations

from pathlib import Path

from tracking_pipeline.application.services import resolve_dataset_root
from tracking_pipeline.config.models import PipelineConfig
from tracking_pipeline.config.validation import ConfigError
from tracking_pipeline.infrastructure.visualization.live_snapshot_loader import LiveSnapshotLoader
from tracking_pipeline.infrastructure.visualization.live_web_server import LiveWebViewerServer


def live_web(
    config: PipelineConfig,
    project_root: Path,
    *,
    run_id: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    if str(config.output.mode) != "dataset":
        raise ConfigError("live-web currently requires output.mode=dataset")

    dataset_root = resolve_dataset_root(config, project_root)
    loader = LiveSnapshotLoader(dataset_root, config)
    server = LiveWebViewerServer(loader, host=host, port=port)
    server.serve(run_id=run_id)
