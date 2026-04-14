from __future__ import annotations

from pathlib import Path

from tracking_pipeline.application.services import resolve_dataset_root
from tracking_pipeline.config.models import PipelineConfig
from tracking_pipeline.config.validation import ConfigError
from tracking_pipeline.infrastructure.visualization.live_snapshot_loader import LiveSnapshotLoader
from tracking_pipeline.infrastructure.visualization.open3d_live_viewer import Open3DLiveViewer


def live_view(config: PipelineConfig, project_root: Path, run_id: str | None = None) -> None:
    if str(config.output.mode) != "dataset":
        raise ConfigError("live-view currently requires output.mode=dataset")

    dataset_root = resolve_dataset_root(config, project_root)
    loader = LiveSnapshotLoader(dataset_root, config)
    viewer = Open3DLiveViewer(
        config.visualization,
        loader,
        track_exit_edge_margin=config.output.track_exit_edge_margin,
        require_track_exit=config.output.require_track_exit,
        track_exit_line_axis=config.aggregation.frame_selection_line_axis,
    )
    viewer.live_view(run_id=run_id)
