from __future__ import annotations

from pathlib import Path

import pytest

from tracking_pipeline.application.live_view import live_view
from tracking_pipeline.config.models import (
    InputConfig,
    OutputConfig,
    PipelineConfig,
    PreprocessingConfig,
)
from tracking_pipeline.config.validation import ConfigError


def _config(output_mode: str = "dataset") -> PipelineConfig:
    return PipelineConfig(
        input=InputConfig(paths=["dummy.pb"]),
        preprocessing=PreprocessingConfig(lane_box=[-1.0, 1.0, 0.0, 10.0, 0.0, 2.0]),
        output=OutputConfig(mode=output_mode, dataset_root_dir="dataset"),
    )


def test_live_view_rejects_non_dataset_mode(tmp_path: Path) -> None:
    config = _config(output_mode="run")

    with pytest.raises(ConfigError, match="output.mode=dataset"):
        live_view(config, tmp_path)


def test_live_view_resolves_dataset_root_and_starts_viewer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _config(output_mode="dataset")
    dataset_root = tmp_path / "dataset_export"
    seen: dict[str, object] = {}

    class _FakeLoader:
        def __init__(self, resolved_root: Path, resolved_config: PipelineConfig):
            seen["loader_root"] = resolved_root
            seen["loader_config"] = resolved_config

    class _FakeViewer:
        def __init__(
            self,
            visualization_config,
            loader,
            track_exit_edge_margin: float,
            require_track_exit: bool,
            track_exit_line_axis: str,
        ):
            seen["viewer_visualization"] = visualization_config
            seen["viewer_loader"] = loader
            seen["track_exit_edge_margin"] = track_exit_edge_margin
            seen["require_track_exit"] = require_track_exit
            seen["track_exit_line_axis"] = track_exit_line_axis

        def live_view(self, run_id: str | None = None) -> None:
            seen["run_id"] = run_id

    monkeypatch.setattr("tracking_pipeline.application.live_view.resolve_dataset_root", lambda *_args: dataset_root)
    monkeypatch.setattr("tracking_pipeline.application.live_view.LiveSnapshotLoader", _FakeLoader)
    monkeypatch.setattr("tracking_pipeline.application.live_view.Open3DLiveViewer", _FakeViewer)

    live_view(config, tmp_path, run_id="run_20260407")

    assert seen["loader_root"] == dataset_root
    assert seen["loader_config"] is config
    assert seen["viewer_visualization"] is config.visualization
    assert seen["track_exit_edge_margin"] == config.output.track_exit_edge_margin
    assert seen["require_track_exit"] is config.output.require_track_exit
    assert seen["track_exit_line_axis"] == config.aggregation.frame_selection_line_axis
    assert seen["run_id"] == "run_20260407"
