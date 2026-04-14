from __future__ import annotations

from pathlib import Path

import pytest

from tracking_pipeline.application.live_web import live_web
from tracking_pipeline.config.models import InputConfig, OutputConfig, PipelineConfig, PreprocessingConfig
from tracking_pipeline.config.validation import ConfigError


def _config(output_mode: str = "dataset") -> PipelineConfig:
    return PipelineConfig(
        input=InputConfig(paths=["dummy.pb"]),
        preprocessing=PreprocessingConfig(lane_box=[-1.0, 1.0, 0.0, 10.0, 0.0, 2.0]),
        output=OutputConfig(mode=output_mode, dataset_root_dir="dataset"),
    )


def test_live_web_rejects_non_dataset_mode(tmp_path: Path) -> None:
    config = _config(output_mode="run")

    with pytest.raises(ConfigError, match="output.mode=dataset"):
        live_web(config, tmp_path)


def test_live_web_resolves_dataset_root_and_starts_server(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _config(output_mode="dataset")
    dataset_root = tmp_path / "dataset_export"
    seen: dict[str, object] = {}

    class _FakeLoader:
        def __init__(self, resolved_root: Path, resolved_config: PipelineConfig):
            seen["loader_root"] = resolved_root
            seen["loader_config"] = resolved_config

    class _FakeServer:
        def __init__(self, loader, *, host: str, port: int):
            seen["server_loader"] = loader
            seen["server_host"] = host
            seen["server_port"] = port

        def serve(self, run_id: str | None = None) -> None:
            seen["run_id"] = run_id

    monkeypatch.setattr("tracking_pipeline.application.live_web.resolve_dataset_root", lambda *_args: dataset_root)
    monkeypatch.setattr("tracking_pipeline.application.live_web.LiveSnapshotLoader", _FakeLoader)
    monkeypatch.setattr("tracking_pipeline.application.live_web.LiveWebViewerServer", _FakeServer)

    live_web(config, tmp_path, run_id="run_20260407", host="0.0.0.0", port=9001)

    assert seen["loader_root"] == dataset_root
    assert seen["loader_config"] is config
    assert seen["server_host"] == "0.0.0.0"
    assert seen["server_port"] == 9001
    assert seen["run_id"] == "run_20260407"
