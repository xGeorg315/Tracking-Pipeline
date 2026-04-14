from __future__ import annotations

from pathlib import Path

from tracking_pipeline.application.factories import build_artifact_writer, build_clusterer, build_reader
from tracking_pipeline.config.models import (
    AggregationConfig,
    ClusteringConfig,
    InputConfig,
    OutputConfig,
    PipelineConfig,
    PostprocessingConfig,
    PreprocessingConfig,
    QB2LiveInputConfig,
    QB2LiveMQTTConfig,
    TrackingConfig,
    VisualizationConfig,
)
from tracking_pipeline.infrastructure.clustering.voxel_grid_connected_components import VoxelGridConnectedComponentsClusterer
from tracking_pipeline.infrastructure.io.dataset_artifact_writer import DatasetArtifactWriter
from tracking_pipeline.infrastructure.readers.qb2_live_reader import QB2LiveReader


def test_build_clusterer_supports_voxel_grid_connected_components() -> None:
    config = PipelineConfig(
        input=InputConfig(paths=["dummy.pb"]),
        preprocessing=PreprocessingConfig(lane_box=[-1.0, 1.0, 0.0, 10.0, 0.0, 2.0]),
        clustering=ClusteringConfig(algorithm="voxel_grid_connected_components", voxel_size=0.3),
        tracking=TrackingConfig(),
        aggregation=AggregationConfig(),
        postprocessing=PostprocessingConfig(),
        output=OutputConfig(),
        visualization=VisualizationConfig(),
    )

    clusterer = build_clusterer(config)

    assert isinstance(clusterer, VoxelGridConnectedComponentsClusterer)


def test_build_reader_supports_qb2_live() -> None:
    config = PipelineConfig(
        input=InputConfig(
            paths=["qb2_live://class_qb2@10.16.3.160"],
            format="qb2_live",
            qb2_live=QB2LiveInputConfig(
                sensor_name="class_qb2",
                ip="10.16.3.160",
                api_key="secret",
                mqtt=QB2LiveMQTTConfig(host="10.16.3.111", topic="blickfeld/states_160"),
            ),
        ),
        preprocessing=PreprocessingConfig(lane_box=[-1.0, 1.0, 0.0, 10.0, 0.0, 2.0]),
        clustering=ClusteringConfig(),
        tracking=TrackingConfig(),
        aggregation=AggregationConfig(),
        postprocessing=PostprocessingConfig(),
        output=OutputConfig(),
        visualization=VisualizationConfig(),
    )

    reader = build_reader(config)

    assert isinstance(reader, QB2LiveReader)


def test_build_artifact_writer_supports_dataset_mode(tmp_path: Path) -> None:
    config = PipelineConfig(
        input=InputConfig(paths=["dummy.pb"]),
        preprocessing=PreprocessingConfig(lane_box=[-1.0, 1.0, 0.0, 10.0, 0.0, 2.0]),
        clustering=ClusteringConfig(),
        tracking=TrackingConfig(),
        aggregation=AggregationConfig(),
        postprocessing=PostprocessingConfig(),
        output=OutputConfig(mode="dataset", dataset_root_dir=str(tmp_path / "dataset")),
        visualization=VisualizationConfig(),
    )

    writer = build_artifact_writer(config, tmp_path)

    assert isinstance(writer, DatasetArtifactWriter)
