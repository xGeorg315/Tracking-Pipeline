from __future__ import annotations

from tracking_pipeline.application.factories import build_clusterer
from tracking_pipeline.config.models import (
    AggregationConfig,
    ClusteringConfig,
    InputConfig,
    OutputConfig,
    PipelineConfig,
    PostprocessingConfig,
    PreprocessingConfig,
    TrackingConfig,
    VisualizationConfig,
)
from tracking_pipeline.infrastructure.clustering.voxel_grid_connected_components import VoxelGridConnectedComponentsClusterer


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
