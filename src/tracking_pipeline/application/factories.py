from __future__ import annotations

from pathlib import Path

from tracking_pipeline.application.ports import Accumulator, ArtifactWriter, Clusterer, FrameReader, ReplayViewer, TrackPostprocessor, Tracker
from tracking_pipeline.config.models import PipelineConfig
from tracking_pipeline.domain.value_objects import LaneBox
from tracking_pipeline.infrastructure.aggregation.occupancy_consensus_fusion import OccupancyConsensusFusionAccumulator
from tracking_pipeline.infrastructure.aggregation.registration_voxel_fusion import RegistrationVoxelFusionAccumulator
from tracking_pipeline.infrastructure.aggregation.voxel_fusion import VoxelFusionAccumulator
from tracking_pipeline.infrastructure.aggregation.weighted_voxel_fusion import WeightedVoxelFusionAccumulator
from tracking_pipeline.infrastructure.clustering.dbscan_clusterer import DBSCANClusterer
from tracking_pipeline.infrastructure.clustering.euclidean_clustering import EuclideanClusteringClusterer
from tracking_pipeline.infrastructure.clustering.ground_removed_dbscan import GroundRemovedDBSCANClusterer
from tracking_pipeline.infrastructure.clustering.hdbscan_clusterer import HDBSCANClusterer
from tracking_pipeline.infrastructure.clustering.range_image_connected_components import RangeImageConnectedComponentsClusterer
from tracking_pipeline.infrastructure.clustering.range_image_depth_jump import RangeImageDepthJumpClusterer
from tracking_pipeline.infrastructure.clustering.beam_neighbor_region_growing import BeamNeighborRegionGrowingClusterer
from tracking_pipeline.infrastructure.io.artifact_writer import JsonArtifactWriter
from tracking_pipeline.infrastructure.postprocessing.articulated_vehicle_merge import ArticulatedVehicleMergePostprocessor
from tracking_pipeline.infrastructure.postprocessing.co_moving_track_merge import CoMovingTrackMergePostprocessor
from tracking_pipeline.infrastructure.postprocessing.track_quality_scoring import TrackQualityScoringPostprocessor
from tracking_pipeline.infrastructure.postprocessing.tracklet_stitching import TrackletStitchingPostprocessor
from tracking_pipeline.infrastructure.postprocessing.trajectory_smoothing import TrajectorySmoothingPostprocessor
from tracking_pipeline.infrastructure.readers.a42_pb_reader import A42PBReader
from tracking_pipeline.infrastructure.tracking.euclidean_nn import EuclideanNNTracker
from tracking_pipeline.infrastructure.tracking.hungarian_kalman import HungarianKalmanTracker
from tracking_pipeline.infrastructure.tracking.kalman_nn import KalmanNNTracker
from tracking_pipeline.infrastructure.visualization.open3d_replay_viewer import Open3DReplayViewer


def build_lane_box(config: PipelineConfig) -> LaneBox:
    return LaneBox.from_values(config.preprocessing.lane_box)


def build_reader(config: PipelineConfig) -> FrameReader:
    if config.input.format == "a42_pb":
        return A42PBReader(read_intensity=config.visualization.color_by_intensity or config.output.save_aggregate_intensity)
    raise ValueError(f"Unsupported input format: {config.input.format}")


def build_clusterer(config: PipelineConfig) -> Clusterer:
    algorithm = config.clustering.algorithm
    if algorithm == "dbscan":
        return DBSCANClusterer(config.clustering)
    if algorithm == "euclidean_clustering":
        return EuclideanClusteringClusterer(config.clustering)
    if algorithm == "ground_removed_dbscan":
        return GroundRemovedDBSCANClusterer(config.clustering)
    if algorithm == "hdbscan":
        return HDBSCANClusterer(config.clustering)
    if algorithm == "range_image_connected_components":
        return RangeImageConnectedComponentsClusterer(config.clustering)
    if algorithm == "range_image_depth_jump":
        return RangeImageDepthJumpClusterer(config.clustering)
    if algorithm == "beam_neighbor_region_growing":
        return BeamNeighborRegionGrowingClusterer(config.clustering)
    raise ValueError(f"Unsupported clusterer: {algorithm}")


def build_tracker(config: PipelineConfig) -> Tracker:
    algorithm = config.tracking.algorithm
    if algorithm == "euclidean_nn":
        return EuclideanNNTracker(config.tracking)
    if algorithm == "kalman_nn":
        return KalmanNNTracker(config.tracking)
    if algorithm == "hungarian_kalman":
        return HungarianKalmanTracker(config.tracking)
    raise ValueError(f"Unsupported tracker: {algorithm}")


def build_track_postprocessors(config: PipelineConfig) -> list[TrackPostprocessor]:
    processors: list[TrackPostprocessor] = []
    if config.postprocessing.enable_tracklet_stitching:
        processors.append(TrackletStitchingPostprocessor(config.postprocessing))
    if config.postprocessing.enable_articulated_vehicle_merge:
        processors.append(
            ArticulatedVehicleMergePostprocessor(
                config.postprocessing,
                config.aggregation.frame_selection_line_axis,
            )
        )
    if config.postprocessing.enable_co_moving_track_merge:
        processors.append(
            CoMovingTrackMergePostprocessor(
                config.postprocessing,
                config.aggregation.frame_selection_line_axis,
            )
        )
    if config.postprocessing.enable_trajectory_smoothing:
        processors.append(TrajectorySmoothingPostprocessor(config.postprocessing))
    if config.postprocessing.enable_track_quality_scoring:
        processors.append(
            TrackQualityScoringPostprocessor(
                config.postprocessing,
                config.aggregation.frame_selection_line_axis,
                config.aggregation.long_vehicle_length_threshold,
            )
        )
    return processors


def build_accumulator(config: PipelineConfig) -> Accumulator:
    algorithm = config.aggregation.algorithm
    if algorithm == "voxel_fusion":
        return VoxelFusionAccumulator(config.aggregation, config.output, config.tracking)
    if algorithm == "registration_voxel_fusion":
        return RegistrationVoxelFusionAccumulator(config.aggregation, config.output, config.tracking)
    if algorithm == "weighted_voxel_fusion":
        return WeightedVoxelFusionAccumulator(config.aggregation, config.output, config.tracking)
    if algorithm == "occupancy_consensus_fusion":
        return OccupancyConsensusFusionAccumulator(config.aggregation, config.output, config.tracking)
    raise ValueError(f"Unsupported accumulator: {algorithm}")


def build_artifact_writer(project_root: Path) -> ArtifactWriter:
    return JsonArtifactWriter(project_root)


def build_viewer(config: PipelineConfig) -> ReplayViewer:
    return Open3DReplayViewer(
        config.visualization,
        track_exit_edge_margin=config.output.track_exit_edge_margin,
        require_track_exit=config.output.require_track_exit,
        track_exit_line_axis=config.aggregation.frame_selection_line_axis,
    )
