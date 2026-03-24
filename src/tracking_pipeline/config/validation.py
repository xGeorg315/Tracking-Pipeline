from __future__ import annotations

from pathlib import Path

from tracking_pipeline.config.models import BenchmarkConfig, PipelineConfig


SUPPORTED_INPUT_FORMATS = {"a42_pb"}
SUPPORTED_CLUSTERERS = {
    "dbscan",
    "euclidean_clustering",
    "ground_removed_dbscan",
    "hdbscan",
    "voxel_grid_connected_components",
    "range_image_connected_components",
    "range_image_depth_jump",
    "beam_neighbor_region_growing",
}
SUPPORTED_TRACKERS = {"euclidean_nn", "kalman_nn", "hungarian_kalman"}
SUPPORTED_ACCUMULATORS = {
    "voxel_fusion",
    "registration_voxel_fusion",
    "weighted_voxel_fusion",
    "occupancy_consensus_fusion",
}
SUPPORTED_FRAME_SELECTION_METHODS = {
    "auto",
    "all_track_frames",
    "line_touch_last_k",
    "keyframe_motion",
    "length_coverage",
    "quality_coverage",
    "tail_coverage",
    "center_diversity",
    "max_extent",
}
SUPPORTED_REGISTRATION_BACKENDS = {
    "small_gicp",
    "icp_point_to_plane",
    "generalized_icp",
    "feature_global_then_local",
    "kiss_matcher",
    "kiss_matcher_then_icp",
}
SUPPORTED_FUSION_WEIGHT_MODES = {"uniform", "point_count", "quality"}


class ConfigError(ValueError):
    """Raised when the pipeline configuration is invalid."""


def validate_config(config: PipelineConfig) -> None:
    if config.input.format not in SUPPORTED_INPUT_FORMATS:
        raise ConfigError(f"Unsupported input format: {config.input.format}")
    if not config.input.paths:
        raise ConfigError("input.paths must not be empty")
    for input_path in config.input.paths:
        if not Path(input_path).exists():
            raise ConfigError(f"Input path does not exist: {input_path}")
    if config.clustering.algorithm not in SUPPORTED_CLUSTERERS:
        raise ConfigError(f"Unsupported clustering algorithm: {config.clustering.algorithm}")
    if config.tracking.algorithm not in SUPPORTED_TRACKERS:
        raise ConfigError(f"Unsupported tracking algorithm: {config.tracking.algorithm}")
    if config.aggregation.algorithm not in SUPPORTED_ACCUMULATORS:
        raise ConfigError(f"Unsupported aggregation algorithm: {config.aggregation.algorithm}")
    if config.aggregation.frame_selection_method not in SUPPORTED_FRAME_SELECTION_METHODS:
        raise ConfigError(f"Unsupported frame selection method: {config.aggregation.frame_selection_method}")
    if config.aggregation.registration_backend not in SUPPORTED_REGISTRATION_BACKENDS:
        raise ConfigError(f"Unsupported registration backend: {config.aggregation.registration_backend}")
    if config.aggregation.fusion_weight_mode not in SUPPORTED_FUSION_WEIGHT_MODES:
        raise ConfigError(f"Unsupported fusion weight mode: {config.aggregation.fusion_weight_mode}")
    if not isinstance(config.aggregation.symmetry_completion, bool):
        raise ConfigError("aggregation.symmetry_completion must be a boolean")
    if not isinstance(config.aggregation.motion_deskew, bool):
        raise ConfigError("aggregation.motion_deskew must be a boolean")
    if not isinstance(config.aggregation.truncate_after_lane_end_touch, bool):
        raise ConfigError("aggregation.truncate_after_lane_end_touch must be a boolean")
    if not isinstance(config.aggregation.enable_registration_underfill_fallback, bool):
        raise ConfigError("aggregation.enable_registration_underfill_fallback must be a boolean")
    if not isinstance(config.aggregation.enable_confidence_point_cap, bool):
        raise ConfigError("aggregation.enable_confidence_point_cap must be a boolean")
    if not isinstance(config.visualization.show_full_frame_pcd, bool):
        raise ConfigError("visualization.show_full_frame_pcd must be a boolean")
    if not isinstance(config.visualization.show_tracker_debug, bool):
        raise ConfigError("visualization.show_tracker_debug must be a boolean")
    if not isinstance(config.visualization.show_track_outcome_debug, bool):
        raise ConfigError("visualization.show_track_outcome_debug must be a boolean")
    if not isinstance(config.visualization.show_articulated_merge_debug, bool):
        raise ConfigError("visualization.show_articulated_merge_debug must be a boolean")
    if len(config.preprocessing.lane_box) != 6:
        raise ConfigError("preprocessing.lane_box must contain exactly 6 values")
    if config.preprocessing.bootstrap_frames < 0:
        raise ConfigError("preprocessing.bootstrap_frames must be >= 0")
    if config.aggregation.top_k_frames < 1:
        raise ConfigError("aggregation.top_k_frames must be >= 1")
    if config.aggregation.keyframe_keep < 1:
        raise ConfigError("aggregation.keyframe_keep must be >= 1")
    if config.aggregation.chunk_min_points_ratio_to_peak < 0 or config.aggregation.chunk_min_points_ratio_to_peak > 1:
        raise ConfigError("aggregation.chunk_min_points_ratio_to_peak must be within [0, 1]")
    if config.aggregation.chunk_min_extent_ratio_to_peak < 0 or config.aggregation.chunk_min_extent_ratio_to_peak > 1:
        raise ConfigError("aggregation.chunk_min_extent_ratio_to_peak must be within [0, 1]")
    if config.aggregation.chunk_min_segment_length < 1:
        raise ConfigError("aggregation.chunk_min_segment_length must be >= 1")
    if config.aggregation.min_saved_aggregate_points < 0:
        raise ConfigError("aggregation.min_saved_aggregate_points must be >= 0")
    if config.aggregation.registration_min_kept_chunks < 1:
        raise ConfigError("aggregation.registration_min_kept_chunks must be >= 1")
    if config.aggregation.confidence_point_cap_max_points < 1:
        raise ConfigError("aggregation.confidence_point_cap_max_points must be >= 1")
    if config.aggregation.confidence_point_cap_bins < 1:
        raise ConfigError("aggregation.confidence_point_cap_bins must be >= 1")
    if config.tracking.min_track_hits < 1:
        raise ConfigError("tracking.min_track_hits must be >= 1")
    if config.postprocessing.stitching_max_gap < 0:
        raise ConfigError("postprocessing.stitching_max_gap must be >= 0")
    if config.postprocessing.articulated_gap_eval_window_frames < 1:
        raise ConfigError("postprocessing.articulated_gap_eval_window_frames must be >= 1")
    if config.postprocessing.articulated_min_overlap_frames < 1:
        raise ConfigError("postprocessing.articulated_min_overlap_frames must be >= 1")
    if config.postprocessing.articulated_min_overlap_ratio < 0 or config.postprocessing.articulated_min_overlap_ratio > 1:
        raise ConfigError("postprocessing.articulated_min_overlap_ratio must be within [0, 1]")
    if config.postprocessing.articulated_max_lateral_offset < 0:
        raise ConfigError("postprocessing.articulated_max_lateral_offset must be >= 0")
    if config.postprocessing.articulated_max_vertical_offset < 0:
        raise ConfigError("postprocessing.articulated_max_vertical_offset must be >= 0")
    if config.postprocessing.articulated_max_hitch_gap < 0:
        raise ConfigError("postprocessing.articulated_max_hitch_gap must be >= 0")
    if config.postprocessing.articulated_max_hitch_gap_std < 0:
        raise ConfigError("postprocessing.articulated_max_hitch_gap_std must be >= 0")
    if config.postprocessing.articulated_max_speed_delta < 0:
        raise ConfigError("postprocessing.articulated_max_speed_delta must be >= 0")
    if config.postprocessing.articulated_min_combined_length < 0:
        raise ConfigError("postprocessing.articulated_min_combined_length must be >= 0")
    if config.postprocessing.parallel_merge_min_overlap_frames < 1:
        raise ConfigError("postprocessing.parallel_merge_min_overlap_frames must be >= 1")
    if config.postprocessing.parallel_merge_min_overlap_ratio < 0 or config.postprocessing.parallel_merge_min_overlap_ratio > 1:
        raise ConfigError("postprocessing.parallel_merge_min_overlap_ratio must be within [0, 1]")
    if config.postprocessing.smoothing_window < 1:
        raise ConfigError("postprocessing.smoothing_window must be >= 1")
    if config.aggregation.min_track_quality_for_save < 0 or config.aggregation.min_track_quality_for_save > 1:
        raise ConfigError("aggregation.min_track_quality_for_save must be within [0, 1]")
    if config.aggregation.min_track_quality_for_save_long_vehicle < 0 or config.aggregation.min_track_quality_for_save_long_vehicle > 1:
        raise ConfigError("aggregation.min_track_quality_for_save_long_vehicle must be within [0, 1]")
    if config.aggregation.long_vehicle_length_threshold <= 0:
        raise ConfigError("aggregation.long_vehicle_length_threshold must be > 0")
    if config.aggregation.length_coverage_bins < 2:
        raise ConfigError("aggregation.length_coverage_bins must be >= 2")
    if config.clustering.sensor_range_max <= config.clustering.sensor_range_min:
        raise ConfigError("clustering.sensor_range_max must be greater than clustering.sensor_range_min")
    if config.clustering.algorithm == "voxel_grid_connected_components" and config.clustering.voxel_size <= 0:
        raise ConfigError("clustering.voxel_size must be > 0 for voxel_grid_connected_components")
    if config.clustering.sensor_min_component_size < 1:
        raise ConfigError("clustering.sensor_min_component_size must be >= 1")
    if config.clustering.sensor_neighbor_rows < 0 or config.clustering.sensor_neighbor_cols < 0:
        raise ConfigError("clustering sensor neighbor windows must be >= 0")
    if config.clustering.sensor_ground_row_ignore < 0:
        raise ConfigError("clustering.sensor_ground_row_ignore must be >= 0")


def validate_benchmark_config(config: BenchmarkConfig) -> None:
    if not config.sequences:
        raise ConfigError("benchmark.sequences must not be empty")
    if not config.presets:
        raise ConfigError("benchmark.presets must not be empty")
    if config.warmup_runs < 0:
        raise ConfigError("benchmark.warmup_runs must be >= 0")
    if config.measure_runs < 1:
        raise ConfigError("benchmark.measure_runs must be >= 1")
