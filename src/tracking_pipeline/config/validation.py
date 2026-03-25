from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

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
SUPPORTED_CLASSIFICATION_BACKENDS = {"pointnext"}
SUPPORTED_CLASSIFICATION_DEVICES = {"auto", "cpu", "cuda", "mps"}


class ConfigError(ValueError):
    """Raised when the pipeline configuration is invalid."""


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


def _resolve_support_path(value: str | Path, config: PipelineConfig) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    base_dir = None if config.config_path is None else config.config_path.parent
    candidates: list[Path] = []
    if base_dir is not None:
        candidates.append(base_dir / path)
        if base_dir.parent != base_dir:
            candidates.append(base_dir.parent / path)
    candidates.append(Path.cwd() / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _load_classification_model_cfg(config: PipelineConfig) -> dict[str, Any]:
    cfg_path = _resolve_support_path(config.classification.model_cfg_path, config)
    raw = _read_yaml(cfg_path)
    base_path = cfg_path.with_name("default.yaml")
    if cfg_path.name != "default.yaml" and base_path.exists():
        raw = _deep_merge(_read_yaml(base_path), raw)
    return raw


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
    if config.classification.backend not in SUPPORTED_CLASSIFICATION_BACKENDS:
        raise ConfigError(f"Unsupported classification backend: {config.classification.backend}")
    if config.classification.device not in SUPPORTED_CLASSIFICATION_DEVICES:
        raise ConfigError(f"Unsupported classification device: {config.classification.device}")
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
    if not isinstance(config.aggregation.enable_tail_bridge, bool):
        raise ConfigError("aggregation.enable_tail_bridge must be a boolean")
    if not isinstance(config.aggregation.enable_post_filter_stat_outlier_removal, bool):
        raise ConfigError("aggregation.enable_post_filter_stat_outlier_removal must be a boolean")
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
    if not isinstance(config.classification.enabled, bool):
        raise ConfigError("classification.enabled must be a boolean")
    if not isinstance(config.class_normalization.enabled, bool):
        raise ConfigError("class_normalization.enabled must be a boolean")
    if not isinstance(config.class_normalization.aliases, dict):
        raise ConfigError("class_normalization.aliases must be a mapping")
    for raw_name, canonical_name in config.class_normalization.aliases.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ConfigError("class_normalization.aliases keys must be non-empty strings")
        if not isinstance(canonical_name, str) or not canonical_name.strip():
            raise ConfigError("class_normalization.aliases values must be non-empty strings")
    if config.classification.enabled:
        pointnext_root = _resolve_support_path(config.classification.pointnext_root, config)
        checkpoint_path = _resolve_support_path(config.classification.checkpoint_path, config)
        model_cfg_path = _resolve_support_path(config.classification.model_cfg_path, config)
        if not pointnext_root.exists() or not pointnext_root.is_dir():
            raise ConfigError(f"classification.pointnext_root does not exist: {pointnext_root}")
        if not checkpoint_path.exists() or not checkpoint_path.is_file():
            raise ConfigError(f"classification.checkpoint_path does not exist: {checkpoint_path}")
        if not model_cfg_path.exists() or not model_cfg_path.is_file():
            raise ConfigError(f"classification.model_cfg_path does not exist: {model_cfg_path}")
        if not config.classification.class_names:
            raise ConfigError("classification.class_names must not be empty when classification.enabled is true")
        if any(not str(class_name).strip() for class_name in config.classification.class_names):
            raise ConfigError("classification.class_names must not contain empty values")

        model_cfg = _load_classification_model_cfg(config)
        model_data = model_cfg.get("model") or {}
        cls_args = model_data.get("cls_args") or {}
        expected_num_classes = cls_args.get("num_classes")
        if expected_num_classes is not None and len(config.classification.class_names) != int(expected_num_classes):
            raise ConfigError(
                "classification.class_names length must match the PointNeXt model num_classes "
                f"({int(expected_num_classes)})"
            )
        extra_global_channels = int(model_data.get("extra_global_channels", 0) or 0)
        if extra_global_channels != 0:
            raise ConfigError("PointNeXt classification currently supports only model.extra_global_channels == 0")
        encoder_args = model_data.get("encoder_args") or {}
        in_channels = int(encoder_args.get("in_channels", 3) or 0)
        if in_channels != 3:
            raise ConfigError("PointNeXt classification currently supports only model.encoder_args.in_channels == 3")


def validate_benchmark_config(config: BenchmarkConfig) -> None:
    if not config.sequences:
        raise ConfigError("benchmark.sequences must not be empty")
    if not config.presets:
        raise ConfigError("benchmark.presets must not be empty")
    if config.warmup_runs < 0:
        raise ConfigError("benchmark.warmup_runs must be >= 0")
    if config.measure_runs < 1:
        raise ConfigError("benchmark.measure_runs must be >= 1")
