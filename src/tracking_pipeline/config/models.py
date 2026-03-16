from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class InputConfig:
    paths: list[str]
    format: str = "a42_pb"


@dataclass(slots=True)
class PreprocessingConfig:
    lane_box: list[float]
    bootstrap_frames: int = 10


@dataclass(slots=True)
class ClusteringConfig:
    algorithm: str = "dbscan"
    eps: float = 1.15
    min_points: int = 20
    vehicle_min_points: int = 20
    vehicle_max_points: int = 10000
    plane_distance_threshold: float = 0.12
    plane_ransac_n: int = 3
    plane_num_iterations: int = 120
    ground_normal_z_min: float = 0.75
    hdbscan_min_cluster_size: int = 20
    hdbscan_min_samples: int = 10
    sensor_range_min: float = 0.0
    sensor_range_max: float = 120.0
    sensor_depth_jump_ratio: float = 0.08
    sensor_depth_jump_abs: float = 0.45
    sensor_min_component_size: int = 8
    sensor_neighbor_rows: int = 1
    sensor_neighbor_cols: int = 1
    sensor_ground_row_ignore: int = 0


@dataclass(slots=True)
class TrackingConfig:
    algorithm: str = "kalman_nn"
    max_dist: float = 3.4
    max_missed: int = 12
    min_track_hits: int = 4
    sticky_extra_dist_per_missed: float = 0.55
    sticky_max_dist: float = 6.2
    kf_init_var: float = 5.0
    kf_process_var: float = 0.08
    kf_meas_var: float = 0.60
    association_size_weight: float = 0.15


@dataclass(slots=True)
class AggregationConfig:
    algorithm: str = "voxel_fusion"
    frame_selection_method: str = "auto"
    use_all_frames: bool = True
    top_k_frames: int = 10
    keyframe_keep: int = 8
    chunk_quality_filter: bool = True
    chunk_min_points_ratio_to_peak: float = 0.40
    chunk_min_extent_ratio_to_peak: float = 0.35
    chunk_min_segment_length: int = 4
    frame_selection_line_axis: str = "y"
    frame_selection_line_ratio: float = 0.10
    frame_selection_touch_margin: float = 0.12
    frame_downsample_voxel: float = 0.07
    shape_consistency_filter: bool = False
    shape_consistency_max_extent_ratio: float = 2.0
    registration_backend: str = "small_gicp"
    registration_max_corr_dist: float = 0.95
    registration_max_iter: int = 80
    registration_min_fitness: float = 0.25
    registration_max_translation: float = 3.2
    global_registration_voxel: float = 0.12
    fusion_voxel_size: float = 0.05
    fusion_min_observations: int = 1
    fusion_weight_mode: str = "point_count"
    consensus_ratio: float = 0.35
    min_track_quality_for_save: float = 0.0
    min_saved_aggregate_points: int = 180
    long_vehicle_mode: bool = False
    long_vehicle_length_threshold: float = 4.5
    length_coverage_bins: int = 10
    min_track_quality_for_save_long_vehicle: float = 0.0
    tail_bridge_longitudinal_gap_max: float = 1.5
    tail_bridge_lateral_gap_max: float = 0.8
    tail_bridge_vertical_gap_max: float = 0.5
    post_filter_stat_nb_neighbors: int = 12
    post_filter_stat_std_ratio: float = 2.3
    aggregate_voxel: float = 0.06


@dataclass(slots=True)
class PostprocessingConfig:
    enable_tracklet_stitching: bool = False
    stitching_max_gap: int = 4
    stitching_max_center_dist: float = 2.5
    enable_co_moving_track_merge: bool = False
    parallel_merge_max_lateral_offset: float = 0.8
    parallel_merge_max_longitudinal_gap: float = 4.0
    parallel_merge_min_overlap_frames: int = 5
    parallel_merge_min_overlap_ratio: float = 0.6
    enable_trajectory_smoothing: bool = False
    smoothing_window: int = 3
    enable_track_quality_scoring: bool = True


@dataclass(slots=True)
class OutputConfig:
    root_dir: str = "runs"
    save_world: bool = False
    save_aggregate_intensity: bool = False
    require_track_exit: bool = True
    track_exit_edge_margin: float = 0.9


@dataclass(slots=True)
class VisualizationConfig:
    enabled: bool = True
    color_by_intensity: bool = False
    max_points: int = 120000
    max_cluster_points: int = 15000
    max_assoc_dist: float = 4.2


@dataclass(slots=True)
class PipelineConfig:
    input: InputConfig
    preprocessing: PreprocessingConfig
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    postprocessing: PostprocessingConfig = field(default_factory=PostprocessingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    config_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.config_path is not None:
            data["config_path"] = str(self.config_path)
        return data


@dataclass(slots=True)
class BenchmarkConfig:
    sequences: list[str]
    presets: list[str]
    output_root: str = "benchmarks"
    name: str = "curated_proxy"
    warmup_runs: int = 1
    measure_runs: int = 3
    config_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.config_path is not None:
            data["config_path"] = str(self.config_path)
        return data
