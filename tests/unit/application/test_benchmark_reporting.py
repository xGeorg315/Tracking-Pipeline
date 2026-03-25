from __future__ import annotations

from pathlib import Path

from tracking_pipeline.application.benchmark_run import BenchmarkRunner


def test_aggregation_component_metrics_from_summary_include_ms_per_track() -> None:
    runner = BenchmarkRunner(Path(__file__).resolve().parents[3])

    metrics = runner._aggregation_component_metrics_from_summary(
        {
            "registration": {"wall_seconds": 0.8, "cpu_seconds": 0.3, "call_count": 4},
            "fusion_core": {"wall_seconds": 1.2, "cpu_seconds": 0.6, "call_count": 4},
            "fusion_post": {"wall_seconds": 0.4, "cpu_seconds": 0.2, "call_count": 3},
            "fusion_total": {"wall_seconds": 1.6, "cpu_seconds": 0.8, "call_count": 4},
        },
        finished_track_count=4,
    )

    assert metrics["aggregation_registration_wall_seconds"] == 0.8
    assert metrics["aggregation_registration_call_count"] == 4
    assert metrics["aggregation_registration_wall_ms_per_track"] == 200.0
    assert metrics["aggregation_fusion_total_wall_seconds"] == 1.6
    assert metrics["aggregation_fusion_total_wall_ms_per_track"] == 400.0


def test_aggregate_rows_include_hz_statistics() -> None:
    runner = BenchmarkRunner(Path(__file__).resolve().parents[3])

    aggregated = runner._aggregate_rows(
        [
            {
                "sequence_name": "seq_a",
                "preset_name": "preset_a",
                "tracker_algorithm": "kalman_nn",
                "clusterer_algorithm": "euclidean_clustering",
                "accumulator_algorithm": "voxel_fusion",
                "registration_backend": "none",
                "frame_selection_method": "all_track_frames",
                "long_vehicle_mode": False,
                "frame_count": 10,
                "finished_track_count": 2,
                "saved_aggregates": 1,
                "aggregate_save_rate": 0.5,
                "registration_attempts": 0,
                "registration_accept_rate": 0.0,
                "track_quality_mean": 0.1,
                "mean_selected_frames_per_saved_aggregate": 2.0,
                "mean_points_per_saved_aggregate": 3.0,
                "mean_longitudinal_extent_saved": 4.0,
                "p90_longitudinal_extent_saved": 4.5,
                "mean_component_count_saved": 1.0,
                "mean_tail_bridge_count_saved": 0.0,
                "long_vehicle_saved_count": 0,
                "aggregate_status_counts": {"saved": 1},
                "run_dir": "/tmp/run1",
                "runtime_seconds": 2.0,
                "total_wall_seconds": 2.0,
                "total_cpu_seconds": 1.0,
                "compute_wall_seconds": 1.25,
                "compute_cpu_seconds": 0.9,
                "io_wall_seconds": 0.75,
                "total_hz": 5.0,
                "compute_hz": 8.0,
                "peak_rss_mb": 100.0,
                "wall_ms_per_frame": 200.0,
                "cpu_ms_per_frame": 100.0,
                "accumulate_wall_ms_per_track": 50.0,
            },
            {
                "sequence_name": "seq_a",
                "preset_name": "preset_a",
                "tracker_algorithm": "kalman_nn",
                "clusterer_algorithm": "euclidean_clustering",
                "accumulator_algorithm": "voxel_fusion",
                "registration_backend": "none",
                "frame_selection_method": "all_track_frames",
                "long_vehicle_mode": False,
                "frame_count": 10,
                "finished_track_count": 2,
                "saved_aggregates": 1,
                "aggregate_save_rate": 0.5,
                "registration_attempts": 0,
                "registration_accept_rate": 0.0,
                "track_quality_mean": 0.1,
                "mean_selected_frames_per_saved_aggregate": 2.0,
                "mean_points_per_saved_aggregate": 3.0,
                "mean_longitudinal_extent_saved": 4.0,
                "p90_longitudinal_extent_saved": 4.5,
                "mean_component_count_saved": 1.0,
                "mean_tail_bridge_count_saved": 0.0,
                "long_vehicle_saved_count": 0,
                "aggregate_status_counts": {"saved": 1},
                "run_dir": "/tmp/run2",
                "runtime_seconds": 4.0,
                "total_wall_seconds": 4.0,
                "total_cpu_seconds": 2.0,
                "compute_wall_seconds": 2.0,
                "compute_cpu_seconds": 1.5,
                "io_wall_seconds": 2.0,
                "total_hz": 2.5,
                "compute_hz": 5.0,
                "peak_rss_mb": 110.0,
                "wall_ms_per_frame": 400.0,
                "cpu_ms_per_frame": 200.0,
                "accumulate_wall_ms_per_track": 75.0,
            },
        ]
    )

    assert aggregated["total_hz_median"] == 3.75
    assert aggregated["total_hz_min"] == 2.5
    assert aggregated["total_hz_max"] == 5.0
    assert aggregated["compute_hz_median"] == 6.5
    assert aggregated["compute_hz_min"] == 5.0
    assert aggregated["compute_hz_max"] == 8.0


def test_component_matching_uses_identical_remaining_context() -> None:
    runner = BenchmarkRunner(Path(__file__).resolve().parents[3])
    rows = [
        {
            "sequence_name": "seq_a",
            "preset_name": "gicp",
            "tracker_algorithm": "kalman_nn",
            "clusterer_algorithm": "euclidean_clustering",
            "accumulator_algorithm": "registration_voxel_fusion",
            "registration_backend": "generalized_icp",
            "frame_selection_method": "keyframe_motion",
            "long_vehicle_mode": False,
            "total_wall_seconds_median": 1.2,
            "total_hz_median": 8.3,
            "compute_hz_median": 10.0,
            "aggregation_registration_wall_seconds_median": 0.4,
            "aggregation_fusion_total_wall_seconds_median": 0.6,
            "compute_wall_seconds_median": 0.9,
            "total_cpu_seconds_median": 0.8,
            "peak_rss_mb_max": 120.0,
        },
        {
            "sequence_name": "seq_a",
            "preset_name": "feature",
            "tracker_algorithm": "kalman_nn",
            "clusterer_algorithm": "euclidean_clustering",
            "accumulator_algorithm": "registration_voxel_fusion",
            "registration_backend": "feature_global_then_local",
            "frame_selection_method": "keyframe_motion",
            "long_vehicle_mode": False,
            "total_wall_seconds_median": 1.5,
            "total_hz_median": 6.7,
            "compute_hz_median": 8.3,
            "aggregation_registration_wall_seconds_median": 0.7,
            "aggregation_fusion_total_wall_seconds_median": 0.8,
            "compute_wall_seconds_median": 1.2,
            "total_cpu_seconds_median": 1.1,
            "peak_rss_mb_max": 128.0,
        },
        {
            "sequence_name": "seq_a",
            "preset_name": "other_clusterer",
            "tracker_algorithm": "kalman_nn",
            "clusterer_algorithm": "range_image_connected_components",
            "accumulator_algorithm": "registration_voxel_fusion",
            "registration_backend": "generalized_icp",
            "frame_selection_method": "keyframe_motion",
            "long_vehicle_mode": False,
            "total_wall_seconds_median": 1.7,
            "total_hz_median": 5.9,
            "compute_hz_median": 7.1,
            "aggregation_registration_wall_seconds_median": 0.5,
            "aggregation_fusion_total_wall_seconds_median": 0.9,
            "compute_wall_seconds_median": 1.4,
            "total_cpu_seconds_median": 1.0,
            "peak_rss_mb_max": 130.0,
        },
    ]

    matched_groups = runner._matched_component_groups(rows, "registration_backend")

    assert len(matched_groups) == 1
    context, group_rows = matched_groups[0]
    assert "clusterer_algorithm=euclidean_clustering" in context
    assert [row["preset_name"] for row in group_rows] == ["gicp", "feature"]


def test_component_report_lists_registration_backend_section() -> None:
    runner = BenchmarkRunner(Path(__file__).resolve().parents[3])
    rows = [
        {
            "sequence_name": "seq_a",
            "preset_name": "gicp",
            "tracker_algorithm": "kalman_nn",
            "clusterer_algorithm": "euclidean_clustering",
            "accumulator_algorithm": "registration_voxel_fusion",
            "registration_backend": "generalized_icp",
            "frame_selection_method": "keyframe_motion",
            "long_vehicle_mode": False,
            "total_wall_seconds_median": 1.2,
            "total_hz_median": 8.3,
            "compute_hz_median": 10.0,
            "aggregation_registration_wall_seconds_median": 0.4,
            "aggregation_fusion_total_wall_seconds_median": 0.6,
            "compute_wall_seconds_median": 0.9,
            "total_cpu_seconds_median": 0.8,
            "peak_rss_mb_max": 120.0,
        },
        {
            "sequence_name": "seq_a",
            "preset_name": "feature",
            "tracker_algorithm": "kalman_nn",
            "clusterer_algorithm": "euclidean_clustering",
            "accumulator_algorithm": "registration_voxel_fusion",
            "registration_backend": "feature_global_then_local",
            "frame_selection_method": "keyframe_motion",
            "long_vehicle_mode": False,
            "total_wall_seconds_median": 1.5,
            "total_hz_median": 6.7,
            "compute_hz_median": 8.3,
            "aggregation_registration_wall_seconds_median": 0.7,
            "aggregation_fusion_total_wall_seconds_median": 0.8,
            "compute_wall_seconds_median": 1.2,
            "total_cpu_seconds_median": 1.1,
            "peak_rss_mb_max": 128.0,
        },
    ]

    markdown = runner._render_component_markdown(rows)

    assert "## registration_backend" in markdown
    assert "Total Hz Median" in markdown
    assert "Compute Hz Median" in markdown
    assert "ICP/Registration Median (s)" in markdown
    assert "Fusion Post Median (s)" in markdown
    assert "Post-Filter Median (s)" in markdown
    assert "feature_global_then_local" in markdown
    assert "generalized_icp" in markdown
