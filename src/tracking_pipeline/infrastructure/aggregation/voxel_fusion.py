from __future__ import annotations

import math
from typing import Any

import numpy as np
import open3d as o3d

from tracking_pipeline.config.models import AggregationConfig, OutputConfig, TrackingConfig
from tracking_pipeline.domain.models import AggregateResult, Track
from tracking_pipeline.domain.rules import (
    axis_to_index,
    compute_extent,
    filter_chunks_by_shape_consistency,
    find_lane_end_touch_index,
    orthogonal_axes,
    select_best_frames_for_aggregation,
    track_exit_line_value,
    track_exited_lane_box,
)
from tracking_pipeline.domain.value_objects import LaneBox
from tracking_pipeline.shared.geometry import ensure_aligned_optional


class VoxelFusionAccumulator:
    fusion_method = "voxel_fusion"

    def __init__(self, config: AggregationConfig, output_config: OutputConfig, tracking_config: TrackingConfig):
        self.config = config
        self.output_config = output_config
        self.tracking_config = tracking_config

    def accumulate(self, track: Track, lane_box: LaneBox) -> AggregateResult:
        if track.hit_count < int(self.tracking_config.min_track_hits):
            return self._result(track, lane_box, np.zeros((0, 3), dtype=np.float32), [], "skipped_min_hits", self._base_metrics(track))
        if self.output_config.require_track_exit and not track_exited_lane_box(
            track,
            lane_box,
            edge_margin=self.output_config.track_exit_edge_margin,
            axis=self.config.frame_selection_line_axis,
        ):
            return self._result(track, lane_box, np.zeros((0, 3), dtype=np.float32), [], "skipped_track_exit", self._base_metrics(track))

        long_vehicle_mode_applied = self._track_is_long_vehicle(track)
        frame_selection_method = self._effective_frame_selection_method(track, long_vehicle_mode_applied)
        all_world_chunks = list(track.world_points)
        all_chunk_intensity = ensure_aligned_optional(track.world_intensity, len(all_world_chunks))
        all_point_timestamps_ns = ensure_aligned_optional(track.point_timestamps_ns, len(all_world_chunks))
        (
            world_chunks,
            chunk_intensity,
            chunk_point_timestamps_ns,
            track_centers,
            track_frame_ids,
            track_frame_timestamps_ns,
            chunk_extents,
            lane_end_touch_info,
        ) = self._truncate_after_lane_end_touch(track, lane_box, all_world_chunks, all_chunk_intensity, all_point_timestamps_ns)
        available_frame_ids = list(track_frame_ids)
        world_chunks, track_centers, track_frame_ids, chunk_quality_info = self._filter_chunks_by_quality_window(
            list(world_chunks),
            list(track_centers),
            list(track_frame_ids),
            chunk_extents,
        )
        chunk_intensity = self._select_optional_by_frame_ids(available_frame_ids, chunk_intensity, track_frame_ids)
        chunk_point_timestamps_ns = self._select_optional_by_frame_ids(available_frame_ids, chunk_point_timestamps_ns, track_frame_ids)
        track_frame_timestamps_ns = self._select_scalar_by_frame_ids(available_frame_ids, track_frame_timestamps_ns, track_frame_ids)
        selected_chunks, selected_centers, selected_frame_ids, selection_info = select_best_frames_for_aggregation(
            chunks=list(world_chunks),
            centers=list(track_centers),
            frame_ids=list(track_frame_ids),
            frame_selection_method=frame_selection_method,
            use_all_frames=self.config.use_all_frames,
            top_k=self.config.top_k_frames,
            keyframe_keep=self.config.keyframe_keep,
            length_coverage_bins=self.config.length_coverage_bins,
            lane_box=lane_box,
            line_axis=self.config.frame_selection_line_axis,
            line_ratio=self.config.frame_selection_line_ratio,
            line_touch_margin=self.config.frame_selection_touch_margin,
        )
        selected_intensity = self._select_optional_by_frame_ids(track_frame_ids, chunk_intensity, selected_frame_ids)
        selected_point_timestamps_ns = self._select_optional_by_frame_ids(track_frame_ids, chunk_point_timestamps_ns, selected_frame_ids)
        selected_frame_timestamps_ns = self._select_scalar_by_frame_ids(track_frame_ids, track_frame_timestamps_ns, selected_frame_ids)
        selection_info = {**lane_end_touch_info, **chunk_quality_info, **selection_info}
        if self.config.shape_consistency_filter:
            pre_shape_frame_ids = [int(frame_id) for frame_id in selected_frame_ids]
            selected_chunks, selected_centers, selected_frame_ids, shape_info = filter_chunks_by_shape_consistency(
                selected_chunks,
                selected_centers,
                selected_frame_ids,
                self.config.shape_consistency_max_extent_ratio,
            )
            selected_intensity = self._select_optional_by_frame_ids(
                pre_shape_frame_ids,
                selected_intensity,
                selected_frame_ids,
            )
            selected_point_timestamps_ns = self._select_optional_by_frame_ids(
                pre_shape_frame_ids,
                selected_point_timestamps_ns,
                selected_frame_ids,
            )
            selected_frame_timestamps_ns = self._select_scalar_by_frame_ids(
                pre_shape_frame_ids,
                selected_frame_timestamps_ns,
                selected_frame_ids,
            )
            selection_info = {**selection_info, **shape_info}
        if not selected_chunks:
            metrics = {**self._base_metrics(track), **selection_info, **self._motion_deskew_metrics()}
            return self._result(track, lane_box, np.zeros((0, 3), dtype=np.float32), [], "empty_selection", metrics)

        selected_chunks, selected_intensity, motion_deskew_metrics = self._apply_motion_deskew(
            track,
            selected_chunks,
            selected_intensity,
            selected_centers,
            selected_frame_ids,
            selected_frame_timestamps_ns,
            selected_point_timestamps_ns,
            long_vehicle_mode_applied,
        )
        if not self.output_config.save_world:
            selected_chunks = [
                (np.asarray(points, dtype=np.float32) - np.asarray(center, dtype=np.float32)).astype(np.float32, copy=False)
                for points, center in zip(selected_chunks, selected_centers)
            ]

        selected_triplets = [
            (
                *self._voxel_downsample(points, intensity, self.config.frame_downsample_voxel),
                center,
                frame_id,
            )
            for points, intensity, center, frame_id in zip(selected_chunks, selected_intensity, selected_centers, selected_frame_ids)
        ]
        selected_triplets = [(points, intensity, center, frame_id) for points, intensity, center, frame_id in selected_triplets if len(points) > 0]
        prepared_chunks = [points for points, _, _, _ in selected_triplets]
        prepared_intensity = [intensity for _, intensity, _, _ in selected_triplets]
        selected_centers = [center for _, _, center, _ in selected_triplets]
        selected_frame_ids = [int(frame_id) for _, _, _, frame_id in selected_triplets]
        if not prepared_chunks:
            metrics = {**self._base_metrics(track), **selection_info, **motion_deskew_metrics}
            return self._result(
                track,
                lane_box,
                np.zeros((0, 3), dtype=np.float32),
                selected_frame_ids,
                "empty_prepared_chunks",
                metrics,
                prepared_chunk_count=0,
            )

        accumulation_input, registration_metrics = self._prepare_for_fusion(prepared_chunks)
        (
            accumulation_input,
            prepared_intensity,
            selected_centers,
            selected_frame_ids,
            registration_metrics,
        ) = self._apply_registration_subset(
            accumulation_input,
            prepared_chunks,
            prepared_intensity,
            selected_centers,
            selected_frame_ids,
            registration_metrics,
        )
        chunk_weights = self._chunk_weights(track, accumulation_input, registration_metrics, long_vehicle_mode_applied)
        confidence_chunk_weights = self._confidence_chunk_weights(accumulation_input, chunk_weights, registration_metrics)
        fused_xyz, fused_intensity, fused_confidence, raw_points_total, fusion_voxels_total, fusion_voxels_kept = self._fuse_chunks(
            accumulation_input,
            prepared_intensity,
            chunk_weights,
            confidence_chunk_weights,
            min_observations=self._required_observations(len(accumulation_input)),
        )
        if len(fused_xyz) == 0:
            metrics = {**self._base_metrics(track), **selection_info, **motion_deskew_metrics, **registration_metrics}
            return self._result(
                track,
                lane_box,
                np.zeros((0, 3), dtype=np.float32),
                selected_frame_ids,
                "empty_fused",
                metrics,
                prepared_chunk_count=len(accumulation_input),
            )

        filtered_xyz, filtered_intensity, filtered_confidence, prefilter_points, stat_filtered_points, final_points = self._post_filter(
            fused_xyz,
            fused_intensity,
            fused_confidence,
        )
        filtered_xyz, filtered_intensity, filtered_confidence, component_count_post_fusion, tail_bridge_count, longitudinal_extent = self._apply_tail_bridge(
            filtered_xyz,
            filtered_intensity,
            filtered_confidence,
            long_vehicle_mode_applied,
        )
        final_points = len(filtered_xyz)
        symmetry_metrics = self._symmetry_completion_metrics(filtered_xyz, selected_centers)
        dimension_metrics = self._vehicle_dimension_metrics(filtered_xyz)
        metrics: dict[str, Any] = {
            **self._base_metrics(track),
            **selection_info,
            **motion_deskew_metrics,
            **registration_metrics,
            "alignment_method": registration_metrics.get("alignment_method", "none"),
            "frame_selection_method": selection_info.get("strategy", self.config.frame_selection_method),
            "fusion_method": self.fusion_method,
            "chunk_weights": chunk_weights,
            "raw_point_count": raw_points_total,
            "fusion_voxels_total": fusion_voxels_total,
            "fusion_voxels_kept": fusion_voxels_kept,
            "point_count_after_fusion": prefilter_points,
            "point_count_after_stat_filter": stat_filtered_points,
            "point_count_after_downsample": final_points,
            "longitudinal_extent": longitudinal_extent,
            "component_count_post_fusion": component_count_post_fusion,
            "tail_bridge_count": tail_bridge_count,
            "long_vehicle_mode_applied": bool(long_vehicle_mode_applied),
            "min_saved_aggregate_points": int(self.config.min_saved_aggregate_points),
            "mode": "world" if self.output_config.save_world else "local",
            **dimension_metrics,
            **symmetry_metrics,
        }
        if final_points == 0:
            return self._result(
                track,
                lane_box,
                np.zeros((0, 3), dtype=np.float32),
                selected_frame_ids,
                "empty_filtered",
                metrics,
                prepared_chunk_count=len(accumulation_input),
                point_count_after_downsample=0,
                point_count_before_confidence_cap=0,
                point_count_after_confidence_cap=0,
            )
        if final_points < int(self.config.min_saved_aggregate_points):
            return self._result(
                track,
                lane_box,
                np.zeros((0, 3), dtype=np.float32),
                selected_frame_ids,
                "skipped_min_saved_points",
                metrics,
                prepared_chunk_count=len(accumulation_input),
                point_count_after_downsample=final_points,
                point_count_before_confidence_cap=final_points,
                point_count_after_confidence_cap=final_points,
            )
        quality_threshold = self._quality_threshold(long_vehicle_mode_applied)
        if float(track.quality_score or 0.0) < quality_threshold:
            metrics["quality_threshold"] = quality_threshold
            return self._result(
                track,
                lane_box,
                np.zeros((0, 3), dtype=np.float32),
                selected_frame_ids,
                "skipped_quality_threshold",
                metrics,
                prepared_chunk_count=len(accumulation_input),
                point_count_after_downsample=final_points,
                quality_threshold=quality_threshold,
                point_count_before_confidence_cap=final_points,
                point_count_after_confidence_cap=final_points,
            )
        point_count_before_confidence_cap = int(len(filtered_xyz))
        capped_xyz, capped_intensity, capped_confidence, confidence_cap_applied = self._apply_confidence_point_cap(
            filtered_xyz,
            filtered_intensity,
            filtered_confidence,
        )
        completed_xyz, completed_intensity, _, symmetry_metrics = self._apply_symmetry_completion(
            capped_xyz,
            capped_intensity,
            capped_confidence,
            selected_centers,
            component_count_post_fusion,
        )
        metrics.update(symmetry_metrics)
        metrics.update(self._vehicle_dimension_metrics(completed_xyz))
        metrics["point_count_after_downsample"] = int(len(completed_xyz))
        return self._result(
            track,
            lane_box,
            completed_xyz,
            selected_frame_ids,
            "saved",
            metrics,
            intensity=completed_intensity,
            prepared_chunk_count=len(accumulation_input),
            point_count_after_downsample=len(completed_xyz),
            quality_threshold=quality_threshold,
            point_count_before_confidence_cap=point_count_before_confidence_cap,
            point_count_after_confidence_cap=len(capped_xyz),
            confidence_point_cap_applied=confidence_cap_applied,
        )

    def _base_metrics(self, track: Track) -> dict[str, Any]:
        metrics = {
            "quality_score": 0.0 if track.quality_score is None else float(track.quality_score),
            "quality_metrics": track.quality_metrics,
            "track_source_ids": track.source_track_ids or [track.track_id],
        }
        if bool(track.state.get("articulated_vehicle")):
            metrics["articulated_vehicle"] = True
            component_ids = track.state.get("articulated_component_track_ids")
            if component_ids:
                metrics["articulated_component_track_ids"] = list(component_ids)
            gap_mean = track.state.get("articulated_rear_gap_mean")
            if gap_mean is not None:
                metrics["articulated_rear_gap_mean"] = float(gap_mean)
            gap_std = track.state.get("articulated_rear_gap_std")
            if gap_std is not None:
                metrics["articulated_rear_gap_std"] = float(gap_std)
            object_kind = track.state.get("object_kind")
            if object_kind:
                metrics["object_kind"] = str(object_kind)
        return metrics

    def _result(
        self,
        track: Track,
        lane_box: LaneBox,
        points: np.ndarray,
        selected_frame_ids: list[int],
        status: str,
        metrics: dict[str, Any],
        *,
        intensity: np.ndarray | None = None,
        prepared_chunk_count: int = 0,
        point_count_after_downsample: int | None = None,
        quality_threshold: float | None = None,
        point_count_before_confidence_cap: int | None = None,
        point_count_after_confidence_cap: int | None = None,
        confidence_point_cap_applied: bool = False,
    ) -> AggregateResult:
        cap_before = int(len(points)) if point_count_before_confidence_cap is None else int(point_count_before_confidence_cap)
        cap_after = int(len(points)) if point_count_after_confidence_cap is None else int(point_count_after_confidence_cap)
        merged_metrics = {
            **metrics,
            **self._confidence_point_cap_metrics(
                point_count_before_cap=cap_before,
                point_count_after_cap=cap_after,
                applied=confidence_point_cap_applied,
            ),
            **self._decision_metrics(
                track,
                lane_box,
                status,
                selected_frame_ids,
                prepared_chunk_count=prepared_chunk_count,
                point_count_after_downsample=point_count_after_downsample,
                quality_threshold=quality_threshold,
            ),
        }
        return AggregateResult(
            track.track_id,
            np.asarray(points, dtype=np.float32),
            [int(frame_id) for frame_id in selected_frame_ids],
            status,
            merged_metrics,
            intensity=None if intensity is None else np.asarray(intensity, dtype=np.float32),
        )

    def _confidence_point_cap_metrics(
        self,
        *,
        point_count_before_cap: int,
        point_count_after_cap: int,
        applied: bool,
    ) -> dict[str, Any]:
        return {
            "confidence_point_cap_enabled": bool(self.config.enable_confidence_point_cap),
            "confidence_point_cap_applied": bool(applied),
            "confidence_point_cap_target": int(self.config.confidence_point_cap_max_points),
            "confidence_point_cap_bins": int(self.config.confidence_point_cap_bins),
            "point_count_before_confidence_cap": int(point_count_before_cap),
            "point_count_after_confidence_cap": int(point_count_after_cap),
        }

    def _decision_metrics(
        self,
        track: Track,
        lane_box: LaneBox,
        status: str,
        selected_frame_ids: list[int],
        *,
        prepared_chunk_count: int,
        point_count_after_downsample: int | None,
        quality_threshold: float | None,
    ) -> dict[str, Any]:
        decision_stage, decision_reason_code, decision_reason_label = self._decision_identity(status)
        quality_score = 0.0 if track.quality_score is None else float(track.quality_score)
        track_exit_metrics = self._track_exit_metrics(track, lane_box)
        point_count = 0 if point_count_after_downsample is None else int(point_count_after_downsample)
        selected_count = int(len(selected_frame_ids))
        summary = self._decision_summary(
            decision_reason_code,
            hit_count=int(track.hit_count),
            min_track_hits=int(self.tracking_config.min_track_hits),
            track_exit_line_value=float(track_exit_metrics["track_exit_line_value"]),
            track_center_line_coordinate=float(track_exit_metrics["track_center_line_coordinate"]),
            quality_score=quality_score,
            quality_threshold=quality_threshold,
            point_count_after_downsample=point_count,
            min_saved_aggregate_points=int(self.config.min_saved_aggregate_points),
            selected_frame_count=selected_count,
            prepared_chunk_count=int(prepared_chunk_count),
        )
        return {
            "decision_stage": decision_stage,
            "decision_reason_code": decision_reason_code,
            "decision_reason_label": decision_reason_label,
            "decision_summary": summary,
            "hit_count": int(track.hit_count),
            "min_track_hits": int(self.tracking_config.min_track_hits),
            "quality_score": quality_score,
            "quality_threshold": 0.0 if quality_threshold is None else float(quality_threshold),
            "selected_frame_count": selected_count,
            "prepared_chunk_count": int(prepared_chunk_count),
            "point_count_after_downsample": point_count,
            "min_saved_aggregate_points": int(self.config.min_saved_aggregate_points),
            **track_exit_metrics,
        }

    def _decision_identity(self, status: str) -> tuple[str, str, str]:
        mapping = {
            "skipped_min_hits": ("tracking_gate", "min_hits", "Minimum Hits"),
            "skipped_track_exit": ("exit_gate", "track_exit", "Track Exit"),
            "empty_selection": ("selection", "empty_selection", "Empty Selection"),
            "empty_prepared_chunks": ("preparation", "empty_prepared_chunks", "Empty Prepared Chunks"),
            "empty_fused": ("fusion", "empty_output", "Empty Output"),
            "empty_filtered": ("fusion", "empty_output", "Empty Output"),
            "skipped_min_saved_points": ("save_gate", "min_saved_points", "Minimum Saved Points"),
            "skipped_quality_threshold": ("quality_gate", "quality_threshold", "Quality Threshold"),
            "saved": ("saved", "saved", "Saved"),
        }
        return mapping.get(status, ("other", status, status.replace("_", " ")))

    def _decision_summary(
        self,
        decision_reason_code: str,
        *,
        hit_count: int,
        min_track_hits: int,
        track_exit_line_value: float,
        track_center_line_coordinate: float,
        quality_score: float,
        quality_threshold: float | None,
        point_count_after_downsample: int,
        min_saved_aggregate_points: int,
        selected_frame_count: int,
        prepared_chunk_count: int,
    ) -> str:
        if decision_reason_code == "min_hits":
            return f"min_hits {hit_count}/{min_track_hits}"
        if decision_reason_code == "track_exit":
            return f"track_exit line={track_exit_line_value:.2f} center={track_center_line_coordinate:.2f}"
        if decision_reason_code == "empty_selection":
            return f"empty_selection selected={selected_frame_count}"
        if decision_reason_code == "empty_prepared_chunks":
            return f"empty_prepared_chunks prepared={prepared_chunk_count}"
        if decision_reason_code == "empty_output":
            return f"empty_output points={point_count_after_downsample}"
        if decision_reason_code == "min_saved_points":
            return f"min_saved_points {point_count_after_downsample}/{min_saved_aggregate_points}"
        if decision_reason_code == "quality_threshold":
            threshold = 0.0 if quality_threshold is None else float(quality_threshold)
            return f"quality {quality_score:.2f}<{threshold:.2f}"
        if decision_reason_code == "saved":
            return f"saved points={point_count_after_downsample}"
        return decision_reason_code

    def _track_exit_metrics(self, track: Track, lane_box: LaneBox) -> dict[str, Any]:
        axis_idx = axis_to_index(self.config.frame_selection_line_axis)
        line_value = track_exit_line_value(lane_box, axis_idx, edge_margin=self.output_config.track_exit_edge_margin)
        if not track.centers:
            return {
                "track_exited": False,
                "track_exit_edge_margin": float(self.output_config.track_exit_edge_margin),
                "closest_edge_distance": -1.0,
                "distance_to_exit_line": -1.0,
                "track_exit_line_axis": ["x", "y", "z"][axis_idx],
                "track_exit_line_side": "min",
                "track_exit_line_value": float(line_value),
                "track_center_line_coordinate": -1.0,
                "track_passed_exit_line": False,
            }
        last = np.asarray(track.centers[-1], dtype=np.float64)
        center_coordinate = float(last[axis_idx]) if last.shape[0] > axis_idx else -1.0
        distance_to_exit_line = float(center_coordinate - float(line_value))
        return {
            "track_exited": bool(
                track_exited_lane_box(
                    track,
                    lane_box,
                    edge_margin=self.output_config.track_exit_edge_margin,
                    axis=self.config.frame_selection_line_axis,
                )
            ),
            "track_exit_edge_margin": float(self.output_config.track_exit_edge_margin),
            "closest_edge_distance": distance_to_exit_line,
            "distance_to_exit_line": distance_to_exit_line,
            "track_exit_line_axis": ["x", "y", "z"][axis_idx],
            "track_exit_line_side": "min",
            "track_exit_line_value": float(line_value),
            "track_center_line_coordinate": center_coordinate,
            "track_passed_exit_line": bool(center_coordinate <= float(line_value)),
        }

    def _symmetry_completion_metrics(
        self,
        xyz: np.ndarray,
        selected_centers: list[np.ndarray],
    ) -> dict[str, Any]:
        _, lateral_idx, _ = self._symmetry_axes()
        return {
            "symmetry_completion_enabled": bool(self.config.symmetry_completion),
            "symmetry_completion_applied": False,
            "symmetry_completion_lateral_axis": ["x", "y", "z"][lateral_idx],
            "symmetry_completion_plane_coordinate": self._symmetry_plane_coordinate(selected_centers, xyz, lateral_idx),
            "point_count_before_symmetry_completion": int(len(xyz)),
            "symmetry_completion_generated_points": 0,
            "point_count_after_symmetry_completion": int(len(xyz)),
            "symmetry_completion_source_side": "none",
            "symmetry_completion_source_slice_count": 0,
            "symmetry_completion_target_slice_count": 0,
            "symmetry_completion_candidate_count": 0,
            "symmetry_completion_overlap_rejected_count": 0,
            "symmetry_completion_continuity_rejected_count": 0,
            "symmetry_completion_capped_count": 0,
            "symmetry_completion_skipped_reason": "disabled" if not self.config.symmetry_completion else "not_run",
        }

    def _vehicle_dimension_metrics(self, xyz: np.ndarray) -> dict[str, Any]:
        axis_idx, lateral_idx, vertical_idx = self._symmetry_axes()
        extent = compute_extent(np.asarray(xyz, dtype=np.float32))
        axis_names = ["x", "y", "z"]
        return {
            "vehicle_length": float(extent[axis_idx]),
            "vehicle_width": float(extent[lateral_idx]),
            "vehicle_height": float(extent[vertical_idx]),
            "vehicle_length_axis": axis_names[axis_idx],
            "vehicle_width_axis": axis_names[lateral_idx],
            "vehicle_height_axis": axis_names[vertical_idx],
            "extent_x": float(extent[0]),
            "extent_y": float(extent[1]),
            "extent_z": float(extent[2]),
        }

    def _apply_symmetry_completion(
        self,
        xyz: np.ndarray,
        intensity: np.ndarray | None,
        confidence: np.ndarray | None,
        selected_centers: list[np.ndarray],
        component_count_post_fusion: int,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, dict[str, Any]]:
        points = np.asarray(xyz, dtype=np.float32)
        metrics = self._symmetry_completion_metrics(points, selected_centers)
        if not self.config.symmetry_completion or len(points) == 0:
            metrics["symmetry_completion_skipped_reason"] = "disabled" if not self.config.symmetry_completion else "empty"
            return (
                points,
                None if intensity is None else np.asarray(intensity, dtype=np.float32),
                None if confidence is None else np.asarray(confidence, dtype=np.float32),
                metrics,
            )
        _ = component_count_post_fusion

        axis_idx, lateral_idx, vertical_idx = self._symmetry_axes()
        plane_coordinate = float(metrics["symmetry_completion_plane_coordinate"])
        completion_voxel = self._symmetry_completion_voxel()
        intensity_values = None
        if intensity is not None and len(np.asarray(intensity, dtype=np.float32)) == len(points):
            intensity_values = np.asarray(intensity, dtype=np.float32)
        confidence_values = None
        if confidence is not None and len(np.asarray(confidence, dtype=np.float32)) == len(points):
            confidence_values = np.asarray(confidence, dtype=np.float32)

        slice_records, side_point_clouds = self._symmetry_slice_records(
            points,
            intensity_values,
            plane_coordinate,
            axis_idx,
            lateral_idx,
            vertical_idx,
            completion_voxel,
        )
        positive_slice_count = sum(1 for records in slice_records.values() if 1 in records)
        negative_slice_count = sum(1 for records in slice_records.values() if -1 in records)
        positive_count = int(len(side_point_clouds.get(1, np.zeros((0, 3), dtype=np.float32))))
        negative_count = int(len(side_point_clouds.get(-1, np.zeros((0, 3), dtype=np.float32))))
        if positive_count == 0 and negative_count == 0:
            metrics["symmetry_completion_skipped_reason"] = "no_off_plane_points"
            return points, intensity_values, confidence_values, metrics

        dominant_side = self._symmetry_half_source_side(points, plane_coordinate, lateral_idx, completion_voxel)
        source_slice_count = positive_slice_count if dominant_side > 0 else negative_slice_count
        target_slice_count = negative_slice_count if dominant_side > 0 else positive_slice_count
        metrics["symmetry_completion_source_side"] = "positive" if dominant_side > 0 else "negative"
        metrics["symmetry_completion_source_slice_count"] = int(source_slice_count)
        metrics["symmetry_completion_target_slice_count"] = int(target_slice_count)

        signed_distance = points[:, lateral_idx].astype(np.float64) - plane_coordinate
        positive_mask = signed_distance > (0.5 * completion_voxel)
        negative_mask = signed_distance < (-0.5 * completion_voxel)
        center_mask = ~(positive_mask | negative_mask)
        source_mask = positive_mask if dominant_side > 0 else negative_mask
        kept_mask = center_mask | source_mask
        source_points = np.asarray(points[source_mask], dtype=np.float32)
        if len(source_points) == 0:
            metrics["symmetry_completion_skipped_reason"] = "no_source_side"
            return points, intensity_values, confidence_values, metrics

        mirrored_points = source_points.copy()
        mirrored_points[:, lateral_idx] = np.float32((2.0 * plane_coordinate) - mirrored_points[:, lateral_idx])
        metrics["symmetry_completion_candidate_count"] = int(len(source_points))

        kept_points = np.asarray(points[kept_mask], dtype=np.float32)
        combined_xyz = np.vstack([kept_points, mirrored_points]).astype(np.float32, copy=False)
        combined_intensity = None
        if intensity_values is not None:
            kept_intensity = np.asarray(intensity_values[kept_mask], dtype=np.float32)
            source_intensity = np.asarray(intensity_values[source_mask], dtype=np.float32)
            combined_intensity = np.concatenate([kept_intensity, source_intensity], axis=0).astype(np.float32, copy=False)
        combined_confidence = None
        if confidence_values is not None:
            kept_confidence = np.asarray(confidence_values[kept_mask], dtype=np.float32)
            source_confidence = np.asarray(confidence_values[source_mask], dtype=np.float32)
            combined_confidence = np.concatenate([kept_confidence, source_confidence], axis=0).astype(np.float32, copy=False)
        combined_xyz, combined_intensity, combined_confidence = self._aggregate_per_voxel(
            combined_xyz,
            combined_intensity,
            combined_confidence,
            completion_voxel,
        )
        metrics.update(
            {
                "symmetry_completion_applied": True,
                "symmetry_completion_generated_points": int(len(source_points)),
                "point_count_after_symmetry_completion": int(len(combined_xyz)),
                "symmetry_completion_skipped_reason": "applied",
            }
        )
        return combined_xyz, combined_intensity, combined_confidence, metrics

    def _symmetry_axes(self) -> tuple[int, int, int]:
        axis_idx = axis_to_index(self.config.frame_selection_line_axis)
        lateral_idx, vertical_idx = orthogonal_axes(axis_idx)
        return axis_idx, lateral_idx, vertical_idx

    def _symmetry_completion_voxel(self) -> float:
        return max(0.05, float(self.config.aggregate_voxel), float(self.config.fusion_voxel_size))

    def _symmetry_slice_records(
        self,
        points: np.ndarray,
        intensity: np.ndarray | None,
        plane_coordinate: float,
        axis_idx: int,
        lateral_idx: int,
        vertical_idx: int,
        completion_voxel: float,
    ) -> tuple[dict[tuple[int, int], dict[int, dict[str, Any]]], dict[int, np.ndarray]]:
        off_plane_mask = np.abs(np.asarray(points, dtype=np.float32)[:, lateral_idx] - float(plane_coordinate)) >= (0.5 * completion_voxel)
        off_plane_points = np.asarray(points, dtype=np.float32)[off_plane_mask]
        off_plane_intensity = None if intensity is None else np.asarray(intensity, dtype=np.float32)[off_plane_mask]
        if len(off_plane_points) == 0:
            return {}, {1: np.zeros((0, 3), dtype=np.float32), -1: np.zeros((0, 3), dtype=np.float32)}

        side_distances = off_plane_points[:, lateral_idx].astype(np.float64) - float(plane_coordinate)
        side_point_clouds = {
            1: np.asarray(off_plane_points[side_distances > 0.0], dtype=np.float32),
            -1: np.asarray(off_plane_points[side_distances < 0.0], dtype=np.float32),
        }
        raw_buckets: dict[tuple[int, int, int], dict[str, Any]] = {}
        for index, point in enumerate(off_plane_points):
            signed_distance = float(point[lateral_idx] - plane_coordinate)
            side = 1 if signed_distance > 0.0 else -1
            bucket = raw_buckets.setdefault(
                (
                    int(np.floor(float(point[axis_idx]) / completion_voxel)),
                    int(np.floor(float(point[vertical_idx]) / completion_voxel)),
                    int(side),
                ),
                {
                    "points": [],
                    "distances": [],
                    "intensity": [],
                },
            )
            bucket["points"].append(np.asarray(point, dtype=np.float32))
            bucket["distances"].append(abs(signed_distance))
            if off_plane_intensity is not None:
                bucket["intensity"].append(float(off_plane_intensity[index]))

        slice_records: dict[tuple[int, int], dict[int, dict[str, Any]]] = {}
        for (longitudinal_bin, vertical_bin, side), bucket in raw_buckets.items():
            bucket_points = np.asarray(bucket["points"], dtype=np.float32)
            bucket_distances = np.asarray(bucket["distances"], dtype=np.float32)
            lateral_bins = np.floor(bucket_distances / completion_voxel).astype(np.int32)
            outer_bin = int(np.max(lateral_bins))
            outer_mask = lateral_bins == outer_bin
            outer_points = bucket_points[outer_mask]
            representative_point = np.asarray(np.mean(outer_points, axis=0), dtype=np.float32)
            representative_intensity = None
            if off_plane_intensity is not None and bucket["intensity"]:
                bucket_intensity = np.asarray(bucket["intensity"], dtype=np.float32)
                representative_intensity = float(np.mean(bucket_intensity[outer_mask]))
            slice_records.setdefault((int(longitudinal_bin), int(vertical_bin)), {})[int(side)] = {
                "point": representative_point,
                "intensity": representative_intensity,
                "support_count": int(len(bucket_points)),
                "outer_lateral_distance": float(np.mean(bucket_distances[outer_mask])),
            }
        return slice_records, side_point_clouds

    def _symmetry_half_source_side(
        self,
        points: np.ndarray,
        plane_coordinate: float,
        lateral_idx: int,
        completion_voxel: float,
    ) -> int:
        signed_distance = np.asarray(points, dtype=np.float32)[:, lateral_idx].astype(np.float64) - float(plane_coordinate)
        positive_mask = signed_distance > (0.5 * completion_voxel)
        negative_mask = signed_distance < (-0.5 * completion_voxel)
        positive_count = int(np.sum(positive_mask))
        negative_count = int(np.sum(negative_mask))
        if positive_count > negative_count:
            return 1
        if negative_count > positive_count:
            return -1
        positive_extent = 0.0 if positive_count == 0 else float(np.mean(np.abs(signed_distance[positive_mask])))
        negative_extent = 0.0 if negative_count == 0 else float(np.mean(np.abs(signed_distance[negative_mask])))
        if positive_extent > negative_extent:
            return 1
        if negative_extent > positive_extent:
            return -1
        return 1

    def _symmetry_plane_coordinate(
        self,
        selected_centers: list[np.ndarray],
        xyz: np.ndarray,
        lateral_idx: int,
    ) -> float:
        base_plane = 0.0
        if self.output_config.save_world:
            if selected_centers:
                center_values = np.asarray(selected_centers, dtype=np.float64)
                if center_values.ndim == 2 and center_values.shape[1] > lateral_idx:
                    base_plane = float(np.median(center_values[:, lateral_idx]))
            else:
                points = np.asarray(xyz, dtype=np.float64)
                if points.ndim == 2 and len(points) > 0 and points.shape[1] > lateral_idx:
                    base_plane = float(np.mean(points[:, lateral_idx]))

        points = np.asarray(xyz, dtype=np.float64)
        if points.ndim != 2 or len(points) == 0 or points.shape[1] <= lateral_idx:
            return float(base_plane)
        values = points[:, lateral_idx]
        values = values[np.isfinite(values)]
        if len(values) == 0:
            return float(base_plane)

        completion_voxel = self._symmetry_completion_voxel()
        centered = values - float(base_plane)
        has_negative = bool(np.any(centered < (-0.5 * completion_voxel)))
        has_positive = bool(np.any(centered > (0.5 * completion_voxel)))
        if not (has_negative and has_positive):
            return float(base_plane)

        lower = float(np.percentile(values, 5.0))
        upper = float(np.percentile(values, 95.0))
        robust_midpoint = 0.5 * (lower + upper)
        if abs(robust_midpoint - float(base_plane)) < (0.5 * completion_voxel):
            return float(base_plane)
        return float(robust_midpoint)

    def _prepare_for_fusion(self, chunks: list[np.ndarray]) -> tuple[list[np.ndarray], dict[str, Any]]:
        return chunks, {
            "alignment_method": "none",
            "registration_backend": "disabled",
            "registration_pairs": 0,
            "registration_accepted": 0,
            "registration_rejected": 0,
            "registration_input_chunk_count": int(len(chunks)),
            "registration_output_chunk_count": int(len(chunks)),
            "registration_dropped_count": 0,
            "registration_keep_indices": list(range(len(chunks))),
            "registration_chunk_weights": [1.0 for _ in chunks],
        }

    def _apply_registration_subset(
        self,
        accumulation_input: list[np.ndarray],
        prepared_chunks: list[np.ndarray],
        prepared_intensity: list[np.ndarray | None],
        selected_centers: list[np.ndarray],
        selected_frame_ids: list[int],
        registration_metrics: dict[str, Any],
    ) -> tuple[list[np.ndarray], list[np.ndarray | None], list[np.ndarray], list[int], dict[str, Any]]:
        input_count = len(prepared_intensity)
        keep_indices = registration_metrics.get("registration_keep_indices")
        if not isinstance(keep_indices, list):
            keep_indices = list(range(len(accumulation_input)))
        keep_indices = [
            int(index)
            for index in keep_indices
            if 0 <= int(index) < input_count
        ]
        if len(keep_indices) != len(accumulation_input):
            aligned_count = min(len(accumulation_input), input_count)
            keep_indices = list(range(aligned_count))
            accumulation_input = list(accumulation_input[:aligned_count])
        attempt_output_count = int(len(keep_indices))
        attempt_dropped_count = max(0, int(input_count - attempt_output_count))
        synced_metrics = dict(registration_metrics)
        synced_metrics["registration_input_chunk_count"] = int(input_count)
        chunk_weights = synced_metrics.get("registration_chunk_weights")
        if not isinstance(chunk_weights, list) or len(chunk_weights) != len(keep_indices):
            attempt_chunk_weights = [1.0 for _ in keep_indices]
        else:
            attempt_chunk_weights = [float(weight) for weight in chunk_weights]
        fallback_min_kept_chunks = max(1, int(self.config.registration_min_kept_chunks))
        fallback_enabled = bool(self.config.enable_registration_underfill_fallback)
        fallback_applied = (
            fallback_enabled
            and not bool(synced_metrics.get("registration_skipped", False))
            and attempt_output_count < fallback_min_kept_chunks
            and attempt_output_count < input_count
        )
        if fallback_applied:
            effective_chunks = list(prepared_chunks)
            effective_intensity = list(prepared_intensity)
            effective_centers = list(selected_centers)
            effective_frame_ids = [int(frame_id) for frame_id in selected_frame_ids]
            effective_keep_indices = list(range(input_count))
            effective_chunk_weights = [1.0 for _ in effective_keep_indices]
        else:
            effective_chunks = list(accumulation_input)
            effective_intensity = [prepared_intensity[index] for index in keep_indices]
            effective_centers = [selected_centers[index] for index in keep_indices]
            effective_frame_ids = [int(selected_frame_ids[index]) for index in keep_indices]
            effective_keep_indices = list(keep_indices)
            effective_chunk_weights = attempt_chunk_weights
        synced_metrics["registration_fallback_applied"] = bool(fallback_applied)
        synced_metrics["registration_fallback_min_kept_chunks"] = int(fallback_min_kept_chunks)
        synced_metrics["registration_attempt_output_chunk_count"] = attempt_output_count
        synced_metrics["registration_attempt_dropped_count"] = attempt_dropped_count
        synced_metrics["registration_attempt_keep_indices"] = list(keep_indices)
        synced_metrics["registration_attempt_chunk_weights"] = list(attempt_chunk_weights)
        synced_metrics["registration_output_chunk_count"] = int(len(effective_keep_indices))
        synced_metrics["registration_dropped_count"] = max(0, int(input_count - len(effective_keep_indices)))
        synced_metrics["registration_keep_indices"] = list(effective_keep_indices)
        synced_metrics["registration_chunk_weights"] = list(effective_chunk_weights)
        return (
            effective_chunks,
            effective_intensity,
            effective_centers,
            effective_frame_ids,
            synced_metrics,
        )

    def _chunk_extents_for_track(self, track: Track, chunks: list[np.ndarray]) -> list[np.ndarray]:
        if len(track.bbox_extents) == len(chunks):
            return [np.asarray(extent, dtype=np.float32) for extent in track.bbox_extents]
        return [compute_extent(chunk) for chunk in chunks]

    def _select_optional_by_frame_ids(
        self,
        source_frame_ids: list[int],
        values: list[np.ndarray | None],
        target_frame_ids: list[int],
    ) -> list[np.ndarray | None]:
        if not target_frame_ids:
            return []
        mapping = {int(frame_id): values[index] if index < len(values) else None for index, frame_id in enumerate(source_frame_ids)}
        return [mapping.get(int(frame_id)) for frame_id in target_frame_ids]

    def _select_scalar_by_frame_ids(
        self,
        source_frame_ids: list[int],
        values: list[int],
        target_frame_ids: list[int],
    ) -> list[int]:
        if not target_frame_ids:
            return []
        mapping = {int(frame_id): int(values[index]) if index < len(values) else -1 for index, frame_id in enumerate(source_frame_ids)}
        return [int(mapping.get(int(frame_id), -1)) for frame_id in target_frame_ids]

    def _motion_deskew_metrics(self) -> dict[str, Any]:
        return {
            "motion_deskew_enabled": bool(self.config.motion_deskew),
            "motion_deskew_applied": False,
            "motion_deskew_skipped_reason": "disabled" if not self.config.motion_deskew else "not_run",
            "motion_deskew_corrected_chunk_count": 0,
            "motion_deskew_confident_chunk_count": 0,
            "motion_deskew_mean_speed_mps": 0.0,
            "motion_deskew_mean_abs_shift_m": 0.0,
            "motion_deskew_mean_time_span_ms": 0.0,
        }

    def _apply_motion_deskew(
        self,
        track: Track,
        chunks: list[np.ndarray],
        intensity: list[np.ndarray | None],
        centers: list[np.ndarray],
        frame_ids: list[int],
        frame_timestamps_ns: list[int],
        point_timestamps_ns: list[np.ndarray | None],
        long_vehicle_mode_applied: bool,
    ) -> tuple[list[np.ndarray], list[np.ndarray | None], dict[str, Any]]:
        metrics = self._motion_deskew_metrics()
        if not self.config.motion_deskew:
            return list(chunks), list(intensity), metrics
        if not chunks:
            metrics["motion_deskew_skipped_reason"] = "empty_selection"
            return [], list(intensity), metrics
        if not self._motion_deskew_candidate(track, long_vehicle_mode_applied):
            metrics["motion_deskew_skipped_reason"] = "not_elongated_candidate"
            return list(chunks), list(intensity), metrics
        _ = centers, frame_timestamps_ns

        axis_idx = axis_to_index(self.config.frame_selection_line_axis)
        corrected_chunks: list[np.ndarray] = []
        has_any_point_timestamps = False
        has_any_velocity = False
        corrected_chunk_count = 0
        mean_speed_values: list[float] = []
        mean_abs_shift_values: list[float] = []
        mean_time_span_values: list[float] = []
        for chunk, chunk_point_timestamps, frame_id in zip(chunks, point_timestamps_ns, frame_ids):
            points = np.asarray(chunk, dtype=np.float32)
            if len(points) == 0:
                corrected_chunks.append(points)
                continue
            if chunk_point_timestamps is None:
                corrected_chunks.append(points)
                continue
            point_times = np.asarray(chunk_point_timestamps, dtype=np.int64)
            if len(point_times) != len(points):
                corrected_chunks.append(points)
                continue
            has_any_point_timestamps = True
            projected_speed_mps = self._track_velocity_along_axis(track, int(frame_id), axis_idx)
            if projected_speed_mps is None:
                corrected_chunks.append(points)
                continue
            has_any_velocity = True
            if abs(float(projected_speed_mps)) < 1.0:
                corrected_chunks.append(points)
                continue
            centered_time_ns = point_times.astype(np.float64) - float(np.median(point_times))
            dt_seconds = centered_time_ns * 1e-9
            corrected = points.copy()
            corrected[:, axis_idx] = corrected[:, axis_idx] - (float(projected_speed_mps) * dt_seconds).astype(np.float32)
            corrected_chunks.append(corrected.astype(np.float32, copy=False))
            corrected_chunk_count += 1
            mean_speed_values.append(abs(float(projected_speed_mps)))
            mean_abs_shift_values.append(float(np.mean(np.abs(float(projected_speed_mps) * dt_seconds))))
            mean_time_span_values.append(float((int(np.max(point_times)) - int(np.min(point_times))) * 1e-6))

        metrics["motion_deskew_confident_chunk_count"] = int(corrected_chunk_count)
        metrics["motion_deskew_corrected_chunk_count"] = int(corrected_chunk_count)
        if corrected_chunk_count == 0:
            if not has_any_point_timestamps:
                metrics["motion_deskew_skipped_reason"] = "no_point_timestamps"
            elif not has_any_velocity:
                metrics["motion_deskew_skipped_reason"] = "no_velocity_estimate"
            else:
                metrics["motion_deskew_skipped_reason"] = "insufficient_velocity"
            return corrected_chunks, list(intensity), metrics

        metrics.update(
            {
                "motion_deskew_applied": True,
                "motion_deskew_skipped_reason": "applied",
                "motion_deskew_mean_speed_mps": float(np.mean(mean_speed_values)),
                "motion_deskew_mean_abs_shift_m": float(np.mean(mean_abs_shift_values)),
                "motion_deskew_mean_time_span_ms": float(np.mean(mean_time_span_values)),
            }
        )
        return corrected_chunks, list(intensity), metrics

    def _motion_deskew_candidate(self, track: Track, long_vehicle_mode_applied: bool) -> bool:
        if long_vehicle_mode_applied or self.config.long_vehicle_mode:
            return True
        is_long_vehicle = track.quality_metrics.get("is_long_vehicle")
        if bool(is_long_vehicle):
            return True
        extents = (
            np.asarray(track.bbox_extents, dtype=np.float64)
            if track.bbox_extents
            else np.asarray([compute_extent(points) for points in track.world_points], dtype=np.float64)
        )
        if extents.size == 0:
            return False
        axis_idx = axis_to_index(self.config.frame_selection_line_axis)
        max_extent = float(np.max(extents[:, axis_idx]))
        return max_extent >= (0.85 * float(self.config.long_vehicle_length_threshold))

    def _track_velocity_along_axis(self, track: Track, frame_id: int, axis_idx: int) -> float | None:
        if len(track.centers) < 2 or len(track.frame_ids) != len(track.centers):
            return None
        if len(track.frame_timestamps_ns) != len(track.frame_ids):
            return None
        try:
            observation_index = next(index for index, value in enumerate(track.frame_ids) if int(value) == int(frame_id))
        except StopIteration:
            return None
        if observation_index <= 0:
            return self._velocity_between(track, observation_index, min(observation_index + 1, len(track.centers) - 1), axis_idx)
        if observation_index >= (len(track.centers) - 1):
            return self._velocity_between(track, max(observation_index - 1, 0), observation_index, axis_idx)
        return self._velocity_between(track, observation_index - 1, observation_index + 1, axis_idx)

    def _velocity_between(self, track: Track, left_index: int, right_index: int, axis_idx: int) -> float | None:
        if left_index == right_index:
            return None
        left_time_ns = int(track.frame_timestamps_ns[left_index])
        right_time_ns = int(track.frame_timestamps_ns[right_index])
        dt_ns = right_time_ns - left_time_ns
        if dt_ns <= 0:
            return None
        displacement = np.asarray(track.centers[right_index], dtype=np.float64) - np.asarray(track.centers[left_index], dtype=np.float64)
        return float(displacement[axis_idx] / (float(dt_ns) * 1e-9))

    def _truncate_after_lane_end_touch(
        self,
        track: Track,
        lane_box: LaneBox,
        chunks: list[np.ndarray],
        chunk_intensity: list[np.ndarray | None],
        chunk_point_timestamps_ns: list[np.ndarray | None],
    ) -> tuple[list[np.ndarray], list[np.ndarray | None], list[np.ndarray | None], list[np.ndarray], list[int], list[int], list[np.ndarray], dict[str, Any]]:
        axis_idx = axis_to_index(self.config.frame_selection_line_axis)
        lane_end_value, _ = self._lane_end_touch_bounds(lane_box, axis_idx)
        frame_timestamps_ns = self._track_frame_timestamps(track)
        metrics: dict[str, Any] = {
            "lane_end_touch_filter_enabled": bool(self.config.truncate_after_lane_end_touch),
            "lane_end_touch_found": False,
            "lane_end_touch_frame_id": -1,
            "lane_end_touch_index": -1,
            "lane_end_touch_axis": ["x", "y", "z"][axis_idx],
            "lane_end_touch_value": float(lane_end_value),
            "lane_end_touch_kept_frame_count": int(len(track.frame_ids)),
        }
        if not self.config.truncate_after_lane_end_touch:
            return (
                list(chunks),
                list(chunk_intensity),
                list(chunk_point_timestamps_ns),
                list(track.centers),
                list(track.frame_ids),
                list(frame_timestamps_ns),
                self._chunk_extents_for_track(track, chunks),
                metrics,
            )

        touch_idx = find_lane_end_touch_index(
            list(track.world_points),
            list(track.centers),
            lane_box,
            axis_idx,
            self.config.frame_selection_touch_margin,
        )
        if touch_idx is None:
            return (
                list(chunks),
                list(chunk_intensity),
                list(chunk_point_timestamps_ns),
                list(track.centers),
                list(track.frame_ids),
                list(frame_timestamps_ns),
                self._chunk_extents_for_track(track, chunks),
                metrics,
            )

        keep_count = max(0, int(touch_idx) + 1)
        metrics.update(
            {
                "lane_end_touch_found": True,
                "lane_end_touch_frame_id": int(track.frame_ids[touch_idx]) if touch_idx < len(track.frame_ids) else -1,
                "lane_end_touch_index": int(touch_idx),
                "lane_end_touch_kept_frame_count": int(keep_count),
            }
        )
        kept_chunks = list(chunks[:keep_count])
        return (
            kept_chunks,
            list(chunk_intensity[:keep_count]),
            list(chunk_point_timestamps_ns[:keep_count]),
            list(track.centers[:keep_count]),
            list(track.frame_ids[:keep_count]),
            list(frame_timestamps_ns[:keep_count]),
            self._chunk_extents_for_track(track, kept_chunks),
            metrics,
        )

    def _track_frame_timestamps(self, track: Track) -> list[int]:
        if len(track.frame_timestamps_ns) == len(track.frame_ids):
            return [int(value) for value in track.frame_timestamps_ns]
        return [-1 for _ in track.frame_ids]

    def _lane_end_touch_bounds(self, lane_box: LaneBox, axis_idx: int) -> tuple[float, float]:
        if axis_idx == 0:
            return float(lane_box.x_min), float(lane_box.x_max)
        if axis_idx == 1:
            return float(lane_box.y_min), float(lane_box.y_max)
        return float(lane_box.z_min), float(lane_box.z_max)

    def _filter_chunks_by_quality_window(
        self,
        chunks: list[np.ndarray],
        centers: list[np.ndarray],
        frame_ids: list[int],
        chunk_extents: list[np.ndarray],
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[int], dict[str, Any]]:
        total = len(chunks)
        empty_metrics = {
            "chunk_quality_filter_enabled": bool(self.config.chunk_quality_filter),
            "chunk_quality_total": total,
            "chunk_quality_kept": total,
            "chunk_quality_segment_start_frame": -1 if not frame_ids else int(frame_ids[0]),
            "chunk_quality_segment_end_frame": -1 if not frame_ids else int(frame_ids[-1]),
            "peak_chunk_point_count": 0,
            "peak_chunk_extent_norm": 0.0,
        }
        if not chunks:
            return [], [], [], empty_metrics
        if not self.config.chunk_quality_filter:
            return list(chunks), list(centers), list(frame_ids), empty_metrics

        point_counts = np.asarray([len(chunk) for chunk in chunks], dtype=np.int32)
        extent_norms = np.asarray(
            [float(np.linalg.norm(np.asarray(extent, dtype=np.float64))) for extent in chunk_extents],
            dtype=np.float64,
        )
        peak_point_count = int(np.max(point_counts)) if len(point_counts) else 0
        peak_extent_norm = float(np.max(extent_norms)) if len(extent_norms) else 0.0
        point_threshold = float(self.config.chunk_min_points_ratio_to_peak) * float(max(peak_point_count, 1))
        extent_threshold = float(self.config.chunk_min_extent_ratio_to_peak) * float(max(peak_extent_norm, 1e-6))
        good_mask = (point_counts.astype(np.float64) >= point_threshold) & (extent_norms >= extent_threshold)

        if not np.any(good_mask):
            peak_index = int(np.argmax(point_counts)) if len(point_counts) else 0
            start_idx, end_idx = self._expand_segment_around_peak(peak_index, total)
        else:
            quality_scores = (
                0.6 * (point_counts.astype(np.float64) / max(float(peak_point_count), 1.0))
                + 0.4 * (extent_norms / max(float(peak_extent_norm), 1e-6))
            )
            peak_score = float(np.max(quality_scores))
            peak_indices = {int(index) for index in np.flatnonzero(np.isclose(quality_scores, peak_score))}
            runs = self._good_chunk_runs(good_mask)
            candidate_runs = [run for run in runs if any(index in peak_indices for index in range(run[0], run[1] + 1))]
            if not candidate_runs:
                candidate_runs = runs
            if candidate_runs:
                start_idx, end_idx = max(
                    candidate_runs,
                    key=lambda run: (
                        run[1] - run[0] + 1,
                        float(np.mean(point_counts[run[0] : run[1] + 1])),
                        -run[0],
                    ),
                )
            else:
                peak_index = int(np.argmax(point_counts)) if len(point_counts) else 0
                start_idx, end_idx = peak_index, peak_index
            if (end_idx - start_idx + 1) < int(self.config.chunk_min_segment_length):
                peak_index = int(max(peak_indices, key=lambda index: (quality_scores[index], point_counts[index]))) if peak_indices else int(np.argmax(point_counts))
                start_idx, end_idx = self._expand_segment_around_peak(peak_index, total)

        kept_indices = list(range(start_idx, end_idx + 1))
        metrics = {
            "chunk_quality_filter_enabled": True,
            "chunk_quality_total": total,
            "chunk_quality_kept": len(kept_indices),
            "chunk_quality_segment_start_frame": int(frame_ids[start_idx]),
            "chunk_quality_segment_end_frame": int(frame_ids[end_idx]),
            "peak_chunk_point_count": peak_point_count,
            "peak_chunk_extent_norm": peak_extent_norm,
        }
        return (
            [chunks[index] for index in kept_indices],
            [centers[index] for index in kept_indices],
            [frame_ids[index] for index in kept_indices],
            metrics,
        )

    def _expand_segment_around_peak(self, peak_index: int, total: int) -> tuple[int, int]:
        window = min(total, max(1, int(self.config.chunk_min_segment_length)))
        if total <= window:
            return 0, max(total - 1, 0)
        half = window // 2
        start = max(0, int(peak_index) - half)
        end = start + window - 1
        if end >= total:
            end = total - 1
            start = end - window + 1
        return int(start), int(end)

    def _good_chunk_runs(self, good_mask: np.ndarray) -> list[tuple[int, int]]:
        runs: list[tuple[int, int]] = []
        start = None
        for index, is_good in enumerate(good_mask.tolist()):
            if is_good and start is None:
                start = index
            elif not is_good and start is not None:
                runs.append((start, index - 1))
                start = None
        if start is not None:
            runs.append((start, len(good_mask) - 1))
        return runs

    def _required_observations(self, chunk_count: int) -> int:
        _ = chunk_count
        return max(1, int(self.config.fusion_min_observations))

    def _chunk_weights(
        self,
        track: Track,
        chunks: list[np.ndarray],
        registration_metrics: dict[str, Any],
        long_vehicle_mode_applied: bool,
    ) -> list[float]:
        reg_weights = registration_metrics.get("registration_chunk_weights")
        if self.fusion_method == "weighted_voxel_fusion":
            if self.config.fusion_weight_mode == "uniform":
                base_weights = [1.0 for _ in chunks]
            elif self.config.fusion_weight_mode == "quality":
                base_weights = self._quality_weighted_chunk_weights(track, chunks, long_vehicle_mode_applied)
            else:
                base_weights = [float(max(1, len(chunk))) for chunk in chunks]
            if reg_weights and len(reg_weights) == len(chunks):
                return [float(base) * float(reg) for base, reg in zip(base_weights, reg_weights)]
            return base_weights
        return [1.0 for _ in chunks]

    def _confidence_chunk_weights(
        self,
        chunks: list[np.ndarray],
        chunk_weights: list[float],
        registration_metrics: dict[str, Any],
    ) -> list[float]:
        if self.fusion_method == "weighted_voxel_fusion":
            return [float(weight) for weight in chunk_weights]
        reg_weights = registration_metrics.get("registration_chunk_weights")
        if reg_weights and len(reg_weights) == len(chunks):
            return [float(weight) for weight in reg_weights]
        return [1.0 for _ in chunks]

    def _quality_weighted_chunk_weights(self, track: Track, chunks: list[np.ndarray], long_vehicle_mode_applied: bool) -> list[float]:
        if not chunks:
            return []
        axis_idx = axis_to_index(self.config.frame_selection_line_axis)
        lateral_idx, vertical_idx = orthogonal_axes(axis_idx)
        quality = max(0.1, float(track.quality_score or 0.0))
        point_counts = np.asarray([max(1, len(chunk)) for chunk in chunks], dtype=np.float64)
        extents = np.asarray([self._chunk_extent(chunk) for chunk in chunks], dtype=np.float64)
        median_points = float(np.median(point_counts)) if len(point_counts) else 1.0
        median_extent = np.median(extents, axis=0) if len(extents) else np.ones((3,), dtype=np.float64)

        weights = []
        for count, extent in zip(point_counts, extents):
            point_dev = abs(float(count) - median_points) / max(median_points, 1.0)
            cross_section_dev = (
                abs(float(extent[lateral_idx]) - float(median_extent[lateral_idx])) / max(float(median_extent[lateral_idx]), 1e-6)
                + abs(float(extent[vertical_idx]) - float(median_extent[vertical_idx])) / max(float(median_extent[vertical_idx]), 1e-6)
            ) * 0.5
            length_dev = abs(float(extent[axis_idx]) - float(median_extent[axis_idx])) / max(float(median_extent[axis_idx]), 1e-6)
            if long_vehicle_mode_applied:
                consistency_weight = max(0.1, 1.0 / (1.0 + point_dev + cross_section_dev))
            else:
                consistency_weight = max(0.1, 1.0 / (1.0 + point_dev + cross_section_dev + length_dev))
            weights.append(float(quality * consistency_weight))
        return weights

    def _chunk_extent(self, chunk: np.ndarray) -> np.ndarray:
        if len(chunk) == 0:
            return np.zeros((3,), dtype=np.float32)
        return (np.max(chunk, axis=0) - np.min(chunk, axis=0)).astype(np.float32)

    def _fuse_chunks(
        self,
        chunks: list[np.ndarray],
        intensities: list[np.ndarray | None],
        chunk_weights: list[float],
        confidence_chunk_weights: list[float],
        min_observations: int,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, int, int, int]:
        return self._voxel_accumulate(
            chunks,
            intensities,
            chunk_weights,
            confidence_chunk_weights,
            self.config.fusion_voxel_size,
            min_observations,
        )

    def _voxel_downsample(
        self,
        xyz: np.ndarray,
        intensity: np.ndarray | None,
        voxel_size: float,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if len(xyz) == 0 or voxel_size <= 0:
            return np.asarray(xyz, dtype=np.float32), None if intensity is None else np.asarray(intensity, dtype=np.float32)
        return self._mean_per_voxel(xyz, intensity, voxel_size)

    def _voxel_accumulate(
        self,
        chunks: list[np.ndarray],
        intensities: list[np.ndarray | None],
        chunk_weights: list[float],
        confidence_chunk_weights: list[float],
        voxel_size: float,
        min_observations: int,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, int, int, int]:
        voxel_sum: dict[tuple[int, int, int], np.ndarray] = {}
        voxel_weight_sum: dict[tuple[int, int, int], float] = {}
        voxel_hits: dict[tuple[int, int, int], int] = {}
        voxel_intensity_sum: dict[tuple[int, int, int], float] = {}
        voxel_confidence_sum: dict[tuple[int, int, int], float] = {}
        has_intensity = len(intensities) == len(chunks) and all(
            intensity is not None and len(np.asarray(intensity, dtype=np.float32)) == len(points)
            for points, intensity in zip(chunks, intensities)
        )
        raw_points = 0
        for chunk_index, points in enumerate(chunks):
            if len(points) == 0:
                continue
            raw_points += len(points)
            weight = float(chunk_weights[chunk_index]) if chunk_index < len(chunk_weights) else 1.0
            confidence_weight = float(confidence_chunk_weights[chunk_index]) if chunk_index < len(confidence_chunk_weights) else 1.0
            voxels = np.floor(points / float(voxel_size)).astype(np.int32)
            unique, inverse = np.unique(voxels, axis=0, return_inverse=True)
            intensity_values = np.asarray(intensities[chunk_index], dtype=np.float32) if has_intensity else None
            for index, voxel in enumerate(unique):
                key = (int(voxel[0]), int(voxel[1]), int(voxel[2]))
                mask = inverse == index
                point_mean = points[mask].mean(axis=0)
                intensity_mean = float(np.mean(intensity_values[mask])) if intensity_values is not None else 0.0
                if key not in voxel_sum:
                    voxel_sum[key] = point_mean * weight
                    voxel_weight_sum[key] = weight
                    voxel_hits[key] = 1
                    voxel_confidence_sum[key] = confidence_weight
                    if intensity_values is not None:
                        voxel_intensity_sum[key] = intensity_mean * weight
                else:
                    voxel_sum[key] += point_mean * weight
                    voxel_weight_sum[key] += weight
                    voxel_hits[key] += 1
                    voxel_confidence_sum[key] += confidence_weight
                    if intensity_values is not None:
                        voxel_intensity_sum[key] += intensity_mean * weight
        if not voxel_sum:
            return np.zeros((0, 3), dtype=np.float32), None, np.zeros((0,), dtype=np.float32), raw_points, 0, 0
        kept = []
        kept_intensity = []
        kept_confidence = []
        for key, hits in voxel_hits.items():
            if hits < max(1, int(min_observations)):
                continue
            kept.append(voxel_sum[key] / max(voxel_weight_sum[key], 1e-6))
            kept_confidence.append(float(voxel_confidence_sum[key]))
            if has_intensity:
                kept_intensity.append(voxel_intensity_sum[key] / max(voxel_weight_sum[key], 1e-6))
        if not kept:
            return np.zeros((0, 3), dtype=np.float32), None, np.zeros((0,), dtype=np.float32), raw_points, len(voxel_sum), 0
        xyz = np.asarray(kept, dtype=np.float32)
        intensity = np.asarray(kept_intensity, dtype=np.float32) if has_intensity else None
        confidence = np.asarray(kept_confidence, dtype=np.float32)
        return xyz, intensity, confidence, raw_points, len(voxel_sum), len(kept)

    def _post_filter(
        self,
        xyz: np.ndarray,
        intensity: np.ndarray | None,
        confidence: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, int, int, int]:
        if len(xyz) == 0:
            empty_confidence = None if confidence is None else np.zeros((0,), dtype=np.float32)
            return np.zeros((0, 3), dtype=np.float32), None, empty_confidence, 0, 0, 0
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(xyz, dtype=np.float64))
        raw_count = len(pcd.points)
        stat_count = raw_count
        filtered_intensity = None if intensity is None else np.asarray(intensity, dtype=np.float32).copy()
        filtered_confidence = None if confidence is None else np.asarray(confidence, dtype=np.float32).copy()
        if raw_count >= max(8, int(self.config.post_filter_stat_nb_neighbors)):
            pcd, keep_indices = pcd.remove_statistical_outlier(
                nb_neighbors=int(self.config.post_filter_stat_nb_neighbors),
                std_ratio=float(self.config.post_filter_stat_std_ratio),
            )
            stat_count = len(pcd.points)
            keep = np.asarray(keep_indices, dtype=np.int32)
            if filtered_intensity is not None:
                filtered_intensity = filtered_intensity[keep]
            if filtered_confidence is not None:
                filtered_confidence = filtered_confidence[keep]
        current_xyz = np.asarray(pcd.points, dtype=np.float32)
        if self.config.aggregate_voxel > 0 and len(current_xyz) > 0:
            current_xyz, filtered_intensity, filtered_confidence = self._aggregate_per_voxel(
                current_xyz,
                filtered_intensity,
                filtered_confidence,
                self.config.aggregate_voxel,
            )
        return current_xyz, filtered_intensity, filtered_confidence, raw_count, stat_count, len(current_xyz)

    def _effective_frame_selection_method(self, track: Track, long_vehicle_mode_applied: bool) -> str:
        method = str(self.config.frame_selection_method)
        if long_vehicle_mode_applied and method == "keyframe_motion":
            return "length_coverage"
        if long_vehicle_mode_applied and method == "auto" and not self.config.use_all_frames:
            return "length_coverage"
        return method

    def _quality_threshold(self, long_vehicle_mode_applied: bool) -> float:
        if long_vehicle_mode_applied:
            return float(self.config.min_track_quality_for_save_long_vehicle)
        return float(self.config.min_track_quality_for_save)

    def _track_is_long_vehicle(self, track: Track) -> bool:
        if self.config.long_vehicle_mode:
            return True
        is_long_vehicle = track.quality_metrics.get("is_long_vehicle")
        if is_long_vehicle is not None:
            return bool(is_long_vehicle)
        extents = (
            np.asarray(track.bbox_extents, dtype=np.float64)
            if track.bbox_extents
            else np.asarray([compute_extent(points) for points in track.world_points], dtype=np.float64)
        )
        if extents.size == 0:
            return False
        axis_idx = axis_to_index(self.config.frame_selection_line_axis)
        long_extent_p75 = float(np.percentile(extents[:, axis_idx], 75))
        return long_extent_p75 >= float(self.config.long_vehicle_length_threshold)

    def _apply_tail_bridge(
        self,
        xyz: np.ndarray,
        intensity: np.ndarray | None,
        confidence: np.ndarray | None,
        long_vehicle_mode_applied: bool,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, int, int, float]:
        longitudinal_extent = self._longitudinal_extent(xyz)
        if len(xyz) == 0:
            empty_confidence = None if confidence is None else np.zeros((0,), dtype=np.float32)
            return np.zeros((0, 3), dtype=np.float32), None, empty_confidence, 0, 0, longitudinal_extent
        components = self._components(xyz, intensity, confidence)
        if not long_vehicle_mode_applied or len(components) <= 1:
            return np.asarray(xyz, dtype=np.float32), intensity, confidence, len(components), 0, longitudinal_extent

        bridge_points, bridge_intensity, bridge_confidence, bridge_count = self._tail_bridge_components(components)
        if len(bridge_points) == 0:
            return np.asarray(xyz, dtype=np.float32), intensity, confidence, len(components), 0, longitudinal_extent

        bridged_xyz = np.vstack([np.asarray(xyz, dtype=np.float32), bridge_points.astype(np.float32)])
        bridged_intensity = None
        if intensity is not None and bridge_intensity is not None:
            bridged_intensity = np.concatenate(
                [np.asarray(intensity, dtype=np.float32), np.asarray(bridge_intensity, dtype=np.float32)],
                axis=0,
            ).astype(np.float32, copy=False)
        bridged_confidence = None
        if confidence is not None and bridge_confidence is not None:
            bridged_confidence = np.concatenate(
                [np.asarray(confidence, dtype=np.float32), np.asarray(bridge_confidence, dtype=np.float32)],
                axis=0,
            ).astype(np.float32, copy=False)
        bridged_xyz, bridged_intensity, bridged_confidence, _, _, _ = self._post_filter(
            bridged_xyz,
            bridged_intensity,
            bridged_confidence,
        )
        bridged_components = self._components(bridged_xyz, bridged_intensity, bridged_confidence)
        return bridged_xyz, bridged_intensity, bridged_confidence, len(bridged_components), bridge_count, self._longitudinal_extent(bridged_xyz)

    def _components(
        self,
        xyz: np.ndarray,
        intensity: np.ndarray | None,
        confidence: np.ndarray | None,
    ) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray | None]]:
        points = np.asarray(xyz, dtype=np.float32)
        if len(points) == 0:
            return []
        if len(points) < 3:
            return [(
                points,
                None if intensity is None else np.asarray(intensity, dtype=np.float32),
                None if confidence is None else np.asarray(confidence, dtype=np.float32),
            )]
        eps = max(0.35, float(max(self.config.aggregate_voxel, self.config.fusion_voxel_size)) * 5.0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
        labels = np.asarray(pcd.cluster_dbscan(eps=eps, min_points=1, print_progress=False), dtype=np.int32)
        components = []
        intensity_values = None if intensity is None else np.asarray(intensity, dtype=np.float32)
        confidence_values = None if confidence is None else np.asarray(confidence, dtype=np.float32)
        for label in sorted(set(labels.tolist())):
            if label < 0:
                continue
            mask = labels == label
            component_points = points[mask]
            if len(component_points) == 0:
                continue
            component_intensity = None if intensity_values is None else intensity_values[mask]
            component_confidence = None if confidence_values is None else confidence_values[mask]
            components.append((np.asarray(component_points, dtype=np.float32), component_intensity, component_confidence))
        return components

    def _tail_bridge_components(
        self,
        components: list[tuple[np.ndarray, np.ndarray | None, np.ndarray | None]],
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, int]:
        axis_idx = axis_to_index(self.config.frame_selection_line_axis)
        lateral_idx, vertical_idx = orthogonal_axes(axis_idx)
        sorted_components = sorted(components, key=lambda component: float(np.mean(component[0][:, axis_idx])))
        step = max(0.08, float(max(self.config.aggregate_voxel, self.config.fusion_voxel_size)) * 0.9)
        bridges = []
        bridge_intensity_parts = []
        bridge_confidence_parts = []
        bridge_count = 0
        for left, right in zip(sorted_components[:-1], sorted_components[1:]):
            left_points, left_intensity, left_confidence = left
            right_points, right_intensity, right_confidence = right
            left_min = np.min(left_points, axis=0)
            left_max = np.max(left_points, axis=0)
            right_min = np.min(right_points, axis=0)
            right_max = np.max(right_points, axis=0)
            longitudinal_gap = self._axis_gap(left_min[axis_idx], left_max[axis_idx], right_min[axis_idx], right_max[axis_idx])
            lateral_gap = self._axis_gap(left_min[lateral_idx], left_max[lateral_idx], right_min[lateral_idx], right_max[lateral_idx])
            vertical_gap = self._axis_gap(left_min[vertical_idx], left_max[vertical_idx], right_min[vertical_idx], right_max[vertical_idx])
            if longitudinal_gap <= 0.0 or longitudinal_gap > float(self.config.tail_bridge_longitudinal_gap_max):
                continue
            if lateral_gap > float(self.config.tail_bridge_lateral_gap_max):
                continue
            if vertical_gap > float(self.config.tail_bridge_vertical_gap_max):
                continue
            start = np.mean(left_points, axis=0)
            end = np.mean(right_points, axis=0)
            start[axis_idx] = float(left_max[axis_idx])
            end[axis_idx] = float(right_min[axis_idx])
            steps = max(2, int(math.ceil(longitudinal_gap / max(step, 1e-3))) + 1)
            alphas = np.linspace(0.0, 1.0, steps, dtype=np.float32)[1:-1]
            if len(alphas) == 0:
                continue
            bridges.append(np.asarray([(1.0 - alpha) * start + alpha * end for alpha in alphas], dtype=np.float32))
            if left_intensity is not None and right_intensity is not None:
                left_mean = float(np.mean(np.asarray(left_intensity, dtype=np.float32)))
                right_mean = float(np.mean(np.asarray(right_intensity, dtype=np.float32)))
                bridge_intensity_parts.append(
                    np.asarray([(1.0 - alpha) * left_mean + alpha * right_mean for alpha in alphas], dtype=np.float32)
                )
            else:
                bridge_intensity_parts.append(None)
            if left_confidence is not None and right_confidence is not None:
                left_conf_mean = float(np.mean(np.asarray(left_confidence, dtype=np.float32)))
                right_conf_mean = float(np.mean(np.asarray(right_confidence, dtype=np.float32)))
                bridge_confidence_parts.append(
                    np.asarray([(1.0 - alpha) * left_conf_mean + alpha * right_conf_mean for alpha in alphas], dtype=np.float32)
                )
            else:
                bridge_confidence_parts.append(None)
            bridge_count += 1
        if not bridges:
            return np.zeros((0, 3), dtype=np.float32), None, None, 0
        bridge_intensity = None
        if bridge_intensity_parts and all(values is not None for values in bridge_intensity_parts):
            bridge_intensity = np.concatenate(
                [np.asarray(values, dtype=np.float32) for values in bridge_intensity_parts if values is not None],
                axis=0,
            ).astype(np.float32, copy=False)
        bridge_confidence = None
        if bridge_confidence_parts and all(values is not None for values in bridge_confidence_parts):
            bridge_confidence = np.concatenate(
                [np.asarray(values, dtype=np.float32) for values in bridge_confidence_parts if values is not None],
                axis=0,
            ).astype(np.float32, copy=False)
        return np.vstack(bridges).astype(np.float32), bridge_intensity, bridge_confidence, bridge_count

    def _mean_per_voxel(
        self,
        xyz: np.ndarray,
        intensity: np.ndarray | None,
        voxel_size: float,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        reduced_xyz, reduced_intensity, _ = self._aggregate_per_voxel(xyz, intensity, None, voxel_size)
        return reduced_xyz, reduced_intensity

    def _aggregate_per_voxel(
        self,
        xyz: np.ndarray,
        intensity: np.ndarray | None,
        confidence: np.ndarray | None,
        voxel_size: float,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        if len(xyz) == 0:
            return (
                np.zeros((0, 3), dtype=np.float32),
                None if intensity is None else np.zeros((0,), dtype=np.float32),
                None if confidence is None else np.zeros((0,), dtype=np.float32),
            )
        voxels = np.floor(np.asarray(xyz, dtype=np.float32) / float(voxel_size)).astype(np.int32)
        unique, inverse = np.unique(voxels, axis=0, return_inverse=True)
        intensity_values = None
        if intensity is not None and len(np.asarray(intensity, dtype=np.float32)) == len(xyz):
            intensity_values = np.asarray(intensity, dtype=np.float32)
        confidence_values = None
        if confidence is not None and len(np.asarray(confidence, dtype=np.float32)) == len(xyz):
            confidence_values = np.asarray(confidence, dtype=np.float32)
        reduced_points = []
        reduced_intensity = []
        reduced_confidence = []
        for index in range(len(unique)):
            mask = inverse == index
            reduced_points.append(np.asarray(xyz, dtype=np.float32)[mask].mean(axis=0))
            if intensity_values is not None:
                reduced_intensity.append(float(np.mean(intensity_values[mask])))
            if confidence_values is not None:
                reduced_confidence.append(float(np.sum(confidence_values[mask])))
        return (
            np.asarray(reduced_points, dtype=np.float32),
            None if intensity_values is None else np.asarray(reduced_intensity, dtype=np.float32),
            None if confidence_values is None else np.asarray(reduced_confidence, dtype=np.float32),
        )

    def _apply_confidence_point_cap(
        self,
        xyz: np.ndarray,
        intensity: np.ndarray | None,
        confidence: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, bool]:
        points = np.asarray(xyz, dtype=np.float32)
        intensity_values = None if intensity is None else np.asarray(intensity, dtype=np.float32)
        confidence_values = None if confidence is None else np.asarray(confidence, dtype=np.float32)
        if not self.config.enable_confidence_point_cap:
            return points, intensity_values, confidence_values, False
        target = int(self.config.confidence_point_cap_max_points)
        if len(points) <= target:
            return points, intensity_values, confidence_values, False
        if confidence_values is None or len(confidence_values) != len(points):
            confidence_values = np.ones((len(points),), dtype=np.float32)
        selected_indices = self._confidence_point_cap_indices(points, confidence_values, target, int(self.config.confidence_point_cap_bins))
        capped_points = points[selected_indices]
        capped_intensity = None if intensity_values is None else intensity_values[selected_indices]
        capped_confidence = confidence_values[selected_indices]
        return capped_points, capped_intensity, capped_confidence, True

    def _confidence_point_cap_indices(
        self,
        points: np.ndarray,
        confidence: np.ndarray,
        target: int,
        bin_count: int,
    ) -> np.ndarray:
        point_count = int(len(points))
        if point_count <= target:
            return np.arange(point_count, dtype=np.int32)
        axis_idx = axis_to_index(self.config.frame_selection_line_axis)
        axis_values = np.asarray(points, dtype=np.float32)[:, axis_idx]
        if bin_count <= 1 or not np.all(np.isfinite(axis_values)):
            return self._top_confidence_indices(confidence, np.arange(point_count, dtype=np.int32), target)
        axis_min = float(np.min(axis_values))
        axis_max = float(np.max(axis_values))
        axis_span = axis_max - axis_min
        if axis_span <= 1e-6:
            return self._top_confidence_indices(confidence, np.arange(point_count, dtype=np.int32), target)
        normalized = (axis_values - axis_min) / axis_span
        bin_indices = np.clip(np.floor(normalized * float(bin_count)).astype(np.int32), 0, int(bin_count) - 1)
        occupied_bins = sorted(int(bin_id) for bin_id in np.unique(bin_indices))
        if not occupied_bins:
            return self._top_confidence_indices(confidence, np.arange(point_count, dtype=np.int32), target)
        base_budget = int(target // len(occupied_bins))
        selected_mask = np.zeros((point_count,), dtype=bool)
        selected_parts: list[np.ndarray] = []
        for bin_id in occupied_bins:
            bin_member_indices = np.flatnonzero(bin_indices == int(bin_id)).astype(np.int32)
            if len(bin_member_indices) == 0 or base_budget <= 0:
                continue
            ranked = self._top_confidence_indices(confidence, bin_member_indices, min(base_budget, len(bin_member_indices)))
            selected_parts.append(ranked)
            selected_mask[ranked] = True
        selected_count = int(sum(len(part) for part in selected_parts))
        remaining_budget = max(0, int(target) - selected_count)
        if remaining_budget > 0:
            remaining_indices = np.flatnonzero(~selected_mask).astype(np.int32)
            if len(remaining_indices) > 0:
                selected_parts.append(self._top_confidence_indices(confidence, remaining_indices, remaining_budget))
        if not selected_parts:
            return self._top_confidence_indices(confidence, np.arange(point_count, dtype=np.int32), target)
        selected = np.concatenate(selected_parts, axis=0).astype(np.int32, copy=False)
        if len(selected) > target:
            selected = self._top_confidence_indices(confidence, selected, target)
        return np.sort(selected, kind="stable")

    def _top_confidence_indices(
        self,
        confidence: np.ndarray,
        candidate_indices: np.ndarray,
        keep_count: int,
    ) -> np.ndarray:
        if keep_count <= 0 or len(candidate_indices) == 0:
            return np.zeros((0,), dtype=np.int32)
        candidates = np.asarray(candidate_indices, dtype=np.int32)
        order = np.argsort(-np.asarray(confidence, dtype=np.float32)[candidates], kind="stable")
        return candidates[order[: int(keep_count)]]

    def _axis_gap(self, left_min: float, left_max: float, right_min: float, right_max: float) -> float:
        return max(0.0, max(float(right_min) - float(left_max), float(left_min) - float(right_max)))

    def _longitudinal_extent(self, xyz: np.ndarray) -> float:
        if len(xyz) == 0:
            return 0.0
        axis_idx = axis_to_index(self.config.frame_selection_line_axis)
        values = np.asarray(xyz, dtype=np.float64)[:, axis_idx]
        return float(np.max(values) - np.min(values))
