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
    orthogonal_axes,
    select_best_frames_for_aggregation,
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
            return AggregateResult(track.track_id, np.zeros((0, 3), dtype=np.float32), [], "skipped_min_hits", self._base_metrics(track))
        if self.output_config.require_track_exit and not track_exited_lane_box(track, lane_box, edge_margin=self.output_config.track_exit_edge_margin):
            return AggregateResult(track.track_id, np.zeros((0, 3), dtype=np.float32), [], "skipped_track_exit", self._base_metrics(track))

        long_vehicle_mode_applied = self._track_is_long_vehicle(track)
        frame_selection_method = self._effective_frame_selection_method(track, long_vehicle_mode_applied)
        chunks = track.world_points if self.output_config.save_world else track.local_points
        chunk_intensity = ensure_aligned_optional(
            track.world_intensity if self.output_config.save_world else track.local_intensity,
            len(chunks),
        )
        chunk_extents = self._chunk_extents_for_track(track, chunks)
        chunks, track_centers, track_frame_ids, chunk_quality_info = self._filter_chunks_by_quality_window(
            list(chunks),
            list(track.centers),
            list(track.frame_ids),
            chunk_extents,
        )
        chunk_intensity = self._select_optional_by_frame_ids(track.frame_ids, chunk_intensity, track_frame_ids)
        selected_chunks, selected_centers, selected_frame_ids, selection_info = select_best_frames_for_aggregation(
            chunks=list(chunks),
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
        selection_info = {**chunk_quality_info, **selection_info}
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
            selection_info = {**selection_info, **shape_info}
        if not selected_chunks:
            metrics = {**self._base_metrics(track), **selection_info}
            return AggregateResult(track.track_id, np.zeros((0, 3), dtype=np.float32), [], "empty_selection", metrics)

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
            metrics = {**self._base_metrics(track), **selection_info}
            return AggregateResult(track.track_id, np.zeros((0, 3), dtype=np.float32), selected_frame_ids, "empty_prepared_chunks", metrics)

        accumulation_input, registration_metrics = self._prepare_for_fusion(prepared_chunks)
        (
            accumulation_input,
            prepared_intensity,
            selected_centers,
            selected_frame_ids,
            registration_metrics,
        ) = self._apply_registration_subset(
            accumulation_input,
            prepared_intensity,
            selected_centers,
            selected_frame_ids,
            registration_metrics,
        )
        chunk_weights = self._chunk_weights(track, accumulation_input, registration_metrics, long_vehicle_mode_applied)
        fused_xyz, fused_intensity, raw_points_total, fusion_voxels_total, fusion_voxels_kept = self._fuse_chunks(
            accumulation_input,
            prepared_intensity,
            chunk_weights,
            min_observations=self._required_observations(len(accumulation_input)),
        )
        if len(fused_xyz) == 0:
            metrics = {**self._base_metrics(track), **selection_info, **registration_metrics}
            return AggregateResult(track.track_id, np.zeros((0, 3), dtype=np.float32), selected_frame_ids, "empty_fused", metrics)

        filtered_xyz, filtered_intensity, prefilter_points, stat_filtered_points, final_points = self._post_filter(fused_xyz, fused_intensity)
        filtered_xyz, filtered_intensity, component_count_post_fusion, tail_bridge_count, longitudinal_extent = self._apply_tail_bridge(
            filtered_xyz,
            filtered_intensity,
            long_vehicle_mode_applied,
        )
        final_points = len(filtered_xyz)
        metrics: dict[str, Any] = {
            **self._base_metrics(track),
            **selection_info,
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
        }
        if final_points == 0:
            return AggregateResult(track.track_id, np.zeros((0, 3), dtype=np.float32), selected_frame_ids, "empty_filtered", metrics)
        if final_points < int(self.config.min_saved_aggregate_points):
            return AggregateResult(
                track.track_id,
                np.zeros((0, 3), dtype=np.float32),
                selected_frame_ids,
                "skipped_min_saved_points",
                metrics,
            )
        quality_threshold = self._quality_threshold(long_vehicle_mode_applied)
        if float(track.quality_score or 0.0) < quality_threshold:
            metrics["quality_threshold"] = quality_threshold
            return AggregateResult(track.track_id, np.zeros((0, 3), dtype=np.float32), selected_frame_ids, "skipped_quality_threshold", metrics)
        return AggregateResult(track.track_id, filtered_xyz, selected_frame_ids, "saved", metrics, intensity=filtered_intensity)

    def _base_metrics(self, track: Track) -> dict[str, Any]:
        return {
            "quality_score": 0.0 if track.quality_score is None else float(track.quality_score),
            "quality_metrics": track.quality_metrics,
            "track_source_ids": track.source_track_ids or [track.track_id],
        }

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
        synced_metrics = dict(registration_metrics)
        synced_metrics["registration_input_chunk_count"] = int(input_count)
        synced_metrics["registration_output_chunk_count"] = int(len(keep_indices))
        synced_metrics["registration_dropped_count"] = max(0, int(input_count - len(keep_indices)))
        synced_metrics["registration_keep_indices"] = keep_indices
        chunk_weights = synced_metrics.get("registration_chunk_weights")
        if not isinstance(chunk_weights, list) or len(chunk_weights) != len(keep_indices):
            synced_metrics["registration_chunk_weights"] = [1.0 for _ in keep_indices]
        else:
            synced_metrics["registration_chunk_weights"] = [float(weight) for weight in chunk_weights]
        return (
            list(accumulation_input),
            [prepared_intensity[index] for index in keep_indices],
            [selected_centers[index] for index in keep_indices],
            [int(selected_frame_ids[index]) for index in keep_indices],
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
        min_observations: int,
    ) -> tuple[np.ndarray, np.ndarray | None, int, int, int]:
        return self._voxel_accumulate(chunks, intensities, chunk_weights, self.config.fusion_voxel_size, min_observations)

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
        voxel_size: float,
        min_observations: int,
    ) -> tuple[np.ndarray, np.ndarray | None, int, int, int]:
        voxel_sum: dict[tuple[int, int, int], np.ndarray] = {}
        voxel_weight_sum: dict[tuple[int, int, int], float] = {}
        voxel_hits: dict[tuple[int, int, int], int] = {}
        voxel_intensity_sum: dict[tuple[int, int, int], float] = {}
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
                    if intensity_values is not None:
                        voxel_intensity_sum[key] = intensity_mean * weight
                else:
                    voxel_sum[key] += point_mean * weight
                    voxel_weight_sum[key] += weight
                    voxel_hits[key] += 1
                    if intensity_values is not None:
                        voxel_intensity_sum[key] += intensity_mean * weight
        if not voxel_sum:
            return np.zeros((0, 3), dtype=np.float32), None, raw_points, 0, 0
        kept = []
        kept_intensity = []
        for key, hits in voxel_hits.items():
            if hits < max(1, int(min_observations)):
                continue
            kept.append(voxel_sum[key] / max(voxel_weight_sum[key], 1e-6))
            if has_intensity:
                kept_intensity.append(voxel_intensity_sum[key] / max(voxel_weight_sum[key], 1e-6))
        if not kept:
            return np.zeros((0, 3), dtype=np.float32), None, raw_points, len(voxel_sum), 0
        xyz = np.asarray(kept, dtype=np.float32)
        intensity = np.asarray(kept_intensity, dtype=np.float32) if has_intensity else None
        return xyz, intensity, raw_points, len(voxel_sum), len(kept)

    def _post_filter(
        self,
        xyz: np.ndarray,
        intensity: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray | None, int, int, int]:
        if len(xyz) == 0:
            return np.zeros((0, 3), dtype=np.float32), None, 0, 0, 0
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(xyz, dtype=np.float64))
        raw_count = len(pcd.points)
        stat_count = raw_count
        filtered_intensity = None if intensity is None else np.asarray(intensity, dtype=np.float32).copy()
        if raw_count >= max(8, int(self.config.post_filter_stat_nb_neighbors)):
            pcd, keep_indices = pcd.remove_statistical_outlier(
                nb_neighbors=int(self.config.post_filter_stat_nb_neighbors),
                std_ratio=float(self.config.post_filter_stat_std_ratio),
            )
            stat_count = len(pcd.points)
            if filtered_intensity is not None:
                filtered_intensity = filtered_intensity[np.asarray(keep_indices, dtype=np.int32)]
        current_xyz = np.asarray(pcd.points, dtype=np.float32)
        if self.config.aggregate_voxel > 0 and len(current_xyz) > 0:
            current_xyz, filtered_intensity = self._mean_per_voxel(current_xyz, filtered_intensity, self.config.aggregate_voxel)
        return current_xyz, filtered_intensity, raw_count, stat_count, len(current_xyz)

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
        long_vehicle_mode_applied: bool,
    ) -> tuple[np.ndarray, np.ndarray | None, int, int, float]:
        longitudinal_extent = self._longitudinal_extent(xyz)
        if len(xyz) == 0:
            return np.zeros((0, 3), dtype=np.float32), None, 0, 0, longitudinal_extent
        components = self._components(xyz, intensity)
        if not long_vehicle_mode_applied or len(components) <= 1:
            return np.asarray(xyz, dtype=np.float32), intensity, len(components), 0, longitudinal_extent

        bridge_points, bridge_intensity, bridge_count = self._tail_bridge_components(components)
        if len(bridge_points) == 0:
            return np.asarray(xyz, dtype=np.float32), intensity, len(components), 0, longitudinal_extent

        bridged_xyz = np.vstack([np.asarray(xyz, dtype=np.float32), bridge_points.astype(np.float32)])
        bridged_intensity = None
        if intensity is not None and bridge_intensity is not None:
            bridged_intensity = np.concatenate(
                [np.asarray(intensity, dtype=np.float32), np.asarray(bridge_intensity, dtype=np.float32)],
                axis=0,
            ).astype(np.float32, copy=False)
        bridged_xyz, bridged_intensity, _, _, _ = self._post_filter(bridged_xyz, bridged_intensity)
        bridged_components = self._components(bridged_xyz, bridged_intensity)
        return bridged_xyz, bridged_intensity, len(bridged_components), bridge_count, self._longitudinal_extent(bridged_xyz)

    def _components(self, xyz: np.ndarray, intensity: np.ndarray | None) -> list[tuple[np.ndarray, np.ndarray | None]]:
        points = np.asarray(xyz, dtype=np.float32)
        if len(points) == 0:
            return []
        if len(points) < 3:
            return [(points, None if intensity is None else np.asarray(intensity, dtype=np.float32))]
        eps = max(0.35, float(max(self.config.aggregate_voxel, self.config.fusion_voxel_size)) * 5.0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
        labels = np.asarray(pcd.cluster_dbscan(eps=eps, min_points=1, print_progress=False), dtype=np.int32)
        components = []
        intensity_values = None if intensity is None else np.asarray(intensity, dtype=np.float32)
        for label in sorted(set(labels.tolist())):
            if label < 0:
                continue
            mask = labels == label
            component_points = points[mask]
            if len(component_points) == 0:
                continue
            component_intensity = None if intensity_values is None else intensity_values[mask]
            components.append((np.asarray(component_points, dtype=np.float32), component_intensity))
        return components

    def _tail_bridge_components(
        self,
        components: list[tuple[np.ndarray, np.ndarray | None]],
    ) -> tuple[np.ndarray, np.ndarray | None, int]:
        axis_idx = axis_to_index(self.config.frame_selection_line_axis)
        lateral_idx, vertical_idx = orthogonal_axes(axis_idx)
        sorted_components = sorted(components, key=lambda component: float(np.mean(component[0][:, axis_idx])))
        step = max(0.08, float(max(self.config.aggregate_voxel, self.config.fusion_voxel_size)) * 0.9)
        bridges = []
        bridge_intensity_parts = []
        bridge_count = 0
        for left, right in zip(sorted_components[:-1], sorted_components[1:]):
            left_points, left_intensity = left
            right_points, right_intensity = right
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
            bridge_count += 1
        if not bridges:
            return np.zeros((0, 3), dtype=np.float32), None, 0
        bridge_intensity = None
        if bridge_intensity_parts and all(values is not None for values in bridge_intensity_parts):
            bridge_intensity = np.concatenate(
                [np.asarray(values, dtype=np.float32) for values in bridge_intensity_parts if values is not None],
                axis=0,
            ).astype(np.float32, copy=False)
        return np.vstack(bridges).astype(np.float32), bridge_intensity, bridge_count

    def _mean_per_voxel(
        self,
        xyz: np.ndarray,
        intensity: np.ndarray | None,
        voxel_size: float,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if len(xyz) == 0:
            return np.zeros((0, 3), dtype=np.float32), None if intensity is None else np.zeros((0,), dtype=np.float32)
        voxels = np.floor(np.asarray(xyz, dtype=np.float32) / float(voxel_size)).astype(np.int32)
        unique, inverse = np.unique(voxels, axis=0, return_inverse=True)
        intensity_values = None
        if intensity is not None and len(np.asarray(intensity, dtype=np.float32)) == len(xyz):
            intensity_values = np.asarray(intensity, dtype=np.float32)
        reduced_points = []
        reduced_intensity = []
        for index in range(len(unique)):
            mask = inverse == index
            reduced_points.append(np.asarray(xyz, dtype=np.float32)[mask].mean(axis=0))
            if intensity_values is not None:
                reduced_intensity.append(float(np.mean(intensity_values[mask])))
        return (
            np.asarray(reduced_points, dtype=np.float32),
            None if intensity_values is None else np.asarray(reduced_intensity, dtype=np.float32),
        )

    def _axis_gap(self, left_min: float, left_max: float, right_min: float, right_max: float) -> float:
        return max(0.0, max(float(right_min) - float(left_max), float(left_min) - float(right_max)))

    def _longitudinal_extent(self, xyz: np.ndarray) -> float:
        if len(xyz) == 0:
            return 0.0
        axis_idx = axis_to_index(self.config.frame_selection_line_axis)
        values = np.asarray(xyz, dtype=np.float64)[:, axis_idx]
        return float(np.max(values) - np.min(values))
