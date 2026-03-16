from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import open3d as o3d

from tracking_pipeline.config.models import AggregationConfig
from tracking_pipeline.domain.rules import is_valid_transform
from tracking_pipeline.shared.geometry import estimate_normals, np_to_o3d, transform_points, voxel_downsample_numpy


class _BaseRegistrationBackend:
    name = "base"

    def __init__(self, config: AggregationConfig):
        self.config = config

    def align_chunks(self, chunks: list[np.ndarray]) -> tuple[list[np.ndarray], dict[str, Any]]:
        if len(chunks) <= 1:
            return list(chunks), self._empty_info(len(chunks))
        aligned = [chunks[0].copy()]
        model = chunks[0].copy()
        chunk_weights = [1.0]
        keep_indices = [0]
        pairs = 0
        accepted = 0
        fitness_values: list[float] = []
        rmse_values: list[float] = []
        backend_counts: Counter[str] = Counter()

        for source_index, source in enumerate(chunks[1:], start=1):
            if len(source) < 8 or len(model) < 8:
                backend_counts["skipped_insufficient_points"] += 1
                continue
            pairs += 1
            try:
                transform, fitness, rmse = self._register_pair(source, model)
                backend_counts[self.name] += 1
                fitness_values.append(float(fitness))
                rmse_values.append(float(rmse))
                if float(fitness) >= float(self.config.registration_min_fitness) and is_valid_transform(
                    transform,
                    max_translation=self.config.registration_max_translation,
                ):
                    aligned_source = transform_points(source, transform)
                    accepted += 1
                    keep_indices.append(source_index)
                    chunk_weights.append(max(0.1, float(fitness)))
                    aligned.append(aligned_source)
                    model = voxel_downsample_numpy(
                        np.vstack([model, aligned_source]),
                        max(0.03, self.config.frame_downsample_voxel),
                    )
                else:
                    backend_counts["rejected"] += 1
            except Exception:
                backend_counts["error"] += 1

        return aligned, {
            "alignment_method": self.name,
            "registration_backend": self.name,
            "registration_pairs": pairs,
            "registration_accepted": accepted,
            "registration_rejected": pairs - accepted,
            "registration_mean_fitness": float(np.mean(fitness_values)) if fitness_values else 0.0,
            "registration_mean_rmse": float(np.mean(rmse_values)) if rmse_values else 0.0,
            "registration_backend_counts": dict(backend_counts),
            "registration_pair_fitness": fitness_values,
            "registration_chunk_weights": chunk_weights,
            "registration_input_chunk_count": int(len(chunks)),
            "registration_output_chunk_count": int(len(aligned)),
            "registration_dropped_count": int(len(chunks) - len(aligned)),
            "registration_keep_indices": keep_indices,
            "registration_skipped": False,
        }

    def _empty_info(self, chunk_count: int = 0) -> dict[str, Any]:
        keep_indices = list(range(int(chunk_count)))
        return {
            "alignment_method": self.name,
            "registration_backend": self.name,
            "registration_pairs": 0,
            "registration_accepted": 0,
            "registration_rejected": 0,
            "registration_mean_fitness": 0.0,
            "registration_mean_rmse": 0.0,
            "registration_backend_counts": {},
            "registration_pair_fitness": [],
            "registration_chunk_weights": [1.0 for _ in keep_indices],
            "registration_input_chunk_count": int(chunk_count),
            "registration_output_chunk_count": int(chunk_count),
            "registration_dropped_count": 0,
            "registration_keep_indices": keep_indices,
            "registration_skipped": False,
        }

    def _register_pair(self, source_xyz: np.ndarray, target_xyz: np.ndarray) -> tuple[np.ndarray, float, float]:
        raise NotImplementedError


class SmallGICPRegistrationBackend(_BaseRegistrationBackend):
    name = "small_gicp"

    def __init__(self, config: AggregationConfig):
        super().__init__(config)
        self._small_gicp = self._try_import()

    def align_chunks(self, chunks: list[np.ndarray]) -> tuple[list[np.ndarray], dict[str, Any]]:
        if self._small_gicp is None:
            return list(chunks), {
                **self._empty_info(len(chunks)),
                "registration_backend": "unavailable",
                "alignment_method": "unavailable",
                "registration_skipped": True,
            }
        return super().align_chunks(chunks)

    def _try_import(self):
        try:
            import small_gicp  # type: ignore
        except Exception:
            return None
        return small_gicp

    def _register_pair(self, source_xyz: np.ndarray, target_xyz: np.ndarray) -> tuple[np.ndarray, float, float]:
        small_gicp = self._small_gicp
        assert small_gicp is not None
        source = np.asarray(source_xyz, dtype=np.float64)
        target = np.asarray(target_xyz, dtype=np.float64)
        if hasattr(small_gicp, "align"):
            kwargs_variants = [
                {
                    "registration_type": "GICP",
                    "max_correspondence_distance": float(self.config.registration_max_corr_dist),
                    "max_iterations": int(self.config.registration_max_iter),
                },
                {"max_correspondence_distance": float(self.config.registration_max_corr_dist)},
                {},
            ]
            for kwargs in kwargs_variants:
                try:
                    result = small_gicp.align(target, source, **kwargs)
                    transform = self._extract_transform(result)
                    if transform is not None:
                        return transform, *self._extract_metrics(result, len(source))
                except Exception:
                    continue
        if hasattr(small_gicp, "align_points"):
            for kwargs in ({"max_correspondence_distance": float(self.config.registration_max_corr_dist)}, {}):
                try:
                    result = small_gicp.align_points(source, target, **kwargs)
                    transform = self._extract_transform(result)
                    if transform is not None:
                        return transform, 1.0, 0.0
                except Exception:
                    continue
        raise RuntimeError("No compatible small_gicp registration signature found")

    def _extract_transform(self, result: Any) -> np.ndarray | None:
        if result is None:
            return None
        if isinstance(result, dict):
            for key in ("transformation", "T", "matrix"):
                if key in result:
                    arr = np.asarray(result[key])
                    if arr.shape == (4, 4):
                        return arr
        if isinstance(result, (list, tuple)):
            for item in result:
                arr = np.asarray(item)
                if arr.shape == (4, 4):
                    return arr
        if hasattr(result, "T_target_source"):
            arr = np.asarray(result.T_target_source)
            if arr.shape == (4, 4):
                return arr
        arr = np.asarray(result)
        return arr if arr.shape == (4, 4) else None

    def _extract_metrics(self, result: Any, source_size: int) -> tuple[float, float]:
        fitness = 1.0
        rmse = 0.0
        if hasattr(result, "num_inliers") and source_size > 0:
            try:
                fitness = max(0.0, min(1.0, float(result.num_inliers) / float(source_size)))
            except Exception:
                pass
        if hasattr(result, "e"):
            try:
                rmse = float(result.e)
            except Exception:
                pass
        elif hasattr(result, "error"):
            try:
                rmse = float(result.error)
            except Exception:
                pass
        return fitness, rmse


class ICPPointToPlaneRegistrationBackend(_BaseRegistrationBackend):
    name = "icp_point_to_plane"

    def _register_pair(self, source_xyz: np.ndarray, target_xyz: np.ndarray) -> tuple[np.ndarray, float, float]:
        source = estimate_normals(source_xyz, radius=max(0.2, self.config.frame_downsample_voxel * 4.0))
        target = estimate_normals(target_xyz, radius=max(0.2, self.config.frame_downsample_voxel * 4.0))
        result = o3d.pipelines.registration.registration_icp(
            source,
            target,
            max_correspondence_distance=float(self.config.registration_max_corr_dist),
            init=np.eye(4),
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(self.config.registration_max_iter)),
        )
        return np.asarray(result.transformation), float(result.fitness), float(result.inlier_rmse)


class GeneralizedICPRegistrationBackend(_BaseRegistrationBackend):
    name = "generalized_icp"

    def _register_pair(self, source_xyz: np.ndarray, target_xyz: np.ndarray) -> tuple[np.ndarray, float, float]:
        source = estimate_normals(source_xyz, radius=max(0.2, self.config.frame_downsample_voxel * 4.0))
        target = estimate_normals(target_xyz, radius=max(0.2, self.config.frame_downsample_voxel * 4.0))
        result = o3d.pipelines.registration.registration_generalized_icp(
            source,
            target,
            max_correspondence_distance=float(self.config.registration_max_corr_dist),
            init=np.eye(4),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(self.config.registration_max_iter)),
        )
        return np.asarray(result.transformation), float(result.fitness), float(result.inlier_rmse)


class FeatureGlobalThenLocalRegistrationBackend(_BaseRegistrationBackend):
    name = "feature_global_then_local"

    def _register_pair(self, source_xyz: np.ndarray, target_xyz: np.ndarray) -> tuple[np.ndarray, float, float]:
        voxel = max(float(self.config.global_registration_voxel), 0.05)
        source_down = voxel_downsample_numpy(source_xyz, voxel)
        target_down = voxel_downsample_numpy(target_xyz, voxel)
        source = estimate_normals(source_down, radius=voxel * 2.0, max_nn=50)
        target = estimate_normals(target_down, radius=voxel * 2.0, max_nn=50)
        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            source,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5.0, max_nn=100),
        )
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            target,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5.0, max_nn=100),
        )
        global_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source,
            target,
            source_fpfh,
            target_fpfh,
            mutual_filter=True,
            max_correspondence_distance=voxel * 2.5,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=4,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel * 2.5),
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 0.999),
        )
        source_full = estimate_normals(source_xyz, radius=max(0.2, self.config.frame_downsample_voxel * 4.0))
        target_full = estimate_normals(target_xyz, radius=max(0.2, self.config.frame_downsample_voxel * 4.0))
        refined = o3d.pipelines.registration.registration_icp(
            source_full,
            target_full,
            max_correspondence_distance=float(self.config.registration_max_corr_dist),
            init=np.asarray(global_result.transformation),
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(self.config.registration_max_iter)),
        )
        return np.asarray(refined.transformation), float(refined.fitness), float(refined.inlier_rmse)


def build_registration_backend(config: AggregationConfig) -> _BaseRegistrationBackend:
    if config.registration_backend == "small_gicp":
        return SmallGICPRegistrationBackend(config)
    if config.registration_backend == "icp_point_to_plane":
        return ICPPointToPlaneRegistrationBackend(config)
    if config.registration_backend == "generalized_icp":
        return GeneralizedICPRegistrationBackend(config)
    if config.registration_backend == "feature_global_then_local":
        return FeatureGlobalThenLocalRegistrationBackend(config)
    raise ValueError(f"Unsupported registration backend: {config.registration_backend}")
