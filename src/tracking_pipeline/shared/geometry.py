from __future__ import annotations

import numpy as np
import open3d as o3d


def np_to_o3d(xyz: np.ndarray, colors: np.ndarray | None = None) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(xyz, dtype=np.float64))
    if colors is not None and len(colors) == len(xyz):
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return pcd


def clamp_visual_points(xyz: np.ndarray, max_points: int) -> np.ndarray:
    indices = clamp_visual_indices(len(xyz), max_points)
    return np.asarray(xyz, dtype=np.float32)[indices]


def clamp_visual_indices(point_count: int, max_points: int) -> np.ndarray:
    if point_count <= max_points:
        return np.arange(point_count, dtype=np.int32)
    rng = np.random.default_rng(42)
    return rng.choice(point_count, size=max_points, replace=False).astype(np.int32)


def clamp_visual_scalar(values: np.ndarray | None, indices: np.ndarray) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float32)
    if len(arr) == 0:
        return arr
    return arr[np.asarray(indices, dtype=np.int32)]


def grayscale_from_intensity(intensity: np.ndarray | None) -> np.ndarray | None:
    if intensity is None:
        return None
    values = np.asarray(intensity, dtype=np.float32).reshape(-1)
    if len(values) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    values = values.copy()
    finite_mask = np.isfinite(values)
    values[~finite_mask] = 0.0
    values = np.maximum(values, 0.0)
    display_values = np.log1p(values).astype(np.float32, copy=False)
    lo = float(np.percentile(display_values, 5.0))
    hi = float(np.percentile(display_values, 95.0))
    if hi > lo + 1e-6:
        normalized = (display_values - lo) / (hi - lo)
    else:
        normalized = display_values
    normalized = np.clip(normalized, 0.0, 1.0).astype(np.float32, copy=False)
    return np.repeat(normalized[:, None], 3, axis=1).astype(np.float32)


def optional_concatenate(parts: list[np.ndarray | None], width: int | None = None) -> np.ndarray | None:
    if not parts:
        return None
    if any(part is None for part in parts):
        return None
    arrays = [np.asarray(part, dtype=np.float32) for part in parts if part is not None]
    if not arrays:
        if width is None:
            return np.zeros((0,), dtype=np.float32)
        return np.zeros((0, int(width)), dtype=np.float32)
    return np.concatenate(arrays, axis=0).astype(np.float32, copy=False)


def apply_mask_optional(values: np.ndarray | None, mask: np.ndarray) -> np.ndarray | None:
    if values is None:
        return None
    return np.asarray(values, dtype=np.float32)[np.asarray(mask, dtype=bool)]


def ensure_aligned_optional(values: list[np.ndarray | None], expected: int) -> list[np.ndarray | None]:
    if len(values) == expected:
        return values
    return [None for _ in range(expected)]


def transform_points(xyz: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if len(xyz) == 0:
        return np.asarray(xyz, dtype=np.float32)
    xyz64 = np.asarray(xyz, dtype=np.float64)
    hom = np.hstack([xyz64, np.ones((len(xyz64), 1), dtype=np.float64)])
    out = (np.asarray(transform, dtype=np.float64) @ hom.T).T[:, :3]
    out = out[np.isfinite(out).all(axis=1)]
    return out.astype(np.float32, copy=False)


def estimate_normals(points: np.ndarray, radius: float = 0.5, max_nn: int = 30) -> o3d.geometry.PointCloud:
    pcd = np_to_o3d(points)
    if len(pcd.points) == 0:
        return pcd
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=float(radius), max_nn=int(max_nn)))
    return pcd


def voxel_downsample_numpy(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if len(points) == 0 or voxel_size <= 0:
        return np.asarray(points, dtype=np.float32)
    pcd = np_to_o3d(points)
    return np.asarray(pcd.voxel_down_sample(float(voxel_size)).points, dtype=np.float32)
