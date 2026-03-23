from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d

from tracking_pipeline.shared.geometry import np_to_o3d


class PCDWriter:
    def write(self, path: Path, points, intensity=None, scalar_field_name: str = "intensity") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        xyz = np.asarray(points, dtype=np.float32)
        intensity_values = None if intensity is None else np.asarray(intensity, dtype=np.float32).reshape(-1)
        if intensity_values is not None and len(intensity_values) == len(xyz):
            field_name = str(scalar_field_name or "intensity")
            pcd = o3d.t.geometry.PointCloud()
            pcd.point["positions"] = o3d.core.Tensor(xyz, dtype=o3d.core.float32)
            pcd.point[field_name] = o3d.core.Tensor(intensity_values.reshape(-1, 1), dtype=o3d.core.float32)
            o3d.t.io.write_point_cloud(str(path), pcd)
            return
        o3d.io.write_point_cloud(str(path), np_to_o3d(xyz))
