from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d

from tracking_pipeline.infrastructure.io.pcd_writer import PCDWriter


def test_pcd_writer_roundtrips_intensity_field(tmp_path: Path) -> None:
    path = tmp_path / "with_intensity.pcd"

    PCDWriter().write(
        path,
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        intensity=np.array([0.25, 0.75], dtype=np.float32),
    )

    loaded = o3d.t.io.read_point_cloud(str(path))

    assert "intensity" in loaded.point
    assert np.allclose(loaded.point["intensity"].numpy().reshape(-1), np.array([0.25, 0.75], dtype=np.float32))


def test_pcd_writer_roundtrips_reflectivity_field_when_configured(tmp_path: Path) -> None:
    path = tmp_path / "with_reflectivity.pcd"

    PCDWriter().write(
        path,
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        intensity=np.array([1.25, 4.75], dtype=np.float32),
        scalar_field_name="reflectivity",
    )

    loaded = o3d.t.io.read_point_cloud(str(path))

    assert "reflectivity" in loaded.point
    assert "intensity" not in loaded.point
    assert np.allclose(loaded.point["reflectivity"].numpy().reshape(-1), np.array([1.25, 4.75], dtype=np.float32))


def test_pcd_writer_keeps_xyz_only_when_intensity_missing(tmp_path: Path) -> None:
    path = tmp_path / "xyz_only.pcd"

    PCDWriter().write(path, np.array([[0.0, 0.0, 0.0]], dtype=np.float32))

    loaded = o3d.t.io.read_point_cloud(str(path))

    assert "intensity" not in loaded.point
