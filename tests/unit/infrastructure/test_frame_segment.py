from __future__ import annotations

from pathlib import Path

import numpy as np

from tracking_pipeline.domain.models import FrameData, LidarScanData, ObjectLabelData, SensorCalibrationData
from tracking_pipeline.infrastructure.io.frame_segment import FrameSegmentReader, FrameSegmentWriter


def test_frame_segment_roundtrips_frame_data(tmp_path: Path) -> None:
    calibration = SensorCalibrationData(
        sensor_name="class_qb2",
        vertical_fov=20.0,
        horizontal_fov=40.0,
        vertical_scanlines=4,
        horizontal_scanlines=8,
        horizontal_angle_spacing=5.0,
        beam_altitude_angles=np.array([-1.0, 1.0], dtype=np.float32),
        beam_azimuth_angles=np.array([0.0, 0.5], dtype=np.float32),
        frame_mode="1",
        scan_pattern="test",
    )
    frame = FrameData(
        frame_index=3,
        timestamp_ns=1234,
        points=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        point_intensity=np.array([0.25, 0.75], dtype=np.float32),
        point_timestamp_ns=np.array([1200, 1210], dtype=np.int64),
        source_path="qb2_live://class_qb2@10.16.3.160",
        source_frame_index=7,
        source_sequence_index=2,
        object_labels=[
            ObjectLabelData(
                object_id=11,
                timestamp_ns=1220,
                points=np.array([[9.0, 0.0, 0.0]], dtype=np.float32),
                obj_class="car",
                obj_class_score=0.9,
                sensor_name="class_qb2",
                frame_index=3,
                source_path="mqtt",
            )
        ],
        scans=[
            LidarScanData(
                sensor_name="class_qb2",
                timestamp_ns=1234,
                xyz=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
                ranges=np.array([3.74], dtype=np.float32),
                row_index=np.array([1], dtype=np.int32),
                col_index=np.array([2], dtype=np.int32),
                calibration=calibration,
                intensity=np.array([0.25], dtype=np.float32),
                point_timestamp_ns=np.array([1200], dtype=np.int64),
            )
        ],
    )

    with FrameSegmentWriter(tmp_path / "segment") as writer:
        writer.write_frame(frame)

    loaded = list(FrameSegmentReader().iter_frames([str(tmp_path / "segment")]))

    assert len(loaded) == 1
    assert loaded[0].frame_index == 3
    assert loaded[0].timestamp_ns == 1234
    assert loaded[0].source_path == "qb2_live://class_qb2@10.16.3.160"
    assert np.array_equal(loaded[0].points, frame.points)
    assert np.array_equal(loaded[0].point_intensity, frame.point_intensity)
    assert np.array_equal(loaded[0].point_timestamp_ns, frame.point_timestamp_ns)
    assert len(loaded[0].object_labels) == 1
    assert loaded[0].object_labels[0].object_id == 11
    assert np.array_equal(loaded[0].object_labels[0].points, frame.object_labels[0].points)
    assert len(loaded[0].scans) == 1
    assert loaded[0].scans[0].calibration.sensor_name == "class_qb2"
    assert np.array_equal(loaded[0].scans[0].row_index, np.array([1], dtype=np.int32))
