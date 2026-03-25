from __future__ import annotations

import numpy as np

from tracking_pipeline.application.class_normalization import ClassNormalizer
from tracking_pipeline.application.class_statistics import build_class_statistics
from tracking_pipeline.config.models import ClassNormalizationConfig
from tracking_pipeline.domain.models import AggregateResult, ObjectLabelData


def test_build_class_statistics_counts_saved_predictions_gt_labels_and_match_rows() -> None:
    aggregate_results = [
        AggregateResult(
            track_id=1,
            points=np.ones((1, 3), dtype=np.float32),
            selected_frame_ids=[1],
            status="saved",
            metrics={"predicted_class_name": "PKW", "gt_matched": True, "gt_obj_class": "PKW"},
        ),
        AggregateResult(
            track_id=2,
            points=np.ones((1, 3), dtype=np.float32),
            selected_frame_ids=[2],
            status="saved",
            metrics={"predicted_class_name": "LKW"},
        ),
        AggregateResult(
            track_id=3,
            points=np.ones((1, 3), dtype=np.float32),
            selected_frame_ids=[3],
            status="saved",
            metrics={"predicted_class_name": "PKW", "gt_matched": True, "gt_obj_class": "Van"},
        ),
        AggregateResult(
            track_id=4,
            points=np.zeros((0, 3), dtype=np.float32),
            selected_frame_ids=[4],
            status="skipped_quality_threshold",
            metrics={"predicted_class_name": "Bus", "gt_matched": True, "gt_obj_class": "Bus"},
        ),
    ]
    latest_object_labels = {
        9: ObjectLabelData(object_id=9, timestamp_ns=0, points=np.ones((1, 3), dtype=np.float32), obj_class="Bus"),
        10: ObjectLabelData(object_id=10, timestamp_ns=1, points=np.ones((1, 3), dtype=np.float32), obj_class="PKW"),
        11: ObjectLabelData(object_id=11, timestamp_ns=2, points=np.ones((1, 3), dtype=np.float32), obj_class="LKW"),
        12: ObjectLabelData(object_id=12, timestamp_ns=3, points=np.ones((1, 3), dtype=np.float32), obj_class="Van"),
    }

    stats = build_class_statistics(aggregate_results, latest_object_labels)

    assert stats["predicted_class_counts"] == {"LKW": 1, "PKW": 2}
    assert stats["gt_class_counts"] == {"Bus": 1, "LKW": 1, "PKW": 1, "Van": 1}
    assert stats["matched_gt_class_counts"] == {"PKW": 1, "Van": 1}
    assert stats["class_comparison_count"] == 2
    assert stats["class_match_count"] == 1
    assert stats["class_mismatch_count"] == 1
    assert stats["class_count_rows"] == [
        {"class_name": "LKW", "predicted_count": 1, "gt_match_count": 0},
        {"class_name": "PKW", "predicted_count": 2, "gt_match_count": 1},
        {"class_name": "Van", "predicted_count": 0, "gt_match_count": 1},
        {"class_name": "TOTAL", "predicted_count": 3, "gt_match_count": 2},
    ]


def test_build_class_statistics_normalizes_prediction_and_gt_into_shared_tls_rows() -> None:
    aggregate_results = [
        AggregateResult(
            track_id=1,
            points=np.ones((1, 3), dtype=np.float32),
            selected_frame_ids=[1],
            status="saved",
            metrics={"predicted_class_name": "PKW", "gt_matched": True, "gt_obj_class": "TLS_VEHICLE_CAR"},
        ),
        AggregateResult(
            track_id=2,
            points=np.ones((1, 3), dtype=np.float32),
            selected_frame_ids=[2],
            status="saved",
            metrics={"predicted_class_name": "Transporter", "gt_matched": True, "gt_obj_class": "van"},
        ),
    ]
    latest_object_labels = {
        9: ObjectLabelData(object_id=9, timestamp_ns=0, points=np.ones((1, 3), dtype=np.float32), obj_class="car"),
        10: ObjectLabelData(object_id=10, timestamp_ns=1, points=np.ones((1, 3), dtype=np.float32), obj_class="TLS_VEHICLE_VAN"),
    }
    normalizer = ClassNormalizer.from_config(
        ClassNormalizationConfig(
            enabled=True,
            aliases={
                "PKW": "TLS_VEHICLE_CAR",
                "Transporter": "TLS_VEHICLE_VAN",
                "car": "TLS_VEHICLE_CAR",
                "van": "TLS_VEHICLE_VAN",
            },
        )
    )

    stats = build_class_statistics(aggregate_results, latest_object_labels, normalizer)

    assert stats["predicted_class_counts"] == {"TLS_VEHICLE_CAR": 1, "TLS_VEHICLE_VAN": 1}
    assert stats["gt_class_counts"] == {"TLS_VEHICLE_CAR": 1, "TLS_VEHICLE_VAN": 1}
    assert stats["matched_gt_class_counts"] == {"TLS_VEHICLE_CAR": 1, "TLS_VEHICLE_VAN": 1}
    assert stats["class_comparison_count"] == 2
    assert stats["class_match_count"] == 2
    assert stats["class_mismatch_count"] == 0
    assert stats["class_count_rows"] == [
        {"class_name": "TLS_VEHICLE_CAR", "predicted_count": 1, "gt_match_count": 1},
        {"class_name": "TLS_VEHICLE_VAN", "predicted_count": 1, "gt_match_count": 1},
        {"class_name": "TOTAL", "predicted_count": 2, "gt_match_count": 2},
    ]
