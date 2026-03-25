from __future__ import annotations

import numpy as np

from tracking_pipeline.application.class_normalization import ClassNormalizer
from tracking_pipeline.config.models import ClassNormalizationConfig
from tracking_pipeline.domain.models import ObjectLabelData


def test_class_normalizer_maps_aliases_case_insensitively() -> None:
    normalizer = ClassNormalizer.from_config(
        ClassNormalizationConfig(
            enabled=True,
            aliases={
                "PKW": "TLS_VEHICLE_CAR",
                "Motorrad": "TLS_VEHICLE_MOTORBIKE",
            },
        )
    )

    assert normalizer.normalize("PKW") == "TLS_VEHICLE_CAR"
    assert normalizer.normalize(" pkw ") == "TLS_VEHICLE_CAR"
    assert normalizer.normalize("MOTORRAD") == "TLS_VEHICLE_MOTORBIKE"


def test_class_normalizer_keeps_unknown_values_and_tls_other() -> None:
    normalizer = ClassNormalizer.from_config(
        ClassNormalizationConfig(
            enabled=True,
            aliases={
                "PKW": "TLS_VEHICLE_CAR",
            },
        )
    )

    assert normalizer.normalize("TLS_VEHICLE_OTHER") == "TLS_VEHICLE_OTHER"
    assert normalizer.normalize("UnknownVehicleClass") == "UnknownVehicleClass"


def test_class_normalizer_normalizes_object_labels() -> None:
    normalizer = ClassNormalizer.from_config(
        ClassNormalizationConfig(
            enabled=True,
            aliases={"car": "TLS_VEHICLE_CAR"},
        )
    )
    label = ObjectLabelData(
        object_id=7,
        timestamp_ns=123,
        points=np.ones((1, 3), dtype=np.float32),
        obj_class=" car ",
        obj_class_score=0.9,
        frame_index=5,
    )

    normalized = normalizer.normalize_object_label(label)

    assert normalized.obj_class == "TLS_VEHICLE_CAR"
    assert normalized.object_id == 7
    assert normalized.timestamp_ns == 123
