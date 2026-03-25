from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tracking_pipeline.application.classification import classify_aggregate_results
from tracking_pipeline.application.class_normalization import ClassNormalizer
from tracking_pipeline.application.factories import build_classifier
from tracking_pipeline.config.models import (
    ClassificationConfig,
    ClassNormalizationConfig,
    InputConfig,
    PipelineConfig,
    PreprocessingConfig,
)
from tracking_pipeline.config.validation import ConfigError
from tracking_pipeline.domain.models import AggregateResult, ClassificationPrediction
from tracking_pipeline.infrastructure.classification.pointnext_classifier import PointNextObjectClassifier


class _FakeClassifier:
    backend = "pointnext"

    def __init__(self):
        self.seen_points: list[np.ndarray] = []

    def classify_points(self, points: np.ndarray) -> ClassificationPrediction:
        arr = np.asarray(points, dtype=np.float32)
        self.seen_points.append(arr.copy())
        return ClassificationPrediction(class_id=2, class_name="truck", score=0.87)


def _base_config(tmp_path: Path, classification: ClassificationConfig) -> PipelineConfig:
    input_path = tmp_path / "sample.pb"
    input_path.write_bytes(b"pb")
    return PipelineConfig(
        input=InputConfig(paths=[str(input_path)]),
        preprocessing=PreprocessingConfig(lane_box=[-1.0, 1.0, 0.0, 10.0, 0.0, 2.0]),
        classification=classification,
    )


def test_classify_aggregate_results_prefers_result_points() -> None:
    classifier = _FakeClassifier()
    result = AggregateResult(
        track_id=7,
        points=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        selected_frame_ids=[100],
        status="saved",
        metrics={},
        candidate_points_world=np.array([[5.0, 5.0, 5.0]], dtype=np.float32),
    )

    classify_aggregate_results([result], classifier)

    assert len(classifier.seen_points) == 1
    assert np.allclose(classifier.seen_points[0], result.points)
    assert result.metrics["predicted_class_id"] == 2
    assert result.metrics["predicted_class_name"] == "truck"
    assert result.metrics["predicted_class_score"] == pytest.approx(0.87)
    assert result.metrics["classification_backend"] == "pointnext"
    assert result.metrics["classification_point_source"] == "result_points"
    assert result.metrics["classification_input_point_count"] == 2


def test_classify_aggregate_results_normalizes_prediction_names() -> None:
    classifier = _FakeClassifier()
    normalizer = ClassNormalizer.from_config(
        ClassNormalizationConfig(enabled=True, aliases={"truck": "TLS_VEHICLE_TRUCK"})
    )
    result = AggregateResult(
        track_id=70,
        points=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        selected_frame_ids=[100],
        status="saved",
        metrics={},
    )

    classify_aggregate_results([result], classifier, normalizer)

    assert result.metrics["predicted_class_name"] == "TLS_VEHICLE_TRUCK"


def test_classify_aggregate_results_falls_back_to_candidate_points() -> None:
    classifier = _FakeClassifier()
    result = AggregateResult(
        track_id=8,
        points=np.zeros((0, 3), dtype=np.float32),
        selected_frame_ids=[100],
        status="skipped_min_saved_points",
        metrics={},
        candidate_points_world=np.array([[0.5, 0.0, 0.0], [0.6, 0.0, 0.0]], dtype=np.float32),
    )

    classify_aggregate_results([result], classifier)

    assert len(classifier.seen_points) == 1
    assert np.allclose(classifier.seen_points[0], result.candidate_points_world)
    assert result.metrics["classification_point_source"] == "candidate_points_world"


def test_classify_aggregate_results_skips_empty_results() -> None:
    classifier = _FakeClassifier()
    result = AggregateResult(
        track_id=9,
        points=np.zeros((0, 3), dtype=np.float32),
        selected_frame_ids=[],
        status="empty_filtered",
        metrics={},
        candidate_points_world=np.zeros((0, 3), dtype=np.float32),
    )

    classify_aggregate_results([result], classifier)

    assert classifier.seen_points == []
    assert result.metrics == {}


def test_classify_aggregate_results_propagates_merged_group_prediction_from_target() -> None:
    classifier = _FakeClassifier()
    lead = AggregateResult(
        track_id=10,
        points=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        selected_frame_ids=[100],
        status="saved",
        metrics={},
    )
    rear = AggregateResult(
        track_id=11,
        points=np.zeros((0, 3), dtype=np.float32),
        selected_frame_ids=[100],
        status="merged_into_long_vehicle_group",
        metrics={"merged_target_track_id": 10},
        candidate_points_world=np.array([[5.0, 0.0, 0.0], [5.5, 0.0, 0.0]], dtype=np.float32),
    )

    classify_aggregate_results([lead, rear], classifier)

    assert len(classifier.seen_points) == 1
    assert np.allclose(classifier.seen_points[0], lead.points)
    assert rear.metrics["predicted_class_id"] == 2
    assert rear.metrics["predicted_class_name"] == "truck"
    assert rear.metrics["predicted_class_score"] == pytest.approx(0.87)
    assert rear.metrics["classification_backend"] == "pointnext"
    assert rear.metrics["classification_point_source"] == "merged_target_track"
    assert rear.metrics["classification_input_point_count"] == 2


def test_build_classifier_rejects_empty_openpoints_directory(tmp_path: Path) -> None:
    pointnext_root = tmp_path / "PointNeXt"
    (pointnext_root / "openpoints").mkdir(parents=True)
    checkpoint_path = tmp_path / "bestckpt.pth"
    checkpoint_path.write_bytes(b"ckpt")
    model_cfg_path = tmp_path / "pointnext-s.yaml"
    model_cfg_path.write_text(
        "\n".join(
            [
                "model:",
                "  NAME: BaseCls",
                "  extra_global_channels: 0",
                "  cls_args:",
                "    NAME: ClsHead",
                "    num_classes: 2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config = _base_config(
        tmp_path,
        ClassificationConfig(
            enabled=True,
            class_names=["car", "truck"],
            pointnext_root=str(pointnext_root),
            checkpoint_path=str(checkpoint_path),
            model_cfg_path=str(model_cfg_path),
        ),
    )

    with pytest.raises(ConfigError, match="openpoints directory is empty"):
        build_classifier(config)


def test_pointnext_classifier_resolve_device_prefers_mps_when_auto_and_cuda_unavailable(monkeypatch) -> None:
    classifier = PointNextObjectClassifier.__new__(PointNextObjectClassifier)
    monkeypatch.setattr("tracking_pipeline.infrastructure.classification.pointnext_classifier.torch.cuda.is_available", lambda: False)
    monkeypatch.setattr(
        "tracking_pipeline.infrastructure.classification.pointnext_classifier.torch.backends.mps.is_available",
        lambda: True,
    )

    device = classifier._resolve_device("auto")

    assert str(device) == "mps"


def test_pointnext_classifier_resolve_device_accepts_explicit_mps(monkeypatch) -> None:
    classifier = PointNextObjectClassifier.__new__(PointNextObjectClassifier)
    monkeypatch.setattr(
        "tracking_pipeline.infrastructure.classification.pointnext_classifier.torch.backends.mps.is_available",
        lambda: True,
    )

    device = classifier._resolve_device("mps")

    assert str(device) == "mps"


def test_pointnext_classifier_resolve_device_rejects_unavailable_explicit_mps(monkeypatch) -> None:
    classifier = PointNextObjectClassifier.__new__(PointNextObjectClassifier)
    monkeypatch.setattr(
        "tracking_pipeline.infrastructure.classification.pointnext_classifier.torch.backends.mps.is_available",
        lambda: False,
    )

    with pytest.raises(ConfigError, match="classification.device is set to mps, but MPS is not available"):
        classifier._resolve_device("mps")
