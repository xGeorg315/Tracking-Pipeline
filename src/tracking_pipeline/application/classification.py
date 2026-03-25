from __future__ import annotations

from tracking_pipeline.application.class_normalization import ClassNormalizer
from tracking_pipeline.application.ports import ObjectClassifier
from tracking_pipeline.domain.models import AggregateResult


CLASSIFICATION_METRIC_KEYS = (
    "predicted_class_id",
    "predicted_class_name",
    "predicted_class_score",
    "classification_backend",
    "classification_point_source",
    "classification_input_point_count",
)


def classify_aggregate_results(
    aggregate_results: list[AggregateResult],
    classifier: ObjectClassifier | None,
    class_normalizer: ClassNormalizer | None = None,
) -> list[AggregateResult]:
    if classifier is None:
        return aggregate_results
    by_track_id = {int(result.track_id): result for result in aggregate_results}
    for result in aggregate_results:
        if _merged_target_track_id(result) is not None:
            continue
        point_source, points = _classification_points(result)
        if points is None:
            continue
        prediction = classifier.classify_points(points)
        _write_prediction_metrics(
            result,
            class_id=int(prediction.class_id),
            class_name=_normalize_class_name(str(prediction.class_name), class_normalizer),
            score=float(prediction.score),
            backend=str(classifier.backend),
            point_source=str(point_source),
            input_point_count=int(len(points)),
        )
    for result in aggregate_results:
        merged_target_track_id = _merged_target_track_id(result)
        if merged_target_track_id is None:
            continue
        merged_target = by_track_id.get(int(merged_target_track_id))
        if merged_target is None:
            continue
        target_metrics = merged_target.metrics
        predicted_class_id = target_metrics.get("predicted_class_id")
        predicted_class_name = target_metrics.get("predicted_class_name")
        predicted_class_score = target_metrics.get("predicted_class_score")
        if predicted_class_id is None and not predicted_class_name:
            continue
        _write_prediction_metrics(
            result,
            class_id=None if predicted_class_id is None else int(predicted_class_id),
            class_name=_normalize_class_name(str(predicted_class_name or ""), class_normalizer),
            score=None if predicted_class_score is None else float(predicted_class_score),
            backend=str(target_metrics.get("classification_backend", classifier.backend)),
            point_source="merged_target_track",
            input_point_count=int(target_metrics.get("classification_input_point_count", 0) or 0),
        )
    return aggregate_results


def _classification_points(result: AggregateResult) -> tuple[str, object] | tuple[None, None]:
    if result.points is not None and len(result.points) > 0:
        return "result_points", result.points
    if result.candidate_points_world is not None and len(result.candidate_points_world) > 0:
        return "candidate_points_world", result.candidate_points_world
    return None, None


def _merged_target_track_id(result: AggregateResult) -> int | None:
    if str(result.status) != "merged_into_long_vehicle_group":
        return None
    merged_target_track_id = result.metrics.get("merged_target_track_id")
    if merged_target_track_id is None:
        return None
    return int(merged_target_track_id)


def _write_prediction_metrics(
    result: AggregateResult,
    *,
    class_id: int | None,
    class_name: str,
    score: float | None,
    backend: str,
    point_source: str,
    input_point_count: int,
) -> None:
    if class_id is None:
        result.metrics.pop("predicted_class_id", None)
    else:
        result.metrics["predicted_class_id"] = int(class_id)
    result.metrics["predicted_class_name"] = str(class_name)
    if score is None:
        result.metrics.pop("predicted_class_score", None)
    else:
        result.metrics["predicted_class_score"] = float(score)
    result.metrics["classification_backend"] = str(backend)
    result.metrics["classification_point_source"] = str(point_source)
    result.metrics["classification_input_point_count"] = int(input_point_count)


def _normalize_class_name(class_name: str, class_normalizer: ClassNormalizer | None) -> str:
    if class_normalizer is None:
        return str(class_name or "")
    return class_normalizer.normalize(class_name)
