from __future__ import annotations

from collections import Counter

from tracking_pipeline.application.class_normalization import ClassNormalizer
from tracking_pipeline.domain.models import AggregateResult, ObjectLabelData


def build_class_statistics(
    aggregate_results: list[AggregateResult],
    latest_object_labels: dict[int, ObjectLabelData],
    class_normalizer: ClassNormalizer | None = None,
) -> dict[str, object]:
    predicted_counts: Counter[str] = Counter()
    gt_counts: Counter[str] = Counter()
    matched_gt_counts: Counter[str] = Counter()
    class_comparison_count = 0
    class_match_count = 0
    class_mismatch_count = 0

    for result in aggregate_results:
        if str(result.status) != "saved":
            continue
        class_name = _normalize_class_name(str(result.metrics.get("predicted_class_name", "") or "").strip(), class_normalizer)
        if class_name:
            predicted_counts[class_name] += 1
        matched = bool(result.metrics.get("gt_matched"))
        gt_class_name = _normalize_class_name(str(result.metrics.get("gt_obj_class", "") or "").strip(), class_normalizer)
        if matched and gt_class_name:
            matched_gt_counts[gt_class_name] += 1
        if matched and class_name and gt_class_name:
            class_comparison_count += 1
            if class_name == gt_class_name:
                class_match_count += 1
            else:
                class_mismatch_count += 1

    for object_id in sorted(latest_object_labels):
        object_label = latest_object_labels[int(object_id)]
        class_name = _normalize_class_name(str(object_label.obj_class or "").strip(), class_normalizer)
        if class_name:
            gt_counts[class_name] += 1

    predicted_count_dict = _sorted_count_dict(predicted_counts)
    gt_count_dict = _sorted_count_dict(gt_counts)
    matched_gt_count_dict = _sorted_count_dict(matched_gt_counts)
    return {
        "predicted_class_counts": predicted_count_dict,
        "gt_class_counts": gt_count_dict,
        "matched_gt_class_counts": matched_gt_count_dict,
        "class_comparison_count": int(class_comparison_count),
        "class_match_count": int(class_match_count),
        "class_mismatch_count": int(class_mismatch_count),
        "class_count_rows": _class_count_rows(predicted_count_dict, matched_gt_count_dict),
    }


def _sorted_count_dict(counter: Counter[str]) -> dict[str, int]:
    return {str(class_name): int(counter[class_name]) for class_name in sorted(counter)}


def _normalize_class_name(class_name: str, class_normalizer: ClassNormalizer | None) -> str:
    if class_normalizer is None:
        return str(class_name or "")
    return class_normalizer.normalize(class_name)


def _class_count_rows(
    predicted_class_counts: dict[str, int],
    matched_gt_class_counts: dict[str, int],
) -> list[dict[str, int | str]]:
    class_names = sorted(set(predicted_class_counts) | set(matched_gt_class_counts))
    if not class_names:
        return []
    rows = [
        {
            "class_name": str(class_name),
            "predicted_count": int(predicted_class_counts.get(class_name, 0)),
            "gt_match_count": int(matched_gt_class_counts.get(class_name, 0)),
        }
        for class_name in class_names
    ]
    rows.append(
        {
            "class_name": "TOTAL",
            "predicted_count": int(sum(predicted_class_counts.values())),
            "gt_match_count": int(sum(matched_gt_class_counts.values())),
        }
    )
    return rows
