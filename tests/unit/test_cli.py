from __future__ import annotations

from tracking_pipeline.cli import _format_class_comparison_line, _format_class_count_table, _format_throughput_line


def test_format_class_count_table_returns_markdown_table() -> None:
    assert _format_class_count_table(
        [
            {"class_name": "Bus", "predicted_count": 1, "gt_match_count": 0},
            {"class_name": "PKW", "predicted_count": 3, "gt_match_count": 2},
            {"class_name": "TOTAL", "predicted_count": 4, "gt_match_count": 2},
        ]
    ) == (
        "Class Counts\n"
        "| Class | Net | GT matches |\n"
        "| --- | ---: | ---: |\n"
        "| Bus | 1 | 0 |\n"
        "| PKW | 3 | 2 |\n"
        "| TOTAL | 4 | 2 |"
    )


def test_format_class_count_table_returns_none_for_empty_stats() -> None:
    assert _format_class_count_table([]) == "Class Counts: none"


def test_format_throughput_line_formats_total_and_compute_hz() -> None:
    assert _format_throughput_line(12.34567, 23.45678) == "throughput total=12.346 Hz compute=23.457 Hz"


def test_format_class_comparison_line_formats_match_and_mismatch_counts() -> None:
    assert _format_class_comparison_line(7, 5, 2) == "class_compare compared=7 matches=5 mismatches=2"


def test_format_class_comparison_line_returns_none_for_empty_comparisons() -> None:
    assert _format_class_comparison_line(0, 0, 0) == "Class Compare: none"
