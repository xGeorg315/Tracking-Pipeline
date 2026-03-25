from __future__ import annotations

import argparse
from pathlib import Path

from tracking_pipeline.application.benchmark_run import BenchmarkRunner
from tracking_pipeline.application.replay_run import replay_run
from tracking_pipeline.application.run_pipeline import run_pipeline
from tracking_pipeline.config.loader import load_benchmark_config, load_config
from tracking_pipeline.infrastructure.logging.run_logger import get_run_logger


def _format_class_count_table(class_count_rows: list[dict[str, int | str]]) -> str:
    if not class_count_rows:
        return "Class Counts: none"
    lines = [
        "Class Counts",
        "| Class | Net | GT matches |",
        "| --- | ---: | ---: |",
    ]
    for row in class_count_rows:
        lines.append(
            "| {class_name} | {predicted_count} | {gt_match_count} |".format(
                class_name=str(row.get("class_name", "")),
                predicted_count=int(row.get("predicted_count", 0)),
                gt_match_count=int(row.get("gt_match_count", 0)),
            )
        )
    return "\n".join(lines)


def _format_throughput_line(total_hz: float, compute_hz: float) -> str:
    return f"throughput total={float(total_hz):.3f} Hz compute={float(compute_hz):.3f} Hz"


def _format_class_comparison_line(class_comparison_count: int, class_match_count: int, class_mismatch_count: int) -> str:
    if int(class_comparison_count) <= 0:
        return "Class Compare: none"
    return (
        f"class_compare compared={int(class_comparison_count)} "
        f"matches={int(class_match_count)} mismatches={int(class_mismatch_count)}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tracking-pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Execute the tracking pipeline")
    run_parser.add_argument("-c", "--config", required=True, help="Path to a YAML config")

    replay_parser = subparsers.add_parser("replay", help="Replay a pipeline run in Open3D")
    replay_parser.add_argument("-c", "--config", required=True, help="Path to a YAML config")

    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark multiple presets on one or more sequences")
    benchmark_parser.add_argument("-c", "--config", required=True, help="Path to a benchmark YAML config")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logger = get_run_logger()
    project_root = Path(__file__).resolve().parents[2]

    if args.command == "run":
        config = load_config(args.config)
        summary = run_pipeline(config, project_root)
        logger.info("Run completed: %s", summary.output_dir)
        logger.info(
            "frames=%s tracks=%s saved_aggregates=%s",
            summary.frame_count,
            summary.finished_track_count,
            summary.saved_aggregates,
        )
        performance = summary.performance
        if performance is not None:
            logger.info("%s", _format_throughput_line(performance.total_hz, performance.compute_hz))
        logger.info("%s", _format_class_count_table(summary.class_count_rows))
        logger.info(
            "%s",
            _format_class_comparison_line(
                summary.class_comparison_count,
                summary.class_match_count,
                summary.class_mismatch_count,
            ),
        )
        return

    if args.command == "replay":
        config = load_config(args.config)
        replay_run(config, project_root)
        return

    if args.command == "benchmark":
        config = load_benchmark_config(args.config)
        output_dir = BenchmarkRunner(project_root).run(config)
        logger.info("Benchmark completed: %s", output_dir)
        return

    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
