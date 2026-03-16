from __future__ import annotations

import argparse
from pathlib import Path

from tracking_pipeline.application.benchmark_run import BenchmarkRunner
from tracking_pipeline.application.replay_run import replay_run
from tracking_pipeline.application.run_pipeline import run_pipeline
from tracking_pipeline.config.loader import load_benchmark_config, load_config
from tracking_pipeline.infrastructure.logging.run_logger import get_run_logger


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
