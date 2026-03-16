from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import yaml

from tracking_pipeline.application.services import build_run_name, resolve_output_root
from tracking_pipeline.config.models import PipelineConfig
from tracking_pipeline.domain.models import AggregateResult, ObjectLabelData, RunSummary, Track
from tracking_pipeline.infrastructure.io.manifest_writer import ManifestWriter
from tracking_pipeline.infrastructure.io.pcd_writer import PCDWriter
from tracking_pipeline.shared.ids import aggregate_file_stem, object_file_stem


class JsonArtifactWriter:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.manifest_writer = ManifestWriter()
        self.pcd_writer = PCDWriter()

    def prepare_run_dir(self, config: PipelineConfig) -> Path:
        root = resolve_output_root(config, self.project_root)
        run_dir = root / build_run_name(config)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "aggregates").mkdir(exist_ok=True)
        (run_dir / "object_list").mkdir(exist_ok=True)
        return run_dir

    def write_config_snapshot(self, run_dir: Path, config: PipelineConfig) -> None:
        with (run_dir / "config.snapshot.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config.to_dict(), handle, sort_keys=False)

    def write_aggregate(self, run_dir: Path, result: AggregateResult, save_intensity: bool = False) -> None:
        stem = aggregate_file_stem(result.track_id)
        self.pcd_writer.write(
            run_dir / "aggregates" / f"{stem}.pcd",
            result.points,
            intensity=result.intensity if save_intensity else None,
        )
        self.manifest_writer.write_json(
            run_dir / "aggregates" / f"{stem}.json",
            {
                "track_id": result.track_id,
                "status": result.status,
                "selected_frame_ids": result.selected_frame_ids,
                "metrics": result.metrics,
            },
        )

    def write_object_list(self, run_dir: Path, object_labels: dict[int, ObjectLabelData]) -> None:
        object_dir = run_dir / "object_list"
        rows = []
        for object_id, object_label in sorted(object_labels.items()):
            stem = object_file_stem(object_id)
            pcd_path = object_dir / f"{stem}.pcd"
            self.pcd_writer.write(pcd_path, object_label.points)
            rows.append(
                {
                    "object_id": int(object_label.object_id),
                    "timestamp_ns": int(object_label.timestamp_ns),
                    "frame_index": int(object_label.frame_index),
                    "sensor_name": object_label.sensor_name,
                    "obj_class": object_label.obj_class,
                    "obj_class_score": float(object_label.obj_class_score),
                    "pcd_path": str(Path("object_list") / f"{stem}.pcd"),
                    "point_count": int(len(object_label.points)),
                    "source_path": object_label.source_path,
                }
            )
        self.manifest_writer.write_jsonl(object_dir / "manifest.jsonl", rows)

    def write_summary(self, run_dir: Path, summary: RunSummary) -> None:
        self.manifest_writer.write_json(run_dir / "summary.json", asdict(summary))

    def write_tracks(self, run_dir: Path, tracks: dict[int, Track], aggregate_results: list[AggregateResult]) -> None:
        by_track_id = {result.track_id: result for result in aggregate_results}
        rows = []
        for track_id, track in sorted(tracks.items()):
            result = by_track_id.get(track_id)
            rows.append(
                {
                    "track_id": track_id,
                    "source_track_ids": track.source_track_ids or [track_id],
                    "frame_ids": track.frame_ids,
                    "hit_count": track.hit_count,
                    "age": track.age,
                    "missed": track.missed,
                    "ended_by_missed": track.ended_by_missed,
                    "quality_score": track.quality_score,
                    "quality_metrics": track.quality_metrics,
                    "selected_frame_ids": [] if result is None else result.selected_frame_ids,
                    "aggregate_status": None if result is None else result.status,
                    "aggregation_metrics": {} if result is None else result.metrics,
                }
            )
        self.manifest_writer.write_jsonl(run_dir / "tracks.jsonl", rows)
