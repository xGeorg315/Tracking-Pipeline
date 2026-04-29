from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import yaml

from tracking_pipeline.application.services import build_run_name, resolve_dataset_root
from tracking_pipeline.config.models import PipelineConfig
from tracking_pipeline.domain.models import AggregateResult, FrameTrackingState, GTMatchResult, ObjectLabelData, RunSummary, Track, TrackOutcomeDebug
from tracking_pipeline.infrastructure.io.artifact_writer import JsonArtifactWriter
from tracking_pipeline.infrastructure.io.manifest_writer import ManifestWriter
from tracking_pipeline.infrastructure.io.pcd_writer import PCDWriter
from tracking_pipeline.shared.ids import aggregate_file_stem, object_file_stem


@dataclass(slots=True)
class _DatasetSampleEntry:
    sample_id: str
    bucket: str
    class_name: str
    day_key: str
    match_payload: dict[str, Any]
    gt_label: ObjectLabelData | None = None
    aggregate_result: AggregateResult | None = None
    save_intensity: bool = False
    gt_state: tuple[Any, ...] | None = None
    pred_state: tuple[Any, ...] | None = None
    match_state: str = ""


class DatasetArtifactWriter:
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.manifest_writer = ManifestWriter()
        self.pcd_writer = PCDWriter()
        self.stats_writer = JsonArtifactWriter(project_root)
        self._dataset_root: Path | None = None
        self._run_id = ""
        self._run_day_key = datetime.now().astimezone().strftime("%Y-%m-%d")
        self._statistics_enabled = True
        self._config_payload: dict[str, Any] | None = None
        self._object_labels: dict[int, ObjectLabelData] = {}
        self._aggregate_results: dict[int, AggregateResult] = {}
        self._aggregate_save_intensity: dict[int, bool] = {}
        self._matches: list[GTMatchResult] = []
        self._unmatched_saved_tracks: list[GTMatchResult] = []
        self._unmatched_gt_objects: list[GTMatchResult] = []
        self._gt_summary: dict[str, Any] = {}
        self._tracks: dict[int, Track] | None = None
        self._aggregate_results_for_stats: list[AggregateResult] = []
        self._tracker_states: list[FrameTrackingState] | None = None
        self._track_outcomes: dict[int, TrackOutcomeDebug] | None = None
        self._class_stats: dict[str, Any] | None = None
        self._summary: RunSummary | None = None
        self._live_status_payload: dict[str, Any] | None = None
        self._current_sample_entries: dict[str, _DatasetSampleEntry] = {}
        self._current_stats_days: set[str] = set()
        self._sample_flush_suspended = 0
        self._sample_flush_dirty = False
        self._stats_flush_suspended = 0
        self._stats_flush_dirty = False

    def prepare_run_dir(self, config: PipelineConfig) -> Path:
        self._dataset_root = resolve_dataset_root(config, self.project_root)
        self._statistics_enabled = bool(config.output.statistics_enabled)
        self._dataset_root.mkdir(parents=True, exist_ok=True)
        self._run_id = build_run_name(config)
        self._run_day_key = datetime.now().astimezone().strftime("%Y-%m-%d")
        if self._statistics_enabled:
            self._active_stats_dir().mkdir(parents=True, exist_ok=True)
        return self._dataset_root

    def write_config_snapshot(self, run_dir: Path, config: PipelineConfig) -> None:
        _ = run_dir
        self._config_payload = config.to_dict()
        if not self._statistics_enabled:
            return
        self._write_config_yaml(self._active_stats_dir() / "config.snapshot.yaml")
        self._flush_stats_dirs()

    def begin_snapshot(self, run_dir: Path) -> None:
        _ = run_dir
        self._aggregate_results = {}
        self._aggregate_save_intensity = {}
        self._matches = []
        self._unmatched_saved_tracks = []
        self._unmatched_gt_objects = []
        self._gt_summary = {}
        self._tracks = None
        self._aggregate_results_for_stats = []
        self._tracker_states = None
        self._track_outcomes = None
        self._class_stats = None
        self._summary = None

    def clear_live_outputs(self, run_dir: Path) -> None:
        _ = run_dir

    def begin_sample_batch(self) -> None:
        self._sample_flush_suspended += 1

    def end_sample_batch(self) -> None:
        if self._sample_flush_suspended <= 0:
            return
        self._sample_flush_suspended -= 1
        if self._sample_flush_suspended == 0 and self._sample_flush_dirty:
            self._sample_flush_dirty = False
            self._flush_dataset_samples()

    def begin_stats_batch(self) -> None:
        self._stats_flush_suspended += 1

    def end_stats_batch(self) -> None:
        if self._stats_flush_suspended <= 0:
            return
        self._stats_flush_suspended -= 1
        if self._stats_flush_suspended == 0 and self._stats_flush_dirty:
            self._stats_flush_dirty = False
            self._flush_stats_dirs()

    def live_status_path(self, run_dir: Path) -> Path:
        _ = run_dir
        return self._active_stats_dir() / "live_status.json"

    def object_list_manifest_path(self, run_dir: Path) -> Path:
        _ = run_dir
        return self._active_stats_dir() / "object_list_manifest.jsonl"

    def live_artifact_dir(self, run_dir: Path) -> Path:
        _ = run_dir
        return self._require_root()

    def write_live_status(self, run_dir: Path, payload: dict[str, Any]) -> None:
        _ = run_dir
        if not self._statistics_enabled:
            return
        self._live_status_payload = dict(payload)
        self.manifest_writer.write_json(self.live_status_path(self._require_root()), self._live_status_payload)

    def write_aggregate(self, run_dir: Path, result: AggregateResult, save_intensity: bool = False) -> None:
        _ = run_dir
        if str(result.status) != "saved":
            return
        self._aggregate_results[int(result.track_id)] = result
        self._aggregate_save_intensity[int(result.track_id)] = bool(save_intensity)

    def write_object_list(self, run_dir: Path, object_labels: dict[int, ObjectLabelData]) -> None:
        _ = run_dir
        self._object_labels = {int(object_id): object_label for object_id, object_label in sorted(object_labels.items())}
        self._schedule_dataset_sample_flush()

    def write_live_object_list_snapshot(self, run_dir: Path, object_labels: dict[int, ObjectLabelData]) -> None:
        _ = run_dir
        self._object_labels = {int(object_id): object_label for object_id, object_label in sorted(object_labels.items())}
        if not self._statistics_enabled:
            return
        self.manifest_writer.write_jsonl(self.object_list_manifest_path(self._require_root()), self._build_object_list_manifest_rows())
        self._flush_stats_dirs()

    def write_gt_matching(
        self,
        run_dir: Path,
        matches: list[GTMatchResult],
        unmatched_saved_tracks: list[GTMatchResult],
        unmatched_gt_objects: list[GTMatchResult],
        summary: dict[str, int | float | str],
    ) -> None:
        _ = run_dir
        self._matches = list(matches)
        self._unmatched_saved_tracks = list(unmatched_saved_tracks)
        self._unmatched_gt_objects = list(unmatched_gt_objects)
        self._gt_summary = dict(summary)
        self._schedule_dataset_sample_flush()

    def write_summary(self, run_dir: Path, summary: RunSummary) -> None:
        _ = run_dir
        self._summary = summary
        self._schedule_stats_flush()

    def write_tracker_debug(self, run_dir: Path, states: list[FrameTrackingState]) -> None:
        _ = run_dir
        self._tracker_states = list(states)
        self._schedule_stats_flush()

    def write_track_outcomes(self, run_dir: Path, track_outcomes: dict[int, TrackOutcomeDebug]) -> None:
        _ = run_dir
        self._track_outcomes = dict(track_outcomes)
        self._schedule_stats_flush()

    def write_class_stats(self, run_dir: Path, class_stats: dict[str, object]) -> None:
        _ = run_dir
        self._class_stats = dict(class_stats)
        self._schedule_stats_flush()

    def write_tracks(self, run_dir: Path, tracks: dict[int, Track], aggregate_results: list[AggregateResult]) -> None:
        _ = run_dir
        self._tracks = dict(tracks)
        self._aggregate_results_for_stats = list(aggregate_results)
        self._schedule_stats_flush()

    def _flush_dataset_samples(self) -> None:
        new_entries = self._build_sample_entries()
        current_entries = dict(self._current_sample_entries)

        removed_ids = sorted(set(current_entries) - set(new_entries))
        for sample_id in removed_ids:
            self._remove_sample_entry(current_entries[sample_id])

        shared_ids = sorted(set(current_entries) & set(new_entries))
        for sample_id in shared_ids:
            current_entry = current_entries[sample_id]
            new_entry = new_entries[sample_id]
            if self._sample_entries_equal(current_entry, new_entry):
                continue
            self._remove_sample_entry(current_entry)
            self._write_sample_entry(new_entry)

        added_ids = sorted(set(new_entries) - set(current_entries))
        for sample_id in added_ids:
            self._write_sample_entry(new_entries[sample_id])

        self._current_sample_entries = new_entries
        if self._statistics_enabled:
            self._write_active_object_list_manifest()
        self._schedule_stats_flush()

    def _build_sample_entries(self) -> dict[str, _DatasetSampleEntry]:
        entries: dict[str, _DatasetSampleEntry] = {}
        labels = {int(object_id): object_label for object_id, object_label in self._object_labels.items()}
        matching_present = bool(self._matches or self._unmatched_saved_tracks or self._unmatched_gt_objects)
        matched_gt_ids: set[int] = set()

        for match in self._matches:
            if not bool(match.matched):
                continue
            track_id = int(match.track_id)
            gt_object_id = int(match.gt_object_id) if match.gt_object_id is not None else None
            if gt_object_id is None:
                continue
            gt_label = labels.get(gt_object_id)
            aggregate_result = self._aggregate_results.get(track_id)
            if gt_label is None or aggregate_result is None:
                continue
            matched_gt_ids.add(gt_object_id)
            gt_class = self._original_class_name(str(match.gt_obj_class or gt_label.obj_class or "UNKNOWN_GT"))
            pred_class = self._original_class_name(str(aggregate_result.metrics.get("predicted_class_name", "") or ""))
            bucket = "gt-pred-same" if gt_class and pred_class and gt_class == pred_class else "gt-pred-different"
            day_key = self._date_key_from_timestamp(int(match.gt_timestamp_ns) if match.gt_timestamp_ns is not None else int(gt_label.timestamp_ns))
            sample_id = self._sample_id(track_id=track_id, gt_object_id=gt_object_id)
            match_payload = dict(asdict(match))
            match_payload.update(
                {
                    "sample_id": sample_id,
                    "run_id": self._run_id,
                    "bucket": bucket,
                    "class_name": gt_class,
                    "predicted_class_name": pred_class,
                    "day": day_key,
                }
            )
            entries[sample_id] = _DatasetSampleEntry(
                sample_id=sample_id,
                bucket=bucket,
                class_name=self._safe_class_name(gt_class or "UNKNOWN_GT"),
                day_key=day_key,
                gt_label=gt_label,
                aggregate_result=aggregate_result,
                save_intensity=bool(self._aggregate_save_intensity.get(track_id, False)),
                match_payload=match_payload,
            )
            self._populate_entry_state(entries[sample_id])

        if self._statistics_enabled:
            unmatched_gt_results = self._unmatched_gt_objects
            if matching_present and unmatched_gt_results:
                for unmatched in unmatched_gt_results:
                    gt_object_id = int(unmatched.gt_object_id) if unmatched.gt_object_id is not None else None
                    if gt_object_id is None:
                        continue
                    gt_label = labels.get(gt_object_id)
                    if gt_label is None:
                        continue
                    entries[self._sample_id(gt_object_id=gt_object_id)] = self._unmatched_gt_entry(gt_label, unmatched)
            else:
                for object_id, object_label in labels.items():
                    if int(object_id) in matched_gt_ids:
                        continue
                    entries[self._sample_id(gt_object_id=int(object_id))] = self._unmatched_gt_entry(object_label, None)

        for unmatched in self._unmatched_saved_tracks:
            track_id = int(unmatched.track_id)
            aggregate_result = self._aggregate_results.get(track_id)
            if aggregate_result is None:
                continue
            pred_class = self._original_class_name(str(aggregate_result.metrics.get("predicted_class_name", "") or "UNKNOWN_PRED"))
            day_key = self._date_key_from_timestamp(int(unmatched.our_last_timestamp_ns))
            sample_id = self._sample_id(track_id=track_id)
            match_payload = dict(asdict(unmatched))
            match_payload.update(
                {
                    "sample_id": sample_id,
                    "run_id": self._run_id,
                    "bucket": "unmatched_pred",
                    "class_name": pred_class,
                    "predicted_class_name": pred_class,
                    "day": day_key,
                }
            )
            entries[sample_id] = _DatasetSampleEntry(
                sample_id=sample_id,
                bucket="unmatched_pred",
                class_name=self._safe_class_name(pred_class or "UNKNOWN_PRED"),
                day_key=day_key,
                aggregate_result=aggregate_result,
                save_intensity=bool(self._aggregate_save_intensity.get(track_id, False)),
                match_payload=match_payload,
            )
            self._populate_entry_state(entries[sample_id])
        return dict(sorted(entries.items()))

    def _unmatched_gt_entry(self, object_label: ObjectLabelData, unmatched: GTMatchResult | None) -> _DatasetSampleEntry:
        gt_object_id = int(object_label.object_id)
        gt_class = self._original_class_name(
            str("" if unmatched is None else unmatched.gt_obj_class) or str(object_label.obj_class or "UNKNOWN_GT")
        )
        day_key = self._date_key_from_timestamp(int(object_label.timestamp_ns))
        sample_id = self._sample_id(gt_object_id=gt_object_id)
        payload = (
            {
                "track_id": -1,
                "gt_object_id": gt_object_id,
                "our_last_timestamp_ns": -1,
                "gt_timestamp_ns": int(object_label.timestamp_ns),
                "timestamp_delta_ns": None,
                "our_last_frame_id": -1,
                "gt_frame_index": int(object_label.frame_index),
                "assignment_cost": None,
                "matched": False,
                "unmatched_reason": "unmatched_gt",
                "gt_obj_class": gt_class,
                "gt_obj_class_score": float(object_label.obj_class_score),
            }
            if unmatched is None
            else dict(asdict(unmatched))
        )
        payload.update(
            {
                "sample_id": sample_id,
                "run_id": self._run_id,
                "bucket": "unmatched_gt",
                "class_name": gt_class,
                "predicted_class_name": "",
                "day": day_key,
            }
        )
        entry = _DatasetSampleEntry(
            sample_id=sample_id,
            bucket="unmatched_gt",
            class_name=self._safe_class_name(gt_class or "UNKNOWN_GT"),
            day_key=day_key,
            gt_label=object_label,
            match_payload=payload,
        )
        self._populate_entry_state(entry)
        return entry

    def _write_sample_entry(self, entry: _DatasetSampleEntry) -> None:
        if entry.gt_label is not None:
            gt_pcd_path = self._gt_pcd_path(entry)
            gt_json_path = self._gt_json_path(entry)
            self.pcd_writer.write(gt_pcd_path, entry.gt_label.points)
            self.manifest_writer.write_json(
                gt_json_path,
                {
                    "sample_id": entry.sample_id,
                    "run_id": self._run_id,
                    "object_id": int(entry.gt_label.object_id),
                    "timestamp_ns": int(entry.gt_label.timestamp_ns),
                    "frame_index": int(entry.gt_label.frame_index),
                    "sensor_name": str(entry.gt_label.sensor_name),
                    "obj_class": str(entry.gt_label.obj_class),
                    "obj_class_score": float(entry.gt_label.obj_class_score),
                    "point_count": int(len(entry.gt_label.points)),
                    "source_path": str(entry.gt_label.source_path),
                    "bucket": entry.bucket,
                    "class_name": entry.class_name,
                    "day": entry.day_key,
                    "pcd_path": str(gt_pcd_path.relative_to(self._require_root())),
                },
            )
            entry.match_payload["gt_path"] = str(gt_json_path.relative_to(self._require_root()))
            entry.match_payload["gt_pcd_path"] = str(gt_pcd_path.relative_to(self._require_root()))
        if entry.aggregate_result is not None:
            pred_pcd_path = self._pred_pcd_path(entry)
            pred_json_path = self._pred_json_path(entry)
            result = entry.aggregate_result
            self.pcd_writer.write(
                pred_pcd_path,
                result.points,
                intensity=result.intensity if entry.save_intensity else None,
                scalar_field_name="reflectivity",
            )
            self.manifest_writer.write_json(
                pred_json_path,
                {
                    "sample_id": entry.sample_id,
                    "run_id": self._run_id,
                    "track_id": int(result.track_id),
                    "status": str(result.status),
                    "selected_frame_ids": [int(frame_id) for frame_id in result.selected_frame_ids],
                    "point_count": int(len(result.points)),
                    "bucket": entry.bucket,
                    "class_name": entry.class_name,
                    "day": entry.day_key,
                    "pcd_path": str(pred_pcd_path.relative_to(self._require_root())),
                    "metrics": dict(result.metrics),
                },
            )
            entry.match_payload["pred_path"] = str(pred_json_path.relative_to(self._require_root()))
            entry.match_payload["pred_pcd_path"] = str(pred_pcd_path.relative_to(self._require_root()))
        match_json_path = self._match_json_path(entry)
        entry.match_payload["match_json_path"] = str(match_json_path.relative_to(self._require_root()))
        self.manifest_writer.write_json(match_json_path, dict(entry.match_payload))
        self._upsert_manifest_row(
            self._bucket_dir(entry) / "gt_matching" / "manifest.jsonl",
            entry.sample_id,
            dict(entry.match_payload),
        )

    def _remove_sample_entry(self, entry: _DatasetSampleEntry) -> None:
        for path in (
            self._gt_pcd_path(entry),
            self._gt_json_path(entry),
            self._pred_pcd_path(entry),
            self._pred_json_path(entry),
            self._match_json_path(entry),
        ):
            if path.exists():
                path.unlink()
                self._prune_empty_parents(path.parent)
        self._upsert_manifest_row(self._bucket_dir(entry) / "gt_matching" / "manifest.jsonl", entry.sample_id, None)

    def _write_active_object_list_manifest(self) -> None:
        self.manifest_writer.write_jsonl(self.object_list_manifest_path(self._require_root()), self._build_object_list_manifest_rows())

    def _build_object_list_manifest_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        current_entries_by_object_id = {
            int(entry.gt_label.object_id): entry
            for entry in self._current_sample_entries.values()
            if entry.gt_label is not None
        }
        for object_id, object_label in self._object_labels.items():
            entry = current_entries_by_object_id.get(int(object_id))
            day_key = self._date_key_from_timestamp(int(object_label.timestamp_ns))
            row = {
                "sample_id": self._sample_id(gt_object_id=int(object_id)),
                "object_id": int(object_label.object_id),
                "timestamp_ns": int(object_label.timestamp_ns),
                "frame_index": int(object_label.frame_index),
                "obj_class": str(object_label.obj_class),
                "bucket": "unmatched_gt",
                "class_name": self._safe_class_name(self._original_class_name(str(object_label.obj_class or "UNKNOWN_GT"))),
                "day": day_key,
            }
            if entry is not None:
                row.update(
                    {
                        "sample_id": entry.sample_id,
                        "bucket": entry.bucket,
                        "class_name": entry.class_name,
                        "day": entry.day_key,
                        "gt_path": str(self._gt_json_path(entry).relative_to(self._require_root())),
                    }
                )
            rows.append(row)
        return rows

    def _schedule_dataset_sample_flush(self) -> None:
        if self._sample_flush_suspended > 0:
            self._sample_flush_dirty = True
            return
        self._flush_dataset_samples()

    def _schedule_stats_flush(self) -> None:
        if not self._statistics_enabled:
            return
        if self._stats_flush_suspended > 0:
            self._stats_flush_dirty = True
            return
        self._flush_stats_dirs()

    def _populate_entry_state(self, entry: _DatasetSampleEntry) -> None:
        entry.gt_state = self._gt_state(entry.gt_label)
        entry.pred_state = self._pred_state(entry.aggregate_result, entry.save_intensity)
        entry.match_state = self._stable_json(self._sanitize_match_payload(entry.match_payload))

    def _sample_entries_equal(self, left: _DatasetSampleEntry, right: _DatasetSampleEntry) -> bool:
        return (
            left.sample_id == right.sample_id
            and left.bucket == right.bucket
            and left.class_name == right.class_name
            and left.day_key == right.day_key
            and bool(left.save_intensity) == bool(right.save_intensity)
            and left.gt_state == right.gt_state
            and left.pred_state == right.pred_state
            and left.match_state == right.match_state
        )

    def _gt_state(self, object_label: ObjectLabelData | None) -> tuple[Any, ...] | None:
        if object_label is None:
            return None
        return (
            int(object_label.object_id),
            int(object_label.timestamp_ns),
            int(object_label.frame_index),
            str(object_label.sensor_name),
            str(object_label.obj_class),
            float(object_label.obj_class_score),
            str(object_label.source_path),
            int(len(object_label.points)),
            tuple(np.asarray(object_label.points, dtype=np.float32).shape),
        )

    def _pred_state(self, result: AggregateResult | None, save_intensity: bool) -> tuple[Any, ...] | None:
        if result is None:
            return None
        intensity = result.intensity if bool(save_intensity) else None
        return (
            int(result.track_id),
            str(result.status),
            tuple(int(frame_id) for frame_id in result.selected_frame_ids),
            int(len(result.points)),
            tuple(np.asarray(result.points, dtype=np.float32).shape),
            None if intensity is None else int(len(intensity)),
            None if intensity is None else tuple(np.asarray(intensity, dtype=np.float32).shape),
            self._stable_json(dict(result.metrics)),
        )

    def _sanitize_match_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            str(key): value
            for key, value in payload.items()
            if str(key) not in {"gt_path", "gt_pcd_path", "pred_path", "pred_pcd_path", "match_json_path"}
        }

    def _stable_json(self, payload: Any) -> str:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=self._json_default)

    @staticmethod
    def _json_default(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    def _flush_stats_dirs(self) -> None:
        if not self._statistics_enabled:
            return
        days = self._current_day_keys()
        root = self._require_root()
        for day_key in sorted(self._current_stats_days - set(days)):
            shutil.rmtree(self._stats_dir(day_key), ignore_errors=True)
        for day_key in days:
            stats_dir = self._stats_dir(day_key)
            stats_dir.mkdir(parents=True, exist_ok=True)
            self._write_config_yaml(stats_dir / "config.snapshot.yaml")
            if self._tracker_states is not None:
                self.stats_writer.write_tracker_debug(stats_dir, self._tracker_states)
            if self._tracks is not None:
                self.stats_writer.write_tracks(stats_dir, self._tracks, self._aggregate_results_for_stats)
            if self._track_outcomes is not None:
                self.stats_writer.write_track_outcomes(stats_dir, self._track_outcomes)
            if self._class_stats is not None:
                self.stats_writer.write_class_stats(stats_dir, self._class_stats)
            if self._matches or self._unmatched_saved_tracks or self._unmatched_gt_objects or self._gt_summary:
                self.stats_writer.write_gt_matching(
                    stats_dir,
                    self._matches,
                    self._unmatched_saved_tracks,
                    self._unmatched_gt_objects,
                    self._gt_summary,
                )
            if self._summary is not None:
                self.stats_writer.write_summary(stats_dir, self._summary)
            if self._live_status_payload is not None:
                self.manifest_writer.write_json(stats_dir / "live_status.json", self._live_status_payload)
        self._current_stats_days = set(days)
        if self._live_status_payload is not None:
            self.manifest_writer.write_json(self.live_status_path(root), self._live_status_payload)

    def _current_day_keys(self) -> list[str]:
        day_keys = {entry.day_key for entry in self._current_sample_entries.values()}
        if not day_keys:
            day_keys.add(self._run_day_key)
        return sorted(day_keys)

    def _write_config_yaml(self, path: Path) -> None:
        if self._config_payload is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self._config_payload, handle, sort_keys=False)

    def _upsert_manifest_row(self, path: Path, sample_id: str, row: dict[str, Any] | None) -> None:
        existing_rows = []
        if path.exists():
            existing_rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        filtered = [existing for existing in existing_rows if str(existing.get("sample_id", "")) != str(sample_id)]
        if row is not None:
            filtered.append(dict(row))
        filtered.sort(key=lambda item: str(item.get("sample_id", "")))
        if not filtered:
            if path.exists():
                path.unlink()
                self._prune_empty_parents(path.parent)
            return
        self.manifest_writer.write_jsonl(path, filtered)

    def _bucket_dir(self, entry: _DatasetSampleEntry) -> Path:
        return self._require_root() / entry.class_name / entry.day_key / entry.bucket

    def _gt_pcd_path(self, entry: _DatasetSampleEntry) -> Path:
        return self._bucket_dir(entry) / "gt" / f"{entry.sample_id}.pcd"

    def _gt_json_path(self, entry: _DatasetSampleEntry) -> Path:
        return self._bucket_dir(entry) / "gt" / f"{entry.sample_id}.json"

    def _pred_pcd_path(self, entry: _DatasetSampleEntry) -> Path:
        return self._bucket_dir(entry) / "pred" / f"{entry.sample_id}.pcd"

    def _pred_json_path(self, entry: _DatasetSampleEntry) -> Path:
        return self._bucket_dir(entry) / "pred" / f"{entry.sample_id}.json"

    def _match_json_path(self, entry: _DatasetSampleEntry) -> Path:
        return self._bucket_dir(entry) / "gt_matching" / f"{entry.sample_id}.json"

    def _stats_dir(self, day_key: str) -> Path:
        return self._require_root() / "_stats" / day_key / self._run_id

    def _active_stats_dir(self) -> Path:
        return self._require_root() / "_stats" / "_active" / self._run_id

    def _sample_id(self, *, track_id: int | None = None, gt_object_id: int | None = None) -> str:
        parts = [f"run_{self._run_id}"]
        if track_id is not None:
            parts.append(aggregate_file_stem(int(track_id)))
        if gt_object_id is not None:
            parts.append(object_file_stem(int(gt_object_id)))
        return "__".join(parts)

    @staticmethod
    def _safe_class_name(value: str) -> str:
        text = str(value or "").strip()
        if not text:
            return "UNKNOWN_GT"
        return "".join(character if character.isalnum() or character in {"-", "_", "."} else "_" for character in text)

    @staticmethod
    def _original_class_name(value: str) -> str:
        return str(value or "").strip()

    def _date_key_from_timestamp(self, timestamp_ns: int) -> str:
        try:
            if int(timestamp_ns) > 0:
                return datetime.fromtimestamp(float(timestamp_ns) / 1_000_000_000.0).astimezone().strftime("%Y-%m-%d")
        except Exception:
            pass
        return self._run_day_key

    def _prune_empty_parents(self, path: Path) -> None:
        root = self._require_root()
        current = path
        while current != root and current.exists():
            try:
                current.rmdir()
            except OSError:
                break
            current = current.parent

    def _require_root(self) -> Path:
        if self._dataset_root is None:
            raise RuntimeError("DatasetArtifactWriter.prepare_run_dir() must be called before writing artifacts")
        return self._dataset_root
