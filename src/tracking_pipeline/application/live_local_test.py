from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Callable, Sequence

import numpy as np
import yaml

from tracking_pipeline.application.run_pipeline import run_pipeline
from tracking_pipeline.config.models import InputConfig, PipelineConfig
from tracking_pipeline.domain.models import AggregateResult, FrameData, Track
from tracking_pipeline.infrastructure.io.frame_segment import FrameSegmentWriter
from tracking_pipeline.infrastructure.logging.run_logger import get_run_logger


SubprocessRunner = Callable[[Sequence[str], Path], subprocess.CompletedProcess]


@dataclass(slots=True)
class AggregatePointCloudEntry:
    track_id: int
    points: np.ndarray
    selected_frame_ids: list[int]
    frame_ids: list[int]
    metrics: dict[str, object]
    source: str = ""


def live_local_test(
    config: PipelineConfig,
    project_root: Path,
    *,
    duration_sec: float = 15.0,
    compare_tolerance: float = 1e-5,
    live_aggregate_timeout_sec: float = 30.0,
    local_cpu_cores: int = 1,
    subprocess_runner: SubprocessRunner | None = None,
) -> dict[str, object] | None:
    logger = get_run_logger()
    observer = LiveLocalAggregateObserver(
        config=config,
        project_root=project_root,
        duration_sec=duration_sec,
        compare_tolerance=compare_tolerance,
        live_aggregate_timeout_sec=live_aggregate_timeout_sec,
        local_cpu_cores=local_cpu_cores,
        subprocess_runner=subprocess_runner,
        logger=logger,
    )
    try:
        run_pipeline(config, project_root, live_observer=observer)
    finally:
        observer.on_run_finished()
        observer.wait_for_report()
    return observer.report


class LiveLocalAggregateObserver:
    def __init__(
        self,
        *,
        config: PipelineConfig,
        project_root: Path,
        duration_sec: float,
        compare_tolerance: float,
        live_aggregate_timeout_sec: float,
        local_cpu_cores: int,
        subprocess_runner: SubprocessRunner | None,
        logger,
    ) -> None:
        self.config = config
        self.project_root = Path(project_root)
        self.duration_sec = max(0.001, float(duration_sec))
        self.compare_tolerance = max(0.0, float(compare_tolerance))
        self.live_aggregate_timeout_sec = max(0.0, float(live_aggregate_timeout_sec))
        self.local_cpu_cores = max(0, int(local_cpu_cores))
        self.subprocess_runner = subprocess_runner or _default_subprocess_runner
        self.logger = logger
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._segment_writer: FrameSegmentWriter | None = None
        self._root_dir: Path | None = None
        self._segment_dir: Path | None = None
        self._local_run_root: Path | None = None
        self._report_path: Path | None = None
        self._started_monotonic: float | None = None
        self._first_timestamp_ns: int | None = None
        self._first_frame_index: int | None = None
        self._last_frame_index: int | None = None
        self._last_timestamp_ns: int | None = None
        self._frame_count = 0
        self._capture_complete = False
        self._comparison_started = False
        self._comparison_thread: threading.Thread | None = None
        self._live_entries_by_track_id: dict[int, AggregatePointCloudEntry] = {}
        self.report: dict[str, object] | None = None

    def on_run_started(self, *, config: PipelineConfig, run_dir: Path) -> None:
        _ = config
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        root_dir = Path(run_dir) / "live_local_test" / timestamp
        segment_dir = root_dir / "segment"
        local_run_root = root_dir / "local_run"
        root_dir.mkdir(parents=True, exist_ok=True)
        local_run_root.mkdir(parents=True, exist_ok=True)
        writer = FrameSegmentWriter(segment_dir)
        with self._lock:
            self._root_dir = root_dir
            self._segment_dir = segment_dir
            self._local_run_root = local_run_root
            self._report_path = root_dir / "compare_report.json"
            self._segment_writer = writer
        self.logger.info("Live-local aggregate test recording segment: %s", segment_dir)

    def on_frame_read(self, *, frame: FrameData) -> None:
        writer_to_close = None
        start_compare = False
        with self._lock:
            if self._capture_complete:
                return
            writer = self._segment_writer
            if writer is None:
                return
            now = time.monotonic()
            if self._started_monotonic is None:
                self._started_monotonic = now
                self._first_timestamp_ns = int(frame.timestamp_ns)
                self._first_frame_index = int(frame.frame_index)
            elapsed = self._elapsed_sec(frame, now)
            if elapsed > self.duration_sec and self._frame_count > 0:
                self._capture_complete = True
                writer_to_close = writer
                self._segment_writer = None
                start_compare = True
            else:
                writer.write_frame(frame)
                self._frame_count += 1
                self._last_frame_index = int(frame.frame_index)
                self._last_timestamp_ns = int(frame.timestamp_ns)
                if elapsed >= self.duration_sec:
                    self._capture_complete = True
                    writer_to_close = writer
                    self._segment_writer = None
                    start_compare = True
        if writer_to_close is not None:
            writer_to_close.close()
        if start_compare:
            self._start_comparison_thread()

    def on_live_aggregates(self, *, tracks: dict[int, Track], aggregate_results: list[AggregateResult]) -> None:
        entries = _entries_from_live_results(tracks, aggregate_results)
        if not entries:
            return
        with self._condition:
            for entry in entries:
                self._live_entries_by_track_id[int(entry.track_id)] = entry
            self._condition.notify_all()

    def on_run_finished(self) -> None:
        writer_to_close = None
        should_start = False
        with self._lock:
            if not self._capture_complete and self._segment_writer is not None:
                self._capture_complete = True
                writer_to_close = self._segment_writer
                self._segment_writer = None
                should_start = True
        if writer_to_close is not None:
            writer_to_close.close()
        if should_start:
            self._start_comparison_thread()

    def wait_for_report(self, *, timeout_sec: float | None = None) -> dict[str, object] | None:
        thread = self._comparison_thread
        if thread is not None:
            thread.join(timeout=None if timeout_sec is None else max(0.0, float(timeout_sec)))
        return self.report

    def _elapsed_sec(self, frame: FrameData, now: float) -> float:
        first_timestamp_ns = self._first_timestamp_ns
        if first_timestamp_ns is not None and int(frame.timestamp_ns) >= int(first_timestamp_ns):
            return float(int(frame.timestamp_ns) - int(first_timestamp_ns)) / 1_000_000_000.0
        started_monotonic = self._started_monotonic
        return 0.0 if started_monotonic is None else max(0.0, float(now - started_monotonic))

    def _start_comparison_thread(self) -> None:
        with self._lock:
            if self._comparison_started:
                return
            self._comparison_started = True
        thread = threading.Thread(
            target=self._comparison_main,
            name="tracking_pipeline_live_local_compare",
            daemon=True,
        )
        self._comparison_thread = thread
        thread.start()

    def _comparison_main(self) -> None:
        try:
            report = self._build_report()
        except Exception as exc:  # pragma: no cover - defensive report path
            report = {
                "passed": False,
                "status": "error",
                "error": str(exc),
            }
        self.report = report
        report_path = self._report_path
        if report_path is not None:
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summary = dict(report.get("summary", {}) if isinstance(report.get("summary"), dict) else {})
        self.logger.info(
            "Live-local aggregate compare: status=%s compared=%s passed=%s failed=%s skipped=%s report=%s",
            report.get("status"),
            summary.get("compared", 0),
            summary.get("passed", 0),
            summary.get("failed", 0),
            summary.get("skipped", 0),
            report_path,
        )

    def _build_report(self) -> dict[str, object]:
        segment_dir = self._segment_dir
        local_run_root = self._local_run_root
        report_path = self._report_path
        if segment_dir is None or local_run_root is None or report_path is None:
            return {"passed": False, "status": "not_started", "summary": _empty_summary()}
        if self._frame_count <= 0:
            return {
                "passed": False,
                "status": "empty_segment",
                "segment_dir": str(segment_dir),
                "summary": _empty_summary(),
            }
        local_config_path = report_path.parent / "local_config.yaml"
        _write_local_config(
            self.config,
            local_config_path,
            segment_dir=segment_dir,
            local_run_root=local_run_root,
            local_cpu_cores=self.local_cpu_cores,
        )
        completed = self.subprocess_runner(
            [sys.executable, "-m", "tracking_pipeline.cli", "run", "-c", str(local_config_path)],
            self.project_root,
        )
        if int(completed.returncode) != 0:
            return {
                "passed": False,
                "status": "local_run_failed",
                "segment_dir": str(segment_dir),
                "local_config_path": str(local_config_path),
                "local_run_root": str(local_run_root),
                "stdout": str(completed.stdout),
                "stderr": str(completed.stderr),
                "summary": _empty_summary(),
            }
        local_output_dir = _find_local_output_dir(local_run_root)
        local_entries = _load_local_entries(local_output_dir)
        window_start = 0 if self._first_frame_index is None else int(self._first_frame_index)
        window_end = window_start if self._last_frame_index is None else int(self._last_frame_index)
        margin = max(0, int(self.config.tracking.max_missed))
        required_keys = _eligible_keys(local_entries, window_start, window_end, margin)
        self._wait_for_live_keys(required_keys, window_start, window_end, margin)
        with self._condition:
            live_entries = list(self._live_entries_by_track_id.values())
        report = compare_aggregate_pointclouds(
            live_entries,
            local_entries,
            window_start_frame=window_start,
            window_end_frame=window_end,
            incomplete_margin_frames=margin,
            tolerance=self.compare_tolerance,
        )
        report.update(
            {
                "segment_dir": str(segment_dir),
                "local_config_path": str(local_config_path),
                "local_output_dir": str(local_output_dir),
                "report_path": str(report_path),
                "duration_sec": float(self.duration_sec),
                "compare_tolerance": float(self.compare_tolerance),
                "live_aggregate_timeout_sec": float(self.live_aggregate_timeout_sec),
                "frame_window": {
                    "start_frame_index": int(window_start),
                    "end_frame_index": int(window_end),
                    "frame_count": int(self._frame_count),
                    "start_timestamp_ns": self._first_timestamp_ns,
                    "end_timestamp_ns": self._last_timestamp_ns,
                },
            }
        )
        return report

    def _wait_for_live_keys(
        self,
        required_keys: set[str],
        window_start: int,
        window_end: int,
        margin: int,
    ) -> None:
        if not required_keys or self.live_aggregate_timeout_sec <= 0.0:
            return
        deadline = time.monotonic() + float(self.live_aggregate_timeout_sec)
        with self._condition:
            while True:
                live_keys = _eligible_keys(
                    list(self._live_entries_by_track_id.values()),
                    window_start,
                    window_end,
                    margin,
                )
                if required_keys.issubset(live_keys):
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return
                self._condition.wait(timeout=min(0.25, remaining))


def compare_aggregate_pointclouds(
    live_entries: list[AggregatePointCloudEntry],
    local_entries: list[AggregatePointCloudEntry],
    *,
    window_start_frame: int,
    window_end_frame: int,
    incomplete_margin_frames: int,
    tolerance: float,
) -> dict[str, object]:
    live_by_key, live_skipped = _eligible_by_key(
        live_entries,
        int(window_start_frame),
        int(window_end_frame),
        int(incomplete_margin_frames),
        side="live",
    )
    local_by_key, local_skipped = _eligible_by_key(
        local_entries,
        int(window_start_frame),
        int(window_end_frame),
        int(incomplete_margin_frames),
        side="local",
    )
    comparisons: list[dict[str, object]] = []
    missing_live = []
    for key, local_entry in sorted(local_by_key.items()):
        live_entry = live_by_key.get(key)
        if live_entry is None:
            missing_live.append(_entry_summary(local_entry, key=key, reason="missing_live"))
            continue
        comparisons.append(_compare_entry_pair(key, live_entry, local_entry, float(tolerance)))
    missing_local = [
        _entry_summary(entry, key=key, reason="missing_local")
        for key, entry in sorted(live_by_key.items())
        if key not in local_by_key
    ]
    failed = [row for row in comparisons if not bool(row.get("passed"))]
    passed = [row for row in comparisons if bool(row.get("passed"))]
    skipped = live_skipped + local_skipped
    status = "passed"
    if missing_live or missing_local or failed:
        status = "failed"
    elif not comparisons:
        status = "no_comparable_aggregates"
    return {
        "passed": status == "passed",
        "status": status,
        "summary": {
            "compared": int(len(comparisons)),
            "passed": int(len(passed)),
            "failed": int(len(failed)),
            "missing_live": int(len(missing_live)),
            "missing_local": int(len(missing_local)),
            "skipped": int(len(skipped)),
        },
        "comparisons": comparisons,
        "missing_live": missing_live,
        "missing_local": missing_local,
        "skipped": skipped,
    }


def _eligible_by_key(
    entries: list[AggregatePointCloudEntry],
    window_start: int,
    window_end: int,
    incomplete_margin: int,
    *,
    side: str,
) -> tuple[dict[str, AggregatePointCloudEntry], list[dict[str, object]]]:
    eligible: dict[str, AggregatePointCloudEntry] = {}
    skipped = []
    for entry in entries:
        reason = _skip_reason(entry, window_start, window_end, incomplete_margin)
        key = _entry_key(entry)
        if reason:
            skipped.append(_entry_summary(entry, key=key, reason=reason, side=side))
            continue
        eligible[key] = entry
    return eligible, skipped


def _eligible_keys(
    entries: list[AggregatePointCloudEntry],
    window_start: int,
    window_end: int,
    incomplete_margin: int,
) -> set[str]:
    eligible, _ = _eligible_by_key(entries, window_start, window_end, incomplete_margin, side="")
    return set(eligible)


def _skip_reason(
    entry: AggregatePointCloudEntry,
    window_start: int,
    window_end: int,
    incomplete_margin: int,
) -> str:
    if str(entry.metrics.get("status", "saved")) != "saved":
        return "not_saved"
    frame_ids = _entry_frame_ids(entry)
    if frame_ids:
        min_frame = min(frame_ids)
        max_frame = max(frame_ids)
        if min_frame < int(window_start) or max_frame > int(window_end):
            return "outside_window"
        cutoff = int(window_end) - max(0, int(incomplete_margin)) + 1
        if max_frame >= cutoff:
            return "window_incomplete"
    return ""


def _entry_frame_ids(entry: AggregatePointCloudEntry) -> list[int]:
    if entry.frame_ids:
        return [int(frame_id) for frame_id in entry.frame_ids]
    return [int(frame_id) for frame_id in entry.selected_frame_ids]


def _entry_key(entry: AggregatePointCloudEntry) -> str:
    gt_object_id = entry.metrics.get("gt_object_id")
    if gt_object_id is not None:
        return f"gt:{int(gt_object_id)}"
    return f"track:{int(entry.track_id)}"


def _compare_entry_pair(
    key: str,
    live_entry: AggregatePointCloudEntry,
    local_entry: AggregatePointCloudEntry,
    tolerance: float,
) -> dict[str, object]:
    live_points = _canonical_points(live_entry.points)
    local_points = _canonical_points(local_entry.points)
    row: dict[str, object] = {
        "key": key,
        "track_id_live": int(live_entry.track_id),
        "track_id_local": int(local_entry.track_id),
        "point_count_live": int(len(live_points)),
        "point_count_local": int(len(local_points)),
        "selected_frame_ids_live": [int(frame_id) for frame_id in live_entry.selected_frame_ids],
        "selected_frame_ids_local": [int(frame_id) for frame_id in local_entry.selected_frame_ids],
    }
    if len(live_points) != len(local_points):
        row.update({"passed": False, "reason": "point_count_mismatch", "max_abs_diff": None, "rms_diff": None})
        return row
    if len(live_points) == 0:
        row.update({"passed": True, "reason": "", "max_abs_diff": 0.0, "rms_diff": 0.0})
        return row
    diff = np.asarray(live_points - local_points, dtype=np.float64)
    max_abs = float(np.max(np.abs(diff)))
    rms = float(np.sqrt(np.mean(np.square(diff))))
    row.update(
        {
            "passed": bool(max_abs <= float(tolerance)),
            "reason": "" if max_abs <= float(tolerance) else "coordinate_mismatch",
            "max_abs_diff": max_abs,
            "rms_diff": rms,
        }
    )
    return row


def _canonical_points(points: np.ndarray) -> np.ndarray:
    xyz = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(xyz) <= 1:
        return xyz.copy()
    order = np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))
    return xyz[order].copy()


def _entry_summary(
    entry: AggregatePointCloudEntry,
    *,
    key: str,
    reason: str,
    side: str | None = None,
) -> dict[str, object]:
    row = {
        "key": key,
        "track_id": int(entry.track_id),
        "reason": reason,
        "point_count": int(len(entry.points)),
        "selected_frame_ids": [int(frame_id) for frame_id in entry.selected_frame_ids],
        "frame_ids": [int(frame_id) for frame_id in entry.frame_ids],
        "source": str(entry.source),
    }
    if side:
        row["side"] = str(side)
    return row


def _entries_from_live_results(
    tracks: dict[int, Track],
    aggregate_results: list[AggregateResult],
) -> list[AggregatePointCloudEntry]:
    entries = []
    for result in aggregate_results:
        track = tracks.get(int(result.track_id))
        metrics = copy.deepcopy(dict(result.metrics))
        metrics["status"] = str(result.status)
        entries.append(
            AggregatePointCloudEntry(
                track_id=int(result.track_id),
                points=np.asarray(result.points, dtype=np.float32).copy(),
                selected_frame_ids=[int(frame_id) for frame_id in result.selected_frame_ids],
                frame_ids=[] if track is None else [int(frame_id) for frame_id in track.frame_ids],
                metrics=metrics,
                source="live",
            )
        )
    return entries


def _load_local_entries(output_dir: Path) -> list[AggregatePointCloudEntry]:
    aggregate_dir = Path(output_dir) / "aggregates"
    track_rows = _load_track_rows(Path(output_dir) / "tracks.jsonl")
    entries = []
    for metadata_path in sorted(aggregate_dir.glob("*.json")):
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        track_id = int(payload.get("track_id", 0))
        pcd_path = metadata_path.with_suffix(".pcd")
        if not pcd_path.is_file():
            continue
        metrics = dict(payload.get("metrics", {}) or {})
        metrics["status"] = str(payload.get("status", ""))
        track_row = track_rows.get(track_id, {})
        entries.append(
            AggregatePointCloudEntry(
                track_id=track_id,
                points=_read_pcd_points(pcd_path),
                selected_frame_ids=[int(frame_id) for frame_id in payload.get("selected_frame_ids", []) or []],
                frame_ids=[int(frame_id) for frame_id in track_row.get("frame_ids", []) or []],
                metrics=metrics,
                source=str(pcd_path),
            )
        )
    return entries


def _load_track_rows(path: Path) -> dict[int, dict[str, object]]:
    if not path.is_file():
        return {}
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        rows[int(payload.get("track_id", 0))] = payload
    return rows


def _read_pcd_points(path: Path) -> np.ndarray:
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(path))
    return np.asarray(pcd.points, dtype=np.float32)


def _write_local_config(
    source_config: PipelineConfig,
    config_path: Path,
    *,
    segment_dir: Path,
    local_run_root: Path,
    local_cpu_cores: int,
) -> None:
    local_config = copy.deepcopy(source_config)
    local_config.input = InputConfig(paths=[str(segment_dir)], format="frame_segment", qb2_live=None)
    local_config.output.mode = "run"
    local_config.output.root_dir = str(local_run_root)
    local_config.output.statistics_enabled = True
    local_config.output.final_full_recompute = True
    local_config.classification.enabled = False
    local_config.visualization.enabled = False
    local_config.visualization.live_web_enabled = False
    local_config.runtime.cpu_cores = int(local_cpu_cores)
    local_config.config_path = config_path
    config_path.write_text(yaml.safe_dump(local_config.to_dict(), sort_keys=False), encoding="utf-8")


def _find_local_output_dir(local_run_root: Path) -> Path:
    summaries = sorted(Path(local_run_root).glob("*/summary.json"), key=lambda path: path.stat().st_mtime)
    if not summaries:
        raise FileNotFoundError(f"Local run did not write summary.json under {local_run_root}")
    return summaries[-1].parent


def _default_subprocess_runner(command: Sequence[str], cwd: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    src_path = str(Path(cwd) / "src")
    existing_pythonpath = str(env.get("PYTHONPATH", ""))
    env["PYTHONPATH"] = src_path if not existing_pythonpath else f"{src_path}{os.pathsep}{existing_pythonpath}"
    return subprocess.run(
        list(command),
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _empty_summary() -> dict[str, int]:
    return {
        "compared": 0,
        "passed": 0,
        "failed": 0,
        "missing_live": 0,
        "missing_local": 0,
        "skipped": 0,
    }
