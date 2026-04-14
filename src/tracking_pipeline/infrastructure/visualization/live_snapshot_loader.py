from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from tracking_pipeline.config.models import PipelineConfig, VisualizationConfig
from tracking_pipeline.domain.models import DetectionDebugState, FrameTrackerDebug, TrackDebugState, TrackOutcomeDebug
from tracking_pipeline.domain.value_objects import LaneBox


@dataclass(slots=True)
class LiveTrackerFrameSnapshot:
    frame_index: int = -1
    cluster_metrics: dict[str, Any] = field(default_factory=dict)
    tracker_metrics: dict[str, Any] = field(default_factory=dict)
    tracker_debug: FrameTrackerDebug | None = None


@dataclass(slots=True)
class LiveRunContext:
    run_id: str
    active_dir: Path | None = None
    stats_dir: Path | None = None


@dataclass(slots=True)
class LiveSnapshot:
    dataset_root: Path
    run_id: str = ""
    active_dir: Path | None = None
    stats_dir: Path | None = None
    waiting: bool = False
    warnings: list[str] = field(default_factory=list)
    live_status: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    object_list_rows: list[dict[str, Any]] = field(default_factory=list)
    tracker_frame: LiveTrackerFrameSnapshot | None = None
    track_rows: dict[int, dict[str, Any]] = field(default_factory=dict)
    track_outcomes: dict[int, TrackOutcomeDebug] = field(default_factory=dict)
    lane_box: LaneBox | None = None
    visualization_config: VisualizationConfig | None = None
    require_track_exit: bool = True
    track_exit_edge_margin: float = 0.0
    track_exit_line_axis: str = "y"


class LiveSnapshotLoader:
    def __init__(self, dataset_root: Path, config: PipelineConfig):
        self.dataset_root = Path(dataset_root)
        self._fallback_lane_box = LaneBox.from_values(config.preprocessing.lane_box)
        self._fallback_visualization = config.visualization
        self._fallback_require_track_exit = bool(config.output.require_track_exit)
        self._fallback_track_exit_edge_margin = float(config.output.track_exit_edge_margin)
        self._fallback_track_exit_line_axis = str(config.aggregation.frame_selection_line_axis)
        self._cache: dict[str, tuple[int, Any]] = {}

    def load(self, run_id: str | None = None, force: bool = False) -> LiveSnapshot:
        context = self.resolve_run_context(run_id)
        if context is None:
            return LiveSnapshot(
                dataset_root=self.dataset_root,
                waiting=True,
                warnings=[],
                lane_box=self._fallback_lane_box,
                visualization_config=self._fallback_visualization,
                require_track_exit=self._fallback_require_track_exit,
                track_exit_edge_margin=self._fallback_track_exit_edge_margin,
                track_exit_line_axis=self._fallback_track_exit_line_axis,
            )

        warnings: list[str] = []
        active_dir = context.active_dir
        stats_dir = context.stats_dir

        config_payload = self._read_yaml(
            self._first_existing(
                [
                    None if stats_dir is None else stats_dir / "config.snapshot.yaml",
                    None if active_dir is None else active_dir / "config.snapshot.yaml",
                ]
            ),
            default=None,
            force=force,
            warnings=warnings,
            label="config snapshot",
        )
        lane_box = self._lane_box_from_payload(config_payload)
        visualization_config = self._visualization_from_payload(config_payload)
        require_track_exit = self._require_track_exit_from_payload(config_payload)
        track_exit_edge_margin = self._track_exit_edge_margin_from_payload(config_payload)
        track_exit_line_axis = self._track_exit_line_axis_from_payload(config_payload)

        live_status = self._read_json(
            None if active_dir is None else active_dir / "live_status.json",
            default={},
            force=force,
            warnings=warnings,
            label="live status",
        )
        object_list_rows = self._read_jsonl(
            None if active_dir is None else active_dir / "object_list_manifest.jsonl",
            parser=self._parse_json_row,
            default=[],
            force=force,
            warnings=warnings,
            label="object list manifest",
        )
        tracker_frame = self._read_jsonl(
            None if stats_dir is None else stats_dir / "tracker_debug.jsonl",
            parser=self._parse_tracker_debug_row,
            default=[],
            force=force,
            warnings=warnings,
            label="tracker debug",
        )
        track_rows = self._read_jsonl(
            None if stats_dir is None else stats_dir / "tracks.jsonl",
            parser=self._parse_track_row,
            default=[],
            force=force,
            warnings=warnings,
            label="tracks",
        )
        track_outcomes = self._read_jsonl(
            None if stats_dir is None else stats_dir / "track_outcomes.jsonl",
            parser=self._parse_track_outcome_row,
            default=[],
            force=force,
            warnings=warnings,
            label="track outcomes",
        )
        summary = self._read_json(
            None if stats_dir is None else stats_dir / "summary.json",
            default={},
            force=force,
            warnings=warnings,
            label="summary",
        )

        tracker_frame_value = tracker_frame[-1] if tracker_frame else None
        track_row_map = {
            int(row["track_id"]): row
            for row in track_rows
            if isinstance(row, dict) and row.get("track_id") is not None
        }
        track_outcome_map = {
            int(outcome.track_id): outcome
            for outcome in track_outcomes
        }

        return LiveSnapshot(
            dataset_root=self.dataset_root,
            run_id=context.run_id,
            active_dir=active_dir,
            stats_dir=stats_dir,
            waiting=False,
            warnings=warnings,
            live_status=live_status,
            summary=summary,
            object_list_rows=object_list_rows,
            tracker_frame=tracker_frame_value,
            track_rows=track_row_map,
            track_outcomes=track_outcome_map,
            lane_box=lane_box,
            visualization_config=visualization_config,
            require_track_exit=require_track_exit,
            track_exit_edge_margin=track_exit_edge_margin,
            track_exit_line_axis=track_exit_line_axis,
        )

    def resolve_run_context(self, run_id: str | None = None) -> LiveRunContext | None:
        if run_id:
            normalized_run_id = str(run_id).strip()
            if not normalized_run_id:
                return None
            active_dir = self._active_root() / normalized_run_id
            return LiveRunContext(
                run_id=normalized_run_id,
                active_dir=active_dir if active_dir.exists() else None,
                stats_dir=self._latest_stats_dir(normalized_run_id),
            )

        active_dirs = []
        active_root = self._active_root()
        if active_root.exists():
            for child in active_root.iterdir():
                if not child.is_dir():
                    continue
                live_status_path = child / "live_status.json"
                score_path = live_status_path if live_status_path.exists() else child
                try:
                    mtime_ns = int(score_path.stat().st_mtime_ns)
                except OSError:
                    continue
                active_dirs.append((mtime_ns, child))
        if not active_dirs:
            return None
        _, active_dir = max(active_dirs, key=lambda item: (item[0], item[1].name))
        run_id_value = str(active_dir.name)
        return LiveRunContext(
            run_id=run_id_value,
            active_dir=active_dir,
            stats_dir=self._latest_stats_dir(run_id_value),
        )

    def _latest_stats_dir(self, run_id: str) -> Path | None:
        stats_root = self.dataset_root / "_stats"
        if not stats_root.exists():
            return None
        candidates: list[tuple[int, Path]] = []
        pattern = f"*/{run_id}"
        for path in stats_root.glob(pattern):
            if not path.is_dir() or path.parent.name == "_active":
                continue
            try:
                mtime_ns = int(path.stat().st_mtime_ns)
            except OSError:
                continue
            candidates.append((mtime_ns, path))
        if not candidates:
            return None
        return max(candidates, key=lambda item: (item[0], item[1].as_posix()))[1]

    def _active_root(self) -> Path:
        return self.dataset_root / "_stats" / "_active"

    @staticmethod
    def _first_existing(paths: list[Path | None]) -> Path | None:
        for path in paths:
            if path is not None and path.exists():
                return path
        return None

    def _read_json(
        self,
        path: Path | None,
        *,
        default: Any,
        force: bool,
        warnings: list[str],
        label: str,
    ) -> Any:
        return self._read_cached(
            path,
            parser=lambda text: json.loads(text),
            default=default,
            force=force,
            warnings=warnings,
            label=label,
        )

    def _read_yaml(
        self,
        path: Path | None,
        *,
        default: Any,
        force: bool,
        warnings: list[str],
        label: str,
    ) -> Any:
        return self._read_cached(
            path,
            parser=lambda text: yaml.safe_load(text),
            default=default,
            force=force,
            warnings=warnings,
            label=label,
        )

    def _read_jsonl(
        self,
        path: Path | None,
        *,
        parser: Callable[[dict[str, Any]], Any],
        default: list[Any],
        force: bool,
        warnings: list[str],
        label: str,
    ) -> list[Any]:
        return self._read_cached(
            path,
            parser=lambda text: [parser(json.loads(line)) for line in text.splitlines() if line.strip()],
            default=default,
            force=force,
            warnings=warnings,
            label=label,
            preserve_on_empty=True,
        )

    def _read_cached(
        self,
        path: Path | None,
        *,
        parser: Callable[[str], Any],
        default: Any,
        force: bool,
        warnings: list[str],
        label: str,
        preserve_on_empty: bool = False,
    ) -> Any:
        if path is None:
            return default
        cache_key = str(path.resolve())
        cached = self._cache.get(cache_key)
        if not path.exists():
            if cached is not None:
                warnings.append(f"stale/partial snapshot: {label} temporarily unavailable")
                return cached[1]
            return default
        try:
            mtime_ns = int(path.stat().st_mtime_ns)
        except OSError:
            if cached is not None:
                warnings.append(f"stale/partial snapshot: {label} temporarily unavailable")
                return cached[1]
            return default
        if cached is not None and not force and cached[0] == mtime_ns:
            return cached[1]
        try:
            text = path.read_text(encoding="utf-8")
            if preserve_on_empty and not text.strip():
                if cached is not None:
                    warnings.append(f"stale/partial snapshot: {label} updating")
                    return cached[1]
                return default
            value = parser(text)
        except Exception:
            if cached is not None:
                warnings.append(f"stale/partial snapshot: {label} updating")
                return cached[1]
            return default
        self._cache[cache_key] = (mtime_ns, value)
        return value

    @staticmethod
    def _parse_json_row(payload: dict[str, Any]) -> dict[str, Any]:
        return dict(payload)

    @staticmethod
    def _parse_tracker_debug_row(payload: dict[str, Any]) -> LiveTrackerFrameSnapshot:
        tracker_debug_payload = dict(payload.get("tracker_debug") or {})
        track_states = [
            TrackDebugState(
                track_id=int(track_state.get("track_id", -1)),
                predicted_center=LiveSnapshotLoader._optional_array(track_state.get("predicted_center")),
                output_center=LiveSnapshotLoader._optional_array(track_state.get("output_center")),
                status=str(track_state.get("status", "")),
                matched_detection_id=LiveSnapshotLoader._optional_int(track_state.get("matched_detection_id")),
                gate_radius=LiveSnapshotLoader._optional_float(track_state.get("gate_radius")),
                missed_before=LiveSnapshotLoader._int_with_default(track_state.get("missed_before"), 0),
                missed_after=LiveSnapshotLoader._int_with_default(track_state.get("missed_after"), 0),
            )
            for track_state in tracker_debug_payload.get("track_states", [])
        ]
        detection_states = [
            DetectionDebugState(
                detection_id=int(detection_state.get("detection_id", -1)),
                center=np.asarray(detection_state.get("center", [0.0, 0.0, 0.0]), dtype=np.float32),
                status=str(detection_state.get("status", "")),
                matched_track_id=LiveSnapshotLoader._optional_int(detection_state.get("matched_track_id")),
                spawned_track_id=LiveSnapshotLoader._optional_int(detection_state.get("spawned_track_id")),
                spawn_suppressed=bool(detection_state.get("spawn_suppressed", False)),
                tracking_halo_only=bool(detection_state.get("tracking_halo_only", False)),
            )
            for detection_state in tracker_debug_payload.get("detection_states", [])
        ]
        tracker_debug = None
        if tracker_debug_payload:
            tracker_debug = FrameTrackerDebug(
                assignment_method=str(tracker_debug_payload.get("assignment_method", "")),
                track_states=track_states,
                detection_states=detection_states,
                matched_count=LiveSnapshotLoader._int_with_default(tracker_debug_payload.get("matched_count"), 0),
                missed_count=LiveSnapshotLoader._int_with_default(tracker_debug_payload.get("missed_count"), 0),
                spawned_count=LiveSnapshotLoader._int_with_default(tracker_debug_payload.get("spawned_count"), 0),
                suppressed_count=LiveSnapshotLoader._int_with_default(tracker_debug_payload.get("suppressed_count"), 0),
                halo_detection_count=LiveSnapshotLoader._int_with_default(
                    tracker_debug_payload.get("halo_detection_count"),
                    0,
                ),
            )
        return LiveTrackerFrameSnapshot(
            frame_index=LiveSnapshotLoader._int_with_default(payload.get("frame_index"), -1),
            cluster_metrics=dict(payload.get("cluster_metrics") or {}),
            tracker_metrics=dict(payload.get("tracker_metrics") or {}),
            tracker_debug=tracker_debug,
        )

    @staticmethod
    def _parse_track_row(payload: dict[str, Any]) -> dict[str, Any]:
        row = dict(payload)
        track_id = row.get("track_id")
        if track_id is not None:
            row["track_id"] = int(track_id)
        return row

    @staticmethod
    def _parse_track_outcome_row(payload: dict[str, Any]) -> TrackOutcomeDebug:
        return TrackOutcomeDebug(
            track_id=LiveSnapshotLoader._int_with_default(payload.get("track_id"), -1),
            status=str(payload.get("status", "")),
            decision_stage=str(payload.get("decision_stage", "")),
            decision_reason_code=str(payload.get("decision_reason_code", "")),
            decision_summary=str(payload.get("decision_summary", "")),
            last_frame_id=LiveSnapshotLoader._int_with_default(payload.get("last_frame_id"), -1),
            last_playback_index=LiveSnapshotLoader._int_with_default(payload.get("last_playback_index"), -1),
            last_center=LiveSnapshotLoader._optional_array(payload.get("last_center")),
            hit_count=LiveSnapshotLoader._int_with_default(payload.get("hit_count"), 0),
            age=LiveSnapshotLoader._int_with_default(payload.get("age"), 0),
            missed=LiveSnapshotLoader._int_with_default(payload.get("missed"), 0),
            ended_by_missed=bool(payload.get("ended_by_missed", False)),
            quality_score=LiveSnapshotLoader._optional_float(payload.get("quality_score")),
            selected_frame_ids=[int(frame_id) for frame_id in payload.get("selected_frame_ids", [])],
            tracker_debug_summary={
                str(key): int(value)
                for key, value in dict(payload.get("tracker_debug_summary") or {}).items()
            },
            predicted_class_id=LiveSnapshotLoader._optional_int(payload.get("predicted_class_id")),
            predicted_class_name=str(payload.get("predicted_class_name", "")),
            predicted_class_score=LiveSnapshotLoader._optional_float(payload.get("predicted_class_score")),
            classification_backend=str(payload.get("classification_backend", "")),
            classification_point_source=str(payload.get("classification_point_source", "")),
            classification_input_point_count=LiveSnapshotLoader._int_with_default(
                payload.get("classification_input_point_count"),
                0,
            ),
            gt_obj_class=str(payload.get("gt_obj_class", "")),
            gt_obj_class_score=LiveSnapshotLoader._optional_float(payload.get("gt_obj_class_score")),
        )

    @staticmethod
    def _optional_array(value: Any) -> np.ndarray | None:
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.float32)
        if arr.size == 0:
            return None
        return arr

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value is None:
            return None
        return int(value)

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        if value is None:
            return None
        return float(value)

    @staticmethod
    def _int_with_default(value: Any, default: int) -> int:
        if value is None:
            return int(default)
        return int(value)

    def _lane_box_from_payload(self, payload: dict[str, Any] | None) -> LaneBox:
        if not isinstance(payload, dict):
            return self._fallback_lane_box
        values = payload.get("preprocessing", {}).get("lane_box")
        if isinstance(values, list) and len(values) == 6:
            return LaneBox.from_values(values)
        return self._fallback_lane_box

    def _visualization_from_payload(self, payload: dict[str, Any] | None) -> VisualizationConfig:
        if not isinstance(payload, dict):
            return self._fallback_visualization
        values = dict(payload.get("visualization") or {})
        return VisualizationConfig(
            enabled=bool(values.get("enabled", self._fallback_visualization.enabled)),
            color_by_intensity=bool(values.get("color_by_intensity", self._fallback_visualization.color_by_intensity)),
            show_full_frame_pcd=bool(values.get("show_full_frame_pcd", self._fallback_visualization.show_full_frame_pcd)),
            show_tracker_debug=bool(values.get("show_tracker_debug", self._fallback_visualization.show_tracker_debug)),
            show_track_outcome_debug=bool(
                values.get("show_track_outcome_debug", self._fallback_visualization.show_track_outcome_debug)
            ),
            show_articulated_merge_debug=bool(
                values.get("show_articulated_merge_debug", self._fallback_visualization.show_articulated_merge_debug)
            ),
            max_points=int(values.get("max_points", self._fallback_visualization.max_points) or self._fallback_visualization.max_points),
            max_cluster_points=int(
                values.get("max_cluster_points", self._fallback_visualization.max_cluster_points)
                or self._fallback_visualization.max_cluster_points
            ),
            max_assoc_dist=float(
                values.get("max_assoc_dist", self._fallback_visualization.max_assoc_dist)
                or self._fallback_visualization.max_assoc_dist
            ),
        )

    def _require_track_exit_from_payload(self, payload: dict[str, Any] | None) -> bool:
        if not isinstance(payload, dict):
            return self._fallback_require_track_exit
        output = dict(payload.get("output") or {})
        return bool(output.get("require_track_exit", self._fallback_require_track_exit))

    def _track_exit_edge_margin_from_payload(self, payload: dict[str, Any] | None) -> float:
        if not isinstance(payload, dict):
            return self._fallback_track_exit_edge_margin
        output = dict(payload.get("output") or {})
        return float(output.get("track_exit_edge_margin", self._fallback_track_exit_edge_margin))

    def _track_exit_line_axis_from_payload(self, payload: dict[str, Any] | None) -> str:
        if not isinstance(payload, dict):
            return self._fallback_track_exit_line_axis
        aggregation = dict(payload.get("aggregation") or {})
        return str(aggregation.get("frame_selection_line_axis", self._fallback_track_exit_line_axis))
