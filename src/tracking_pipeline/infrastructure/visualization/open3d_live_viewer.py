from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np
import open3d as o3d

from tracking_pipeline.config.models import VisualizationConfig
from tracking_pipeline.domain.models import TrackDebugState
from tracking_pipeline.infrastructure.visualization.live_snapshot_loader import LiveSnapshot, LiveSnapshotLoader
from tracking_pipeline.infrastructure.visualization.open3d_replay_viewer import Open3DReplayViewer, _OutcomeEvent

gui = o3d.visualization.gui
rendering = o3d.visualization.rendering


@dataclass(slots=True)
class _LiveUiState:
    paused: bool = False
    show_tracker_debug: bool = False
    show_track_outcomes: bool = False
    show_help: bool = False
    dynamic_geometry_names: list[str] = field(default_factory=list)
    active_3d_labels: list[gui.Label3D] = field(default_factory=list)
    last_poll_monotonic: float = 0.0
    static_signature: tuple[object, ...] | None = None


class Open3DLiveViewer(Open3DReplayViewer):
    _WINDOW_WIDTH = 1400
    _WINDOW_HEIGHT = 900
    _POLL_INTERVAL_SEC = 1.0
    _HELP_TEXT = "Space pause  T tracker  F outcomes  R refresh  H help"

    def __init__(
        self,
        config: VisualizationConfig,
        loader: LiveSnapshotLoader,
        track_exit_edge_margin: float = 0.0,
        require_track_exit: bool = True,
        track_exit_line_axis: str = "y",
    ):
        super().__init__(
            config,
            track_exit_edge_margin=track_exit_edge_margin,
            require_track_exit=require_track_exit,
            track_exit_line_axis=track_exit_line_axis,
        )
        self.loader = loader

    def live_view(self, run_id: str | None = None) -> None:
        app = gui.Application.instance
        if not Open3DReplayViewer._APP_INITIALIZED:
            app.initialize()
            Open3DReplayViewer._APP_INITIALIZED = True

        initial_snapshot = self.loader.load(run_id=run_id, force=True)
        initial_config = initial_snapshot.visualization_config or self.config
        ui_state = _LiveUiState(
            paused=False,
            show_tracker_debug=bool(initial_config.show_tracker_debug),
            show_track_outcomes=bool(initial_config.show_track_outcome_debug),
        )

        window = app.create_window("Tracking Pipeline Live View", self._WINDOW_WIDTH, self._WINDOW_HEIGHT)
        scene_widget = gui.SceneWidget()
        scene_widget.scene = rendering.Open3DScene(window.renderer)
        scene_widget.scene.set_background(np.array([0.03, 0.03, 0.03, 1.0], dtype=np.float32))

        status_label = gui.Label("")
        status_label.text_color = gui.Color(0.95, 0.95, 0.95)
        status_label.background_color = gui.Color(0.08, 0.08, 0.08, 0.75)

        summary_label = gui.Label("")
        summary_label.text_color = gui.Color(0.30, 1.00, 0.45)
        summary_label.background_color = gui.Color(0.08, 0.08, 0.08, 0.75)

        tracker_debug_label = gui.Label("")
        tracker_debug_label.text_color = gui.Color(0.45, 0.90, 1.00)
        tracker_debug_label.background_color = gui.Color(0.08, 0.08, 0.08, 0.75)
        tracker_debug_label.visible = False

        help_label = gui.Label("")
        help_label.text_color = gui.Color(1.00, 0.92, 0.35)
        help_label.background_color = gui.Color(0.08, 0.08, 0.08, 0.75)
        help_label.visible = False

        window.add_child(scene_widget)
        window.add_child(status_label)
        window.add_child(summary_label)
        window.add_child(tracker_debug_label)
        window.add_child(help_label)

        current_snapshot = {"value": initial_snapshot}

        def on_layout(layout_context: gui.LayoutContext) -> None:
            rect = window.content_rect
            scene_widget.frame = rect
            margin = int(round(0.5 * layout_context.theme.font_size))

            status_pref = status_label.calc_preferred_size(layout_context, gui.Widget.Constraints())
            status_width = min(status_pref.width + 2 * margin, max(260, int(rect.width * 0.38)))
            status_height = status_pref.height + margin
            status_label.frame = gui.Rect(
                rect.get_right() - status_width - margin,
                rect.y + margin,
                status_width,
                status_height,
            )

            summary_pref = summary_label.calc_preferred_size(layout_context, gui.Widget.Constraints())
            summary_width = min(summary_pref.width + 2 * margin, max(260, int(rect.width * 0.42)))
            summary_height = summary_pref.height + margin
            summary_label.frame = gui.Rect(
                rect.x + margin,
                rect.y + margin,
                summary_width,
                summary_height,
            )

            tracker_pref = tracker_debug_label.calc_preferred_size(layout_context, gui.Widget.Constraints())
            tracker_width = min(tracker_pref.width + 2 * margin, max(240, int(rect.width * 0.38)))
            tracker_height = tracker_pref.height + margin
            tracker_debug_label.frame = gui.Rect(
                rect.x + margin,
                summary_label.frame.get_bottom() + margin,
                tracker_width,
                tracker_height,
            )

            help_pref = help_label.calc_preferred_size(layout_context, gui.Widget.Constraints())
            help_width = min(help_pref.width + 2 * margin, max(260, int(rect.width * 0.42)))
            help_height = help_pref.height + margin
            help_label.frame = gui.Rect(
                rect.x + margin,
                rect.get_bottom() - help_height - margin,
                help_width,
                help_height,
            )

        def refresh_snapshot(force: bool) -> bool:
            snapshot = self.loader.load(run_id=run_id, force=force)
            current_snapshot["value"] = snapshot
            self._render_current_snapshot(
                scene_widget,
                status_label,
                summary_label,
                tracker_debug_label,
                help_label,
                ui_state,
                snapshot,
            )
            window.set_needs_layout()
            return True

        def on_key(event: gui.KeyEvent) -> bool:
            if event.type != gui.KeyEvent.DOWN:
                return False
            if event.key == gui.KeyName.SPACE:
                ui_state.paused = not ui_state.paused
                return refresh_snapshot(force=False)
            if event.key == gui.KeyName.T:
                ui_state.show_tracker_debug = not ui_state.show_tracker_debug
                return refresh_snapshot(force=False)
            if event.key == gui.KeyName.F:
                ui_state.show_track_outcomes = not ui_state.show_track_outcomes
                return refresh_snapshot(force=False)
            if event.key == gui.KeyName.R:
                return refresh_snapshot(force=True)
            if event.key == gui.KeyName.H:
                ui_state.show_help = not ui_state.show_help
                return refresh_snapshot(force=False)
            return False

        def on_tick() -> bool:
            now = time.monotonic()
            if ui_state.paused:
                return False
            if now - float(ui_state.last_poll_monotonic) < float(self._POLL_INTERVAL_SEC):
                return False
            ui_state.last_poll_monotonic = float(now)
            return refresh_snapshot(force=False)

        window.set_on_layout(on_layout)
        window.set_on_key(on_key)
        window.set_on_tick_event(on_tick)

        self._render_current_snapshot(
            scene_widget,
            status_label,
            summary_label,
            tracker_debug_label,
            help_label,
            ui_state,
            current_snapshot["value"],
        )
        window.set_needs_layout()
        app.run()

    def _render_current_snapshot(
        self,
        scene_widget: gui.SceneWidget,
        status_label: gui.Label,
        summary_label: gui.Label,
        tracker_debug_label: gui.Label,
        help_label: gui.Label,
        ui_state: _LiveUiState,
        snapshot: LiveSnapshot,
    ) -> None:
        self._clear_dynamic_content(scene_widget, ui_state)
        self._update_static_scene(scene_widget, ui_state, snapshot)

        status_label.text = self._build_status_text(snapshot, paused=ui_state.paused)
        summary_label.text = self._build_summary_text(snapshot)
        tracker_debug_label.text = self._build_tracker_debug_text(snapshot, ui_state.show_tracker_debug)
        tracker_debug_label.visible = bool(tracker_debug_label.text)
        help_label.text = self._HELP_TEXT if ui_state.show_help else ""
        help_label.visible = bool(help_label.text)

        if ui_state.show_tracker_debug:
            self._render_tracker_overlay(scene_widget, ui_state, snapshot)
        if ui_state.show_track_outcomes:
            self._render_outcome_overlay(scene_widget, ui_state, snapshot)

    def _update_static_scene(self, scene_widget: gui.SceneWidget, ui_state: _LiveUiState, snapshot: LiveSnapshot) -> None:
        lane_box = snapshot.lane_box
        if lane_box is None:
            return
        self.require_track_exit = bool(snapshot.require_track_exit)
        self.track_exit_edge_margin = float(snapshot.track_exit_edge_margin)
        self.track_exit_line_axis = str(snapshot.track_exit_line_axis)
        signature = (
            float(lane_box.x_min),
            float(lane_box.x_max),
            float(lane_box.y_min),
            float(lane_box.y_max),
            float(lane_box.z_min),
            float(lane_box.z_max),
            bool(snapshot.require_track_exit),
            float(snapshot.track_exit_edge_margin),
            str(snapshot.track_exit_line_axis),
        )
        if signature == ui_state.static_signature:
            return
        ui_state.static_signature = signature
        if scene_widget.scene.has_geometry("lane_box"):
            scene_widget.scene.remove_geometry("lane_box")
        scene_widget.scene.add_geometry("lane_box", self._lane_box_lineset(lane_box), self._line_material((1.0, 0.85, 0.10), 2.0))
        if scene_widget.scene.has_geometry("track_exit_line"):
            scene_widget.scene.remove_geometry("track_exit_line")
        track_exit_geometry = self._track_exit_lineset(lane_box)
        if track_exit_geometry is not None:
            scene_widget.scene.add_geometry(
                "track_exit_line",
                track_exit_geometry,
                self._line_material((0.25, 0.95, 1.0), 1.5),
            )
        self._setup_initial_camera(scene_widget, lane_box)

    def _render_tracker_overlay(self, scene_widget: gui.SceneWidget, ui_state: _LiveUiState, snapshot: LiveSnapshot) -> None:
        tracker_frame = snapshot.tracker_frame
        if tracker_frame is None or tracker_frame.tracker_debug is None:
            return
        debug = tracker_frame.tracker_debug

        detection_by_id = {int(state.detection_id): state for state in debug.detection_states}
        for detection_state in debug.detection_states:
            color, radius = self._detection_marker_style(detection_state.status, detection_state.tracking_halo_only)
            label = None
            label_color = None
            if detection_state.status == "spawn_suppressed":
                label = f"suppressed d{int(detection_state.detection_id)}"
                label_color = self._gui_color(color)
            elif detection_state.tracking_halo_only:
                label = f"halo d{int(detection_state.detection_id)}"
                label_color = self._gui_color(color)
            self._add_debug_marker(
                scene_widget,
                ui_state,
                f"live_detection_{int(detection_state.detection_id)}",
                detection_state.center,
                radius=radius,
                color=color,
                label=label,
                label_color=label_color,
            )

        for track_state in debug.track_states:
            self._render_track_state(scene_widget, ui_state, track_state, detection_by_id)

    def _render_track_state(
        self,
        scene_widget: gui.SceneWidget,
        ui_state: _LiveUiState,
        track_state: TrackDebugState,
        detection_by_id: dict[int, object],
    ) -> None:
        track_id = int(track_state.track_id)
        status = str(track_state.status)
        predicted = None if track_state.predicted_center is None else np.asarray(track_state.predicted_center, dtype=np.float32)
        output = None if track_state.output_center is None else np.asarray(track_state.output_center, dtype=np.float32)

        if predicted is not None:
            predicted_color = (0.20, 0.90, 1.00) if status != "missed" else (1.00, 0.60, 0.15)
            predicted_label = None
            if status == "missed":
                predicted_label = f"missed #{track_id}"
            elif status == "matched":
                predicted_label = f"pred #{track_id}"
            self._add_debug_marker(
                scene_widget,
                ui_state,
                f"live_track_pred_{track_id}",
                predicted,
                radius=0.08,
                color=predicted_color,
                label=predicted_label,
                label_color=self._gui_color(predicted_color) if predicted_label else None,
            )

        if output is not None:
            output_color = {
                "matched": (0.20, 1.00, 0.35),
                "spawned": (0.20, 0.45, 1.00),
                "missed": (1.00, 0.60, 0.15),
            }.get(status, (0.90, 0.90, 0.90))
            output_label = None
            if status == "spawned":
                output_label = f"spawn #{track_id}"
            elif status == "matched":
                output_label = f"track #{track_id}"
            self._add_debug_marker(
                scene_widget,
                ui_state,
                f"live_track_out_{track_id}",
                output,
                radius=0.10,
                color=output_color,
                label=output_label,
                label_color=self._gui_color(output_color) if output_label else None,
            )

        if predicted is not None and output is not None:
            self._add_debug_line(
                scene_widget,
                ui_state,
                f"live_track_assoc_{track_id}",
                predicted,
                output,
                color=(0.20, 1.00, 0.35),
                line_width=2.2,
            )

        if status == "matched" and track_state.matched_detection_id is not None:
            detection_state = detection_by_id.get(int(track_state.matched_detection_id))
            if detection_state is not None and output is not None:
                detection_center = np.asarray(detection_state.center, dtype=np.float32)
                self._add_debug_line(
                    scene_widget,
                    ui_state,
                    f"live_track_detection_{track_id}",
                    output,
                    detection_center,
                    color=(0.35, 0.85, 1.00),
                    line_width=1.4,
                )

    def _render_outcome_overlay(self, scene_widget: gui.SceneWidget, ui_state: _LiveUiState, snapshot: LiveSnapshot) -> None:
        for event in self._build_outcome_events(snapshot):
            beacon = o3d.geometry.TriangleMesh.create_sphere(radius=0.08)
            beacon.compute_vertex_normals()
            beacon.translate(np.asarray(event.center, dtype=np.float64))
            color = self._outcome_color(event)
            self._add_geometry(
                scene_widget,
                ui_state,
                f"live_outcome_{int(event.track_id)}",
                beacon,
                self._mesh_material(color),
            )
            label = scene_widget.add_3d_label(np.asarray(event.center, dtype=np.float32), self._outcome_label_text(event))
            label.color = self._gui_color(color)
            label.scale = 1.0
            ui_state.active_3d_labels.append(label)

    @staticmethod
    def _detection_marker_style(status: str, halo_only: bool) -> tuple[tuple[float, float, float], float]:
        if halo_only:
            return (0.95, 0.30, 1.00), 0.06
        return {
            "matched": ((0.20, 1.00, 0.35), 0.05),
            "spawned": ((0.20, 0.45, 1.00), 0.07),
            "spawn_suppressed": ((1.00, 0.20, 0.20), 0.09),
        }.get(str(status), ((0.90, 0.90, 0.90), 0.05))

    @staticmethod
    def _build_outcome_events(snapshot: LiveSnapshot) -> list[_OutcomeEvent]:
        events: list[_OutcomeEvent] = []
        for track_id, outcome in sorted(snapshot.track_outcomes.items()):
            if outcome.last_center is None:
                continue
            events.append(
                _OutcomeEvent(
                    track_id=int(track_id),
                    status=str(outcome.status),
                    decision_reason_code=str(outcome.decision_reason_code),
                    decision_summary=str(outcome.decision_summary),
                    playback_start_index=int(outcome.last_playback_index),
                    playback_end_index=int(outcome.last_playback_index),
                    frame_index=int(outcome.last_frame_id),
                    center=np.asarray(outcome.last_center, dtype=np.float32).copy(),
                    predicted_class_name=str(outcome.predicted_class_name or ""),
                    predicted_class_score=None
                    if outcome.predicted_class_score is None
                    else float(outcome.predicted_class_score),
                    gt_obj_class=str(outcome.gt_obj_class or ""),
                )
            )
        return events

    @staticmethod
    def _build_status_text(snapshot: LiveSnapshot, paused: bool) -> str:
        if snapshot.waiting:
            state = "paused" if paused else "live"
            return f"Waiting for live run ({state})"
        live_status = snapshot.live_status
        phase = str(live_status.get("pipeline_phase", "unknown"))
        last_frame_value = live_status.get("last_processed_frame_index")
        processed_frames_value = live_status.get("processed_frames")
        last_frame_index = -1 if last_frame_value is None else int(last_frame_value)
        processed_frames = 0 if processed_frames_value is None else int(processed_frames_value)
        state = "paused" if paused else "live"
        return (
            f"Run {snapshot.run_id}\n"
            f"phase={phase} frame={last_frame_index} processed={processed_frames} refresh={state}"
        )

    @classmethod
    def _build_summary_text(cls, snapshot: LiveSnapshot) -> str:
        if snapshot.waiting:
            return f"Waiting for active snapshot under {snapshot.dataset_root}"
        live_status = snapshot.live_status
        summary = snapshot.summary
        lines = [
            f"active={int(live_status.get('active_track_count', 0) or 0)} "
            f"finished={int(live_status.get('finished_track_count', 0) or 0)} "
            f"saved={int(live_status.get('saved_aggregates', 0) or 0)}",
            f"gt={int(live_status.get('object_list_exported_count', len(snapshot.object_list_rows)) or 0)} "
            f"hz={float(live_status.get('processing_recent_hz', 0.0) or 0.0):.2f}/"
            f"{float(live_status.get('processing_total_hz', 0.0) or 0.0):.2f}",
        ]
        if summary:
            lines.append(
                "gt_match "
                f"matched={int(summary.get('gt_match_matched_count', 0) or 0)} "
                f"unmatched_gt={int(summary.get('gt_match_unmatched_gt_count', 0) or 0)}"
            )
        if snapshot.warnings:
            lines.append(snapshot.warnings[-1])
        return "\n".join(lines)

    @classmethod
    def _build_tracker_debug_text(cls, snapshot: LiveSnapshot, enabled: bool) -> str:
        if snapshot.tracker_frame is None:
            return ""
        return cls._build_tracker_debug_hud_text(snapshot.tracker_frame.tracker_debug, enabled)
