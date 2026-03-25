from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import open3d as o3d

from tracking_pipeline.config.models import VisualizationConfig
from tracking_pipeline.domain.models import (
    AggregateResult,
    ArticulatedMergeDebugEvent,
    Detection,
    FrameTrackerDebug,
    FrameTrackingState,
    TrackOutcomeDebug,
)
from tracking_pipeline.domain.rules import axis_to_index, track_exit_line_value
from tracking_pipeline.domain.value_objects import LaneBox
from tracking_pipeline.shared.geometry import clamp_visual_indices, clamp_visual_scalar, grayscale_from_intensity, np_to_o3d

gui = o3d.visualization.gui
rendering = o3d.visualization.rendering


@dataclass(slots=True, frozen=True)
class _OutcomeEvent:
    track_id: int
    status: str
    decision_reason_code: str
    decision_summary: str
    playback_start_index: int
    playback_end_index: int
    frame_index: int
    center: np.ndarray
    predicted_class_name: str = ""
    predicted_class_score: float | None = None
    gt_obj_class: str = ""


@dataclass(slots=True)
class _ReplayUiState:
    index: int = 0
    paused: bool = True
    show_aggregate: bool = False
    show_full_frame_pcd: bool = False
    show_tracker_debug: bool = False
    show_track_outcome_debug: bool = False
    show_articulated_merge_debug: bool = False
    dynamic_geometry_names: list[str] = field(default_factory=list)
    active_3d_labels: list[gui.Label3D] = field(default_factory=list)


class Open3DReplayViewer:
    _SAVE_INDICATOR_DURATION_FRAMES = 20
    _ARTICULATED_MERGE_OUTCOME_DURATION_FRAMES = 20
    _WINDOW_WIDTH = 1400
    _WINDOW_HEIGHT = 900
    _APP_INITIALIZED = False

    def __init__(
        self,
        config: VisualizationConfig,
        track_exit_edge_margin: float = 0.0,
        require_track_exit: bool = True,
        track_exit_line_axis: str = "y",
    ):
        self.config = config
        self.track_exit_edge_margin = float(track_exit_edge_margin)
        self.require_track_exit = bool(require_track_exit)
        self.track_exit_line_axis = str(track_exit_line_axis)

    def replay(
        self,
        states: list[FrameTrackingState],
        lane_box: LaneBox,
        aggregate_results: dict[int, AggregateResult],
        track_outcomes: dict[int, TrackOutcomeDebug],
        articulated_merge_debug_events: list[ArticulatedMergeDebugEvent],
    ) -> None:
        if not states:
            return

        app = gui.Application.instance
        if not self.__class__._APP_INITIALIZED:
            app.initialize()
            self.__class__._APP_INITIALIZED = True

        outcome_events = self._build_outcome_events(track_outcomes, self._SAVE_INDICATOR_DURATION_FRAMES)
        merge_outcome_events = self._build_articulated_merge_outcome_events(
            articulated_merge_debug_events,
            self._ARTICULATED_MERGE_OUTCOME_DURATION_FRAMES,
        )
        ui_state = _ReplayUiState(
            show_full_frame_pcd=bool(self.config.show_full_frame_pcd),
            show_tracker_debug=bool(self.config.show_tracker_debug),
            show_track_outcome_debug=bool(self.config.show_track_outcome_debug),
            show_articulated_merge_debug=bool(self.config.show_articulated_merge_debug),
        )

        window = app.create_window("Tracking Pipeline Replay", self._WINDOW_WIDTH, self._WINDOW_HEIGHT)
        scene_widget = gui.SceneWidget()
        scene_widget.scene = rendering.Open3DScene(window.renderer)
        scene_widget.scene.set_background(np.array([0.03, 0.03, 0.03, 1.0], dtype=np.float32))

        frame_label = gui.Label("")
        frame_label.text_color = gui.Color(0.95, 0.95, 0.95)
        frame_label.background_color = gui.Color(0.08, 0.08, 0.08, 0.75)

        save_label = gui.Label("")
        save_label.text_color = gui.Color(0.30, 1.00, 0.45)
        save_label.background_color = gui.Color(0.08, 0.08, 0.08, 0.75)
        save_label.visible = False

        tracker_debug_label = gui.Label("")
        tracker_debug_label.text_color = gui.Color(0.45, 0.90, 1.00)
        tracker_debug_label.background_color = gui.Color(0.08, 0.08, 0.08, 0.75)
        tracker_debug_label.visible = False

        merge_debug_label = gui.Label("")
        merge_debug_label.text_color = gui.Color(1.00, 0.92, 0.35)
        merge_debug_label.background_color = gui.Color(0.08, 0.08, 0.08, 0.75)
        merge_debug_label.visible = False

        window.add_child(scene_widget)
        window.add_child(frame_label)
        window.add_child(save_label)
        window.add_child(tracker_debug_label)
        window.add_child(merge_debug_label)

        lane_box_geometry = self._lane_box_lineset(lane_box)
        scene_widget.scene.add_geometry("lane_box", lane_box_geometry, self._line_material((1.0, 0.85, 0.10), line_width=2.0))
        track_exit_line_geometry = self._track_exit_lineset(lane_box)
        if track_exit_line_geometry is not None:
            scene_widget.scene.add_geometry(
                "track_exit_line",
                track_exit_line_geometry,
                self._line_material((0.25, 0.95, 1.0), line_width=1.5),
            )
        self._setup_initial_camera(scene_widget, lane_box)

        def on_layout(layout_context: gui.LayoutContext) -> None:
            rect = window.content_rect
            scene_widget.frame = rect
            margin = int(round(0.5 * layout_context.theme.font_size))

            frame_pref = frame_label.calc_preferred_size(layout_context, gui.Widget.Constraints())
            frame_width = frame_pref.width + 2 * margin
            frame_height = frame_pref.height + margin
            frame_label.frame = gui.Rect(
                rect.get_right() - frame_width - margin,
                rect.y + margin,
                frame_width,
                frame_height,
            )

            save_pref = save_label.calc_preferred_size(layout_context, gui.Widget.Constraints())
            save_width = min(save_pref.width + 2 * margin, max(180, int(rect.width * 0.35)))
            save_height = save_pref.height + margin
            save_label.frame = gui.Rect(
                rect.x + margin,
                rect.y + margin,
                save_width,
                save_height,
            )

            debug_pref = tracker_debug_label.calc_preferred_size(layout_context, gui.Widget.Constraints())
            debug_width = min(debug_pref.width + 2 * margin, max(220, int(rect.width * 0.38)))
            debug_height = debug_pref.height + margin
            debug_y = save_label.frame.get_bottom() + margin if save_label.visible else rect.y + margin
            tracker_debug_label.frame = gui.Rect(
                rect.x + margin,
                debug_y,
                debug_width,
                debug_height,
            )

            merge_pref = merge_debug_label.calc_preferred_size(layout_context, gui.Widget.Constraints())
            merge_width = min(merge_pref.width + 2 * margin, max(260, int(rect.width * 0.42)))
            merge_height = merge_pref.height + margin
            merge_y = tracker_debug_label.frame.get_bottom() + margin if tracker_debug_label.visible else debug_y
            merge_debug_label.frame = gui.Rect(
                rect.x + margin,
                merge_y,
                merge_width,
                merge_height,
            )

        def on_key(event: gui.KeyEvent) -> bool:
            if event.type != gui.KeyEvent.DOWN:
                return False
            if event.key == gui.KeyName.SPACE:
                ui_state.paused = not ui_state.paused
                return True
            if event.key == gui.KeyName.N:
                ui_state.index = min(ui_state.index + 1, len(states) - 1)
                self._render_current(
                    scene_widget,
                    frame_label,
                    save_label,
                    tracker_debug_label,
                    merge_debug_label,
                    ui_state,
                    states,
                    aggregate_results,
                    track_outcomes,
                    outcome_events,
                    articulated_merge_debug_events,
                    merge_outcome_events,
                )
                window.set_needs_layout()
                return True
            if event.key == gui.KeyName.B:
                ui_state.index = max(ui_state.index - 1, 0)
                self._render_current(
                    scene_widget,
                    frame_label,
                    save_label,
                    tracker_debug_label,
                    merge_debug_label,
                    ui_state,
                    states,
                    aggregate_results,
                    track_outcomes,
                    outcome_events,
                    articulated_merge_debug_events,
                    merge_outcome_events,
                )
                window.set_needs_layout()
                return True
            if event.key == gui.KeyName.A:
                ui_state.show_aggregate = not ui_state.show_aggregate
                self._render_current(
                    scene_widget,
                    frame_label,
                    save_label,
                    tracker_debug_label,
                    merge_debug_label,
                    ui_state,
                    states,
                    aggregate_results,
                    track_outcomes,
                    outcome_events,
                    articulated_merge_debug_events,
                    merge_outcome_events,
                )
                window.set_needs_layout()
                return True
            if event.key == gui.KeyName.P:
                ui_state.show_full_frame_pcd = not ui_state.show_full_frame_pcd
                self._render_current(
                    scene_widget,
                    frame_label,
                    save_label,
                    tracker_debug_label,
                    merge_debug_label,
                    ui_state,
                    states,
                    aggregate_results,
                    track_outcomes,
                    outcome_events,
                    articulated_merge_debug_events,
                    merge_outcome_events,
                )
                window.set_needs_layout()
                return True
            if event.key == gui.KeyName.T:
                ui_state.show_tracker_debug = not ui_state.show_tracker_debug
                self._render_current(
                    scene_widget,
                    frame_label,
                    save_label,
                    tracker_debug_label,
                    merge_debug_label,
                    ui_state,
                    states,
                    aggregate_results,
                    track_outcomes,
                    outcome_events,
                    articulated_merge_debug_events,
                    merge_outcome_events,
                )
                window.set_needs_layout()
                return True
            if event.key == gui.KeyName.F:
                ui_state.show_track_outcome_debug = not ui_state.show_track_outcome_debug
                self._render_current(
                    scene_widget,
                    frame_label,
                    save_label,
                    tracker_debug_label,
                    merge_debug_label,
                    ui_state,
                    states,
                    aggregate_results,
                    track_outcomes,
                    outcome_events,
                    articulated_merge_debug_events,
                    merge_outcome_events,
                )
                window.set_needs_layout()
                return True
            if event.key == gui.KeyName.M:
                ui_state.show_articulated_merge_debug = not ui_state.show_articulated_merge_debug
                self._render_current(
                    scene_widget,
                    frame_label,
                    save_label,
                    tracker_debug_label,
                    merge_debug_label,
                    ui_state,
                    states,
                    aggregate_results,
                    track_outcomes,
                    outcome_events,
                    articulated_merge_debug_events,
                    merge_outcome_events,
                )
                window.set_needs_layout()
                return True
            return False

        def on_tick() -> bool:
            if not ui_state.paused and ui_state.index < len(states) - 1:
                ui_state.index += 1
                self._render_current(
                    scene_widget,
                    frame_label,
                    save_label,
                    tracker_debug_label,
                    merge_debug_label,
                    ui_state,
                    states,
                    aggregate_results,
                    track_outcomes,
                    outcome_events,
                    articulated_merge_debug_events,
                    merge_outcome_events,
                )
                window.set_needs_layout()
                return True
            return False

        window.set_on_layout(on_layout)
        window.set_on_key(on_key)
        window.set_on_tick_event(on_tick)

        self._render_current(
            scene_widget,
            frame_label,
            save_label,
            tracker_debug_label,
            merge_debug_label,
            ui_state,
            states,
            aggregate_results,
            track_outcomes,
            outcome_events,
            articulated_merge_debug_events,
            merge_outcome_events,
        )
        window.set_needs_layout()
        app.run()

    def _render_current(
        self,
        scene_widget: gui.SceneWidget,
        frame_label: gui.Label,
        save_label: gui.Label,
        tracker_debug_label: gui.Label,
        merge_debug_label: gui.Label,
        ui_state: _ReplayUiState,
        states: list[FrameTrackingState],
        aggregate_results: dict[int, AggregateResult],
        track_outcomes: dict[int, TrackOutcomeDebug],
        outcome_events: dict[int, list[_OutcomeEvent]],
        articulated_merge_debug_events: list[ArticulatedMergeDebugEvent],
        merge_outcome_events: dict[int, list[ArticulatedMergeDebugEvent]],
    ) -> None:
        self._clear_dynamic_content(scene_widget, ui_state)

        frame = states[ui_state.index]
        frame_label.text = self._build_frame_status_text(ui_state.index, len(states), frame.frame_index)

        active_events = self._active_outcome_events(outcome_events, ui_state.index, ui_state.show_track_outcome_debug)
        save_label.text = self._build_outcome_hud_text(active_events)
        save_label.visible = bool(save_label.text)
        tracker_debug_label.text = self._build_tracker_debug_hud_text(frame.tracker_debug, ui_state.show_tracker_debug)
        tracker_debug_label.visible = bool(tracker_debug_label.text)
        active_merge_events = self._active_articulated_merge_events(
            articulated_merge_debug_events,
            ui_state.index,
            ui_state.show_articulated_merge_debug,
        )
        merge_debug_label.text = self._build_articulated_merge_hud_text(active_merge_events, ui_state.index, ui_state.show_articulated_merge_debug)
        merge_debug_label.visible = bool(merge_debug_label.text)

        full_frame_points, full_frame_intensity = self._visible_full_frame_points(frame, ui_state.show_full_frame_pcd)
        if len(full_frame_points) > 0:
            full_frame_colors = self._point_colors(full_frame_intensity)
            full_frame_pcd = np_to_o3d(full_frame_points, colors=full_frame_colors)
            self._add_geometry(
                scene_widget,
                ui_state,
                "full_frame_points",
                full_frame_pcd,
                self._point_material((1.0, 1.0, 1.0), point_size=1.5, use_vertex_colors=full_frame_colors is not None),
            )

        lane_indices = clamp_visual_indices(len(frame.lane_points), self.config.max_points)
        lane_points = np.asarray(frame.lane_points, dtype=np.float32)[lane_indices]
        if len(lane_points) > 0:
            lane_intensity = clamp_visual_scalar(frame.lane_intensity, lane_indices)
            lane_colors = self._point_colors(lane_intensity)
            lane_pcd = np_to_o3d(lane_points, colors=lane_colors)
            self._add_geometry(
                scene_widget,
                ui_state,
                "lane_points",
                lane_pcd,
                self._point_material((0.35, 0.35, 0.35), point_size=2.0, use_vertex_colors=lane_colors is not None),
            )

        active_outcome_by_track = {event.track_id: event for event in active_events}
        active_track_ids: set[int] = set()

        for active_track in frame.active_tracks:
            track_id = int(active_track.track_id)
            active_track_ids.add(track_id)
            outcome_event = active_outcome_by_track.get(track_id)
            aggregate_result = aggregate_results.get(track_id)
            is_saved_flash = outcome_event is not None and outcome_event.status == "saved"
            cluster_color = self._track_color(track_id)
            highlight_color = self._outcome_color(outcome_event) if outcome_event is not None else tuple(cluster_color)
            rendered_cluster_color = self._track_render_color(cluster_color, active_track.status)
            rendered_highlight_color = self._track_render_color(highlight_color, active_track.status)
            cluster_point_size = self._track_point_size(active_track.status)
            bbox_line_width = self._track_bbox_line_width(active_track.status, is_saved_flash)
            marker_radius = self._track_marker_radius(active_track.status, is_saved_flash)

            cluster_indices = clamp_visual_indices(len(active_track.points), self.config.max_cluster_points)
            cluster_vis = np.asarray(active_track.points, dtype=np.float32)[cluster_indices]
            if len(cluster_vis) > 0:
                cluster_intensity = clamp_visual_scalar(active_track.intensity, cluster_indices)
                cluster_colors = self._point_colors(cluster_intensity)
                cluster_pcd = np_to_o3d(cluster_vis, colors=cluster_colors)
                self._add_geometry(
                    scene_widget,
                    ui_state,
                    f"cluster_{track_id}",
                    cluster_pcd,
                    self._point_material(rendered_cluster_color, point_size=cluster_point_size, use_vertex_colors=cluster_colors is not None),
                )

                bbox = o3d.geometry.AxisAlignedBoundingBox(
                    min_bound=np.min(cluster_vis, axis=0),
                    max_bound=np.max(cluster_vis, axis=0),
                )
                self._add_geometry(
                    scene_widget,
                    ui_state,
                    f"bbox_{track_id}",
                    o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(bbox),
                    self._line_material(rendered_highlight_color, line_width=bbox_line_width),
                )

            marker = o3d.geometry.TriangleMesh.create_sphere(radius=marker_radius)
            marker.compute_vertex_normals()
            marker.translate(np.asarray(active_track.center, dtype=np.float64))
            self._add_geometry(
                scene_widget,
                ui_state,
                f"marker_{track_id}",
                marker,
                self._mesh_material(rendered_highlight_color),
            )

            if outcome_event is not None:
                label = scene_widget.add_3d_label(
                    np.asarray(active_track.center, dtype=np.float32),
                    self._outcome_label_text(outcome_event),
                )
                label_color = self._gui_color(self._outcome_color(outcome_event))
                label.color = label_color
                label.scale = 1.0
                ui_state.active_3d_labels.append(label)
                if ui_state.show_track_outcome_debug and outcome_event.status != "saved":
                    self._add_outcome_tail(scene_widget, ui_state, states, outcome_event)
            elif aggregate_result is not None:
                prediction_label = self._active_track_prediction_label_text(track_id, aggregate_result)
                if prediction_label:
                    label = scene_widget.add_3d_label(
                        np.asarray(active_track.center, dtype=np.float32),
                        prediction_label,
                    )
                    label.color = self._gui_color(rendered_highlight_color)
                    label.scale = 0.95
                    ui_state.active_3d_labels.append(label)

            if track_id in self._visible_aggregate_track_ids(frame, aggregate_results, ui_state.show_aggregate):
                aggregate_result = aggregate_results[track_id]
                aggregate_indices = clamp_visual_indices(len(aggregate_result.points), self.config.max_points)
                aggregate_points = np.asarray(aggregate_result.points, dtype=np.float32)[aggregate_indices]
                aggregate_intensity = clamp_visual_scalar(aggregate_result.intensity, aggregate_indices)
                aggregate_colors = self._point_colors(aggregate_intensity)
                aggregate_pcd = np_to_o3d(aggregate_points, colors=aggregate_colors)
                self._add_geometry(
                    scene_widget,
                    ui_state,
                    f"aggregate_{track_id}",
                    aggregate_pcd,
                    self._point_material((1.0, 0.20, 0.20), point_size=2.6, use_vertex_colors=aggregate_colors is not None),
                )
                aggregate_label = self._aggregate_prediction_label_text(track_id, aggregate_result)
                if aggregate_label:
                    label_center = (
                        np.mean(aggregate_points, axis=0, dtype=np.float32)
                        if len(aggregate_points) > 0
                        else np.asarray(active_track.center, dtype=np.float32)
                    )
                    label = scene_widget.add_3d_label(np.asarray(label_center, dtype=np.float32), aggregate_label)
                    label.color = gui.Color(1.00, 0.45, 0.45)
                    label.scale = 0.95
                    ui_state.active_3d_labels.append(label)

        if ui_state.show_tracker_debug and frame.tracker_debug is not None:
            self._render_tracker_debug_overlay(scene_widget, ui_state, frame)

        if ui_state.show_articulated_merge_debug:
            self._render_articulated_merge_overlay(scene_widget, ui_state, frame, active_merge_events, ui_state.index)
            for merge_event in self._active_articulated_merge_outcome_events(merge_outcome_events, ui_state.index, True):
                self._add_articulated_merge_outcome_beacon(scene_widget, ui_state, merge_event)

        for event in active_events:
            if event.track_id in active_track_ids:
                continue
            beacon = o3d.geometry.TriangleMesh.create_sphere(radius=0.10)
            beacon.compute_vertex_normals()
            beacon.translate(np.asarray(event.center, dtype=np.float64))
            self._add_geometry(
                scene_widget,
                ui_state,
                f"outcome_beacon_{event.track_id}",
                beacon,
                self._mesh_material(self._outcome_color(event)),
            )
            label = scene_widget.add_3d_label(np.asarray(event.center, dtype=np.float32), self._outcome_label_text(event))
            label.color = self._gui_color(self._outcome_color(event))
            label.scale = 1.0
            ui_state.active_3d_labels.append(label)
            if ui_state.show_track_outcome_debug and event.status != "saved":
                self._add_outcome_tail(scene_widget, ui_state, states, event)

    def _clear_dynamic_content(self, scene_widget: gui.SceneWidget, ui_state: _ReplayUiState) -> None:
        for name in ui_state.dynamic_geometry_names:
            if scene_widget.scene.has_geometry(name):
                scene_widget.scene.remove_geometry(name)
        ui_state.dynamic_geometry_names.clear()

        for label in ui_state.active_3d_labels:
            scene_widget.remove_3d_label(label)
        ui_state.active_3d_labels.clear()

    def _setup_initial_camera(self, scene_widget: gui.SceneWidget, lane_box: LaneBox) -> None:
        bounds = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=np.array([lane_box.x_min, lane_box.y_min, lane_box.z_min], dtype=np.float64),
            max_bound=np.array([lane_box.x_max, lane_box.y_max, lane_box.z_max], dtype=np.float64),
        )
        scene_widget.setup_camera(60.0, bounds, np.asarray(bounds.get_center(), dtype=np.float32))

    def _add_geometry(
        self,
        scene_widget: gui.SceneWidget,
        ui_state: _ReplayUiState,
        name: str,
        geometry: o3d.geometry.Geometry,
        material: rendering.MaterialRecord,
    ) -> None:
        if scene_widget.scene.has_geometry(name):
            scene_widget.scene.remove_geometry(name)
        scene_widget.scene.add_geometry(name, geometry, material)
        ui_state.dynamic_geometry_names.append(name)

    def _lane_box_lineset(self, lane_box: LaneBox) -> o3d.geometry.LineSet:
        aabb = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=np.array([lane_box.x_min, lane_box.y_min, lane_box.z_min], dtype=np.float64),
            max_bound=np.array([lane_box.x_max, lane_box.y_max, lane_box.z_max], dtype=np.float64),
        )
        lines = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(aabb)
        lines.paint_uniform_color((1.0, 0.85, 0.1))
        return lines

    def _track_exit_lineset(self, lane_box: LaneBox) -> o3d.geometry.LineSet | None:
        if not self.require_track_exit or self.track_exit_edge_margin <= 0.0:
            return None
        axis_idx = axis_to_index(self.track_exit_line_axis)
        line_value = track_exit_line_value(lane_box, axis_idx, edge_margin=self.track_exit_edge_margin)
        if axis_idx == 0:
            start = np.array([line_value, lane_box.y_min, lane_box.z_min], dtype=np.float64)
            end = np.array([line_value, lane_box.y_max, lane_box.z_min], dtype=np.float64)
        elif axis_idx == 1:
            start = np.array([lane_box.x_min, line_value, lane_box.z_min], dtype=np.float64)
            end = np.array([lane_box.x_max, line_value, lane_box.z_min], dtype=np.float64)
        else:
            start = np.array([lane_box.x_min, lane_box.y_min, line_value], dtype=np.float64)
            end = np.array([lane_box.x_max, lane_box.y_min, line_value], dtype=np.float64)
        if np.allclose(start, end):
            return None
        lines = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(np.asarray([start, end], dtype=np.float64)),
            lines=o3d.utility.Vector2iVector(np.array([[0, 1]], dtype=np.int32)),
        )
        lines.paint_uniform_color((0.25, 0.95, 1.0))
        return lines

    def _point_material(
        self,
        color: tuple[float, float, float],
        point_size: float,
        use_vertex_colors: bool = False,
    ) -> rendering.MaterialRecord:
        material = rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        base_color = (1.0, 1.0, 1.0) if use_vertex_colors else color
        material.base_color = [float(base_color[0]), float(base_color[1]), float(base_color[2]), 1.0]
        material.point_size = float(point_size)
        return material

    def _line_material(self, color: tuple[float, float, float], line_width: float) -> rendering.MaterialRecord:
        material = rendering.MaterialRecord()
        material.shader = "unlitLine"
        material.base_color = [float(color[0]), float(color[1]), float(color[2]), 1.0]
        material.line_width = float(line_width)
        return material

    def _mesh_material(self, color: tuple[float, float, float]) -> rendering.MaterialRecord:
        material = rendering.MaterialRecord()
        material.shader = "defaultLit"
        material.base_color = [float(color[0]), float(color[1]), float(color[2]), 1.0]
        return material

    @staticmethod
    def _build_frame_status_text(playback_index: int, total_frames: int, frame_index: int) -> str:
        return f"Frame {int(frame_index)} ({int(playback_index) + 1}/{int(total_frames)})"

    @staticmethod
    def _build_save_events(
        states: list[FrameTrackingState],
        aggregate_results: dict[int, AggregateResult],
        duration_frames: int,
    ) -> dict[int, list[_OutcomeEvent]]:
        saved_track_ids = {
            int(track_id)
            for track_id, result in aggregate_results.items()
            if result.status == "saved"
        }
        last_seen: dict[int, tuple[int, int, np.ndarray]] = {}
        for playback_index, state in enumerate(states):
            for active_track in state.active_tracks:
                track_id = int(active_track.track_id)
                if track_id not in saved_track_ids:
                    continue
                last_seen[track_id] = (
                    int(playback_index),
                    int(state.frame_index),
                    np.asarray(active_track.center, dtype=np.float32).copy(),
                )

        events_by_frame: dict[int, list[_OutcomeEvent]] = {}
        max_index = max(0, len(states) - 1)
        keep_frames = max(1, int(duration_frames))
        for track_id, (start_index, frame_index, center) in last_seen.items():
            end_index = min(start_index + keep_frames - 1, max_index)
            event = _OutcomeEvent(
                track_id=int(track_id),
                status="saved",
                decision_reason_code="saved",
                decision_summary="saved",
                playback_start_index=int(start_index),
                playback_end_index=int(end_index),
                frame_index=int(frame_index),
                center=np.asarray(center, dtype=np.float32),
            )
            for playback_index in range(start_index, end_index + 1):
                events_by_frame.setdefault(int(playback_index), []).append(event)

        for playback_index in list(events_by_frame.keys()):
            events_by_frame[playback_index] = sorted(events_by_frame[playback_index], key=lambda event: event.track_id)
        return events_by_frame

    @staticmethod
    def _active_save_events(save_events: dict[int, list[_OutcomeEvent]], playback_index: int) -> list[_OutcomeEvent]:
        return list(save_events.get(int(playback_index), []))

    @staticmethod
    def _build_save_hud_text(active_events: list[_OutcomeEvent]) -> str:
        if not active_events:
            return ""
        return "\n".join(f"Saved: Track {event.track_id}" for event in active_events)

    @staticmethod
    def _build_outcome_events(
        track_outcomes: dict[int, TrackOutcomeDebug],
        duration_frames: int,
    ) -> dict[int, list[_OutcomeEvent]]:
        events_by_frame: dict[int, list[_OutcomeEvent]] = {}
        keep_frames = max(1, int(duration_frames))
        for track_id, outcome in sorted(track_outcomes.items()):
            if int(outcome.last_playback_index) < 0 or outcome.last_center is None:
                continue
            start_index = int(outcome.last_playback_index)
            end_index = start_index + keep_frames - 1
            event = _OutcomeEvent(
                track_id=int(track_id),
                status=str(outcome.status),
                decision_reason_code=str(outcome.decision_reason_code),
                decision_summary=str(outcome.decision_summary),
                playback_start_index=start_index,
                playback_end_index=end_index,
                frame_index=int(outcome.last_frame_id),
                center=np.asarray(outcome.last_center, dtype=np.float32).copy(),
                predicted_class_name=str(outcome.predicted_class_name or ""),
                predicted_class_score=None
                if outcome.predicted_class_score is None
                else float(outcome.predicted_class_score),
                gt_obj_class=str(outcome.gt_obj_class or ""),
            )
            for playback_index in range(start_index, end_index + 1):
                events_by_frame.setdefault(int(playback_index), []).append(event)
        for playback_index in list(events_by_frame.keys()):
            events_by_frame[playback_index] = sorted(events_by_frame[playback_index], key=lambda event: event.track_id)
        return events_by_frame

    @staticmethod
    def _active_outcome_events(
        outcome_events: dict[int, list[_OutcomeEvent]],
        playback_index: int,
        show_track_outcome_debug: bool,
    ) -> list[_OutcomeEvent]:
        events = list(outcome_events.get(int(playback_index), []))
        if show_track_outcome_debug:
            return events
        return [event for event in events if event.status == "saved"]

    @staticmethod
    def _build_outcome_hud_text(active_events: list[_OutcomeEvent]) -> str:
        if not active_events:
            return ""
        lines: list[str] = []
        for event in active_events:
            if event.status == "saved":
                lines.append(f"Saved #{int(event.track_id)}")
            else:
                lines.append(f"Skip #{int(event.track_id)} {event.decision_summary}")
        return "\n".join(lines)

    @staticmethod
    def _build_tracker_debug_hud_text(debug: FrameTrackerDebug | None, enabled: bool) -> str:
        if not enabled or debug is None:
            return ""
        lines = [
            f"Tracker Debug ({debug.assignment_method})",
            f"matched: {int(debug.matched_count)}",
            f"missed: {int(debug.missed_count)}",
            f"spawned: {int(debug.spawned_count)}",
            f"suppressed: {int(debug.suppressed_count)}",
        ]
        if int(debug.halo_detection_count) > 0:
            lines.append(f"halo detections: {int(debug.halo_detection_count)}")
        return "\n".join(lines)

    @staticmethod
    def _build_articulated_merge_outcome_events(
        articulated_merge_debug_events: list[ArticulatedMergeDebugEvent],
        duration_frames: int,
    ) -> dict[int, list[ArticulatedMergeDebugEvent]]:
        events_by_frame: dict[int, list[ArticulatedMergeDebugEvent]] = {}
        keep_frames = max(1, int(duration_frames))
        for event in articulated_merge_debug_events:
            if int(event.playback_end_index) < 0 or event.center is None:
                continue
            start_index = int(event.playback_end_index)
            end_index = start_index + keep_frames - 1
            for playback_index in range(start_index, end_index + 1):
                events_by_frame.setdefault(int(playback_index), []).append(event)
        for playback_index in list(events_by_frame.keys()):
            events_by_frame[playback_index] = sorted(
                events_by_frame[playback_index],
                key=lambda event: (event.lead_track_id, event.rear_track_id),
            )
        return events_by_frame

    @staticmethod
    def _active_articulated_merge_events(
        articulated_merge_debug_events: list[ArticulatedMergeDebugEvent],
        playback_index: int,
        enabled: bool,
    ) -> list[ArticulatedMergeDebugEvent]:
        if not enabled:
            return []
        return [
            event
            for event in articulated_merge_debug_events
            if int(event.playback_start_index) <= int(playback_index) <= int(event.playback_end_index)
        ]

    @staticmethod
    def _active_articulated_merge_outcome_events(
        merge_outcome_events: dict[int, list[ArticulatedMergeDebugEvent]],
        playback_index: int,
        enabled: bool,
    ) -> list[ArticulatedMergeDebugEvent]:
        if not enabled:
            return []
        return list(merge_outcome_events.get(int(playback_index), []))

    @classmethod
    def _build_articulated_merge_hud_text(
        cls,
        active_events: list[ArticulatedMergeDebugEvent],
        playback_index: int,
        enabled: bool,
    ) -> str:
        if not enabled or not active_events:
            return ""
        lines: list[str] = []
        for event in active_events:
            status = "accept" if event.accepted else "reject"
            lines.append(f"Merge {int(event.lead_track_id)}+{int(event.rear_track_id)} {status} {event.rejection_reason}")
            lines.append(
                "gap full "
                f"{cls._metric_text(event.full_gap_mean)}/{cls._metric_text(event.full_gap_std)} "
                "tail "
                f"{cls._metric_text(event.tail_gap_mean)}/{cls._metric_text(event.tail_gap_std)}"
            )
        return "\n".join(lines)

    def _visible_full_frame_points(
        self,
        frame: FrameTrackingState,
        show_full_frame_pcd: bool,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if not show_full_frame_pcd or frame.full_frame_points is None or len(frame.full_frame_points) == 0:
            return np.zeros((0, 3), dtype=np.float32), None
        indices = clamp_visual_indices(len(frame.full_frame_points), self.config.max_points)
        points = np.asarray(frame.full_frame_points, dtype=np.float32)[indices]
        intensity = clamp_visual_scalar(frame.full_frame_intensity, indices)
        return points, intensity

    @staticmethod
    def _visible_aggregate_track_ids(
        frame: FrameTrackingState,
        aggregate_results: dict[int, AggregateResult],
        show_aggregate: bool,
    ) -> set[int]:
        if not show_aggregate:
            return set()
        active_track_ids = {int(active_track.track_id) for active_track in frame.active_tracks}
        return {
            track_id
            for track_id in active_track_ids
            if aggregate_results.get(track_id) is not None and aggregate_results[track_id].status == "saved"
        }

    @staticmethod
    def _outcome_color(event: _OutcomeEvent | None) -> tuple[float, float, float]:
        if event is None:
            return 1.0, 1.0, 1.0
        return {
            "saved": (0.20, 1.00, 0.35),
            "min_hits": (0.60, 0.60, 0.60),
            "track_exit": (1.00, 0.85, 0.10),
            "quality_threshold": (1.00, 0.25, 0.80),
            "min_saved_points": (1.00, 0.55, 0.15),
            "empty_selection": (1.00, 0.20, 0.20),
            "empty_prepared_chunks": (1.00, 0.20, 0.20),
            "empty_output": (1.00, 0.20, 0.20),
        }.get(str(event.decision_reason_code), (1.00, 0.20, 0.20))

    @staticmethod
    def _outcome_label_text(event: _OutcomeEvent) -> str:
        prediction_suffix = Open3DReplayViewer._classification_suffix(
            event.predicted_class_name,
            event.predicted_class_score,
            event.gt_obj_class,
        )
        if event.status == "saved":
            return f"saved #{int(event.track_id)}{prediction_suffix}"
        return f"skip #{int(event.track_id)} {event.decision_summary}{prediction_suffix}"

    @staticmethod
    def _prediction_suffix(class_name: str, score: float | None) -> str:
        if not class_name:
            return ""
        if score is None or not np.isfinite(float(score)):
            return f" {str(class_name)}"
        return f" {str(class_name)} {float(score):.2f}"

    @classmethod
    def _classification_suffix(cls, predicted_class_name: str, predicted_class_score: float | None, gt_obj_class: str) -> str:
        prediction_suffix = cls._prediction_suffix(predicted_class_name, predicted_class_score)
        gt_class = str(gt_obj_class or "")
        if prediction_suffix and gt_class:
            return f"{prediction_suffix} | gt:{gt_class}"
        if gt_class:
            return f" gt:{gt_class}"
        return prediction_suffix

    @classmethod
    def _active_track_prediction_label_text(cls, track_id: int, aggregate_result: AggregateResult | None) -> str:
        if aggregate_result is None:
            return ""
        metrics = aggregate_result.metrics or {}
        suffix = cls._classification_suffix(
            str(metrics.get("predicted_class_name", "")),
            None if metrics.get("predicted_class_score") is None else float(metrics.get("predicted_class_score")),
            str(metrics.get("gt_obj_class", "")),
        )
        if not suffix:
            return ""
        return f"#{int(track_id)}{suffix}"

    @classmethod
    def _aggregate_prediction_label_text(cls, track_id: int, aggregate_result: AggregateResult | None) -> str:
        if aggregate_result is None:
            return ""
        metrics = aggregate_result.metrics or {}
        suffix = cls._classification_suffix(
            str(metrics.get("predicted_class_name", "")),
            None if metrics.get("predicted_class_score") is None else float(metrics.get("predicted_class_score")),
            str(metrics.get("gt_obj_class", "")),
        )
        if not suffix:
            return ""
        return f"agg #{int(track_id)}{suffix}"

    @staticmethod
    def _articulated_merge_color(event: ArticulatedMergeDebugEvent, playback_index: int, final_only: bool = False) -> tuple[float, float, float]:
        if not final_only and int(playback_index) < int(event.playback_end_index):
            return 1.00, 0.85, 0.10
        return (0.20, 1.00, 0.35) if event.accepted else (1.00, 0.20, 0.20)

    @classmethod
    def _articulated_merge_overlay_text(cls, event: ArticulatedMergeDebugEvent) -> str:
        status = "accept" if event.accepted else "reject"
        return (
            f"{status} {int(event.lead_track_id)}+{int(event.rear_track_id)} {event.rejection_reason}\n"
            f"full {cls._metric_text(event.full_gap_mean)}/{cls._metric_text(event.full_gap_std)} "
            f"tail {cls._metric_text(event.tail_gap_mean)}/{cls._metric_text(event.tail_gap_std)}"
        )

    @staticmethod
    def _articulated_merge_outcome_label_text(event: ArticulatedMergeDebugEvent) -> str:
        prefix = "accept" if event.accepted else "reject"
        return f"{prefix} {int(event.lead_track_id)}+{int(event.rear_track_id)} {event.rejection_reason}"

    @staticmethod
    def _metric_text(value: float) -> str:
        if not np.isfinite(float(value)):
            return "-"
        return f"{float(value):.2f}"

    @staticmethod
    def _gui_color(color: tuple[float, float, float]) -> gui.Color:
        return gui.Color(float(color[0]), float(color[1]), float(color[2]))

    def _track_color(self, track_id: int) -> tuple[float, float, float]:
        rng = np.random.default_rng(int(track_id) * 9973 + 17)
        color = rng.uniform(0.2, 1.0, size=3)
        color = color / np.max(color)
        return float(color[0]), float(color[1]), float(color[2])

    @staticmethod
    def _track_render_color(color: tuple[float, float, float], status: str) -> tuple[float, float, float]:
        if status != "predicted":
            return float(color[0]), float(color[1]), float(color[2])
        return tuple(float(channel) * 0.6 for channel in color)

    @staticmethod
    def _track_point_size(status: str) -> float:
        return 2.4 if status == "predicted" else 3.0

    @staticmethod
    def _track_bbox_line_width(status: str, is_saved_flash: bool) -> float:
        if status == "predicted":
            return 1.6 if is_saved_flash else 1.2
        return 3.0 if is_saved_flash else 2.0

    @staticmethod
    def _track_marker_radius(status: str, is_saved_flash: bool) -> float:
        if status == "predicted":
            return 0.11 if is_saved_flash else 0.09
        return 0.14 if is_saved_flash else 0.12

    def _point_colors(self, intensity: np.ndarray | None) -> np.ndarray | None:
        if not self.config.color_by_intensity:
            return None
        return grayscale_from_intensity(intensity)

    def _render_tracker_debug_overlay(self, scene_widget: gui.SceneWidget, ui_state: _ReplayUiState, frame: FrameTrackingState) -> None:
        debug = frame.tracker_debug
        if debug is None:
            return
        detections_by_id = {int(detection.detection_id): detection for detection in frame.detections}

        for track_state in debug.track_states:
            if track_state.status == "matched" and track_state.predicted_center is not None:
                self._add_debug_marker(
                    scene_widget,
                    ui_state,
                    f"debug_prediction_{track_state.track_id}",
                    track_state.predicted_center,
                    radius=0.08,
                    color=(0.20, 0.90, 1.00),
                )
                if track_state.matched_detection_id is not None:
                    detection = detections_by_id.get(int(track_state.matched_detection_id))
                    if detection is not None:
                        self._add_debug_line(
                            scene_widget,
                            ui_state,
                            f"debug_assignment_{track_state.track_id}",
                            track_state.predicted_center,
                            np.asarray(detection.center, dtype=np.float32),
                            color=(0.20, 1.00, 0.35),
                        )
            elif track_state.status == "missed" and track_state.predicted_center is not None:
                self._add_debug_marker(
                    scene_widget,
                    ui_state,
                    f"debug_missed_{track_state.track_id}",
                    track_state.predicted_center,
                    radius=0.10,
                    color=(1.00, 0.60, 0.15),
                    label=f"missed #{int(track_state.track_id)}",
                    label_color=gui.Color(1.00, 0.60, 0.15),
                )
            elif track_state.status == "spawned" and track_state.output_center is not None:
                self._add_debug_marker(
                    scene_widget,
                    ui_state,
                    f"debug_spawn_{track_state.track_id}",
                    track_state.output_center,
                    radius=0.10,
                    color=(0.20, 0.45, 1.00),
                    label=f"spawn #{int(track_state.track_id)}",
                    label_color=gui.Color(0.20, 0.45, 1.00),
                )

        for detection_state in debug.detection_states:
            detection = detections_by_id.get(int(detection_state.detection_id))
            if detection_state.status == "spawn_suppressed":
                self._add_debug_marker(
                    scene_widget,
                    ui_state,
                    f"debug_suppressed_{detection_state.detection_id}",
                    detection_state.center,
                    radius=0.09,
                    color=(1.00, 0.20, 0.20),
                    label=f"suppressed d{int(detection_state.detection_id)}",
                    label_color=gui.Color(1.00, 0.20, 0.20),
                )
                if detection is not None and len(detection.points) > 0:
                    self._add_detection_bbox(
                        scene_widget,
                        ui_state,
                        f"debug_suppressed_bbox_{detection_state.detection_id}",
                        detection,
                        color=(1.00, 0.20, 0.20),
                    )
            if detection_state.tracking_halo_only:
                self._add_debug_marker(
                    scene_widget,
                    ui_state,
                    f"debug_halo_{detection_state.detection_id}",
                    detection_state.center,
                    radius=0.06,
                    color=(0.95, 0.30, 1.00),
                    label=f"halo d{int(detection_state.detection_id)}",
                    label_color=gui.Color(0.95, 0.30, 1.00),
                )

    def _render_articulated_merge_overlay(
        self,
        scene_widget: gui.SceneWidget,
        ui_state: _ReplayUiState,
        frame: FrameTrackingState,
        active_events: list[ArticulatedMergeDebugEvent],
        playback_index: int,
    ) -> None:
        by_track_id = {int(active_track.track_id): active_track for active_track in frame.active_tracks}
        for event in active_events:
            lead_track = by_track_id.get(int(event.lead_track_id))
            rear_track = by_track_id.get(int(event.rear_track_id))
            if lead_track is None or rear_track is None:
                continue
            start = np.asarray(lead_track.center, dtype=np.float32)
            end = np.asarray(rear_track.center, dtype=np.float32)
            midpoint = 0.5 * (start + end)
            color = self._articulated_merge_color(event, playback_index)
            self._add_debug_line(
                scene_widget,
                ui_state,
                f"merge_pair_{int(event.lead_track_id)}_{int(event.rear_track_id)}",
                start,
                end,
                color=color,
                line_width=2.6,
            )
            self._add_debug_marker(
                scene_widget,
                ui_state,
                f"merge_midpoint_{int(event.lead_track_id)}_{int(event.rear_track_id)}",
                midpoint,
                radius=0.08,
                color=color,
                label=self._articulated_merge_overlay_text(event),
                label_color=self._gui_color(color),
            )

    def _add_articulated_merge_outcome_beacon(
        self,
        scene_widget: gui.SceneWidget,
        ui_state: _ReplayUiState,
        event: ArticulatedMergeDebugEvent,
    ) -> None:
        if event.center is None:
            return
        color = self._articulated_merge_color(event, int(event.playback_end_index), final_only=True)
        beacon = o3d.geometry.TriangleMesh.create_sphere(radius=0.08)
        beacon.compute_vertex_normals()
        beacon.translate(np.asarray(event.center, dtype=np.float64))
        self._add_geometry(
            scene_widget,
            ui_state,
            f"merge_outcome_{int(event.lead_track_id)}_{int(event.rear_track_id)}",
            beacon,
            self._mesh_material(color),
        )
        label = scene_widget.add_3d_label(np.asarray(event.center, dtype=np.float32), self._articulated_merge_outcome_label_text(event))
        label.color = self._gui_color(color)
        label.scale = 0.95
        ui_state.active_3d_labels.append(label)

    def _add_debug_marker(
        self,
        scene_widget: gui.SceneWidget,
        ui_state: _ReplayUiState,
        name: str,
        center: np.ndarray,
        radius: float,
        color: tuple[float, float, float],
        label: str | None = None,
        label_color: gui.Color | None = None,
    ) -> None:
        marker = o3d.geometry.TriangleMesh.create_sphere(radius=float(radius))
        marker.compute_vertex_normals()
        marker.translate(np.asarray(center, dtype=np.float64))
        self._add_geometry(scene_widget, ui_state, name, marker, self._mesh_material(color))
        if label:
            text_label = scene_widget.add_3d_label(np.asarray(center, dtype=np.float32), label)
            text_label.color = label_color or gui.Color(float(color[0]), float(color[1]), float(color[2]))
            text_label.scale = 0.9
            ui_state.active_3d_labels.append(text_label)

    def _add_debug_line(
        self,
        scene_widget: gui.SceneWidget,
        ui_state: _ReplayUiState,
        name: str,
        start: np.ndarray,
        end: np.ndarray,
        color: tuple[float, float, float],
        line_width: float = 2.0,
    ) -> None:
        line = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(
                np.asarray([np.asarray(start, dtype=np.float64), np.asarray(end, dtype=np.float64)], dtype=np.float64)
            ),
            lines=o3d.utility.Vector2iVector(np.array([[0, 1]], dtype=np.int32)),
        )
        line.paint_uniform_color(color)
        self._add_geometry(scene_widget, ui_state, name, line, self._line_material(color, line_width=float(line_width)))

    def _add_detection_bbox(
        self,
        scene_widget: gui.SceneWidget,
        ui_state: _ReplayUiState,
        name: str,
        detection: Detection,
        color: tuple[float, float, float],
    ) -> None:
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=np.asarray(detection.min_bound, dtype=np.float64),
            max_bound=np.asarray(detection.max_bound, dtype=np.float64),
        )
        self._add_geometry(
            scene_widget,
            ui_state,
            name,
            o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(bbox),
            self._line_material(color, line_width=2.0),
        )

    def _add_outcome_tail(
        self,
        scene_widget: gui.SceneWidget,
        ui_state: _ReplayUiState,
        states: list[FrameTrackingState],
        event: _OutcomeEvent,
    ) -> None:
        tail_points = self._trajectory_tail_points(states, event.track_id, event.playback_start_index, max_points=8)
        if len(tail_points) < 2:
            return
        points = np.asarray(tail_points, dtype=np.float64)
        lines = np.asarray([[index, index + 1] for index in range(len(points) - 1)], dtype=np.int32)
        line = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines),
        )
        color = self._outcome_color(event)
        line.paint_uniform_color(color)
        self._add_geometry(
            scene_widget,
            ui_state,
            f"outcome_tail_{event.track_id}",
            line,
            self._line_material(color, line_width=2.5),
        )

    @staticmethod
    def _trajectory_tail_points(
        states: list[FrameTrackingState],
        track_id: int,
        last_playback_index: int,
        max_points: int,
    ) -> np.ndarray:
        points: list[np.ndarray] = []
        end_index = min(int(last_playback_index), len(states) - 1)
        for playback_index in range(end_index, -1, -1):
            state = states[playback_index]
            active_track = next((track for track in state.active_tracks if int(track.track_id) == int(track_id)), None)
            if active_track is None:
                if points:
                    break
                continue
            center = np.asarray(active_track.center, dtype=np.float32).copy()
            if not np.isfinite(center).all():
                if points:
                    break
                continue
            if points and np.allclose(center, points[-1], atol=1e-6):
                continue
            points.append(center)
            if len(points) >= int(max_points):
                break
        if not points:
            return np.zeros((0, 3), dtype=np.float32)
        return np.asarray(list(reversed(points)), dtype=np.float32)
