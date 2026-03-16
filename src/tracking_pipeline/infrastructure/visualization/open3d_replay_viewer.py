from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import open3d as o3d

from tracking_pipeline.config.models import VisualizationConfig
from tracking_pipeline.domain.models import AggregateResult, FrameTrackingState
from tracking_pipeline.domain.value_objects import LaneBox
from tracking_pipeline.shared.geometry import clamp_visual_indices, clamp_visual_scalar, grayscale_from_intensity, np_to_o3d

gui = o3d.visualization.gui
rendering = o3d.visualization.rendering


@dataclass(slots=True, frozen=True)
class _SaveEvent:
    track_id: int
    playback_start_index: int
    playback_end_index: int
    frame_index: int
    center: np.ndarray


@dataclass(slots=True)
class _ReplayUiState:
    index: int = 0
    paused: bool = True
    show_aggregate: bool = False
    dynamic_geometry_names: list[str] = field(default_factory=list)
    active_3d_labels: list[gui.Label3D] = field(default_factory=list)


class Open3DReplayViewer:
    _SAVE_INDICATOR_DURATION_FRAMES = 20
    _WINDOW_WIDTH = 1400
    _WINDOW_HEIGHT = 900
    _APP_INITIALIZED = False

    def __init__(self, config: VisualizationConfig):
        self.config = config

    def replay(self, states: list[FrameTrackingState], lane_box: LaneBox, aggregate_results: dict[int, AggregateResult]) -> None:
        if not states:
            return

        app = gui.Application.instance
        if not self.__class__._APP_INITIALIZED:
            app.initialize()
            self.__class__._APP_INITIALIZED = True

        save_events = self._build_save_events(states, aggregate_results, self._SAVE_INDICATOR_DURATION_FRAMES)
        ui_state = _ReplayUiState()

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

        window.add_child(scene_widget)
        window.add_child(frame_label)
        window.add_child(save_label)

        lane_box_geometry = self._lane_box_lineset(lane_box)
        scene_widget.scene.add_geometry("lane_box", lane_box_geometry, self._line_material((1.0, 0.85, 0.10), line_width=2.0))
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

        def on_key(event: gui.KeyEvent) -> bool:
            if event.type != gui.KeyEvent.DOWN:
                return False
            if event.key == gui.KeyName.SPACE:
                ui_state.paused = not ui_state.paused
                return True
            if event.key == gui.KeyName.N:
                ui_state.index = min(ui_state.index + 1, len(states) - 1)
                self._render_current(scene_widget, frame_label, save_label, ui_state, states, aggregate_results, save_events)
                window.set_needs_layout()
                return True
            if event.key == gui.KeyName.B:
                ui_state.index = max(ui_state.index - 1, 0)
                self._render_current(scene_widget, frame_label, save_label, ui_state, states, aggregate_results, save_events)
                window.set_needs_layout()
                return True
            if event.key == gui.KeyName.A:
                ui_state.show_aggregate = not ui_state.show_aggregate
                self._render_current(scene_widget, frame_label, save_label, ui_state, states, aggregate_results, save_events)
                window.set_needs_layout()
                return True
            return False

        def on_tick() -> bool:
            if not ui_state.paused and ui_state.index < len(states) - 1:
                ui_state.index += 1
                self._render_current(scene_widget, frame_label, save_label, ui_state, states, aggregate_results, save_events)
                window.set_needs_layout()
                return True
            return False

        window.set_on_layout(on_layout)
        window.set_on_key(on_key)
        window.set_on_tick_event(on_tick)

        self._render_current(scene_widget, frame_label, save_label, ui_state, states, aggregate_results, save_events)
        window.set_needs_layout()
        app.run()

    def _render_current(
        self,
        scene_widget: gui.SceneWidget,
        frame_label: gui.Label,
        save_label: gui.Label,
        ui_state: _ReplayUiState,
        states: list[FrameTrackingState],
        aggregate_results: dict[int, AggregateResult],
        save_events: dict[int, list[_SaveEvent]],
    ) -> None:
        self._clear_dynamic_content(scene_widget, ui_state)

        frame = states[ui_state.index]
        frame_label.text = self._build_frame_status_text(ui_state.index, len(states), frame.frame_index)

        active_events = self._active_save_events(save_events, ui_state.index)
        save_label.text = self._build_save_hud_text(active_events)
        save_label.visible = bool(save_label.text)

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

        active_saved_track_ids = {event.track_id for event in active_events}
        active_track_ids: set[int] = set()

        for active_track in frame.active_tracks:
            track_id = int(active_track.track_id)
            active_track_ids.add(track_id)
            is_saved_flash = track_id in active_saved_track_ids
            cluster_color = self._track_color(track_id)
            highlight_color = (0.20, 1.00, 0.35) if is_saved_flash else tuple(cluster_color)

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
                    self._point_material(cluster_color, point_size=3.0, use_vertex_colors=cluster_colors is not None),
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
                    self._line_material(highlight_color, line_width=3.0 if is_saved_flash else 2.0),
                )

            marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.14 if is_saved_flash else 0.12)
            marker.compute_vertex_normals()
            marker.translate(np.asarray(active_track.center, dtype=np.float64))
            self._add_geometry(
                scene_widget,
                ui_state,
                f"marker_{track_id}",
                marker,
                self._mesh_material(highlight_color),
            )

            if is_saved_flash:
                label = scene_widget.add_3d_label(np.asarray(active_track.center, dtype=np.float32), f"saved #{track_id}")
                label.color = gui.Color(0.20, 1.00, 0.35)
                label.scale = 1.0
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

        for event in active_events:
            if event.track_id in active_track_ids:
                continue
            beacon = o3d.geometry.TriangleMesh.create_sphere(radius=0.10)
            beacon.compute_vertex_normals()
            beacon.translate(np.asarray(event.center, dtype=np.float64))
            self._add_geometry(
                scene_widget,
                ui_state,
                f"save_beacon_{event.track_id}",
                beacon,
                self._mesh_material((0.20, 1.00, 0.35)),
            )
            label = scene_widget.add_3d_label(np.asarray(event.center, dtype=np.float32), f"saved #{event.track_id}")
            label.color = gui.Color(0.20, 1.00, 0.35)
            label.scale = 1.0
            ui_state.active_3d_labels.append(label)

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
    ) -> dict[int, list[_SaveEvent]]:
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

        events_by_frame: dict[int, list[_SaveEvent]] = {}
        max_index = max(0, len(states) - 1)
        keep_frames = max(1, int(duration_frames))
        for track_id, (start_index, frame_index, center) in last_seen.items():
            end_index = min(start_index + keep_frames - 1, max_index)
            event = _SaveEvent(
                track_id=int(track_id),
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
    def _active_save_events(save_events: dict[int, list[_SaveEvent]], playback_index: int) -> list[_SaveEvent]:
        return list(save_events.get(int(playback_index), []))

    @staticmethod
    def _build_save_hud_text(active_events: list[_SaveEvent]) -> str:
        if not active_events:
            return ""
        return "\n".join(f"Saved: Track {event.track_id}" for event in active_events)

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

    def _track_color(self, track_id: int) -> tuple[float, float, float]:
        rng = np.random.default_rng(int(track_id) * 9973 + 17)
        color = rng.uniform(0.2, 1.0, size=3)
        color = color / np.max(color)
        return float(color[0]), float(color[1]), float(color[2])

    def _point_colors(self, intensity: np.ndarray | None) -> np.ndarray | None:
        if not self.config.color_by_intensity:
            return None
        return grayscale_from_intensity(intensity)
