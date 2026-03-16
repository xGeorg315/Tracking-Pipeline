from __future__ import annotations

import numpy as np

from tracking_pipeline.config.models import VisualizationConfig
from tracking_pipeline.domain.models import ActiveTrackState, AggregateResult, FrameTrackingState
from tracking_pipeline.infrastructure.visualization.open3d_replay_viewer import Open3DReplayViewer


def _state(playback_index: int, track_specs: list[tuple[int, np.ndarray]]) -> FrameTrackingState:
    active_tracks = []
    for track_id, center in track_specs:
        center = np.asarray(center, dtype=np.float32)
        points = np.asarray([center, center + np.array([0.1, 0.0, 0.0], dtype=np.float32)], dtype=np.float32)
        active_tracks.append(ActiveTrackState(track_id=int(track_id), points=points, center=center))
    return FrameTrackingState(
        frame_index=100 + playback_index,
        lane_points=np.zeros((0, 3), dtype=np.float32),
        detections=[],
        active_tracks=active_tracks,
    )


def test_frame_status_text_contains_frame_and_playback_position() -> None:
    text = Open3DReplayViewer._build_frame_status_text(playback_index=4, total_frames=12, frame_index=108)
    assert text == "Frame 108 (5/12)"


def test_build_save_events_uses_last_observed_frame() -> None:
    states = [
        _state(0, []),
        _state(1, [(7, np.array([1.0, 0.0, 0.0], dtype=np.float32))]),
        _state(2, []),
        _state(3, [(7, np.array([2.0, 0.0, 0.0], dtype=np.float32))]),
        _state(4, []),
    ]
    aggregate_results = {
        7: AggregateResult(
            track_id=7,
            points=np.ones((3, 3), dtype=np.float32),
            selected_frame_ids=[101, 103],
            status="saved",
            metrics={},
        )
    }

    events = Open3DReplayViewer._build_save_events(states, aggregate_results, duration_frames=20)
    active = Open3DReplayViewer._active_save_events(events, 3)

    assert len(active) == 1
    assert active[0].track_id == 7
    assert active[0].playback_start_index == 3
    assert active[0].frame_index == 103
    assert np.allclose(active[0].center, np.array([2.0, 0.0, 0.0], dtype=np.float32))


def test_save_event_duration_is_exactly_configured_number_of_frames() -> None:
    states = [_state(index, [(3, np.array([float(index), 0.0, 0.0], dtype=np.float32))] if index == 2 else []) for index in range(25)]
    aggregate_results = {
        3: AggregateResult(
            track_id=3,
            points=np.ones((2, 3), dtype=np.float32),
            selected_frame_ids=[102],
            status="saved",
            metrics={},
        )
    }

    events = Open3DReplayViewer._build_save_events(states, aggregate_results, duration_frames=20)

    assert Open3DReplayViewer._active_save_events(events, 2)[0].track_id == 3
    assert Open3DReplayViewer._active_save_events(events, 21)[0].track_id == 3
    assert Open3DReplayViewer._active_save_events(events, 22) == []


def test_save_hud_text_lists_active_saved_tracks() -> None:
    states = [
        _state(0, [(1, np.array([0.0, 0.0, 0.0], dtype=np.float32))]),
        _state(1, [(1, np.array([0.1, 0.0, 0.0], dtype=np.float32)), (2, np.array([1.0, 0.0, 0.0], dtype=np.float32))]),
    ]
    aggregate_results = {
        1: AggregateResult(track_id=1, points=np.ones((2, 3), dtype=np.float32), selected_frame_ids=[100], status="saved", metrics={}),
        2: AggregateResult(track_id=2, points=np.ones((2, 3), dtype=np.float32), selected_frame_ids=[101], status="saved", metrics={}),
    }
    events = Open3DReplayViewer._build_save_events(states, aggregate_results, duration_frames=20)
    active = Open3DReplayViewer._active_save_events(events, 1)

    assert Open3DReplayViewer._build_save_hud_text(active) == "Saved: Track 1\nSaved: Track 2"


def test_aggregate_toggle_does_not_change_active_save_events() -> None:
    viewer = Open3DReplayViewer(VisualizationConfig())
    frame = _state(0, [(11, np.array([0.0, 0.0, 0.0], dtype=np.float32))])
    aggregate_results = {
        11: AggregateResult(track_id=11, points=np.ones((2, 3), dtype=np.float32), selected_frame_ids=[100], status="saved", metrics={})
    }
    events = viewer._build_save_events([frame], aggregate_results, duration_frames=20)

    active_events = viewer._active_save_events(events, 0)
    assert {event.track_id for event in active_events} == {11}
    assert viewer._visible_aggregate_track_ids(frame, aggregate_results, show_aggregate=False) == set()
    assert viewer._visible_aggregate_track_ids(frame, aggregate_results, show_aggregate=True) == {11}
    assert {event.track_id for event in viewer._active_save_events(events, 0)} == {11}


def test_point_colors_use_grayscale_when_intensity_enabled() -> None:
    viewer = Open3DReplayViewer(VisualizationConfig(color_by_intensity=True))

    colors = viewer._point_colors(np.array([0.0, 0.5, 1.0], dtype=np.float32))

    assert colors is not None
    assert np.allclose(colors, np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]], dtype=np.float32))


def test_point_colors_are_disabled_when_flag_is_off() -> None:
    viewer = Open3DReplayViewer(VisualizationConfig(color_by_intensity=False))

    assert viewer._point_colors(np.array([0.2, 0.8], dtype=np.float32)) is None
