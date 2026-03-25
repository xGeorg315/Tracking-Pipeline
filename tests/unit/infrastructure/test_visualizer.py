from __future__ import annotations

import numpy as np

from tracking_pipeline.config.models import VisualizationConfig
from tracking_pipeline.domain.models import (
    ActiveTrackState,
    AggregateResult,
    ArticulatedMergeDebugEvent,
    DetectionDebugState,
    FrameTrackerDebug,
    FrameTrackingState,
    TrackOutcomeDebug,
    TrackDebugState,
)
from tracking_pipeline.infrastructure.visualization.open3d_replay_viewer import Open3DReplayViewer


def _state(
    playback_index: int,
    track_specs: list[tuple[int, np.ndarray]],
    tracker_debug: FrameTrackerDebug | None = None,
    full_frame_points: np.ndarray | None = None,
    full_frame_intensity: np.ndarray | None = None,
) -> FrameTrackingState:
    active_tracks = []
    for track_spec in track_specs:
        if len(track_spec) == 2:
            track_id, center = track_spec
            status = "matched"
        else:
            track_id, center, status = track_spec
        center = np.asarray(center, dtype=np.float32)
        points = np.asarray([center, center + np.array([0.1, 0.0, 0.0], dtype=np.float32)], dtype=np.float32)
        active_tracks.append(ActiveTrackState(track_id=int(track_id), points=points, center=center, status=str(status)))
    return FrameTrackingState(
        frame_index=100 + playback_index,
        lane_points=np.zeros((0, 3), dtype=np.float32),
        detections=[],
        active_tracks=active_tracks,
        full_frame_points=full_frame_points,
        full_frame_intensity=full_frame_intensity,
        tracker_debug=tracker_debug,
    )


def _merge_event(
    *,
    lead_track_id: int = 3,
    rear_track_id: int = 4,
    accepted: bool = True,
    rejection_reason: str = "tail_gap",
    playback_start_index: int = 2,
    playback_end_index: int = 5,
    center: np.ndarray | None = None,
) -> ArticulatedMergeDebugEvent:
    return ArticulatedMergeDebugEvent(
        lead_track_id=lead_track_id,
        rear_track_id=rear_track_id,
        accepted=accepted,
        rejection_reason=rejection_reason,
        playback_start_index=playback_start_index,
        playback_end_index=playback_end_index,
        full_gap_mean=2.43,
        full_gap_std=1.87,
        tail_gap_mean=1.07,
        tail_gap_std=0.24,
        tail_window_frame_count=5,
        mean_lateral_offset=0.49,
        mean_vertical_offset=0.16,
        center=None if center is None else np.asarray(center, dtype=np.float32),
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


def test_outcome_events_include_saved_and_skipped_tracks() -> None:
    viewer = Open3DReplayViewer(VisualizationConfig())
    track_outcomes = {
        1: TrackOutcomeDebug(
            track_id=1,
            status="saved",
            decision_stage="saved",
            decision_reason_code="saved",
            decision_summary="saved points=248",
            last_frame_id=101,
            last_playback_index=1,
            last_center=np.array([0.1, 0.0, 0.0], dtype=np.float32),
        ),
        2: TrackOutcomeDebug(
            track_id=2,
            status="skipped_quality_threshold",
            decision_stage="quality_gate",
            decision_reason_code="quality_threshold",
            decision_summary="quality 0.28<0.40",
            last_frame_id=101,
            last_playback_index=1,
            last_center=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        ),
    }

    events = viewer._build_outcome_events(track_outcomes, duration_frames=3)

    hidden_failures = viewer._active_outcome_events(events, playback_index=1, show_track_outcome_debug=False)
    all_events = viewer._active_outcome_events(events, playback_index=1, show_track_outcome_debug=True)

    assert [event.track_id for event in hidden_failures] == [1]
    assert [event.track_id for event in all_events] == [1, 2]


def test_outcome_hud_text_and_colors_cover_saved_and_failed_tracks() -> None:
    events = Open3DReplayViewer._build_outcome_events(
        {
            1: TrackOutcomeDebug(
                track_id=1,
                status="saved",
                decision_stage="saved",
                decision_reason_code="saved",
                decision_summary="saved points=248",
                last_frame_id=105,
                last_playback_index=5,
                last_center=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            ),
            3: TrackOutcomeDebug(
                track_id=3,
                status="skipped_min_saved_points",
                decision_stage="save_gate",
                decision_reason_code="min_saved_points",
                decision_summary="min_saved_points 84/180",
                last_frame_id=105,
                last_playback_index=5,
                last_center=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            ),
        },
        duration_frames=2,
    )
    active = Open3DReplayViewer._active_outcome_events(events, playback_index=5, show_track_outcome_debug=True)

    assert Open3DReplayViewer._build_outcome_hud_text(active) == "Saved #1\nSkip #3 min_saved_points 84/180"
    assert Open3DReplayViewer._outcome_color(active[0]) == (0.20, 1.00, 0.35)
    assert Open3DReplayViewer._outcome_color(active[1]) == (1.00, 0.55, 0.15)
    assert Open3DReplayViewer._outcome_label_text(active[1]) == "skip #3 min_saved_points 84/180"


def test_prediction_labels_include_class_and_score_for_outcomes_and_aggregates() -> None:
    events = Open3DReplayViewer._build_outcome_events(
        {
            7: TrackOutcomeDebug(
                track_id=7,
                status="saved",
                decision_stage="saved",
                decision_reason_code="saved",
                decision_summary="saved points=248",
                last_frame_id=105,
                last_playback_index=5,
                last_center=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                predicted_class_name="trailer",
                predicted_class_score=0.88,
                gt_obj_class="LKW-Anhaenger",
            )
        },
        duration_frames=2,
    )
    active = Open3DReplayViewer._active_outcome_events(events, playback_index=5, show_track_outcome_debug=True)
    aggregate_result = AggregateResult(
        track_id=7,
        points=np.ones((2, 3), dtype=np.float32),
        selected_frame_ids=[100],
        status="saved",
        metrics={
            "predicted_class_name": "trailer",
            "predicted_class_score": 0.88,
            "gt_obj_class": "LKW-Anhaenger",
        },
    )

    assert Open3DReplayViewer._outcome_label_text(active[0]) == "saved #7 trailer 0.88 | gt:LKW-Anhaenger"
    assert Open3DReplayViewer._active_track_prediction_label_text(7, aggregate_result) == "#7 trailer 0.88 | gt:LKW-Anhaenger"
    assert Open3DReplayViewer._aggregate_prediction_label_text(7, aggregate_result) == "agg #7 trailer 0.88 | gt:LKW-Anhaenger"


def test_gt_only_labels_render_without_prediction() -> None:
    events = Open3DReplayViewer._build_outcome_events(
        {
            8: TrackOutcomeDebug(
                track_id=8,
                status="saved",
                decision_stage="saved",
                decision_reason_code="saved",
                decision_summary="saved points=128",
                last_frame_id=106,
                last_playback_index=6,
                last_center=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                gt_obj_class="PKW",
            )
        },
        duration_frames=2,
    )
    active = Open3DReplayViewer._active_outcome_events(events, playback_index=6, show_track_outcome_debug=True)
    aggregate_result = AggregateResult(
        track_id=8,
        points=np.ones((2, 3), dtype=np.float32),
        selected_frame_ids=[101],
        status="saved",
        metrics={"gt_obj_class": "PKW"},
    )

    assert Open3DReplayViewer._outcome_label_text(active[0]) == "saved #8 gt:PKW"
    assert Open3DReplayViewer._active_track_prediction_label_text(8, aggregate_result) == "#8 gt:PKW"
    assert Open3DReplayViewer._aggregate_prediction_label_text(8, aggregate_result) == "agg #8 gt:PKW"


def test_tracker_debug_hud_text_lists_frame_decisions() -> None:
    debug = FrameTrackerDebug(
        assignment_method="hungarian",
        track_states=[
            TrackDebugState(track_id=1, status="matched"),
            TrackDebugState(track_id=2, status="missed"),
            TrackDebugState(track_id=3, status="spawned"),
        ],
        detection_states=[
            DetectionDebugState(detection_id=1, center=np.zeros((3,), dtype=np.float32), status="matched"),
            DetectionDebugState(detection_id=2, center=np.zeros((3,), dtype=np.float32), status="spawn_suppressed", tracking_halo_only=True),
        ],
        matched_count=1,
        missed_count=1,
        spawned_count=1,
        suppressed_count=1,
        halo_detection_count=1,
    )

    text = Open3DReplayViewer._build_tracker_debug_hud_text(debug, enabled=True)

    assert text == (
        "Tracker Debug (hungarian)\n"
        "matched: 1\n"
        "missed: 1\n"
        "spawned: 1\n"
        "suppressed: 1\n"
        "halo detections: 1"
    )


def test_tracker_debug_hud_text_is_empty_when_disabled() -> None:
    debug = FrameTrackerDebug(assignment_method="greedy", matched_count=1)

    assert Open3DReplayViewer._build_tracker_debug_hud_text(debug, enabled=False) == ""


def test_articulated_merge_events_and_outcomes_follow_playback_windows() -> None:
    event = _merge_event(center=np.array([2.0, 0.0, 0.0], dtype=np.float32))
    outcome_events = Open3DReplayViewer._build_articulated_merge_outcome_events([event], duration_frames=3)

    assert Open3DReplayViewer._active_articulated_merge_events([event], playback_index=1, enabled=True) == []
    assert Open3DReplayViewer._active_articulated_merge_events([event], playback_index=4, enabled=True) == [event]
    assert Open3DReplayViewer._active_articulated_merge_events([event], playback_index=4, enabled=False) == []
    assert Open3DReplayViewer._active_articulated_merge_outcome_events(outcome_events, playback_index=4, enabled=True) == []
    assert Open3DReplayViewer._active_articulated_merge_outcome_events(outcome_events, playback_index=5, enabled=True) == [event]
    assert Open3DReplayViewer._active_articulated_merge_outcome_events(outcome_events, playback_index=7, enabled=True) == [event]
    assert Open3DReplayViewer._active_articulated_merge_outcome_events(outcome_events, playback_index=8, enabled=True) == []


def test_articulated_merge_hud_and_labels_include_gap_metrics() -> None:
    event = _merge_event()
    reject_event = _merge_event(accepted=False, rejection_reason="gap_tail_std")

    hud = Open3DReplayViewer._build_articulated_merge_hud_text([event], playback_index=4, enabled=True)

    assert hud == "Merge 3+4 accept tail_gap\ngap full 2.43/1.87 tail 1.07/0.24"
    assert Open3DReplayViewer._build_articulated_merge_hud_text([event], playback_index=4, enabled=False) == ""
    assert Open3DReplayViewer._articulated_merge_overlay_text(event) == "accept 3+4 tail_gap\nfull 2.43/1.87 tail 1.07/0.24"
    assert Open3DReplayViewer._articulated_merge_outcome_label_text(reject_event) == "reject 3+4 gap_tail_std"
    assert Open3DReplayViewer._articulated_merge_color(event, playback_index=4) == (1.00, 0.85, 0.10)
    assert Open3DReplayViewer._articulated_merge_color(event, playback_index=5) == (0.20, 1.00, 0.35)
    assert Open3DReplayViewer._articulated_merge_color(reject_event, playback_index=5, final_only=True) == (1.00, 0.20, 0.20)


def test_aggregate_toggle_does_not_change_active_save_events() -> None:
    viewer = Open3DReplayViewer(VisualizationConfig())
    frame = _state(
        0,
        [(11, np.array([0.0, 0.0, 0.0], dtype=np.float32))],
        tracker_debug=FrameTrackerDebug(
            assignment_method="greedy",
            track_states=[TrackDebugState(track_id=11, status="spawned")],
            matched_count=0,
            missed_count=0,
            spawned_count=1,
            suppressed_count=0,
        ),
    )
    aggregate_results = {
        11: AggregateResult(track_id=11, points=np.ones((2, 3), dtype=np.float32), selected_frame_ids=[100], status="saved", metrics={})
    }
    events = viewer._build_save_events([frame], aggregate_results, duration_frames=20)

    active_events = viewer._active_save_events(events, 0)
    assert {event.track_id for event in active_events} == {11}
    assert viewer._visible_aggregate_track_ids(frame, aggregate_results, show_aggregate=False) == set()
    assert viewer._visible_aggregate_track_ids(frame, aggregate_results, show_aggregate=True) == {11}
    assert {event.track_id for event in viewer._active_save_events(events, 0)} == {11}
    assert "spawned: 1" in viewer._build_tracker_debug_hud_text(frame.tracker_debug, enabled=True)


def test_point_colors_use_grayscale_when_intensity_enabled() -> None:
    viewer = Open3DReplayViewer(VisualizationConfig(color_by_intensity=True))
    reflectivity = np.array([0.0, 1.0, 100.0, 10000.0], dtype=np.float32)

    colors = viewer._point_colors(reflectivity)
    display_values = np.log1p(reflectivity)
    lo = float(np.percentile(display_values, 5.0))
    hi = float(np.percentile(display_values, 95.0))
    expected = np.clip((display_values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

    assert colors is not None
    assert np.allclose(colors, np.repeat(expected[:, None], 3, axis=1), atol=1e-6)


def test_point_colors_clip_negative_and_nonfinite_reflectivity_for_display() -> None:
    viewer = Open3DReplayViewer(VisualizationConfig(color_by_intensity=True))

    colors = viewer._point_colors(np.array([-5.0, np.nan, np.inf, 3.0], dtype=np.float32))

    assert colors is not None
    assert colors.shape == (4, 3)
    assert np.all(np.isfinite(colors))
    assert np.all(colors >= 0.0)
    assert np.all(colors <= 1.0)


def test_point_colors_are_disabled_when_flag_is_off() -> None:
    viewer = Open3DReplayViewer(VisualizationConfig(color_by_intensity=False))

    assert viewer._point_colors(np.array([0.2, 0.8], dtype=np.float32)) is None


def test_visible_full_frame_points_respects_toggle_and_intensity() -> None:
    viewer = Open3DReplayViewer(VisualizationConfig(max_points=10))
    frame = _state(
        0,
        [],
        full_frame_points=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        full_frame_intensity=np.array([0.2, 0.8], dtype=np.float32),
    )

    hidden_points, hidden_intensity = viewer._visible_full_frame_points(frame, show_full_frame_pcd=False)
    visible_points, visible_intensity = viewer._visible_full_frame_points(frame, show_full_frame_pcd=True)

    assert hidden_points.shape == (0, 3)
    assert hidden_intensity is None
    assert np.allclose(visible_points, np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32))
    assert np.allclose(visible_intensity, np.array([0.2, 0.8], dtype=np.float32))


def test_visible_full_frame_points_are_empty_without_source_points() -> None:
    viewer = Open3DReplayViewer(VisualizationConfig(show_full_frame_pcd=True))
    frame = _state(0, [])

    visible_points, visible_intensity = viewer._visible_full_frame_points(frame, show_full_frame_pcd=True)

    assert visible_points.shape == (0, 3)
    assert visible_intensity is None


def test_track_exit_line_lineset_builds_y_axis_exit_line_along_x() -> None:
    viewer = Open3DReplayViewer(
        VisualizationConfig(),
        track_exit_edge_margin=0.5,
        require_track_exit=True,
        track_exit_line_axis="y",
    )

    lineset = viewer._track_exit_lineset(type("LaneBoxStub", (), {
        "x_min": -2.0,
        "x_max": 2.0,
        "y_min": 0.0,
        "y_max": 10.0,
        "z_min": 0.0,
        "z_max": 2.0,
    })())

    assert lineset is not None
    points = np.asarray(lineset.points, dtype=np.float64)
    assert np.allclose(points, np.array([[-2.0, 0.5, 0.0], [2.0, 0.5, 0.0]], dtype=np.float64))


def test_track_exit_line_lineset_is_disabled_when_not_required() -> None:
    viewer = Open3DReplayViewer(
        VisualizationConfig(),
        track_exit_edge_margin=0.5,
        require_track_exit=False,
    )

    lineset = viewer._track_exit_lineset(type("LaneBoxStub", (), {
        "x_min": -2.0,
        "x_max": 2.0,
        "y_min": 0.0,
        "y_max": 10.0,
        "z_min": 0.0,
        "z_max": 2.0,
    })())

    assert lineset is None


def test_trajectory_tail_points_follow_recent_track_history() -> None:
    states = [
        _state(0, []),
        _state(1, [(7, np.array([1.0, 0.0, 0.0], dtype=np.float32))]),
        _state(2, [(7, np.array([2.0, 0.0, 0.0], dtype=np.float32))]),
        _state(3, [(7, np.array([3.0, 0.0, 0.0], dtype=np.float32))]),
        _state(4, []),
    ]

    tail = Open3DReplayViewer._trajectory_tail_points(states, track_id=7, last_playback_index=3, max_points=8)

    assert np.allclose(
        tail,
        np.array(
            [
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )


def test_build_save_events_uses_last_predicted_active_state() -> None:
    states = [
        _state(0, []),
        _state(1, [(7, np.array([1.0, 0.0, 0.0], dtype=np.float32))]),
        _state(2, [(7, np.array([2.0, 0.0, 0.0], dtype=np.float32), "predicted")]),
        _state(3, [(7, np.array([3.0, 0.0, 0.0], dtype=np.float32), "predicted")]),
    ]
    aggregate_results = {
        7: AggregateResult(
            track_id=7,
            points=np.ones((3, 3), dtype=np.float32),
            selected_frame_ids=[101],
            status="saved",
            metrics={},
        )
    }

    events = Open3DReplayViewer._build_save_events(states, aggregate_results, duration_frames=20)
    active = Open3DReplayViewer._active_save_events(events, 3)

    assert len(active) == 1
    assert active[0].track_id == 7
    assert active[0].playback_start_index == 3
    assert np.allclose(active[0].center, np.array([3.0, 0.0, 0.0], dtype=np.float32))


def test_trajectory_tail_points_include_predicted_track_frames() -> None:
    states = [
        _state(0, []),
        _state(1, [(7, np.array([1.0, 0.0, 0.0], dtype=np.float32))]),
        _state(2, [(7, np.array([2.0, 0.0, 0.0], dtype=np.float32), "predicted")]),
        _state(3, [(7, np.array([3.0, 0.0, 0.0], dtype=np.float32), "predicted")]),
        _state(4, []),
    ]

    tail = Open3DReplayViewer._trajectory_tail_points(states, track_id=7, last_playback_index=3, max_points=8)

    assert np.allclose(
        tail,
        np.array(
            [
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )


def test_trajectory_tail_points_collapse_stationary_duplicate_centers() -> None:
    states = [
        _state(0, []),
        _state(1, [(7, np.array([3.0, 0.0, 0.0], dtype=np.float32))]),
        _state(2, [(7, np.array([3.0, 0.0, 0.0], dtype=np.float32), "predicted")]),
        _state(3, [(7, np.array([3.0, 0.0, 0.0], dtype=np.float32), "predicted")]),
    ]

    tail = Open3DReplayViewer._trajectory_tail_points(states, track_id=7, last_playback_index=3, max_points=8)

    assert np.allclose(tail, np.array([[3.0, 0.0, 0.0]], dtype=np.float32))
