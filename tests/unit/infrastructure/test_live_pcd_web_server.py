from __future__ import annotations

import json
import numpy as np

from tracking_pipeline.domain.models import ClusterResult, FrameData, FrameTrackingState
from tracking_pipeline.domain.value_objects import LaneBox
from tracking_pipeline.infrastructure.visualization.live_frame_publisher import LiveFramePublisher
from tracking_pipeline.infrastructure.visualization.live_pcd_web_server import LivePCDWebServer


def _publisher() -> LiveFramePublisher:
    return LiveFramePublisher(
        lane_box=LaneBox.from_values([-1.0, 1.0, 0.0, 10.0, 0.0, 2.0]),
        track_exit_line_axis="y",
        track_exit_edge_margin=0.9,
        max_points=4,
        history_sec=1.0,
        point_source="lane",
        color_by_intensity=False,
        show_tracker_debug=True,
        show_track_outcomes=False,
        run_label="embedded_live_run",
        async_publish=False,
    )


def _dispatch(server: LivePCDWebServer, path: str) -> dict[str, object]:
    handler_cls = server._build_handler()
    handler = object.__new__(handler_cls)
    captured: dict[str, object] = {}
    handler.path = path
    handler._write_html = lambda payload: captured.update({"kind": "html", "payload": payload})
    handler._write_json = lambda status, payload, **_kwargs: captured.update(
        {"kind": "json", "status": int(status), "payload": json.loads(json.dumps(payload))}
    )
    handler.do_GET()
    return captured


def test_live_pcd_web_server_handler_serves_meta_frame_and_html() -> None:
    publisher = _publisher()
    publisher.update_status(pipeline_phase="processing_frames", processed_frames=1)
    publisher.publish_frame(
        FrameData(
            frame_index=4,
            timestamp_ns=123_000_000,
            points=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.1, 0.2, 0.3],
                    [0.2, 0.4, 0.6],
                ],
                dtype=np.float32,
            ),
        ),
        ClusterResult(
            lane_points=np.array(
                [
                    [1.0, 2.0, 0.1],
                    [1.1, 2.1, 0.2],
                    [1.2, 2.2, 0.3],
                ],
                dtype=np.float32,
            ),
            detections=[],
        ),
        FrameTrackingState(
            frame_index=4,
            lane_points=np.array(
                [
                    [1.0, 2.0, 0.1],
                    [1.1, 2.1, 0.2],
                    [1.2, 2.2, 0.3],
                ],
                dtype=np.float32,
            ),
            detections=[],
            active_tracks=[],
        ),
    )
    server = LivePCDWebServer(publisher, host="127.0.0.1", port=0)
    meta = _dispatch(server, "/api/live/meta")
    frame = _dispatch(server, "/api/live/frame/1.json")
    batch = _dispatch(server, "/api/live/frames.json?start_sequence_id=1&limit=2")
    html = _dispatch(server, "/")

    assert meta["status"] == 200
    assert meta["payload"]["run_label"] == "embedded_live_run"
    assert meta["payload"]["status"]["pipeline_phase"] == "processing_frames"
    assert meta["payload"]["point_source"] == "lane"
    assert "retain_all_frames" not in meta["payload"]
    assert meta["payload"]["monitoring"]["status_line"].startswith("live phase=processing_frames f=1")
    assert meta["payload"]["sequence_window"]["latest_sequence_id"] == 1
    assert frame["status"] == 200
    assert frame["payload"]["frame_index"] == 4
    assert frame["payload"]["point_count"] == 3
    assert batch["status"] == 200
    assert batch["payload"]["limit"] == 2
    assert [int(row["sequence_id"]) for row in batch["payload"]["frames"]] == [1]
    assert html["kind"] == "html"
    assert "Live Raw PCD Viewer" in html["payload"]
    assert "Monitoring" in html["payload"]
    assert "Saved vehicles" in html["payload"]
    assert "JOURNAL_LOG_MAX_ENTRIES = 240" in html["payload"]
    assert "journal-style status" in html["payload"]
    assert "Frames are fetched sequentially in small batches" in html["payload"]
    assert "<canvas id=\"scene\"></canvas>" in html["payload"]


def test_live_pcd_web_server_handler_returns_404_for_unknown_sequence() -> None:
    response = _dispatch(LivePCDWebServer(_publisher(), host="127.0.0.1", port=0), "/api/live/frame/99.json")

    assert response["status"] == 404
    assert response["payload"]["error"] == "frame_not_found"


def test_live_pcd_web_server_handler_rejects_invalid_batch_query() -> None:
    response = _dispatch(
        LivePCDWebServer(_publisher(), host="127.0.0.1", port=0),
        "/api/live/frames.json?start_sequence_id=nope&limit=2",
    )

    assert response["status"] == 400
    assert response["payload"]["error"] == "invalid_frame_batch_query"
