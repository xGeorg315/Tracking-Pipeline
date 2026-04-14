from __future__ import annotations

from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

from tracking_pipeline.domain.rules import axis_to_index, track_exit_line_value
from tracking_pipeline.domain.value_objects import LaneBox
from tracking_pipeline.infrastructure.visualization.live_snapshot_loader import LiveSnapshot, LiveSnapshotLoader
from tracking_pipeline.infrastructure.visualization.open3d_live_viewer import Open3DLiveViewer

_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Tracking Pipeline Live Web</title>
  <style>
    :root {
      --bg-0: #081018;
      --bg-1: #0f1a25;
      --panel: rgba(9, 15, 23, 0.82);
      --panel-strong: rgba(5, 11, 18, 0.92);
      --line: rgba(255, 255, 255, 0.12);
      --text: #eaf2f7;
      --muted: #8ea3b5;
      --accent: #f0c44f;
      --success: #3cff89;
      --warning: #ffbf52;
      --danger: #ff6b5f;
      --tracker: #6dd8ff;
      --font-ui: "Avenir Next", "Segoe UI", "Helvetica Neue", sans-serif;
      --font-mono: "IBM Plex Mono", "SFMono-Regular", "Consolas", monospace;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      font-family: var(--font-ui);
      background:
        radial-gradient(circle at 20% 10%, rgba(61, 127, 150, 0.16), transparent 30%),
        radial-gradient(circle at 80% 0%, rgba(240, 196, 79, 0.15), transparent 26%),
        linear-gradient(160deg, var(--bg-0), var(--bg-1));
    }

    .shell {
      display: grid;
      grid-template-columns: minmax(260px, 360px) 1fr minmax(260px, 360px);
      gap: 16px;
      min-height: 100vh;
      padding: 16px;
    }

    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      backdrop-filter: blur(14px);
      box-shadow: 0 18px 60px rgba(0, 0, 0, 0.28);
      overflow: hidden;
    }

    .stack {
      display: grid;
      gap: 16px;
      align-content: start;
    }

    .section {
      padding: 18px 18px 16px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }

    .section:last-child { border-bottom: 0; }

    .eyebrow {
      margin: 0 0 6px;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: var(--accent);
    }

    .title {
      margin: 0;
      font-size: 24px;
      line-height: 1.15;
      font-weight: 700;
    }

    .subtitle {
      margin: 8px 0 0;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.45;
    }

    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 14px;
    }

    button {
      border: 1px solid rgba(255, 255, 255, 0.12);
      background: rgba(255, 255, 255, 0.06);
      color: var(--text);
      border-radius: 999px;
      padding: 9px 14px;
      font: inherit;
      font-size: 13px;
      cursor: pointer;
      transition: background 120ms ease, border-color 120ms ease, transform 120ms ease;
    }

    button:hover {
      background: rgba(255, 255, 255, 0.1);
      border-color: rgba(255, 255, 255, 0.24);
      transform: translateY(-1px);
    }

    button.active {
      background: rgba(240, 196, 79, 0.18);
      border-color: rgba(240, 196, 79, 0.36);
    }

    .meta {
      display: grid;
      gap: 10px;
    }

    .meta-row {
      display: grid;
      gap: 4px;
    }

    .meta-label {
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--muted);
    }

    .mono {
      white-space: pre-wrap;
      font-family: var(--font-mono);
      font-size: 13px;
      line-height: 1.45;
    }

    .viewer {
      position: relative;
      min-height: calc(100vh - 32px);
      display: grid;
      grid-template-rows: auto 1fr;
      background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.03), rgba(255, 255, 255, 0)),
        rgba(4, 10, 16, 0.84);
    }

    .viewer-head {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      padding: 18px 20px 12px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }

    .viewer-title {
      margin: 0;
      font-size: 18px;
      font-weight: 700;
    }

    .viewer-note {
      margin: 4px 0 0;
      color: var(--muted);
      font-size: 13px;
    }

    .status-pill {
      align-self: start;
      padding: 8px 12px;
      border-radius: 999px;
      border: 1px solid rgba(255, 255, 255, 0.1);
      background: rgba(255, 255, 255, 0.04);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }

    canvas {
      width: 100%;
      height: 100%;
      display: block;
    }

    .canvas-wrap {
      position: relative;
      min-height: 620px;
    }

    .overlay {
      position: absolute;
      left: 16px;
      bottom: 16px;
      padding: 10px 12px;
      border-radius: 14px;
      background: var(--panel-strong);
      border: 1px solid rgba(255, 255, 255, 0.08);
      font-size: 12px;
      color: var(--muted);
    }

    .help {
      display: none;
    }

    .help.visible {
      display: block;
    }

    @media (max-width: 1220px) {
      .shell {
        grid-template-columns: 1fr;
      }

      .viewer {
        min-height: 72vh;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="stack">
      <section class="panel section">
        <p class="eyebrow">Tracking Pipeline</p>
        <h1 class="title">Live Web Viewer</h1>
        <p class="subtitle">Headless browser view for the current dataset snapshot. The canvas shows a top-down x/y projection of the latest tracker state and outcome markers.</p>
        <div class="toolbar">
          <button id="pauseBtn">Pause</button>
          <button id="trackerBtn" class="active">Tracker</button>
          <button id="outcomeBtn" class="active">Outcomes</button>
          <button id="refreshBtn">Refresh</button>
          <button id="helpBtn">Help</button>
        </div>
      </section>
      <section class="panel section meta">
        <div class="meta-row">
          <div class="meta-label">Status</div>
          <div id="statusText" class="mono"></div>
        </div>
        <div class="meta-row">
          <div class="meta-label">Summary</div>
          <div id="summaryText" class="mono"></div>
        </div>
      </section>
    </div>
    <section class="panel viewer">
      <div class="viewer-head">
        <div>
          <h2 class="viewer-title">Top-Down Snapshot</h2>
          <p class="viewer-note">Lane-box, exit-line, detections, tracks and final outcome beacons. Auto-refresh runs once per second.</p>
        </div>
        <div id="runBadge" class="status-pill">connecting</div>
      </div>
      <div class="canvas-wrap">
        <canvas id="scene"></canvas>
        <div class="overlay">Projection: x/y plane</div>
      </div>
    </section>
    <div class="stack">
      <section class="panel section meta">
        <div class="meta-row">
          <div class="meta-label">Tracker Debug</div>
          <div id="trackerText" class="mono"></div>
        </div>
        <div class="meta-row">
          <div class="meta-label">Warnings</div>
          <div id="warningText" class="mono"></div>
        </div>
      </section>
      <section id="helpBox" class="panel section meta help">
        <div class="meta-row">
          <div class="meta-label">Controls</div>
          <div class="mono">Space pause/resume
T toggle tracker overlay
F toggle outcomes
R force refresh
H toggle help</div>
        </div>
      </section>
    </div>
  </div>
  <script>
    const state = {
      paused: false,
      showTracker: true,
      showOutcomes: true,
      showHelp: false,
      snapshot: null,
      fetchInFlight: false,
      pollTimer: null,
    };

    const canvas = document.getElementById("scene");
    const ctx = canvas.getContext("2d");

    function resizeCanvas() {
      const ratio = Math.max(window.devicePixelRatio || 1, 1);
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.round(rect.width * ratio));
      canvas.height = Math.max(1, Math.round(rect.height * ratio));
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      render();
    }

    function colorCss(rgb, alpha = 1) {
      if (!rgb || rgb.length !== 3) return `rgba(234,242,247,${alpha})`;
      return `rgba(${Math.round(rgb[0] * 255)}, ${Math.round(rgb[1] * 255)}, ${Math.round(rgb[2] * 255)}, ${alpha})`;
    }

    function updateButtons() {
      document.getElementById("pauseBtn").textContent = state.paused ? "Resume" : "Pause";
      document.getElementById("trackerBtn").classList.toggle("active", state.showTracker);
      document.getElementById("outcomeBtn").classList.toggle("active", state.showOutcomes);
      document.getElementById("helpBtn").classList.toggle("active", state.showHelp);
      document.getElementById("helpBox").classList.toggle("visible", state.showHelp);
    }

    function setText(id, text) {
      document.getElementById(id).textContent = text || "";
    }

    function allPoints(snapshot) {
      const pts = [];
      if (!snapshot) return pts;
      const lane = snapshot.lane_box;
      if (lane) {
        pts.push([lane.x_min, lane.y_min], [lane.x_max, lane.y_max]);
      }
      for (const det of snapshot.detections || []) {
        pts.push([det.center[0], det.center[1]]);
      }
      for (const track of snapshot.tracks || []) {
        if (track.predicted_center) pts.push([track.predicted_center[0], track.predicted_center[1]]);
        if (track.output_center) pts.push([track.output_center[0], track.output_center[1]]);
      }
      for (const outcome of snapshot.outcomes || []) {
        pts.push([outcome.center[0], outcome.center[1]]);
      }
      return pts;
    }

    function bounds(snapshot) {
      const pts = allPoints(snapshot);
      if (!pts.length) return { x0: -5, x1: 5, y0: -5, y1: 5 };
      let x0 = pts[0][0], x1 = pts[0][0], y0 = pts[0][1], y1 = pts[0][1];
      for (const [x, y] of pts) {
        x0 = Math.min(x0, x);
        x1 = Math.max(x1, x);
        y0 = Math.min(y0, y);
        y1 = Math.max(y1, y);
      }
      const dx = Math.max(1.0, x1 - x0);
      const dy = Math.max(1.0, y1 - y0);
      const px = dx * 0.12;
      const py = dy * 0.12;
      return { x0: x0 - px, x1: x1 + px, y0: y0 - py, y1: y1 + py };
    }

    function makeProjector(snapshot) {
      const rect = canvas.getBoundingClientRect();
      const width = Math.max(rect.width, 1);
      const height = Math.max(rect.height, 1);
      const box = bounds(snapshot);
      const pad = 34;
      const sx = (width - pad * 2) / Math.max(box.x1 - box.x0, 1e-6);
      const sy = (height - pad * 2) / Math.max(box.y1 - box.y0, 1e-6);
      const scale = Math.min(sx, sy);
      const cx = (box.x0 + box.x1) * 0.5;
      const cy = (box.y0 + box.y1) * 0.5;
      return ([x, y]) => {
        const px = width * 0.5 + (x - cx) * scale;
        const py = height * 0.5 - (y - cy) * scale;
        return [px, py];
      };
    }

    function drawGrid(snapshot, project) {
      const box = bounds(snapshot);
      const stepX = Math.max(1, Math.round((box.x1 - box.x0) / 8));
      const stepY = Math.max(1, Math.round((box.y1 - box.y0) / 8));
      ctx.lineWidth = 1;
      ctx.strokeStyle = "rgba(255,255,255,0.05)";
      for (let x = Math.floor(box.x0 / stepX) * stepX; x <= box.x1; x += stepX) {
        const a = project([x, box.y0]);
        const b = project([x, box.y1]);
        ctx.beginPath();
        ctx.moveTo(a[0], a[1]);
        ctx.lineTo(b[0], b[1]);
        ctx.stroke();
      }
      for (let y = Math.floor(box.y0 / stepY) * stepY; y <= box.y1; y += stepY) {
        const a = project([box.x0, y]);
        const b = project([box.x1, y]);
        ctx.beginPath();
        ctx.moveTo(a[0], a[1]);
        ctx.lineTo(b[0], b[1]);
        ctx.stroke();
      }
    }

    function drawLaneBox(snapshot, project) {
      const lane = snapshot && snapshot.lane_box;
      if (!lane) return;
      const p0 = project([lane.x_min, lane.y_min]);
      const p1 = project([lane.x_max, lane.y_min]);
      const p2 = project([lane.x_max, lane.y_max]);
      const p3 = project([lane.x_min, lane.y_max]);
      ctx.strokeStyle = "rgba(240,196,79,0.95)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(p0[0], p0[1]);
      ctx.lineTo(p1[0], p1[1]);
      ctx.lineTo(p2[0], p2[1]);
      ctx.lineTo(p3[0], p3[1]);
      ctx.closePath();
      ctx.stroke();

      if (snapshot.track_exit && snapshot.track_exit.enabled) {
        ctx.strokeStyle = "rgba(109,216,255,0.95)";
        ctx.lineWidth = 1.5;
        const axis = snapshot.track_exit.axis;
        const value = snapshot.track_exit.value;
        let a, b;
        if (axis === "x") {
          a = project([value, lane.y_min]);
          b = project([value, lane.y_max]);
        } else {
          a = project([lane.x_min, value]);
          b = project([lane.x_max, value]);
        }
        ctx.beginPath();
        ctx.moveTo(a[0], a[1]);
        ctx.lineTo(b[0], b[1]);
        ctx.stroke();
      }
    }

    function drawLabel(text, x, y, color) {
      if (!text) return;
      ctx.font = "12px IBM Plex Mono, Consolas, monospace";
      ctx.fillStyle = color;
      ctx.fillText(text, x + 8, y - 8);
    }

    function drawMarker(x, y, radius, color, strokeColor = null) {
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      if (strokeColor) {
        ctx.lineWidth = 1.2;
        ctx.strokeStyle = strokeColor;
        ctx.stroke();
      }
    }

    function drawDetections(snapshot, project) {
      for (const det of snapshot.detections || []) {
        const [x, y] = project([det.center[0], det.center[1]]);
        const radius = 4 + Math.max(0, det.radius * 22);
        drawMarker(x, y, radius, colorCss(det.color, 0.95), "rgba(255,255,255,0.18)");
        drawLabel(det.label, x, y, colorCss(det.color, 0.95));
      }
    }

    function drawTracks(snapshot, project) {
      const detectionMap = new Map((snapshot.detections || []).map(det => [det.detection_id, det]));
      for (const track of snapshot.tracks || []) {
        const predicted = track.predicted_center ? project([track.predicted_center[0], track.predicted_center[1]]) : null;
        const output = track.output_center ? project([track.output_center[0], track.output_center[1]]) : null;

        if (predicted && output) {
          ctx.strokeStyle = colorCss([0.20, 1.0, 0.35], 0.85);
          ctx.lineWidth = 2.2;
          ctx.beginPath();
          ctx.moveTo(predicted[0], predicted[1]);
          ctx.lineTo(output[0], output[1]);
          ctx.stroke();
        }

        if (track.status === "matched" && output && track.matched_detection_id != null) {
          const det = detectionMap.get(track.matched_detection_id);
          if (det) {
            const detPos = project([det.center[0], det.center[1]]);
            ctx.strokeStyle = colorCss([0.35, 0.85, 1.0], 0.85);
            ctx.lineWidth = 1.4;
            ctx.beginPath();
            ctx.moveTo(output[0], output[1]);
            ctx.lineTo(detPos[0], detPos[1]);
            ctx.stroke();
          }
        }

        if (predicted) {
          drawMarker(predicted[0], predicted[1], 6, colorCss(track.predicted_color, 0.95), "rgba(255,255,255,0.16)");
          drawLabel(track.predicted_label, predicted[0], predicted[1], colorCss(track.predicted_color, 0.95));
        }
        if (output) {
          drawMarker(output[0], output[1], 7, colorCss(track.output_color, 0.95), "rgba(255,255,255,0.18)");
          drawLabel(track.output_label, output[0], output[1], colorCss(track.output_color, 0.95));
        }
      }
    }

    function drawOutcomes(snapshot, project) {
      for (const outcome of snapshot.outcomes || []) {
        const [x, y] = project([outcome.center[0], outcome.center[1]]);
        drawMarker(x, y, 7, colorCss(outcome.color, 0.95), "rgba(255,255,255,0.22)");
        drawLabel(outcome.label, x, y, colorCss(outcome.color, 0.95));
      }
    }

    function render() {
      const rect = canvas.getBoundingClientRect();
      ctx.clearRect(0, 0, rect.width, rect.height);
      if (!state.snapshot) return;
      const snapshot = state.snapshot;
      const project = makeProjector(snapshot);
      drawGrid(snapshot, project);
      drawLaneBox(snapshot, project);
      if (state.showTracker) {
        drawDetections(snapshot, project);
        drawTracks(snapshot, project);
      }
      if (state.showOutcomes) {
        drawOutcomes(snapshot, project);
      }
    }

    async function refresh(force = false) {
      if (state.fetchInFlight) return;
      state.fetchInFlight = true;
      try {
        const resp = await fetch(`/api/snapshot${force ? "?force=1" : ""}`, { cache: "no-store" });
        const payload = await resp.json();
        state.snapshot = payload;
        setText("statusText", payload.status_text);
        setText("summaryText", payload.summary_text);
        setText("trackerText", state.showTracker ? payload.tracker_text : "");
        setText("warningText", (payload.warnings || []).join("\\n"));
        document.getElementById("runBadge").textContent = payload.waiting ? "waiting" : (payload.run_id || "live");
        render();
      } catch (err) {
        setText("warningText", `snapshot fetch failed\\n${String(err)}`);
      } finally {
        state.fetchInFlight = false;
      }
    }

    function schedule() {
      if (state.pollTimer) window.clearInterval(state.pollTimer);
      const interval = (state.snapshot && state.snapshot.poll_interval_ms) || 1000;
      state.pollTimer = window.setInterval(() => {
        if (!state.paused) refresh(false);
      }, interval);
    }

    document.getElementById("pauseBtn").addEventListener("click", () => {
      state.paused = !state.paused;
      updateButtons();
      refresh(false);
    });
    document.getElementById("trackerBtn").addEventListener("click", () => {
      state.showTracker = !state.showTracker;
      updateButtons();
      setText("trackerText", state.showTracker && state.snapshot ? state.snapshot.tracker_text : "");
      render();
    });
    document.getElementById("outcomeBtn").addEventListener("click", () => {
      state.showOutcomes = !state.showOutcomes;
      updateButtons();
      render();
    });
    document.getElementById("refreshBtn").addEventListener("click", () => refresh(true));
    document.getElementById("helpBtn").addEventListener("click", () => {
      state.showHelp = !state.showHelp;
      updateButtons();
    });

    window.addEventListener("keydown", (event) => {
      if (event.code === "Space") {
        event.preventDefault();
        document.getElementById("pauseBtn").click();
      } else if (event.key === "t" || event.key === "T") {
        document.getElementById("trackerBtn").click();
      } else if (event.key === "f" || event.key === "F") {
        document.getElementById("outcomeBtn").click();
      } else if (event.key === "r" || event.key === "R") {
        document.getElementById("refreshBtn").click();
      } else if (event.key === "h" || event.key === "H") {
        document.getElementById("helpBtn").click();
      }
    });

    window.addEventListener("resize", resizeCanvas);
    updateButtons();
    resizeCanvas();
    schedule();
    refresh(true);
  </script>
</body>
</html>
"""


def _vector3(value: np.ndarray | None) -> list[float] | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < 3:
        return None
    return [float(arr[0]), float(arr[1]), float(arr[2])]


def _lane_box_payload(lane_box: LaneBox | None) -> dict[str, float] | None:
    if lane_box is None:
        return None
    return {
        "x_min": float(lane_box.x_min),
        "x_max": float(lane_box.x_max),
        "y_min": float(lane_box.y_min),
        "y_max": float(lane_box.y_max),
        "z_min": float(lane_box.z_min),
        "z_max": float(lane_box.z_max),
    }


def _track_exit_payload(snapshot: LiveSnapshot) -> dict[str, Any]:
    enabled = bool(snapshot.require_track_exit) and float(snapshot.track_exit_edge_margin) > 0.0 and snapshot.lane_box is not None
    value = None
    if enabled and snapshot.lane_box is not None:
        value = float(
            track_exit_line_value(
                snapshot.lane_box,
                axis_to_index(snapshot.track_exit_line_axis),
                edge_margin=float(snapshot.track_exit_edge_margin),
            )
        )
    return {
        "enabled": enabled,
        "axis": str(snapshot.track_exit_line_axis),
        "edge_margin": float(snapshot.track_exit_edge_margin),
        "value": value,
    }


def build_live_web_payload(snapshot: LiveSnapshot) -> dict[str, Any]:
    tracker_debug = None if snapshot.tracker_frame is None else snapshot.tracker_frame.tracker_debug

    detections = []
    for state in [] if tracker_debug is None else tracker_debug.detection_states:
        color, radius = Open3DLiveViewer._detection_marker_style(state.status, state.tracking_halo_only)
        label = ""
        if state.status == "spawn_suppressed":
            label = f"suppressed d{int(state.detection_id)}"
        elif state.tracking_halo_only:
            label = f"halo d{int(state.detection_id)}"
        detections.append(
            {
                "detection_id": int(state.detection_id),
                "center": _vector3(state.center),
                "status": str(state.status),
                "matched_track_id": None if state.matched_track_id is None else int(state.matched_track_id),
                "spawned_track_id": None if state.spawned_track_id is None else int(state.spawned_track_id),
                "spawn_suppressed": bool(state.spawn_suppressed),
                "tracking_halo_only": bool(state.tracking_halo_only),
                "color": list(color),
                "radius": float(radius),
                "label": label,
            }
        )

    tracks = []
    for state in [] if tracker_debug is None else tracker_debug.track_states:
        predicted_color = (0.20, 0.90, 1.00) if str(state.status) != "missed" else (1.00, 0.60, 0.15)
        predicted_label = ""
        if state.status == "missed":
            predicted_label = f"missed #{int(state.track_id)}"
        elif state.status == "matched":
            predicted_label = f"pred #{int(state.track_id)}"
        output_color = {
            "matched": (0.20, 1.00, 0.35),
            "spawned": (0.20, 0.45, 1.00),
            "missed": (1.00, 0.60, 0.15),
        }.get(str(state.status), (0.90, 0.90, 0.90))
        output_label = ""
        if state.status == "spawned":
            output_label = f"spawn #{int(state.track_id)}"
        elif state.status == "matched":
            output_label = f"track #{int(state.track_id)}"
        tracks.append(
            {
                "track_id": int(state.track_id),
                "status": str(state.status),
                "predicted_center": _vector3(state.predicted_center),
                "output_center": _vector3(state.output_center),
                "matched_detection_id": None if state.matched_detection_id is None else int(state.matched_detection_id),
                "predicted_color": list(predicted_color),
                "output_color": list(output_color),
                "predicted_label": predicted_label,
                "output_label": output_label,
            }
        )

    outcomes = []
    for event in Open3DLiveViewer._build_outcome_events(snapshot):
        outcomes.append(
            {
                "track_id": int(event.track_id),
                "status": str(event.status),
                "decision_reason_code": str(event.decision_reason_code),
                "decision_summary": str(event.decision_summary),
                "frame_index": int(event.frame_index),
                "center": _vector3(event.center),
                "color": list(Open3DLiveViewer._outcome_color(event)),
                "label": Open3DLiveViewer._outcome_label_text(event),
            }
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "poll_interval_ms": 1000,
        "waiting": bool(snapshot.waiting),
        "run_id": str(snapshot.run_id),
        "dataset_root": str(snapshot.dataset_root),
        "status_text": Open3DLiveViewer._build_status_text(snapshot, paused=False),
        "summary_text": Open3DLiveViewer._build_summary_text(snapshot),
        "tracker_text": Open3DLiveViewer._build_tracker_debug_text(snapshot, enabled=True),
        "warnings": list(snapshot.warnings),
        "lane_box": _lane_box_payload(snapshot.lane_box),
        "track_exit": _track_exit_payload(snapshot),
        "detections": detections,
        "tracks": tracks,
        "outcomes": outcomes,
    }


class LiveWebViewerServer:
    def __init__(self, loader: LiveSnapshotLoader, *, host: str = "127.0.0.1", port: int = 8765):
        self.loader = loader
        self.host = str(host)
        self.port = int(port)
        self._lock = threading.Lock()

    def serve(self, run_id: str | None = None) -> None:
        server_ref = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path in ("", "/"):
                    self._send_html(_HTML)
                    return
                if parsed.path == "/api/snapshot":
                    params = parse_qs(parsed.query, keep_blank_values=False)
                    force = str(params.get("force", ["0"])[0]).strip().lower() in {"1", "true", "yes"}
                    payload = server_ref.snapshot_payload(run_id=run_id, force=force)
                    self._send_json(payload)
                    return
                if parsed.path == "/api/health":
                    self._send_json({"ok": True, "run_id": run_id or ""})
                    return
                self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

            def log_message(self, format: str, *args: Any) -> None:
                _ = format, args

            def _send_html(self, body: str) -> None:
                data = body.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _send_json(self, payload: dict[str, Any]) -> None:
                data = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        bind_host = self.host if self.host not in {"0.0.0.0", "::"} else "localhost"
        print(
            "Live web viewer listening on "
            f"http://{bind_host}:{int(httpd.server_port)} "
            f"(bound to {self.host}:{int(httpd.server_port)})"
        )
        try:
            httpd.serve_forever()
        finally:
            httpd.server_close()

    def snapshot_payload(self, *, run_id: str | None = None, force: bool = False) -> dict[str, Any]:
        with self._lock:
            snapshot = self.loader.load(run_id=run_id, force=force)
        return build_live_web_payload(snapshot)
