from __future__ import annotations

import gzip
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from typing import Any
from urllib.parse import parse_qs, urlparse

from tracking_pipeline.infrastructure.visualization.live_frame_publisher import LiveFramePublisher

_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Tracking Pipeline Live PCD Viewer</title>
  <style>
    :root {
      --bg: #061018;
      --panel: rgba(9, 16, 23, 0.84);
      --panel-strong: rgba(6, 12, 18, 0.92);
      --line: rgba(255, 255, 255, 0.08);
      --text: #e8f2f8;
      --muted: #89a1b5;
      --accent: #f6c960;
      --cyan: #73d9ff;
      --green: #51ff95;
      --amber: #ffc763;
      --red: #ff766b;
      --font-ui: "Avenir Next", "Segoe UI", sans-serif;
      --font-mono: "IBM Plex Mono", "SFMono-Regular", monospace;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      font-family: var(--font-ui);
      background:
        radial-gradient(circle at 18% 8%, rgba(115, 217, 255, 0.12), transparent 28%),
        radial-gradient(circle at 82% 0%, rgba(246, 201, 96, 0.14), transparent 22%),
        linear-gradient(165deg, #061018, #09141d 52%, #0b1622);
    }

    .shell {
      display: grid;
      grid-template-columns: minmax(300px, 360px) 1fr;
      gap: 16px;
      min-height: 100vh;
      padding: 16px;
    }

    .panel {
      border: 1px solid var(--line);
      border-radius: 20px;
      background: var(--panel);
      backdrop-filter: blur(16px);
      box-shadow: 0 18px 60px rgba(0, 0, 0, 0.28);
      overflow: hidden;
    }

    .sidebar {
      display: grid;
      align-content: start;
      gap: 16px;
    }

    .section {
      padding: 18px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }

    .section:last-child { border-bottom: 0; }

    .eyebrow {
      margin: 0 0 6px;
      color: var(--accent);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.14em;
      text-transform: uppercase;
    }

    .title {
      margin: 0;
      font-size: 26px;
      line-height: 1.08;
    }

    .subtitle {
      margin: 10px 0 0;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.5;
    }

    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 14px;
    }

    button {
      border: 1px solid rgba(255, 255, 255, 0.12);
      background: rgba(255, 255, 255, 0.05);
      color: var(--text);
      border-radius: 999px;
      padding: 9px 14px;
      font: inherit;
      font-size: 13px;
      cursor: pointer;
    }

    button.active {
      border-color: rgba(246, 201, 96, 0.42);
      background: rgba(246, 201, 96, 0.16);
    }

    .grid {
      display: grid;
      gap: 10px;
    }

    .meta-row {
      display: grid;
      gap: 3px;
    }

    .meta-label {
      color: var(--muted);
      font-size: 11px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
    }

    .mono {
      font-family: var(--font-mono);
      font-size: 12px;
      line-height: 1.45;
      white-space: pre-wrap;
    }

    .viewer {
      display: grid;
      grid-template-rows: auto 1fr;
      min-height: calc(100vh - 32px);
      background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.03), rgba(255, 255, 255, 0)),
        rgba(5, 10, 16, 0.84);
    }

    .viewer-head {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      padding: 18px 22px 12px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }

    .viewer-title {
      margin: 0;
      font-size: 18px;
    }

    .viewer-note {
      margin: 6px 0 0;
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

    .canvas-wrap {
      position: relative;
      min-height: 680px;
    }

    canvas {
      width: 100%;
      height: 100%;
      display: block;
    }

    .hud {
      position: absolute;
      left: 18px;
      bottom: 18px;
      max-width: min(420px, calc(100% - 36px));
      padding: 12px 14px;
      border-radius: 16px;
      border: 1px solid rgba(255, 255, 255, 0.08);
      background: var(--panel-strong);
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }

    .help {
      display: none;
      margin-top: 8px;
    }

    .help.visible {
      display: block;
    }

    .legend {
      display: grid;
      gap: 8px;
    }

    .legend-row {
      display: flex;
      align-items: center;
      gap: 10px;
      color: var(--muted);
      font-size: 13px;
    }

    .swatch {
      width: 12px;
      height: 12px;
      border-radius: 999px;
      flex: 0 0 auto;
    }

    @media (max-width: 1180px) {
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
    <div class="sidebar">
      <section class="panel section">
        <p class="eyebrow">Tracking Pipeline</p>
        <h1 class="title">Live Raw PCD Viewer</h1>
        <p class="subtitle">Headless browser viewer for the in-process `qb2_live` stream. The scene plays queued live frames with an adaptive speed so it stays smooth and close to realtime.</p>
        <div class="toolbar">
          <button id="pauseBtn">Pause</button>
          <button id="trackerBtn">Tracker</button>
          <button id="outcomeBtn">Outcomes</button>
          <button id="refreshBtn">Refresh</button>
          <button id="helpBtn">Help</button>
        </div>
        <div id="helpBox" class="help mono">
Space: pause or resume polling
T: toggle tracker overlay
F: toggle outcome overlay
R: force a meta refresh
H: toggle this help

Mouse drag: orbit
Mouse wheel: zoom
        </div>
      </section>

      <section class="panel section">
        <p class="eyebrow">Run</p>
        <div class="grid mono" id="statusBlock">Waiting for meta...</div>
      </section>

      <section class="panel section">
        <p class="eyebrow">Summary</p>
        <div class="grid mono" id="summaryBlock">No summary yet</div>
      </section>

      <section class="panel section">
        <p class="eyebrow">Legend</p>
        <div class="legend">
          <div class="legend-row"><span class="swatch" style="background:#73d9ff"></span> Raw point cloud</div>
          <div class="legend-row"><span class="swatch" style="background:#51ff95"></span> Matched detections, cluster boxes / saved outcomes</div>
          <div class="legend-row"><span class="swatch" style="background:#ffc763"></span> Spawned detections / skipped outcomes</div>
          <div class="legend-row"><span class="swatch" style="background:#ff766b"></span> Missed or invalid track states</div>
        </div>
      </section>
    </div>

    <section class="panel viewer">
      <div class="viewer-head">
        <div>
          <h2 class="viewer-title">Sequential Point Cloud</h2>
          <p class="viewer-note">Continuous live playback. Frames are fetched sequentially in small batches, and the viewer snaps back to the live tail if it falls too far behind.</p>
        </div>
        <div class="status-pill" id="phasePill">WAITING</div>
      </div>
      <div class="canvas-wrap">
        <canvas id="scene"></canvas>
        <div class="hud mono" id="hudText">Waiting for the first frame...</div>
      </div>
    </section>
  </div>

  <script>
    const canvas = document.getElementById("scene");
    const ctx = canvas.getContext("2d");
    const statusBlock = document.getElementById("statusBlock");
    const summaryBlock = document.getElementById("summaryBlock");
    const hudText = document.getElementById("hudText");
    const phasePill = document.getElementById("phasePill");
    const pauseBtn = document.getElementById("pauseBtn");
    const trackerBtn = document.getElementById("trackerBtn");
    const outcomeBtn = document.getElementById("outcomeBtn");
    const refreshBtn = document.getElementById("refreshBtn");
    const helpBtn = document.getElementById("helpBtn");
    const helpBox = document.getElementById("helpBox");

    const META_POLL_INTERVAL_MS = 40;
    const FRAME_BATCH_LIMIT = 4;
    const FRAME_QUEUE_TARGET = 3;
    const FRAME_QUEUE_MAX = 6;
    const LAG_DROP_THRESHOLD = 12;
    const LAG_TAIL_KEEP = 3;
    const PLAYBACK_IDLE_INTERVAL_MS = 64;
    const PLAYBACK_NORMAL_INTERVAL_MS = 46;
    const PLAYBACK_FAST_INTERVAL_MS = 30;
    const PLAYBACK_CATCH_UP_INTERVAL_MS = 18;
    const OUTCOME_VISIBILITY_SEC = 3.0;
    const MAX_VISIBLE_OUTCOMES = 5;

    const state = {
      meta: null,
      currentFrame: null,
      pendingFrames: [],
      displayedSeq: -1,
      fetchedSeq: -1,
      targetLatestSeq: -1,
      droppedFrames: 0,
      runLabel: "",
      paused: false,
      showTracker: true,
      showOutcomes: true,
      helpVisible: false,
      bootstrapToggles: false,
      metaInFlight: false,
      frameInFlight: false,
      framePumpQueued: false,
      playbackTimerId: 0,
      renderQueued: false,
      camera: {
        yaw: -0.75,
        pitch: 0.42,
        distance: 32.0,
        target: [0.0, 0.0, 1.0],
        dragging: false,
        lastX: 0,
        lastY: 0,
        initialized: false,
        userAdjusted: false,
      },
      displayHz: 0,
      lastDisplayedAtMs: 0,
      canvasPixelWidth: 0,
      canvasPixelHeight: 0,
      canvasDpr: 0,
    };

    function clamp(value, lo, hi) {
      return Math.max(lo, Math.min(hi, value));
    }

    function setButtonState(button, active) {
      button.classList.toggle("active", Boolean(active));
    }

    function resizeCanvas(force = false) {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      const nextWidth = Math.max(1, Math.floor(rect.width * dpr));
      const nextHeight = Math.max(1, Math.floor(rect.height * dpr));
      if (
        !force
        && nextWidth === state.canvasPixelWidth
        && nextHeight === state.canvasPixelHeight
        && dpr === state.canvasDpr
      ) {
        return;
      }
      state.canvasPixelWidth = nextWidth;
      state.canvasPixelHeight = nextHeight;
      state.canvasDpr = dpr;
      canvas.width = nextWidth;
      canvas.height = nextHeight;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function scheduleRender() {
      if (state.renderQueued) {
        return;
      }
      state.renderQueued = true;
      requestAnimationFrame(() => {
        state.renderQueued = false;
        render();
      });
    }

    function setPhase(meta) {
      const phase = (((meta || {}).status || {}).pipeline_phase || "waiting_for_frames").replaceAll("_", " ");
      phasePill.textContent = phase.toUpperCase();
    }

    function formatMeta(meta) {
      if (!meta) {
        return "Waiting for meta...";
      }
      const status = meta.status || {};
      const reader = meta.reader || {};
      const seq = meta.sequence_window || {};
      const retention = meta.retain_all_frames ? "full-run" : `${Number(meta.history_sec || 0).toFixed(2)}s`;
      const lines = [];
      lines.push(`run           ${meta.run_label || "live"}`);
      lines.push(`phase         ${status.pipeline_phase || "waiting_for_frames"}`);
      lines.push(`frames        ${status.processed_frames || 0}`);
      lines.push(`stored        ${seq.frame_count || 0} frames (${seq.oldest_sequence_id ?? -1}..${seq.latest_sequence_id ?? -1})`);
      lines.push(`display       batched live queue`);
      lines.push(`active tracks ${status.active_track_count || 0}`);
      lines.push(`saved aggr    ${status.saved_aggregates || 0}`);
      lines.push(`reader        ${reader.reader_state || "n/a"}`);
      lines.push(`raw frames    ${reader.raw_frames_received || 0}`);
      lines.push(`mqtt labels   ${reader.pending_label_count || 0}`);
      lines.push(`retention     ${retention}`);
      lines.push(`max points    ${meta.max_points || 0} / frame`);
      lines.push(`point source  ${meta.point_source || "all"}`);
      return lines.join("\\n");
    }

    function formatSummary(meta) {
      const summary = (meta || {}).summary || {};
      if (!Object.keys(summary).length) {
        return "No summary yet";
      }
      const lines = [];
      lines.push(`finished tracks ${summary.finished_track_count || 0}`);
      lines.push(`saved aggregates ${summary.saved_aggregates || 0}`);
      lines.push(`gt matched      ${summary.gt_match_matched_count || 0}`);
      lines.push(`gt unmatched    ${summary.gt_match_unmatched_gt_count || 0}`);
      const counts = summary.aggregate_status_counts || {};
      if (Object.keys(counts).length) {
        lines.push("");
        for (const [key, value] of Object.entries(counts)) {
          lines.push(`${String(key).padEnd(15, " ")} ${value}`);
        }
      }
      return lines.join("\\n");
    }

    function formatHud(meta, currentFrame) {
      if (!meta) {
        return "Waiting for the first frame...";
      }
      const status = meta.status || {};
      const reader = meta.reader || {};
      const seq = meta.sequence_window || {};
      const shownOutcomes = visibleOutcomes(meta);
      const lag = currentLag(meta);
      const lines = [];
      lines.push(`phase=${status.pipeline_phase || "waiting_for_frames"}  seq=${currentFrame ? currentFrame.sequence_id : -1}`);
      lines.push(`frame=${currentFrame ? currentFrame.frame_index : -1}  ts=${currentFrame ? currentFrame.timestamp_ns : -1}`);
      lines.push(`stored_frames=${seq.frame_count || 0}  queued=${state.pendingFrames.length}  lag=${lag}  display=${state.displayHz > 0 ? state.displayHz.toFixed(1) : "--"}Hz`);
      lines.push(`active_tracks=${status.active_track_count || 0}  outcomes=${shownOutcomes.length}  shown_points=${currentFrame ? currentFrame.point_count || 0 : 0}`);
      lines.push(`displayed=${state.displayedSeq}  fetched=${state.fetchedSeq}  dropped=${state.droppedFrames}`);
      lines.push(`reader=${reader.reader_state || "n/a"}  raw=${reader.raw_frames_received || 0}  mqtt_pending=${reader.pending_label_count || 0}`);
      lines.push(`drag to orbit, wheel to zoom`);
      return lines.join("\\n");
    }

    function fetchJson(path) {
      return fetch(path, { cache: "no-store" }).then((response) => {
        if (!response.ok) {
          throw new Error(`${response.status} ${response.statusText}`);
        }
        return response.json();
      });
    }

    function decodeBase64Float16(payload) {
      if (!payload) {
        return new Float32Array(0);
      }
      const binary = atob(payload);
      const bytes = new Uint8Array(binary.length);
      for (let index = 0; index < binary.length; index += 1) {
        bytes[index] = binary.charCodeAt(index);
      }
      const half = new Uint16Array(bytes.buffer);
      const result = new Float32Array(half.length);
      for (let index = 0; index < half.length; index += 1) {
        const value = half[index];
        const sign = (value & 0x8000) ? -1 : 1;
        const exponent = (value >> 10) & 0x1f;
        const fraction = value & 0x03ff;
        if (exponent === 0) {
          result[index] = fraction === 0 ? sign * 0 : sign * Math.pow(2, -14) * (fraction / 1024);
        } else if (exponent === 0x1f) {
          result[index] = fraction === 0 ? sign * Infinity : NaN;
        } else {
          result[index] = sign * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
        }
      }
      return result;
    }

    function materializeFramePayload(payload) {
      if (!payload || payload.points_xyz) {
        return payload;
      }
      payload.points_xyz = decodeBase64Float16(payload.points_xyz_b64 || "");
      delete payload.points_xyz_b64;
      delete payload.points_xyz_encoding;
      return payload;
    }

    async function fetchFrameBatchJson(startSequenceId, limit) {
      const response = await fetch(
        `/api/live/frames.json?start_sequence_id=${startSequenceId}&limit=${limit}`,
        { cache: "no-store" }
      );
      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText}`);
      }
      const payload = await response.json();
      return payload.frames || [];
    }

    function currentLag(meta = state.meta) {
      const seq = ((meta || {}).sequence_window || {});
      return Math.max(0, Number(seq.latest_sequence_id ?? -1) - Math.max(-1, state.displayedSeq));
    }

    async function pollMeta(force = false) {
      if (state.metaInFlight) {
        return;
      }
      if (state.paused && !force) {
        scheduleRender();
        return;
      }
      state.metaInFlight = true;
      try {
        const meta = await fetchJson("/api/live/meta");
        if (force) {
          resetPollingState();
        }
        applyMeta(meta);
        updateTargetSequence(meta);
        scheduleFramePump();
        schedulePlayback();
        scheduleRender();
      } catch (error) {
        hudText.textContent = `Viewer error: ${error.message}`;
        scheduleRender();
      } finally {
        state.metaInFlight = false;
      }
    }

    function applyMeta(meta) {
      const nextRunLabel = String(meta.run_label || "");
      if (state.runLabel && nextRunLabel && state.runLabel !== nextRunLabel) {
        resetPollingState();
      }
      state.meta = meta;
      state.runLabel = nextRunLabel;
      if (!state.bootstrapToggles) {
        const overlayDefaults = meta.overlay_defaults || {};
        state.showTracker = Boolean(overlayDefaults.show_tracker_debug);
        state.showOutcomes = Boolean(overlayDefaults.show_track_outcomes);
        state.bootstrapToggles = true;
      }
      if (!state.camera.initialized) {
        const lane = meta.lane_box || [-2, 2, 0, 10, 0, 2];
        state.camera.target = [
          0.5 * (lane[0] + lane[1]),
          0.5 * (lane[2] + lane[3]),
          0.5 * (lane[4] + lane[5]),
        ];
        const spanX = Math.abs(lane[1] - lane[0]);
        const spanY = Math.abs(lane[3] - lane[2]);
        const spanZ = Math.abs(lane[5] - lane[4]);
        const diag = Math.max(6.0, Math.sqrt(spanX * spanX + spanY * spanY + spanZ * spanZ));
        state.camera.distance = clamp(state.camera.distance, diag * 0.8, diag * 7.0);
        state.camera.initialized = true;
      }
      statusBlock.textContent = formatMeta(meta);
      summaryBlock.textContent = formatSummary(meta);
      setPhase(meta);
      setButtonState(pauseBtn, state.paused);
      setButtonState(trackerBtn, state.showTracker);
      setButtonState(outcomeBtn, state.showOutcomes);
    }

    function noteDisplayedFrame() {
      const nowMs = Date.now();
      if (state.lastDisplayedAtMs > 0) {
        const deltaMs = Math.max(1, nowMs - state.lastDisplayedAtMs);
        state.displayHz = 1000.0 / deltaMs;
      }
      state.lastDisplayedAtMs = nowMs;
    }

    function updateTargetSequence(meta) {
      const seq = meta.sequence_window || {};
      const oldest = Number(seq.oldest_sequence_id ?? -1);
      const latest = Number(seq.latest_sequence_id ?? -1);
      if (latest < 0 || oldest < 0) {
        state.targetLatestSeq = -1;
        state.currentFrame = null;
        state.pendingFrames = [];
        state.displayedSeq = -1;
        state.fetchedSeq = -1;
        return;
      }
      state.targetLatestSeq = latest;
      const minimumFetchedSeq = oldest - 1;
      if (state.fetchedSeq >= 0 && state.fetchedSeq < minimumFetchedSeq) {
        state.droppedFrames += minimumFetchedSeq - state.fetchedSeq;
      }
      if (state.fetchedSeq < minimumFetchedSeq) {
        state.fetchedSeq = minimumFetchedSeq;
      }
      trimPendingFramesToWindow(oldest);
      trimPendingFramesForLag(meta);
    }

    function trimPendingFramesToWindow(oldestSequenceId) {
      const oldest = Number(oldestSequenceId ?? -1);
      if (oldest < 0 || !state.pendingFrames.length) {
        return;
      }
      let droppedCount = 0;
      state.pendingFrames = state.pendingFrames.filter((frame) => {
        const sequenceId = Number(frame.sequence_id || -1);
        if (sequenceId < oldest) {
          droppedCount += 1;
          return false;
        }
        return true;
      });
      if (droppedCount > 0) {
        state.droppedFrames += droppedCount;
      }
    }

    function trimPendingFramesForLag(meta) {
      const seq = (meta || {}).sequence_window || {};
      const oldest = Number(seq.oldest_sequence_id ?? -1);
      const latest = Number(seq.latest_sequence_id ?? -1);
      if (latest < 0 || oldest < 0) {
        return;
      }
      const lag = currentLag(meta);
      if (lag < LAG_DROP_THRESHOLD && state.pendingFrames.length <= FRAME_QUEUE_MAX) {
        return;
      }
      const tailStart = Math.max(oldest, latest - (LAG_TAIL_KEEP - 1));
      let droppedCount = 0;
      state.pendingFrames = state.pendingFrames.filter((frame) => {
        const sequenceId = Number(frame.sequence_id || -1);
        if (sequenceId < tailStart) {
          droppedCount += 1;
          return false;
        }
        return true;
      });
      if (state.fetchedSeq < tailStart - 1) {
        droppedCount += (tailStart - 1) - state.fetchedSeq;
        state.fetchedSeq = tailStart - 1;
      }
      if (droppedCount > 0) {
        state.droppedFrames += droppedCount;
      }
    }

    function nextBatchStartSequence() {
      const seq = (state.meta || {}).sequence_window || {};
      const oldest = Number(seq.oldest_sequence_id ?? -1);
      const latest = Math.max(Number(seq.latest_sequence_id ?? -1), Number(state.targetLatestSeq ?? -1));
      if (latest < 0 || oldest < 0) {
        return -1;
      }
      if (state.pendingFrames.length >= FRAME_QUEUE_MAX) {
        return -1;
      }
      const nextSeq = Math.max(oldest, state.fetchedSeq + 1);
      if (nextSeq > latest) {
        return -1;
      }
      return nextSeq;
    }

    function nextBatchLimit(startSequenceId) {
      const seq = (state.meta || {}).sequence_window || {};
      const latest = Number(seq.latest_sequence_id ?? -1);
      const queueRoom = Math.max(0, FRAME_QUEUE_MAX - state.pendingFrames.length);
      const remaining = Math.max(0, latest - startSequenceId + 1);
      if (queueRoom <= 0 || remaining <= 0) {
        return 0;
      }
      const desired = state.pendingFrames.length === 0 ? FRAME_BATCH_LIMIT : FRAME_QUEUE_TARGET - state.pendingFrames.length;
      return Math.max(1, Math.min(FRAME_BATCH_LIMIT, queueRoom, remaining, Math.max(1, desired)));
    }

    function playbackIntervalMs() {
      const lag = currentLag();
      const queued = state.pendingFrames.length;
      if (lag >= LAG_DROP_THRESHOLD || queued >= FRAME_QUEUE_MAX) {
        return PLAYBACK_CATCH_UP_INTERVAL_MS;
      }
      if (lag >= 6 || queued >= 6) {
        return PLAYBACK_FAST_INTERVAL_MS;
      }
      if (lag >= 3 || queued >= 3) {
        return PLAYBACK_NORMAL_INTERVAL_MS;
      }
      return PLAYBACK_IDLE_INTERVAL_MS;
    }

    function scheduleFramePump() {
      if (state.framePumpQueued || state.frameInFlight || state.paused) {
        return;
      }
      if (nextBatchStartSequence() < 0) {
        return;
      }
      state.framePumpQueued = true;
      setTimeout(() => {
        state.framePumpQueued = false;
        void pumpFrameBatch();
      }, 0);
    }

    async function pumpFrameBatch() {
      if (state.frameInFlight || state.paused) {
        return;
      }
      const startSequenceId = nextBatchStartSequence();
      if (startSequenceId < 0) {
        return;
      }
      const limit = nextBatchLimit(startSequenceId);
      if (limit <= 0) {
        return;
      }
      state.frameInFlight = true;
      try {
        const frames = await fetchFrameBatchJson(startSequenceId, limit);
        if (!frames.length) {
          return;
        }
        const firstSequenceId = Number(frames[0].sequence_id || startSequenceId);
        if (firstSequenceId > startSequenceId) {
          state.droppedFrames += firstSequenceId - startSequenceId;
        }
        let latestFetchedSeq = state.fetchedSeq;
        for (const frame of frames) {
          const sequenceId = Number(frame.sequence_id || -1);
          if (sequenceId <= latestFetchedSeq) {
            continue;
          }
          state.pendingFrames.push(frame);
          latestFetchedSeq = sequenceId;
        }
        state.fetchedSeq = latestFetchedSeq;
        trimPendingFramesForLag(state.meta);
        schedulePlayback();
        scheduleFramePump();
        scheduleRender();
      } catch (error) {
        hudText.textContent = `Viewer error: ${error.message}`;
        scheduleRender();
      } finally {
        state.frameInFlight = false;
      }
      if (!state.paused && nextBatchStartSequence() >= 0) {
        scheduleFramePump();
      }
    }

    function schedulePlayback() {
      if (state.playbackTimerId || state.paused || state.pendingFrames.length === 0) {
        return;
      }
      state.playbackTimerId = window.setTimeout(() => {
        state.playbackTimerId = 0;
        advancePlayback();
      }, playbackIntervalMs());
    }

    function advancePlayback() {
      if (state.paused) {
        return;
      }
      trimPendingFramesForLag(state.meta);
      const nextFrame = state.pendingFrames.shift() || null;
      if (!nextFrame) {
        scheduleFramePump();
        return;
      }
      state.currentFrame = materializeFramePayload(nextFrame);
      state.displayedSeq = Number(nextFrame.sequence_id || -1);
      noteDisplayedFrame();
      scheduleRender();
      scheduleFramePump();
      schedulePlayback();
    }

    function buildProjectionState() {
      const camera = state.camera;
      return {
        targetX: camera.target[0],
        targetY: camera.target[1],
        targetZ: camera.target[2],
        cosYaw: Math.cos(camera.yaw),
        sinYaw: Math.sin(camera.yaw),
        cosPitch: Math.cos(camera.pitch),
        sinPitch: Math.sin(camera.pitch),
        distance: camera.distance,
        width: canvas.clientWidth || 1,
        height: canvas.clientHeight || 1,
        scale: Math.min(canvas.clientWidth || 1, canvas.clientHeight || 1) * 0.88,
      };
    }

    function viewTransform(point, projection) {
      const px = point[0] - projection.targetX;
      const py = point[1] - projection.targetY;
      const pz = point[2] - projection.targetZ;

      const xYaw = projection.cosYaw * px - projection.sinYaw * py;
      const yYaw = projection.sinYaw * px + projection.cosYaw * py;
      const yPitch = projection.cosPitch * yYaw - projection.sinPitch * pz;
      const zPitch = projection.sinPitch * yYaw + projection.cosPitch * pz;

      const depth = yPitch + projection.distance;
      if (depth <= 0.1) {
        return null;
      }
      return {
        x: projection.width * 0.5 + (xYaw / depth) * projection.scale,
        y: projection.height * 0.58 - (zPitch / depth) * projection.scale,
        depth,
      };
    }

    function drawLine3D(a, b, color, alpha, width, projection) {
      const pa = viewTransform(a, projection);
      const pb = viewTransform(b, projection);
      if (!pa || !pb) {
        return;
      }
      ctx.strokeStyle = color.replace("__ALPHA__", alpha.toFixed(3));
      ctx.lineWidth = width;
      ctx.beginPath();
      ctx.moveTo(pa.x, pa.y);
      ctx.lineTo(pb.x, pb.y);
      ctx.stroke();
    }

    function laneBoxCorners(lane) {
      return [
        [lane[0], lane[2], lane[4]],
        [lane[1], lane[2], lane[4]],
        [lane[1], lane[3], lane[4]],
        [lane[0], lane[3], lane[4]],
        [lane[0], lane[2], lane[5]],
        [lane[1], lane[2], lane[5]],
        [lane[1], lane[3], lane[5]],
        [lane[0], lane[3], lane[5]],
      ];
    }

    function boundsCorners(minBound, maxBound) {
      return [
        [minBound[0], minBound[1], minBound[2]],
        [maxBound[0], minBound[1], minBound[2]],
        [maxBound[0], maxBound[1], minBound[2]],
        [minBound[0], maxBound[1], minBound[2]],
        [minBound[0], minBound[1], maxBound[2]],
        [maxBound[0], minBound[1], maxBound[2]],
        [maxBound[0], maxBound[1], maxBound[2]],
        [minBound[0], maxBound[1], maxBound[2]],
      ];
    }

    function rgbaFromHex(hex, alpha) {
      const normalized = String(hex || "").replace("#", "");
      if (normalized.length !== 6) {
        return `rgba(115, 217, 255, ${alpha})`;
      }
      const r = Number.parseInt(normalized.slice(0, 2), 16);
      const g = Number.parseInt(normalized.slice(2, 4), 16);
      const b = Number.parseInt(normalized.slice(4, 6), 16);
      return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }

    function drawLaneBox(meta, projection) {
      const lane = meta.lane_box;
      if (!lane || lane.length < 6) {
        return;
      }
      const c = laneBoxCorners(lane);
      const edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
      ];
      for (const [a, b] of edges) {
        drawLine3D(c[a], c[b], "rgba(160, 196, 220, __ALPHA__)", 0.38, 1.2, projection);
      }
    }

    function drawExitLine(meta, projection) {
      const lane = meta.lane_box || [];
      const exitLine = meta.exit_line || {};
      if (lane.length < 6) {
        return;
      }
      const axisIndex = Number(exitLine.axis_index ?? 1);
      const value = Number(exitLine.value ?? lane[2]);
      let corners = [];
      if (axisIndex === 0) {
        corners = [
          [value, lane[2], lane[4]],
          [value, lane[3], lane[4]],
          [value, lane[3], lane[5]],
          [value, lane[2], lane[5]],
        ];
      } else if (axisIndex === 1) {
        corners = [
          [lane[0], value, lane[4]],
          [lane[1], value, lane[4]],
          [lane[1], value, lane[5]],
          [lane[0], value, lane[5]],
        ];
      } else {
        corners = [
          [lane[0], lane[2], value],
          [lane[1], lane[2], value],
          [lane[1], lane[3], value],
          [lane[0], lane[3], value],
        ];
      }
      for (let index = 0; index < corners.length; index += 1) {
        drawLine3D(
          corners[index],
          corners[(index + 1) % corners.length],
          "rgba(246, 201, 96, __ALPHA__)",
          0.85,
          2.0,
          projection
        );
      }
    }

    function colorFromIntensity(value, range, alpha) {
      if (!range || range.length < 2) {
        return `rgba(115, 217, 255, ${alpha})`;
      }
      const min = Number(range[0]);
      const max = Number(range[1]);
      const t = max <= min ? 0.5 : clamp((Number(value) - min) / (max - min), 0.0, 1.0);
      const r = Math.round(70 + 185 * t);
      const g = Math.round(150 + 80 * (1.0 - Math.abs(t - 0.4)));
      const b = Math.round(255 - 170 * t);
      return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }

    function drawPoints(frame, meta, projection) {
      const flat = frame.points_xyz || [];
      const lag = currentLag();
      const queued = state.pendingFrames.length;
      const pointCount = Number(frame.point_count || Math.floor(flat.length / 3));
      let pointStride = 1;
      if (lag >= LAG_DROP_THRESHOLD || queued >= FRAME_QUEUE_MAX || pointCount > 30000) {
        pointStride = 4;
      } else if (lag >= 6 || queued >= FRAME_QUEUE_TARGET || pointCount > 18000) {
        pointStride = 2;
      }
      const step = 3 * pointStride;
      ctx.fillStyle = "rgba(115, 217, 255, 0.92)";
      for (let index = 0; index + 2 < flat.length; index += step) {
        const projected = viewTransform([flat[index], flat[index + 1], flat[index + 2]], projection);
        if (!projected) {
          continue;
        }
        const size = clamp(3.2 / projected.depth * 30.0, 1.1, 3.2);
        ctx.fillRect(projected.x - size * 0.5, projected.y - size * 0.5, size, size);
      }
    }

    function statusColor(status) {
      const normalized = String(status || "");
      if (normalized.includes("matched") || normalized === "saved") {
        return "#51ff95";
      }
      if (normalized.includes("spawn") || normalized.includes("skip") || normalized === "track_exit") {
        return "#ffc763";
      }
      if (normalized.includes("miss") || normalized.includes("invalid") || normalized.includes("suppressed")) {
        return "#ff766b";
      }
      return "#73d9ff";
    }

    function drawMarker(point, color, radius, projection, fill = true) {
      const projected = viewTransform(point, projection);
      if (!projected) {
        return null;
      }
      ctx.beginPath();
      ctx.arc(projected.x, projected.y, radius, 0, Math.PI * 2.0);
      if (fill) {
        ctx.fillStyle = color;
        ctx.fill();
      } else {
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
      return projected;
    }

    function drawLabel(projected, text, color) {
      if (!projected) {
        return;
      }
      ctx.fillStyle = color;
      ctx.font = "12px IBM Plex Mono, monospace";
      ctx.fillText(text, projected.x + 8, projected.y - 8);
    }

    function drawTrackerOverlay(frame, projection) {
      for (const detection of (frame.detections || [])) {
        if (!detection.center) {
          continue;
        }
        const color = statusColor(detection.status);
        if (detection.min_bound && detection.max_bound) {
          const corners = boundsCorners(detection.min_bound, detection.max_bound);
          const edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7],
          ];
          for (const [a, b] of edges) {
            drawLine3D(corners[a], corners[b], rgbaFromHex(color, "__ALPHA__"), 0.68, 1.45, projection);
          }
        }
        const projected = drawMarker(detection.center, color, 4.0, projection, true);
        drawLabel(projected, `d${detection.detection_id}`, color);
      }
      for (const track of (frame.track_states || [])) {
        const color = statusColor(track.status);
        if (track.predicted_center && track.output_center) {
          drawLine3D(
            track.predicted_center,
            track.output_center,
            `rgba(115, 217, 255, __ALPHA__)`,
            0.55,
            1.2,
            projection
          );
        }
        const predicted = track.predicted_center
          ? drawMarker(track.predicted_center, "rgba(115, 217, 255, 0.85)", 4.5, projection, false)
          : null;
        const output = track.output_center ? drawMarker(track.output_center, color, 4.5, projection, true) : null;
        drawLabel(output || predicted, `t${track.track_id}`, color);
      }
    }

    function visibleOutcomes(meta) {
      const nowUnixSec = Date.now() / 1000.0;
      return (meta.track_outcomes || [])
        .filter((outcome) => {
          const status = String(outcome.status || "");
          if (!(status === "saved" || status.includes("merged"))) {
            return false;
          }
          const updatedAt = Number(outcome.updated_at_unix_sec || 0);
          return updatedAt > 0 && (nowUnixSec - updatedAt) <= OUTCOME_VISIBILITY_SEC;
        })
        .sort((left, right) => Number(right.updated_at_unix_sec || 0) - Number(left.updated_at_unix_sec || 0))
        .slice(0, MAX_VISIBLE_OUTCOMES);
    }

    function drawOutcomeOverlay(meta, projection) {
      const outcomes = visibleOutcomes(meta);
      for (const outcome of outcomes) {
        if (!outcome.last_center) {
          continue;
        }
        const color = statusColor(outcome.status);
        const projected = drawMarker(outcome.last_center, color, 5.0, projection, true);
        drawLabel(projected, `t${outcome.track_id}:${outcome.status}`, color);
      }
    }

    function render() {
      resizeCanvas();
      const width = canvas.clientWidth || 1;
      const height = canvas.clientHeight || 1;
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "rgba(4, 10, 16, 0.94)";
      ctx.fillRect(0, 0, width, height);

      const meta = state.meta;
      if (!meta) {
        return;
      }
      const projection = buildProjectionState();
      const currentFrame = state.currentFrame;

      drawLaneBox(meta, projection);
      drawExitLine(meta, projection);
      if (currentFrame) {
        drawPoints(currentFrame, meta, projection);
      }
      if (state.showTracker && currentFrame) {
        drawTrackerOverlay(currentFrame, projection);
      }
      if (state.showOutcomes) {
        drawOutcomeOverlay(meta, projection);
      }
      hudText.textContent = formatHud(meta, currentFrame);
    }

    function resetPollingState() {
      if (state.playbackTimerId) {
        clearTimeout(state.playbackTimerId);
      }
      state.currentFrame = null;
      state.pendingFrames = [];
      state.displayedSeq = -1;
      state.fetchedSeq = -1;
      state.targetLatestSeq = -1;
      state.droppedFrames = 0;
      state.displayHz = 0;
      state.lastDisplayedAtMs = 0;
      state.playbackTimerId = 0;
    }

    pauseBtn.addEventListener("click", () => {
      state.paused = !state.paused;
      setButtonState(pauseBtn, state.paused);
      pauseBtn.textContent = state.paused ? "Resume" : "Pause";
      if (!state.paused) {
        scheduleFramePump();
        schedulePlayback();
        void pollMeta(false);
      }
      scheduleRender();
    });

    trackerBtn.addEventListener("click", () => {
      state.showTracker = !state.showTracker;
      setButtonState(trackerBtn, state.showTracker);
      scheduleRender();
    });

    outcomeBtn.addEventListener("click", () => {
      state.showOutcomes = !state.showOutcomes;
      setButtonState(outcomeBtn, state.showOutcomes);
      scheduleRender();
    });

    refreshBtn.addEventListener("click", async () => {
      resetPollingState();
      await pollMeta(true);
      scheduleRender();
    });

    helpBtn.addEventListener("click", () => {
      state.helpVisible = !state.helpVisible;
      helpBox.classList.toggle("visible", state.helpVisible);
      setButtonState(helpBtn, state.helpVisible);
    });

    window.addEventListener("keydown", async (event) => {
      if (event.key === " ") {
        event.preventDefault();
        pauseBtn.click();
      } else if (event.key === "t" || event.key === "T") {
        trackerBtn.click();
      } else if (event.key === "f" || event.key === "F") {
        outcomeBtn.click();
      } else if (event.key === "r" || event.key === "R") {
        await pollMeta(true);
      } else if (event.key === "h" || event.key === "H") {
        helpBtn.click();
      }
    });

    canvas.addEventListener("mousedown", (event) => {
      state.camera.dragging = true;
      state.camera.lastX = event.clientX;
      state.camera.lastY = event.clientY;
      state.camera.userAdjusted = true;
    });

    window.addEventListener("mouseup", () => {
      state.camera.dragging = false;
    });

    window.addEventListener("mousemove", (event) => {
      if (!state.camera.dragging) {
        return;
      }
      const dx = event.clientX - state.camera.lastX;
      const dy = event.clientY - state.camera.lastY;
      state.camera.lastX = event.clientX;
      state.camera.lastY = event.clientY;
      state.camera.yaw -= dx * 0.008;
      state.camera.pitch = clamp(state.camera.pitch - dy * 0.006, -1.35, 1.35);
      state.camera.userAdjusted = true;
      scheduleRender();
    });

    canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      const delta = event.deltaY > 0 ? 1.08 : 0.92;
      state.camera.distance = clamp(state.camera.distance * delta, 4.0, 180.0);
      state.camera.userAdjusted = true;
      scheduleRender();
    }, { passive: false });

    window.addEventListener("resize", () => {
      resizeCanvas(true);
      scheduleRender();
    });

    async function main() {
      await pollMeta(true);
      setInterval(() => {
        pollMeta(false);
      }, META_POLL_INTERVAL_MS);
      scheduleRender();
    }

    main();
  </script>
</body>
</html>
"""


class LivePCDWebServer:
    def __init__(self, publisher: LiveFramePublisher, *, host: str = "0.0.0.0", port: int = 8765):
        self._publisher = publisher
        self._host = str(host)
        self._port = int(port)
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        if self._httpd is None:
            return int(self._port)
        return int(self._httpd.server_address[1])

    def start(self) -> None:
        if self._httpd is not None:
            return
        httpd = ThreadingHTTPServer((self._host, self._port), self._build_handler())
        httpd.daemon_threads = True
        thread = threading.Thread(target=httpd.serve_forever, name="tracking_pipeline_live_pcd_web", daemon=True)
        self._httpd = httpd
        self._thread = thread
        thread.start()

    def stop(self) -> None:
        httpd = self._httpd
        if httpd is None:
            return
        httpd.shutdown()
        httpd.server_close()
        self._httpd = None
        thread = self._thread
        self._thread = None
        if isinstance(thread, threading.Thread):
            thread.join(timeout=2.0)

    def _build_handler(self):
        publisher = self._publisher

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                _ = format, args

            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                path = parsed.path
                query = parse_qs(parsed.query)
                if path == "/":
                    self._write_html(_HTML)
                    return
                if path == "/api/live/meta":
                    self._write_json(HTTPStatus.OK, publisher.current_meta())
                    return
                if path == "/api/live/frames.json":
                    try:
                        start_sequence_id = int((query.get("start_sequence_id") or ["0"])[0])
                        limit = int((query.get("limit") or ["1"])[0])
                    except ValueError:
                        self._write_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_frame_batch_query"})
                        return
                    clamped_limit = max(1, min(limit, 32))
                    self._write_json(
                        HTTPStatus.OK,
                        {
                            "frames": publisher.get_frames(start_sequence_id, limit=clamped_limit),
                            "start_sequence_id": int(start_sequence_id),
                            "limit": int(clamped_limit),
                        },
                        allow_gzip=False,
                    )
                    return
                if path.startswith("/api/live/frame/") and path.endswith(".json"):
                    seq_part = path[len("/api/live/frame/") : -len(".json")]
                    try:
                        sequence_id = int(seq_part)
                    except ValueError:
                        self._write_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_sequence_id"})
                        return
                    payload = publisher.get_frame(sequence_id)
                    if payload is None:
                        self._write_json(HTTPStatus.NOT_FOUND, {"error": "frame_not_found", "sequence_id": sequence_id})
                        return
                    self._write_json(HTTPStatus.OK, payload)
                    return
                self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

            def _write_html(self, payload: str) -> None:
                body = payload.encode("utf-8")
                self._write_bytes(HTTPStatus.OK, "text/html; charset=utf-8", body)

            def _write_json(self, status: HTTPStatus, payload: dict[str, Any], *, allow_gzip: bool = True) -> None:
                body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
                self._write_bytes(status, "application/json; charset=utf-8", body, allow_gzip=allow_gzip)

            def _write_bytes(self, status: HTTPStatus, content_type: str, body: bytes, *, allow_gzip: bool = True) -> None:
                payload = body
                accepted = str(self.headers.get("Accept-Encoding", ""))
                use_gzip = allow_gzip and "gzip" in accepted.lower() and len(body) >= 1024
                if use_gzip:
                    payload = gzip.compress(body, compresslevel=5)
                self.send_response(int(status))
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Vary", "Accept-Encoding")
                if use_gzip:
                    self.send_header("Content-Encoding", "gzip")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                try:
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    return

        return _Handler
