from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from tracking_pipeline.config.models import PipelineConfig
from tracking_pipeline.domain.models import (
    AggregateResult,
    ArticulatedMergeDebugEvent,
    ClassificationPrediction,
    ClusterResult,
    FrameData,
    FrameTrackingState,
    GTMatchResult,
    ObjectLabelData,
    RunSummary,
    Track,
    TrackOutcomeDebug,
)
from tracking_pipeline.domain.value_objects import LaneBox


class FrameReader(Protocol):
    def iter_frames(self, input_paths: list[str]) -> list[FrameData]: ...


class Clusterer(Protocol):
    def cluster(self, frame: FrameData, lane_box: LaneBox) -> ClusterResult: ...


class Tracker(Protocol):
    def step(self, detections, frame_idx: int, frame_timestamp_ns: int) -> FrameTrackingState: ...
    def finalize(self) -> dict[int, Track]: ...


class TrackPostprocessor(Protocol):
    name: str

    def process(self, tracks: dict[int, Track]) -> dict[int, Track]: ...


class Accumulator(Protocol):
    def accumulate(self, track: Track, lane_box: LaneBox) -> AggregateResult: ...


class ArtifactWriter(Protocol):
    def prepare_run_dir(self, config: PipelineConfig) -> Path: ...
    def write_config_snapshot(self, run_dir: Path, config: PipelineConfig) -> None: ...
    def write_aggregate(self, run_dir: Path, result: AggregateResult, save_intensity: bool = False) -> None: ...
    def write_object_list(self, run_dir: Path, object_labels: dict[int, ObjectLabelData]) -> None: ...
    def write_gt_matching(
        self,
        run_dir: Path,
        matches: list[GTMatchResult],
        unmatched_saved_tracks: list[GTMatchResult],
        unmatched_gt_objects: list[GTMatchResult],
        summary: dict[str, int | float | str],
    ) -> None: ...
    def write_tracker_debug(self, run_dir: Path, states: list[FrameTrackingState]) -> None: ...
    def write_track_outcomes(self, run_dir: Path, track_outcomes: dict[int, TrackOutcomeDebug]) -> None: ...
    def write_class_stats(self, run_dir: Path, class_stats: dict[str, object]) -> None: ...
    def write_summary(self, run_dir: Path, summary: RunSummary) -> None: ...
    def write_tracks(self, run_dir: Path, tracks: dict[int, Track], aggregate_results: list[AggregateResult]) -> None: ...


class ObjectClassifier(Protocol):
    backend: str

    def classify_points(self, points: np.ndarray) -> ClassificationPrediction: ...


class ReplayViewer(Protocol):
    def replay(
        self,
        states: list[FrameTrackingState],
        lane_box: LaneBox,
        aggregate_results: dict[int, AggregateResult],
        track_outcomes: dict[int, TrackOutcomeDebug],
        articulated_merge_debug_events: list[ArticulatedMergeDebugEvent],
    ) -> None: ...
