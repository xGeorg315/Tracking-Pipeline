from __future__ import annotations

import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

from tracking_pipeline.domain.models import RunPerformance, StagePerformance

try:
    import resource
except ImportError:  # pragma: no cover - exercised via tests with monkeypatching
    resource = None


STAGE_NAMES = (
    "build_components",
    "prepare_output",
    "read_frames",
    "cluster_frames",
    "tracker_steps",
    "tracker_finalize",
    "postprocess_tracks",
    "accumulate_tracks",
    "write_aggregates",
    "write_object_list",
    "write_tracks",
    "write_summary",
)
COMPUTE_STAGE_NAMES = (
    "cluster_frames",
    "tracker_steps",
    "tracker_finalize",
    "postprocess_tracks",
    "accumulate_tracks",
)
IO_STAGE_NAMES = (
    "read_frames",
    "write_aggregates",
    "write_object_list",
    "write_tracks",
    "write_summary",
)


@dataclass(slots=True)
class PerformanceProfiler:
    _stage_wall: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    _stage_cpu: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    _stage_calls: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _started_wall: float = field(default_factory=time.perf_counter)
    _started_cpu: float = field(default_factory=time.process_time)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        started_wall = time.perf_counter()
        started_cpu = time.process_time()
        try:
            yield
        finally:
            self._stage_wall[name] += max(0.0, time.perf_counter() - started_wall)
            self._stage_cpu[name] += max(0.0, time.process_time() - started_cpu)
            self._stage_calls[name] += 1

    def snapshot(self) -> RunPerformance:
        stages = {
            name: StagePerformance(
                wall_seconds=float(self._stage_wall.get(name, 0.0)),
                cpu_seconds=float(self._stage_cpu.get(name, 0.0)),
                call_count=int(self._stage_calls.get(name, 0)),
            )
            for name in STAGE_NAMES
        }
        compute_wall_seconds = float(sum(stages[name].wall_seconds for name in COMPUTE_STAGE_NAMES))
        compute_cpu_seconds = float(sum(stages[name].cpu_seconds for name in COMPUTE_STAGE_NAMES))
        io_wall_seconds = float(sum(stages[name].wall_seconds for name in IO_STAGE_NAMES))
        return RunPerformance(
            total_wall_seconds=max(0.0, float(time.perf_counter() - self._started_wall)),
            total_cpu_seconds=max(0.0, float(time.process_time() - self._started_cpu)),
            compute_wall_seconds=compute_wall_seconds,
            compute_cpu_seconds=compute_cpu_seconds,
            io_wall_seconds=io_wall_seconds,
            peak_rss_mb=_peak_rss_mb(),
            stages=stages,
        )


def _peak_rss_mb() -> float | None:
    if resource is None:
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF)
    value = float(getattr(usage, "ru_maxrss", 0.0) or 0.0)
    if value <= 0.0:
        return 0.0
    if sys.platform == "darwin":
        return value / (1024.0 * 1024.0)
    return value / 1024.0
