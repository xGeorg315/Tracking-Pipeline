from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TrackAggregatedEvent:
    track_id: int
    status: str


@dataclass(slots=True)
class RunCompletedEvent:
    output_dir: str
    saved_aggregates: int
