from __future__ import annotations

from types import SimpleNamespace

from tracking_pipeline.application import performance


def test_performance_profiler_snapshot_derives_totals(monkeypatch) -> None:
    profiler = performance.PerformanceProfiler()
    profiler._started_wall = 10.0
    profiler._started_cpu = 5.0
    profiler._stage_wall.update(
        {
            "read_frames": 0.2,
            "cluster_frames": 1.2,
            "tracker_steps": 0.8,
            "tracker_finalize": 0.1,
            "postprocess_tracks": 0.3,
            "accumulate_tracks": 1.4,
            "write_tracks": 0.15,
            "write_summary": 0.05,
        }
    )
    profiler._stage_cpu.update(
        {
            "cluster_frames": 0.7,
            "tracker_steps": 0.5,
            "tracker_finalize": 0.05,
            "postprocess_tracks": 0.2,
            "accumulate_tracks": 0.9,
        }
    )
    profiler._stage_calls.update({"cluster_frames": 12, "tracker_steps": 12, "accumulate_tracks": 4})
    monkeypatch.setattr(performance.time, "perf_counter", lambda: 15.0)
    monkeypatch.setattr(performance.time, "process_time", lambda: 7.5)
    monkeypatch.setattr(performance, "_peak_rss_mb", lambda: 123.4)

    snapshot = profiler.snapshot()

    assert snapshot.total_wall_seconds == 5.0
    assert snapshot.total_cpu_seconds == 2.5
    assert snapshot.compute_wall_seconds == 3.8
    assert snapshot.compute_cpu_seconds == 2.35
    assert snapshot.io_wall_seconds == 0.4
    assert snapshot.peak_rss_mb == 123.4
    assert snapshot.stages["cluster_frames"].call_count == 12
    assert snapshot.stages["accumulate_tracks"].wall_seconds == 1.4
    assert "build_components" in snapshot.stages


def test_peak_rss_mb_normalizes_macos_units(monkeypatch) -> None:
    fake_resource = SimpleNamespace(
        RUSAGE_SELF=1,
        getrusage=lambda _: SimpleNamespace(ru_maxrss=5 * 1024 * 1024),
    )
    monkeypatch.setattr(performance, "resource", fake_resource)
    monkeypatch.setattr(performance.sys, "platform", "darwin")

    peak_rss_mb = performance._peak_rss_mb()

    assert peak_rss_mb == 5.0


def test_peak_rss_mb_normalizes_linux_units(monkeypatch) -> None:
    fake_resource = SimpleNamespace(
        RUSAGE_SELF=1,
        getrusage=lambda _: SimpleNamespace(ru_maxrss=2048),
    )
    monkeypatch.setattr(performance, "resource", fake_resource)
    monkeypatch.setattr(performance.sys, "platform", "linux")

    peak_rss_mb = performance._peak_rss_mb()

    assert peak_rss_mb == 2.0
