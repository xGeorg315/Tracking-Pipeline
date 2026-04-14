from __future__ import annotations

import io
import logging
import os
import sys
import types

from tracking_pipeline.application.run_pipeline import _LiveCliStatusWriter, _apply_runtime_limits
from tracking_pipeline.config.models import RuntimeConfig


class _TtyStringIO(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_live_cli_status_writer_updates_in_place_and_finishes_with_newline() -> None:
    stream = _TtyStringIO()
    writer = _LiveCliStatusWriter(stream)

    assert writer.update("mqtt=yes raw=1")
    assert writer.update("mqtt=yes raw=2")
    assert "\n" not in stream.getvalue()
    assert stream.getvalue().count("\r") >= 2
    assert stream.getvalue().count("\033[2K") >= 1

    writer.finish()

    assert stream.getvalue().endswith("\n")


def test_live_cli_status_writer_wraps_to_terminal_width_without_truncation() -> None:
    stream = _TtyStringIO()
    writer = _LiveCliStatusWriter(stream)
    writer._terminal_width = lambda: 24  # type: ignore[method-assign]

    assert writer.update("mqtt=yes raw=123 state=waiting_for_first_raw_frame")

    rendered = stream.getvalue()
    assert "..." not in rendered
    assert "\n" in rendered
    assert rendered.replace("\n", "") == "mqtt=yes raw=123 state=waiting_for_first_raw_frame"


def test_live_cli_status_writer_rewrites_multiline_status_in_place() -> None:
    stream = _TtyStringIO()
    writer = _LiveCliStatusWriter(stream)
    writer._terminal_width = lambda: 24  # type: ignore[method-assign]

    assert writer.update("mqtt=yes raw=123 state=waiting_for_first_raw_frame")
    assert writer.update("mqtt=yes raw=124 state=streaming")

    rendered = stream.getvalue()
    assert "\033[1A" in rendered
    assert rendered.count("\033[2K") >= 2


def test_apply_runtime_limits_uses_affinity_and_sets_thread_env(monkeypatch) -> None:
    affinity_calls: list[tuple[int, set[int]]] = []
    fake_torch = types.SimpleNamespace(
        set_num_threads=lambda count: affinity_calls.append((-1, {int(count)})),
        set_num_interop_threads=lambda count: affinity_calls.append((-2, {int(count)})),
    )
    logger = logging.getLogger("runtime_limits_test")

    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {4, 5, 6, 7})
    monkeypatch.setattr(os, "sched_setaffinity", lambda pid, cpus: affinity_calls.append((int(pid), set(cpus))))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    limits = _apply_runtime_limits(RuntimeConfig(cpu_cores=2), logger)

    assert limits["requested_cpu_cores"] == 2
    assert limits["applied_cpu_cores"] == 2
    assert limits["affinity_applied"] is True
    assert limits["affinity_cpus"] == [4, 5]
    assert affinity_calls[0] == (0, {4, 5})
    assert os.environ["OMP_NUM_THREADS"] == "2"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "2"
    assert os.environ["MKL_NUM_THREADS"] == "2"
    assert os.environ["NUMEXPR_NUM_THREADS"] == "2"
