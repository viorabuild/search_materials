"""Lightweight fallback implementation of a subset of prometheus_client.

This module exists to provide minimal metric collection features in environments
where the real ``prometheus_client`` package cannot be installed (for example,
in offline CI sandboxes). It mirrors only the pieces of functionality that the
project relies on: ``Counter``, ``Histogram`` with timing support, and a simple
HTTP endpoint that exposes metrics in the Prometheus text format.

The goal is not to be feature complete—just compatible enough for local tests
and demos. Whenever the genuine library is available it should be preferred.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import List, Sequence

__all__ = ["Counter", "Histogram", "start_http_server"]


_REGISTRY_LOCK = threading.Lock()
_REGISTRY: List["_MetricBase"] = []


def _register(metric: "_MetricBase") -> None:
    with _REGISTRY_LOCK:
        _REGISTRY.append(metric)


def _collect_metrics() -> str:
    with _REGISTRY_LOCK:
        metrics = list(_REGISTRY)

    lines: List[str] = []
    for metric in metrics:
        lines.extend(metric.render())

    return "\n".join(lines) + ("\n" if lines else "")


class _MetricBase:
    def __init__(self, name: str, documentation: str) -> None:
        self._name = name
        self._documentation = documentation.strip()
        _register(self)

    def render(self) -> Sequence[str]:  # pragma: no cover - implemented in subclasses
        raise NotImplementedError


class _Value:
    def __init__(self, initial: float = 0.0) -> None:
        self._value = float(initial)
        self._lock = threading.Lock()

    def inc(self, amount: float) -> None:
        with self._lock:
            self._value += float(amount)

    def get(self) -> float:
        with self._lock:
            return self._value

    def set(self, value: float) -> None:
        with self._lock:
            self._value = float(value)


class Counter(_MetricBase):
    """A minimal counter implementation compatible with prometheus_client."""

    def __init__(self, name: str, documentation: str) -> None:
        super().__init__(name, documentation)
        self._value = _Value(0.0)

    def inc(self, amount: float = 1.0) -> None:
        self._value.inc(amount)

    def render(self) -> Sequence[str]:
        value = self._value.get()
        return [
            f"# HELP {self._name} {self._documentation}",
            f"# TYPE {self._name} counter",
            f"{self._name} {value}",
        ]


@dataclass
class _Timer:
    histogram: "Histogram"
    start: float | None = None

    def __enter__(self) -> None:
        self.start = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.start is None:
            return
        elapsed = time.perf_counter() - self.start
        self.histogram.observe(elapsed)


class Histogram(_MetricBase):
    """Simplified histogram with fixed buckets and timing helper."""

    DEFAULT_BUCKETS: Sequence[float] = (
        0.005,
        0.01,
        0.025,
        0.05,
        0.075,
        0.1,
        0.25,
        0.5,
        0.75,
        1.0,
        2.5,
        5.0,
        7.5,
        10.0,
    )

    def __init__(self, name: str, documentation: str, buckets: Sequence[float] | None = None) -> None:
        super().__init__(name, documentation)
        self._buckets: Sequence[float] = tuple(sorted(buckets or self.DEFAULT_BUCKETS))
        self._bucket_counts = [0 for _ in self._buckets]
        self._count = _Value(0.0)
        self._sum = _Value(0.0)

    def observe(self, value: float) -> None:
        value = max(float(value), 0.0)
        self._sum.inc(value)
        self._count.inc(1.0)
        for idx, upper in enumerate(self._buckets):
            if value <= upper:
                self._bucket_counts[idx] += 1

    def time(self) -> _Timer:
        return _Timer(histogram=self)

    def render(self) -> Sequence[str]:
        lines = [
            f"# HELP {self._name} {self._documentation}",
            f"# TYPE {self._name} histogram",
        ]

        cumulative = 0
        for upper, count in zip(self._buckets, self._bucket_counts):
            cumulative = count
            lines.append(f"{self._name}_bucket{{le=\"{upper}\"}} {cumulative}")

        total_count = int(self._count.get())
        lines.append(f"{self._name}_bucket{{le=\"+Inf\"}} {total_count}")
        lines.append(f"{self._name}_sum {self._sum.get()}")
        lines.append(f"{self._name}_count {total_count}")
        return lines


class _MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):  # type: ignore[override]
        payload = _collect_metrics().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return


def start_http_server(port: int) -> None:
    """Expose the registered metrics via a lightweight HTTP server."""

    server = ThreadingHTTPServer(("", port), _MetricsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
