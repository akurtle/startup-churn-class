from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from threading import Lock


class APIMetricsCollector:
    def __init__(self) -> None:
        self._lock = Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.requests_total = 0
            self.request_errors_total = 0
            self.request_latency_total_ms = 0.0
            self.predictions_total = 0
            self.prediction_errors_total = 0
            self.prediction_latency_total_ms = 0.0
            self.status_counts = defaultdict(int)
            self.path_counts = defaultdict(int)
            self.last_updated_utc: str | None = None

    def record_request(self, *, path: str, status_code: int, duration_ms: float) -> None:
        with self._lock:
            self.requests_total += 1
            self.request_latency_total_ms += duration_ms
            self.status_counts[str(status_code)] += 1
            self.path_counts[path] += 1

            if status_code >= 400:
                self.request_errors_total += 1

            if path == "/predict":
                self.predictions_total += 1
                self.prediction_latency_total_ms += duration_ms
                if status_code >= 400:
                    self.prediction_errors_total += 1

            self.last_updated_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            request_error_rate = (
                self.request_errors_total / self.requests_total if self.requests_total else 0.0
            )
            prediction_error_rate = (
                self.prediction_errors_total / self.predictions_total
                if self.predictions_total
                else 0.0
            )
            avg_request_latency_ms = (
                self.request_latency_total_ms / self.requests_total if self.requests_total else 0.0
            )
            avg_prediction_latency_ms = (
                self.prediction_latency_total_ms / self.predictions_total
                if self.predictions_total
                else 0.0
            )

            return {
                "requests_total": self.requests_total,
                "request_errors_total": self.request_errors_total,
                "request_error_rate": round(request_error_rate, 4),
                "avg_request_latency_ms": round(avg_request_latency_ms, 2),
                "predictions_total": self.predictions_total,
                "prediction_errors_total": self.prediction_errors_total,
                "prediction_error_rate": round(prediction_error_rate, 4),
                "avg_prediction_latency_ms": round(avg_prediction_latency_ms, 2),
                "status_counts": dict(self.status_counts),
                "path_counts": dict(self.path_counts),
                "last_updated_utc": self.last_updated_utc,
            }


api_metrics = APIMetricsCollector()
