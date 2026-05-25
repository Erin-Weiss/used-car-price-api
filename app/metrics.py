"""
Prometheus metrics for the Used Car Price API.

Exposes three custom metrics for ML model serving observability:
  - prediction_requests_total (Counter): request count by status
  - prediction_latency_seconds (Histogram): prediction duration distribution
  - prediction_errors_total (Counter): error count by error type

Metrics are served at GET /metrics in Prometheus text exposition format.
"""

from __future__ import annotations

import time

from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from starlette.requests import Request
from starlette.responses import Response


# ===============================
# Metric definitions
# ===============================

prediction_requests_total = Counter(
    "prediction_requests_total",
    "Total number of prediction requests received",
    ["status"],
)

prediction_latency_seconds = Histogram(
    "prediction_latency_seconds",
    "Time spent processing a prediction request (seconds)",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

prediction_errors_total = Counter(
    "prediction_errors_total",
    "Total number of failed prediction requests",
    ["error_type"],
)

# Pre-initialize all label combinations so /metrics output
# is predictable before the first request arrives.
prediction_requests_total.labels(status="success")
prediction_requests_total.labels(status="error")
prediction_errors_total.labels(error_type="validation_error")
prediction_errors_total.labels(error_type="server_error")
prediction_errors_total.labels(error_type="unexpected_error")


# ===============================
# Metrics endpoint
# ===============================

async def metrics_endpoint(request: Request) -> Response:
    """Serve all registered metrics in Prometheus text format."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


# ===============================
# Instrumentation middleware
# ===============================

class MetricsMiddleware:
    """ASGI middleware that records latency and status for prediction requests.

    Only instruments POST requests to the /predict endpoint.
    All other routes pass through without instrumentation.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        method = scope.get("method", "")

        if method == "POST" and path.endswith("/predict"):
            start_time = time.perf_counter()
            status_code = 500

            async def send_wrapper(message):
                nonlocal status_code
                if message["type"] == "http.response.start":
                    status_code = message["status"]
                await send(message)

            try:
                await self.app(scope, receive, send_wrapper)
            finally:
                duration = time.perf_counter() - start_time
                prediction_latency_seconds.observe(duration)

                if status_code < 400:
                    prediction_requests_total.labels(status="success").inc()
                elif status_code == 422:
                    prediction_requests_total.labels(status="error").inc()
                    prediction_errors_total.labels(
                        error_type="validation_error"
                    ).inc()
                elif status_code >= 500:
                    prediction_requests_total.labels(status="error").inc()
                    prediction_errors_total.labels(
                        error_type="server_error"
                    ).inc()
                else:
                    prediction_requests_total.labels(status="error").inc()
                    prediction_errors_total.labels(
                        error_type="unexpected_error"
                    ).inc()
        else:
            await self.app(scope, receive, send)


# ===============================
# Setup
# ===============================

def setup_metrics(app) -> None:
    """Register the /metrics endpoint and prediction instrumentation middleware.
    """
    app.add_api_route(
        "/metrics",
        metrics_endpoint,
        methods=["GET"],
        include_in_schema=False,
    )
    app.add_middleware(MetricsMiddleware)