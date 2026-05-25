"""Tests for Prometheus metrics instrumentation."""

from __future__ import annotations


class TestMetricsEndpoint:
    """Verify the /metrics endpoint is accessible and well-formed."""

    def test_metrics_returns_200(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_returns_prometheus_format(self, client):
        response = client.get("/metrics")
        content_type = response.headers.get("content-type", "")
        assert "text/plain" in content_type

    def test_metrics_contains_custom_metrics(self, client):
        response = client.get("/metrics")
        body = response.text
        assert "prediction_requests_total" in body
        assert "prediction_latency_seconds" in body
        assert "prediction_errors_total" in body


class TestMetricsInstrumentation:
    """Verify that predictions update the correct metric labels."""

    def test_successful_prediction_increments_counter(
        self, client, sample_payload
    ):
        response = client.post("/api/v1/predict", json=sample_payload)
        assert response.status_code == 200

        metrics_response = client.get("/metrics")
        body = metrics_response.text
        assert 'prediction_requests_total{status="success"}' in body

    def test_validation_error_increments_error_counter(self, client):
        response = client.post("/api/v1/predict", json={})
        assert response.status_code == 422

        metrics_response = client.get("/metrics")
        body = metrics_response.text
        assert 'prediction_requests_total{status="error"}' in body
        assert 'prediction_errors_total{error_type="validation_error"}' in body

    def test_latency_histogram_records_observations(
        self, client, sample_payload
    ):
        client.post("/api/v1/predict", json=sample_payload)
        metrics_response = client.get("/metrics")
        body = metrics_response.text
        assert "prediction_latency_seconds_bucket" in body

    def test_metrics_listed_in_root(self, client):
        body = client.get("/").json()
        assert body.get("metrics_url") == "/metrics"