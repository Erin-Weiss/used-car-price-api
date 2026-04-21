"""
Tests for operational endpoints: /, /health, /ready.
"""

from __future__ import annotations

import app.main as main_module


# ===============================
# Root endpoint
# ===============================

class TestRoot:
    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_returns_expected_urls(self, client):
        body = client.get("/").json()
        assert body["app_name"] == main_module.settings.app_name
        assert body["version"] == main_module.settings.app_version
        assert body["docs_url"] == f"{main_module.settings.api_prefix}/docs"
        assert body["health_url"] == "/health"
        assert body["ready_url"] == "/ready"
        assert body["predict_url"] == f"{main_module.settings.api_prefix}/predict"


# ===============================
# Health endpoint
# ===============================

class TestHealth:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_expected_payload(self, client):
        body = client.get("/health").json()
        assert body == {
            "status": "ok",
            "app_name": main_module.settings.app_name,
            "version": main_module.settings.app_version,
            "environment": main_module.settings.environment,
        }

    def test_health_environment_is_valid(self, client):
        body = client.get("/health").json()
        assert body["environment"] in ("dev", "test", "prod")


# ===============================
# Readiness endpoint
# ===============================

class TestReadiness:
    def test_ready_returns_200(self, client):
        response = client.get("/ready")
        assert response.status_code == 200

    def test_ready_returns_ready_payload(self, client, monkeypatch):
        monkeypatch.setattr(
            main_module,
            "get_readiness_status",
            lambda: {
                "status": "ready",
                "model_loaded": True,
                "artifacts_loaded": True,
                "feature_pipeline_version": "v3",
            },
        )
        body = client.get("/ready").json()
        assert body == {
            "status": "ready",
            "model_loaded": True,
            "artifacts_loaded": True,
            "feature_pipeline_version": "v3",
        }

    def test_ready_returns_not_ready_payload(self, client, monkeypatch):
        monkeypatch.setattr(
            main_module,
            "get_readiness_status",
            lambda: {
                "status": "not_ready",
                "model_loaded": False,
                "artifacts_loaded": False,
                "feature_pipeline_version": "unknown",
            },
        )
        body = client.get("/ready").json()
        assert body == {
            "status": "not_ready",
            "model_loaded": False,
            "artifacts_loaded": False,
            "feature_pipeline_version": "unknown",
        }
