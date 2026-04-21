"""
Tests for model.py runtime state management.

Tests accessor functions, readiness status, and reset behavior.
"""

from __future__ import annotations

import pytest

from app.model import (
    RuntimeState,
    get_artifacts,
    get_categorical_vocabs,
    get_feature_pipeline_version,
    get_model,
    get_model_metadata,
    get_models_by_manufacturer,
    get_numeric_medians,
    get_readiness_status,
    get_runtime_state,
    is_runtime_ready,
    reset_runtime_state,
)


class TestRuntimeState:
    def test_state_is_loaded(self):
        assert is_runtime_ready()

    def test_get_runtime_state_returns_state(self):
        state = get_runtime_state()
        assert isinstance(state, RuntimeState)

    def test_state_has_model(self):
        model = get_model()
        assert model is not None

    def test_state_has_artifacts(self):
        artifacts = get_artifacts()
        assert isinstance(artifacts, dict)
        assert "model_metadata" in artifacts
        assert "categorical_vocabs" in artifacts
        assert "models_by_manufacturer" in artifacts
        assert "numeric_medians" in artifacts


class TestAccessors:
    def test_get_model_metadata(self):
        metadata = get_model_metadata()
        assert "feature_columns" in metadata
        assert "feature_pipeline_version" in metadata

    def test_get_categorical_vocabs(self):
        vocabs = get_categorical_vocabs()
        assert "manufacturer" in vocabs
        assert isinstance(vocabs["manufacturer"], list)

    def test_get_models_by_manufacturer(self):
        catalog = get_models_by_manufacturer()
        assert "toyota" in catalog
        assert isinstance(catalog["toyota"], list)

    def test_get_numeric_medians(self):
        medians = get_numeric_medians()
        assert "mpg_avg" in medians
        assert isinstance(medians["mpg_avg"], float)

    def test_get_feature_pipeline_version(self):
        version = get_feature_pipeline_version()
        assert version == "v3"


class TestReadinessStatus:
    def test_ready_when_loaded(self):
        status = get_readiness_status()
        assert status["status"] == "ready"
        assert status["model_loaded"] is True
        assert status["artifacts_loaded"] is True
        assert status["feature_pipeline_version"] == "v3"

    def test_not_ready_when_reset(self):
        reset_runtime_state()
        status = get_readiness_status()
        assert status["status"] == "not_ready"
        assert status["model_loaded"] is False
        assert status["artifacts_loaded"] is False


class TestReset:
    def test_reset_clears_state(self):
        assert is_runtime_ready()
        reset_runtime_state()
        assert not is_runtime_ready()

    def test_get_model_raises_after_reset(self):
        reset_runtime_state()
        with pytest.raises(RuntimeError, match="Runtime state not loaded"):
            get_model()

    def test_get_runtime_state_raises_after_reset(self):
        reset_runtime_state()
        with pytest.raises(RuntimeError, match="Runtime state not loaded"):
            get_runtime_state()
