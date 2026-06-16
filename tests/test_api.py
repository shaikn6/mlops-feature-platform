"""
Tests for the api package.
Since the API module only has __init__.py, we test the package
and create a comprehensive FastAPI endpoint test suite covering:
- Health check endpoint
- Feature serving endpoints
- Model registry endpoints
- Monitoring metrics endpoint
- Error handling
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch
from datetime import datetime

import pytest


# ---------------------------------------------------------------------------
# Minimal FastAPI app fixture (simulates the actual API layer)
# ---------------------------------------------------------------------------

@pytest.fixture
def app():
    """Create a minimal FastAPI test app mirroring expected endpoint structure."""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/httpx not installed")

    app = FastAPI(title="MLOps Feature Platform")

    # --- Mock services ---
    mock_feature_server = MagicMock()
    mock_registry = MagicMock()
    mock_drift_detector = MagicMock()
    mock_data_quality = MagicMock()
    mock_alert_manager = MagicMock()
    mock_dashboard = MagicMock()

    # Configure default return values
    mock_feature_server.get_online_features.return_value = [
        {"customer_id": "c1", "avg_spend_7d": 150.0}
    ]
    mock_feature_server.get_fraud_features_online.return_value = [
        {"customer_id": "c1", "avg_spend_7d": 100.0}
    ]
    mock_feature_server.get_credit_features_online.return_value = [
        {"customer_id": "c1", "credit_utilization": 0.35}
    ]

    # Store on app for test access
    app.state.feature_server = mock_feature_server
    app.state.registry = mock_registry
    app.state.dashboard = mock_dashboard

    # --- Endpoints ---

    @app.get("/health")
    def health():
        return {"status": "ok", "version": "1.0.0"}

    @app.get("/ready")
    def ready():
        return {"status": "ready"}

    @app.post("/features/online")
    def get_online_features(payload: dict):
        entity_rows = payload.get("entity_rows", [])
        features = payload.get("features", [])
        if not entity_rows:
            raise HTTPException(status_code=422, detail="entity_rows must not be empty")
        result = mock_feature_server.get_online_features(entity_rows, features)
        return {"data": result}

    @app.post("/features/fraud")
    def get_fraud_features(payload: dict):
        customer_ids = payload.get("customer_ids", [])
        if not customer_ids:
            raise HTTPException(status_code=422, detail="customer_ids must not be empty")
        result = mock_feature_server.get_fraud_features_online(customer_ids)
        return {"data": result, "count": len(result)}

    @app.post("/features/credit")
    def get_credit_features(payload: dict):
        customer_ids = payload.get("customer_ids", [])
        if not customer_ids:
            raise HTTPException(status_code=422, detail="customer_ids required")
        result = mock_feature_server.get_credit_features_online(customer_ids)
        return {"data": result, "count": len(result)}

    @app.get("/models/health")
    def models_health():
        ps = mock_dashboard.get_platform_summary()
        return ps.to_dict() if hasattr(ps, "to_dict") else {"status": "HEALTHY"}

    @app.get("/models/{model_name}/health")
    def model_health(model_name: str):
        snap = mock_dashboard.get_model_health(model_name)
        if snap is None:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        return snap.to_dict() if hasattr(snap, "to_dict") else {"model_name": model_name}

    @app.get("/metrics")
    def prometheus_metrics():
        from fastapi.responses import PlainTextResponse
        output = mock_dashboard.prometheus_metrics()
        return PlainTextResponse(content=output, media_type="text/plain")

    @app.post("/features/apply")
    def apply_features(payload: dict):
        try:
            mock_registry.apply()
            return {"status": "applied"}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.get("/models/retrain-candidates")
    def retrain_candidates():
        models = mock_dashboard.get_models_needing_retrain()
        return {"models": models}

    return app, mock_feature_server, mock_dashboard, mock_registry


@pytest.fixture
def client(app):
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        pytest.skip("httpx not installed")
    application, *mocks = app
    return TestClient(application), *mocks


# ---------------------------------------------------------------------------
# Health endpoints
# ---------------------------------------------------------------------------

class TestHealthEndpoints:
    def test_health_returns_200(self, client):
        tc, *_ = client
        resp = tc.get("/health")
        assert resp.status_code == 200

    def test_health_body_contains_status(self, client):
        tc, *_ = client
        resp = tc.get("/health")
        assert resp.json()["status"] == "ok"

    def test_health_body_contains_version(self, client):
        tc, *_ = client
        resp = tc.get("/health")
        assert "version" in resp.json()

    def test_ready_returns_200(self, client):
        tc, *_ = client
        resp = tc.get("/ready")
        assert resp.status_code == 200

    def test_ready_body(self, client):
        tc, *_ = client
        resp = tc.get("/ready")
        assert resp.json()["status"] == "ready"


# ---------------------------------------------------------------------------
# Feature serving endpoints
# ---------------------------------------------------------------------------

class TestOnlineFeaturesEndpoint:
    def test_valid_request_returns_200(self, client):
        tc, feat_server, *_ = client
        payload = {
            "entity_rows": [{"customer_id": "c1"}],
            "features": ["view:feat_a"],
        }
        resp = tc.post("/features/online", json=payload)
        assert resp.status_code == 200

    def test_valid_request_calls_feature_server(self, client):
        tc, feat_server, *_ = client
        feat_server.get_online_features.reset_mock()
        payload = {
            "entity_rows": [{"customer_id": "c1"}],
            "features": ["view:feat_a"],
        }
        tc.post("/features/online", json=payload)
        feat_server.get_online_features.assert_called_once()

    def test_empty_entity_rows_returns_422(self, client):
        tc, *_ = client
        payload = {"entity_rows": [], "features": ["view:feat"]}
        resp = tc.post("/features/online", json=payload)
        assert resp.status_code == 422

    def test_response_has_data_key(self, client):
        tc, *_ = client
        payload = {
            "entity_rows": [{"customer_id": "c1"}],
            "features": ["view:feat"],
        }
        resp = tc.post("/features/online", json=payload)
        assert "data" in resp.json()

    def test_feature_server_error_propagates(self, client):
        tc, feat_server, *_ = client
        feat_server.get_online_features.side_effect = RuntimeError("Redis down")
        payload = {
            "entity_rows": [{"customer_id": "c1"}],
            "features": ["view:feat"],
        }
        resp = tc.post("/features/online", json=payload)
        assert resp.status_code == 500


class TestFraudFeaturesEndpoint:
    def test_valid_request_returns_200(self, client):
        tc, *_ = client
        resp = tc.post("/features/fraud", json={"customer_ids": ["c1", "c2"]})
        assert resp.status_code == 200

    def test_empty_ids_returns_422(self, client):
        tc, *_ = client
        resp = tc.post("/features/fraud", json={"customer_ids": []})
        assert resp.status_code == 422

    def test_response_has_count(self, client):
        tc, *_ = client
        resp = tc.post("/features/fraud", json={"customer_ids": ["c1"]})
        data = resp.json()
        assert "count" in data


class TestCreditFeaturesEndpoint:
    def test_valid_request_returns_200(self, client):
        tc, *_ = client
        resp = tc.post("/features/credit", json={"customer_ids": ["c1"]})
        assert resp.status_code == 200

    def test_empty_ids_returns_422(self, client):
        tc, *_ = client
        resp = tc.post("/features/credit", json={"customer_ids": []})
        assert resp.status_code == 422

    def test_response_has_data_and_count(self, client):
        tc, *_ = client
        resp = tc.post("/features/credit", json={"customer_ids": ["c1"]})
        body = resp.json()
        assert "data" in body
        assert "count" in body


# ---------------------------------------------------------------------------
# Model monitoring endpoints
# ---------------------------------------------------------------------------

class TestModelsHealthEndpoint:
    def test_returns_200(self, client):
        tc, _, dashboard, *_ = client
        dashboard.get_platform_summary.return_value = MagicMock(
            to_dict=lambda: {"platform_status": "HEALTHY", "models": []}
        )
        resp = tc.get("/models/health")
        assert resp.status_code == 200

    def test_calls_dashboard(self, client):
        tc, _, dashboard, *_ = client
        dashboard.get_platform_summary.return_value = MagicMock(
            to_dict=lambda: {"status": "ok"}
        )
        dashboard.get_platform_summary.reset_mock()
        tc.get("/models/health")
        dashboard.get_platform_summary.assert_called_once()


class TestSingleModelHealthEndpoint:
    def test_known_model_returns_200(self, client):
        tc, _, dashboard, *_ = client
        dashboard.get_model_health.return_value = MagicMock(
            to_dict=lambda: {"model_name": "fraud_v3", "health_status": "HEALTHY"}
        )
        resp = tc.get("/models/fraud_v3/health")
        assert resp.status_code == 200

    def test_unknown_model_returns_404(self, client):
        tc, _, dashboard, *_ = client
        dashboard.get_model_health.return_value = None
        resp = tc.get("/models/unknown_model/health")
        assert resp.status_code == 404

    def test_model_name_passed_correctly(self, client):
        tc, _, dashboard, *_ = client
        dashboard.get_model_health.return_value = MagicMock(
            to_dict=lambda: {"model_name": "credit_v2"}
        )
        dashboard.get_model_health.reset_mock()
        tc.get("/models/credit_v2/health")
        dashboard.get_model_health.assert_called_once_with("credit_v2")


class TestPrometheusMetricsEndpoint:
    def test_returns_200(self, client):
        tc, _, dashboard, *_ = client
        dashboard.prometheus_metrics.return_value = "# HELP test gauge\n"
        resp = tc.get("/metrics")
        assert resp.status_code == 200

    def test_content_type_is_text(self, client):
        tc, _, dashboard, *_ = client
        dashboard.prometheus_metrics.return_value = "metric 1.0\n"
        resp = tc.get("/metrics")
        assert "text" in resp.headers.get("content-type", "")

    def test_calls_dashboard_prometheus_metrics(self, client):
        tc, _, dashboard, *_ = client
        dashboard.prometheus_metrics.return_value = ""
        dashboard.prometheus_metrics.reset_mock()
        tc.get("/metrics")
        dashboard.prometheus_metrics.assert_called_once()


class TestApplyFeaturesEndpoint:
    def test_returns_200_on_success(self, client):
        tc, _, _, registry = client
        registry.apply.return_value = None
        resp = tc.post("/features/apply", json={})
        assert resp.status_code == 200

    def test_returns_applied_status(self, client):
        tc, _, _, registry = client
        registry.apply.return_value = None
        resp = tc.post("/features/apply", json={})
        assert resp.json()["status"] == "applied"

    def test_registry_error_returns_500(self, client):
        tc, _, _, registry = client
        registry.apply.side_effect = RuntimeError("Feast apply failed")
        resp = tc.post("/features/apply", json={})
        assert resp.status_code == 500


class TestRetrainCandidatesEndpoint:
    def test_returns_list(self, client):
        tc, _, dashboard, *_ = client
        dashboard.get_models_needing_retrain.return_value = ["fraud_v3", "credit_v2"]
        resp = tc.get("/models/retrain-candidates")
        assert resp.status_code == 200
        assert resp.json()["models"] == ["fraud_v3", "credit_v2"]

    def test_empty_list_when_none_need_retrain(self, client):
        tc, _, dashboard, *_ = client
        dashboard.get_models_needing_retrain.return_value = []
        resp = tc.get("/models/retrain-candidates")
        assert resp.json()["models"] == []


# ---------------------------------------------------------------------------
# API package structure
# ---------------------------------------------------------------------------

class TestApiPackage:
    def test_api_init_importable(self):
        import api
        assert api.__version__ == "1.0.0"

