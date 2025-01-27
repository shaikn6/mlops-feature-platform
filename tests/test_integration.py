"""
Integration-style tests that exercise multiple components together.
Covers end-to-end flow scenarios with mocked external dependencies.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, call
import json

import numpy as np
import pandas as pd
import pytest

from monitoring.dashboard import ModelHealthSummary, MonitoringDashboard, PlatformSummary
from monitoring.drift_detector import DriftDetector, DriftReport, FeatureDriftResult, _compute_psi
from monitoring.data_quality import DataQualityMonitor, ValidationResult
from monitoring.alert_manager import Alert, AlertManager, AlertResult, AlertSeverity


# ---------------------------------------------------------------------------
# Full drift → alert → dashboard cycle
# ---------------------------------------------------------------------------

class TestDriftAlertDashboardCycle:
    def _build_drift_report(self, model_name: str, ratio: float, retrain: bool):
        feats = [
            FeatureDriftResult(
                feature_name=f"feature_{i}",
                stattest_name="ks",
                drift_score=0.01,
                drifted=True,
                psi=0.3,
                psi_label="HIGH",
            )
            for i in range(int(ratio * 10))
        ]
        return DriftReport(
            model_name=model_name,
            reference_date=datetime(2024, 1, 1),
            current_date=datetime(2024, 6, 1),
            feature_results=feats,
            total_features=10,
            drifted_features=len(feats),
            dataset_drift_ratio=ratio,
            dataset_drifted=ratio >= 0.2,
            recommend_retrain=retrain,
        )

    def test_full_drift_cycle(self):
        """
        Simulate: detect drift → record to dashboard → send alert → verify state.
        """
        report = self._build_drift_report("fraud_v3", ratio=0.4, retrain=True)

        dashboard = MonitoringDashboard()
        snap = dashboard.record_drift(report)

        assert snap.recommend_retrain is True
        assert snap.health_status == "CRITICAL"
        assert "fraud_v3" in dashboard.get_models_needing_retrain()

        alerter = AlertManager()
        with patch.object(alerter, "dispatch") as mock_dispatch:
            mock_dispatch.return_value = AlertResult(
                alert=MagicMock(),
                channels_succeeded=["slack"],
            )
            result = alerter.alert_drift(report, "fraud_v3")
            mock_dispatch.assert_called_once()
            dispatched = mock_dispatch.call_args[0][0]
            assert dispatched.severity == AlertSeverity.CRITICAL

    def test_healthy_model_no_alert(self):
        """Models with low drift should not trigger critical alerts."""
        report = self._build_drift_report("fraud_v3", ratio=0.05, retrain=False)

        dashboard = MonitoringDashboard()
        snap = dashboard.record_drift(report)
        assert snap.health_status in ("HEALTHY", "DEGRADED")

        alerter = AlertManager()
        with patch.object(alerter, "dispatch") as mock_dispatch:
            mock_dispatch.return_value = MagicMock()
            alerter.alert_drift(report, "fraud_v3")
            dispatched = mock_dispatch.call_args[0][0]
            assert dispatched.severity != AlertSeverity.CRITICAL

    def test_multiple_models_platform_status(self):
        """Platform status should reflect worst model state."""
        dashboard = MonitoringDashboard()

        # Model 1: critical
        r1 = self._build_drift_report("fraud_v3", ratio=0.5, retrain=True)
        dashboard.record_drift(r1)

        # Model 2: healthy
        r2 = self._build_drift_report("credit_v2", ratio=0.02, retrain=False)
        dashboard.record_drift(r2)

        ps = dashboard.get_platform_summary()
        assert ps.platform_status == "CRITICAL"
        assert ps.critical_count == 1
        assert len(ps.models) == 2


# ---------------------------------------------------------------------------
# DQ → Alert → Dashboard cycle
# ---------------------------------------------------------------------------

class TestDQAlertCycle:
    def _make_dq_result(self, suite, pct, success):
        return ValidationResult(
            suite_name=suite,
            success=success,
            evaluated_expectations=10,
            successful_expectations=int(pct / 10),
            failed_expectations=10 - int(pct / 10),
            success_percent=pct,
        )

    def test_dq_failure_triggers_critical_alert(self):
        result = self._make_dq_result("transactions_suite", pct=60.0, success=False)
        alerter = AlertManager()
        with patch.object(alerter, "dispatch") as mock_dispatch:
            mock_dispatch.return_value = MagicMock()
            alerter.alert_data_quality_failure(result, "feature_pipeline")
            alert = mock_dispatch.call_args[0][0]
            assert alert.severity == AlertSeverity.CRITICAL

    def test_dq_recorded_on_dashboard(self):
        result = self._make_dq_result("transactions_suite", pct=95.0, success=True)
        dashboard = MonitoringDashboard()
        snap = dashboard.record_dq(result, "fraud_v3")
        assert snap.dq_pass_rate == pytest.approx(95.0)

    def test_dq_critical_degrades_health(self):
        result = self._make_dq_result("transactions_suite", pct=70.0, success=False)
        dashboard = MonitoringDashboard()
        snap = dashboard.record_dq(result, "fraud_v3")
        assert snap.health_status == "CRITICAL"

    def test_dq_good_keeps_healthy(self):
        result = self._make_dq_result("transactions_suite", pct=100.0, success=True)
        dashboard = MonitoringDashboard()
        snap = dashboard.record_dq(result, "fraud_v3")
        assert snap.health_status == "HEALTHY"


# ---------------------------------------------------------------------------
# Serialization round-trip tests
# ---------------------------------------------------------------------------

class TestSerializationRoundTrips:
    def test_drift_report_to_dict_is_json_serializable(self):
        feat = FeatureDriftResult(
            feature_name="avg_spend_7d",
            stattest_name="ks",
            drift_score=0.03,
            drifted=True,
            psi=0.15,
            psi_label="MEDIUM",
            reference_mean=100.0,
            current_mean=150.0,
            reference_std=20.0,
            current_std=30.0,
        )
        report = DriftReport(
            model_name="fraud_v3",
            reference_date=datetime(2024, 1, 1),
            current_date=datetime(2024, 6, 1),
            feature_results=[feat],
            total_features=1,
            drifted_features=1,
            dataset_drift_ratio=0.4,
            recommend_retrain=True,
        )
        d = report.to_dict()
        json_str = json.dumps(d)
        loaded = json.loads(json_str)
        assert loaded["model_name"] == "fraud_v3"
        assert loaded["feature_results"][0]["psi"] == pytest.approx(0.15)

    def test_validation_result_to_dict_is_json_serializable(self):
        result = ValidationResult(
            suite_name="transactions_suite",
            success=True,
            evaluated_expectations=10,
            successful_expectations=10,
            failed_expectations=0,
            success_percent=100.0,
        )
        d = result.to_dict()
        json_str = json.dumps(d)
        loaded = json.loads(json_str)
        assert loaded["success"] is True

    def test_alert_to_dict_is_json_serializable(self):
        alert = Alert(
            title="Test",
            message="Test message",
            severity=AlertSeverity.WARNING,
            source="test",
            metadata={"key": "value", "count": 5},
        )
        d = alert.to_dict()
        json_str = json.dumps(d)
        loaded = json.loads(json_str)
        assert loaded["severity"] == "warning"

    def test_model_health_to_dict_is_json_serializable(self):
        snap = ModelHealthSummary(
            model_name="fraud_v3",
            model_version="v1",
            evaluated_at=datetime.utcnow(),
            drift_ratio=0.15,
            drifted_features=["feat_a"],
            auc_roc=0.92,
        )
        snap.compute_health_status()
        d = snap.to_dict()
        json_str = json.dumps(d)
        loaded = json.loads(json_str)
        assert loaded["performance"]["auc_roc"] == pytest.approx(0.92)

    def test_platform_summary_to_dict_is_json_serializable(self):
        ps = PlatformSummary(models=[
            ModelHealthSummary(
                model_name="m",
                model_version="v1",
                evaluated_at=datetime.utcnow(),
                health_status="HEALTHY",
            )
        ])
        d = ps.to_dict()
        json_str = json.dumps(d)
        loaded = json.loads(json_str)
        assert loaded["platform_status"] == "HEALTHY"


# ---------------------------------------------------------------------------
# PSI edge cases and numerical stability
# ---------------------------------------------------------------------------

class TestPsiNumericalStability:
    def test_psi_with_large_arrays(self):
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 100_000)
        cur = rng.normal(0.5, 1, 100_000)
        psi = _compute_psi(ref, cur)
        assert 0 <= psi < 100  # sanity bound

    def test_psi_with_small_arrays(self):
        ref = np.array([1.0, 2.0, 3.0])
        cur = np.array([1.0, 2.0, 3.0])
        psi = _compute_psi(ref, cur)
        assert psi >= 0

    def test_psi_uniform_vs_normal(self):
        rng = np.random.default_rng(7)
        ref = rng.normal(0, 1, 2000)
        cur = rng.uniform(-3, 3, 2000)
        psi = _compute_psi(ref, cur)
        assert psi >= 0

    def test_psi_positive_floats(self):
        ref = np.array([0.01, 0.02, 0.03, 0.01, 0.05])
        cur = np.array([0.1, 0.2, 0.3, 0.1, 0.5])
        psi = _compute_psi(ref, cur)
        assert psi >= 0


# ---------------------------------------------------------------------------
# Prometheus metrics formatting
# ---------------------------------------------------------------------------

class TestPrometheusFormatting:
    def test_valid_prometheus_format(self):
        dashboard = MonitoringDashboard()
        from monitoring.drift_detector import DriftReport

        feat = FeatureDriftResult("f", "ks", 0.01, True)
        report = DriftReport(
            model_name="fraud_v3",
            reference_date=datetime(2024, 1, 1),
            current_date=datetime(2024, 6, 1),
            feature_results=[feat],
            total_features=1,
            drifted_features=1,
            dataset_drift_ratio=0.3,
            recommend_retrain=True,
        )
        dashboard.record_drift(report)
        dashboard.record_performance("fraud_v3", auc_roc=0.94)
        output = dashboard.prometheus_metrics()

        # Each line should be comment or metric value
        for line in output.strip().split("\n"):
            assert line.startswith("#") or "{" in line or line == ""

    def test_multiple_models_all_appear(self):
        dashboard = MonitoringDashboard()
        for name in ["fraud_v3", "credit_v2", "churn_v1"]:
            feat = FeatureDriftResult(name, "ks", 0.01, False)
            report = DriftReport(
                model_name=name,
                reference_date=datetime(2024, 1, 1),
                current_date=datetime(2024, 6, 1),
                feature_results=[feat],
                total_features=1,
                drifted_features=0,
                dataset_drift_ratio=0.0,
                recommend_retrain=False,
            )
            dashboard.record_drift(report)

        output = dashboard.prometheus_metrics()
        for name in ["fraud_v3", "credit_v2", "churn_v1"]:
            assert name in output

