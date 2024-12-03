"""
Tests for monitoring.dashboard — MonitoringDashboard, ModelHealthSummary, PlatformSummary
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from monitoring.dashboard import (
    ModelHealthSummary,
    MonitoringDashboard,
    PlatformSummary,
)
from monitoring.drift_detector import DriftReport, FeatureDriftResult
from monitoring.data_quality import ValidationResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_drift_report(model_name="fraud_v3", ratio=0.25, retrain=True, drifted_features=None):
    feat_results = [
        FeatureDriftResult(
            feature_name=f,
            stattest_name="ks",
            drift_score=0.01,
            drifted=True,
        )
        for f in (drifted_features or ["avg_spend_7d", "credit_utilization"])
    ]
    return DriftReport(
        model_name=model_name,
        reference_date=datetime(2024, 1, 1),
        current_date=datetime(2024, 6, 1),
        feature_results=feat_results,
        total_features=len(feat_results),
        drifted_features=len(feat_results),
        dataset_drift_ratio=ratio,
        dataset_drifted=ratio >= 0.2,
        recommend_retrain=retrain,
    )


def _make_dq_result(suite="txn", pct=100.0, success=True):
    return ValidationResult(
        suite_name=suite,
        success=success,
        evaluated_expectations=10,
        successful_expectations=int(pct / 10),
        failed_expectations=10 - int(pct / 10),
        success_percent=pct,
    )


# ---------------------------------------------------------------------------
# ModelHealthSummary
# ---------------------------------------------------------------------------

class TestModelHealthSummary:
    def _make(self, drift_ratio=0.0, dq_pct=100.0, retrain=False):
        m = ModelHealthSummary(
            model_name="fraud_v3",
            model_version="v1",
            evaluated_at=datetime.utcnow(),
            drift_ratio=drift_ratio,
            dq_pass_rate=dq_pct,
            recommend_retrain=retrain,
        )
        return m

    def test_healthy_when_all_good(self):
        m = self._make()
        assert m.compute_health_status() == "HEALTHY"

    def test_degraded_when_drift_above_10pct(self):
        m = self._make(drift_ratio=0.15)
        assert m.compute_health_status() == "DEGRADED"

    def test_degraded_when_dq_below_95(self):
        m = self._make(dq_pct=90.0)
        assert m.compute_health_status() == "DEGRADED"

    def test_critical_when_retrain_recommended(self):
        m = self._make(retrain=True)
        assert m.compute_health_status() == "CRITICAL"

    def test_critical_when_dq_below_80(self):
        m = self._make(dq_pct=75.0)
        assert m.compute_health_status() == "CRITICAL"

    def test_health_status_field_updated(self):
        m = self._make(drift_ratio=0.5, retrain=True)
        m.compute_health_status()
        assert m.health_status == "CRITICAL"

    def test_to_dict_structure(self):
        m = self._make()
        m.compute_health_status()
        d = m.to_dict()
        assert "model_name" in d
        assert "health_status" in d
        assert "drift" in d
        assert "data_quality" in d
        assert "performance" in d

    def test_to_dict_last_check_none_when_not_set(self):
        m = self._make()
        d = m.to_dict()
        assert d["drift"]["last_check"] is None
        assert d["data_quality"]["last_check"] is None

    def test_to_dict_last_check_serialized_when_set(self):
        m = self._make()
        m.last_drift_check = datetime(2024, 6, 1)
        d = m.to_dict()
        assert d["drift"]["last_check"] == "2024-06-01T00:00:00"

    def test_performance_fields_none_by_default(self):
        m = self._make()
        d = m.to_dict()
        assert d["performance"]["auc_roc"] is None


# ---------------------------------------------------------------------------
# PlatformSummary
# ---------------------------------------------------------------------------

class TestPlatformSummary:
    def _make_snapshot(self, status="HEALTHY"):
        s = ModelHealthSummary(
            model_name="m",
            model_version="v1",
            evaluated_at=datetime.utcnow(),
            health_status=status,
        )
        return s

    def test_healthy_count(self):
        ps = PlatformSummary(models=[
            self._make_snapshot("HEALTHY"),
            self._make_snapshot("HEALTHY"),
            self._make_snapshot("DEGRADED"),
        ])
        assert ps.healthy_count == 2

    def test_degraded_count(self):
        ps = PlatformSummary(models=[
            self._make_snapshot("DEGRADED"),
            self._make_snapshot("CRITICAL"),
        ])
        assert ps.degraded_count == 1

    def test_critical_count(self):
        ps = PlatformSummary(models=[self._make_snapshot("CRITICAL")])
        assert ps.critical_count == 1

    def test_platform_status_critical_takes_priority(self):
        ps = PlatformSummary(models=[
            self._make_snapshot("HEALTHY"),
            self._make_snapshot("CRITICAL"),
        ])
        assert ps.platform_status == "CRITICAL"

    def test_platform_status_degraded(self):
        ps = PlatformSummary(models=[
            self._make_snapshot("HEALTHY"),
            self._make_snapshot("DEGRADED"),
        ])
        assert ps.platform_status == "DEGRADED"

    def test_platform_status_healthy_when_all_healthy(self):
        ps = PlatformSummary(models=[self._make_snapshot("HEALTHY")])
        assert ps.platform_status == "HEALTHY"

    def test_to_dict_structure(self):
        ps = PlatformSummary(models=[self._make_snapshot()])
        d = ps.to_dict()
        assert "platform_status" in d
        assert "summary" in d
        assert "models" in d

    def test_empty_models_is_healthy(self):
        ps = PlatformSummary(models=[])
        assert ps.platform_status == "HEALTHY"


# ---------------------------------------------------------------------------
# MonitoringDashboard
# ---------------------------------------------------------------------------

class TestMonitoringDashboard:
    # record_drift
    def test_record_drift_creates_snapshot(self):
        dash = MonitoringDashboard()
        report = _make_drift_report("fraud_v3", ratio=0.25, retrain=True)
        snap = dash.record_drift(report)
        assert snap.model_name == "fraud_v3"

    def test_record_drift_updates_drift_ratio(self):
        dash = MonitoringDashboard()
        report = _make_drift_report("fraud_v3", ratio=0.30, retrain=True)
        snap = dash.record_drift(report)
        assert snap.drift_ratio == pytest.approx(0.30)

    def test_record_drift_sets_recommend_retrain(self):
        dash = MonitoringDashboard()
        report = _make_drift_report(retrain=True)
        snap = dash.record_drift(report)
        assert snap.recommend_retrain is True

    def test_record_drift_sets_drifted_features(self):
        dash = MonitoringDashboard()
        report = _make_drift_report(drifted_features=["feat_a", "feat_b"])
        snap = dash.record_drift(report)
        assert "feat_a" in snap.drifted_features

    def test_record_drift_computes_health_status(self):
        dash = MonitoringDashboard()
        report = _make_drift_report(retrain=True)
        snap = dash.record_drift(report)
        assert snap.health_status in ("HEALTHY", "DEGRADED", "CRITICAL")

    # record_dq
    def test_record_dq_creates_snapshot(self):
        dash = MonitoringDashboard()
        result = _make_dq_result(pct=100.0)
        snap = dash.record_dq(result, "fraud_v3")
        assert snap.model_name == "fraud_v3"

    def test_record_dq_updates_pass_rate(self):
        dash = MonitoringDashboard()
        result = _make_dq_result(pct=88.5)
        snap = dash.record_dq(result, "fraud_v3")
        assert snap.dq_pass_rate == pytest.approx(88.5)

    def test_record_dq_updates_suite_name(self):
        dash = MonitoringDashboard()
        result = _make_dq_result(suite="credit_suite")
        snap = dash.record_dq(result, "credit_v2")
        assert snap.dq_suite_name == "credit_suite"

    def test_record_dq_sets_last_dq_check(self):
        dash = MonitoringDashboard()
        result = _make_dq_result()
        snap = dash.record_dq(result, "m")
        assert snap.last_dq_check is not None

    # record_performance
    def test_record_performance_stores_metrics(self):
        dash = MonitoringDashboard()
        snap = dash.record_performance("fraud_v3", auc_roc=0.95, precision=0.9, recall=0.85, f1=0.87)
        assert snap.auc_roc == pytest.approx(0.95)
        assert snap.precision == pytest.approx(0.9)
        assert snap.recall == pytest.approx(0.85)
        assert snap.f1 == pytest.approx(0.87)

    def test_record_performance_partial_metrics(self):
        dash = MonitoringDashboard()
        snap = dash.record_performance("fraud_v3", auc_roc=0.91)
        assert snap.auc_roc == pytest.approx(0.91)
        assert snap.precision is None

    # get_model_health
    def test_get_model_health_returns_snapshot(self):
        dash = MonitoringDashboard()
        dash.record_drift(_make_drift_report("fraud_v3"))
        snap = dash.get_model_health("fraud_v3")
        assert snap is not None
        assert snap.model_name == "fraud_v3"

    def test_get_model_health_none_for_unknown(self):
        dash = MonitoringDashboard()
        assert dash.get_model_health("nonexistent") is None

    # get_platform_summary
    def test_get_platform_summary_returns_platform_summary(self):
        dash = MonitoringDashboard()
        dash.record_drift(_make_drift_report("fraud_v3"))
        ps = dash.get_platform_summary()
        assert isinstance(ps, PlatformSummary)
        assert len(ps.models) == 1

    def test_get_platform_summary_aggregates_multiple_models(self):
        dash = MonitoringDashboard()
        dash.record_drift(_make_drift_report("fraud_v3"))
        dash.record_drift(_make_drift_report("credit_v2", ratio=0.0, retrain=False))
        ps = dash.get_platform_summary()
        assert len(ps.models) == 2

    # get_critical_models
    def test_get_critical_models_empty_initially(self):
        dash = MonitoringDashboard()
        assert dash.get_critical_models() == []

    def test_get_critical_models_returns_critical(self):
        dash = MonitoringDashboard()
        dash.record_drift(_make_drift_report("fraud_v3", ratio=0.5, retrain=True))
        criticals = dash.get_critical_models()
        assert any(s.model_name == "fraud_v3" for s in criticals)

    # get_models_needing_retrain
    def test_get_models_needing_retrain_empty(self):
        dash = MonitoringDashboard()
        dash.record_drift(_make_drift_report(retrain=False, ratio=0.05))
        assert dash.get_models_needing_retrain() == []

    def test_get_models_needing_retrain_has_model(self):
        dash = MonitoringDashboard()
        dash.record_drift(_make_drift_report("fraud_v3", retrain=True))
        assert "fraud_v3" in dash.get_models_needing_retrain()

    # prometheus_metrics
    def test_prometheus_metrics_returns_string(self):
        dash = MonitoringDashboard()
        dash.record_drift(_make_drift_report("fraud_v3"))
        output = dash.prometheus_metrics()
        assert isinstance(output, str)

    def test_prometheus_metrics_contains_model_name(self):
        dash = MonitoringDashboard()
        dash.record_drift(_make_drift_report("fraud_v3"))
        output = dash.prometheus_metrics()
        assert "fraud_v3" in output

    def test_prometheus_metrics_has_all_metric_types(self):
        dash = MonitoringDashboard()
        dash.record_drift(_make_drift_report("m"))
        dash.record_performance("m", auc_roc=0.92)
        output = dash.prometheus_metrics()
        assert "mlops_model_drift_ratio" in output
        assert "mlops_model_dq_pass_rate" in output
        assert "mlops_model_health_status" in output
        assert "mlops_model_auc_roc" in output

    def test_prometheus_metrics_omits_auc_when_none(self):
        dash = MonitoringDashboard()
        dash.record_drift(_make_drift_report("m"))
        output = dash.prometheus_metrics()
        # no auc_roc since it's None
        assert 'mlops_model_auc_roc{model="m"}' not in output

    def test_prometheus_metrics_format_ends_with_newline(self):
        dash = MonitoringDashboard()
        dash.record_drift(_make_drift_report("m"))
        output = dash.prometheus_metrics()
        assert output.endswith("\n")

    # _get_or_create
    def test_get_or_create_returns_same_instance(self):
        dash = MonitoringDashboard()
        s1 = dash._get_or_create("m", "v1")
        s2 = dash._get_or_create("m", "v1")
        assert s1 is s2

    def test_get_or_create_updates_evaluated_at(self):
        dash = MonitoringDashboard()
        s = dash._get_or_create("m", "v1")
        first_time = s.evaluated_at
        s2 = dash._get_or_create("m", "v1")
        # evaluated_at is updated every time
        assert s2.evaluated_at >= first_time

    def test_get_or_create_updates_version_when_not_unknown(self):
        dash = MonitoringDashboard()
        dash._get_or_create("m", "unknown")
        s = dash._get_or_create("m", "v2")
        assert s.model_version == "v2"

    def test_record_drift_then_dq_same_model(self):
        dash = MonitoringDashboard()
        dash.record_drift(_make_drift_report("fraud_v3", ratio=0.1, retrain=False))
        dash.record_dq(_make_dq_result(pct=88.0), "fraud_v3")
        snap = dash.get_model_health("fraud_v3")
        assert snap.dq_pass_rate == pytest.approx(88.0)
        assert snap.drift_ratio == pytest.approx(0.1)

