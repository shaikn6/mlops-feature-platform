"""
Tests for monitoring.alert_manager
Covers: Alert, AlertResult, AlertManager dispatch, all channel paths,
        high-level alert builders, error paths.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from unittest.mock import MagicMock, patch, call
from email.mime.multipart import MIMEMultipart

import pytest

from monitoring.alert_manager import (
    Alert,
    AlertManager,
    AlertResult,
    AlertSeverity,
)


# ---------------------------------------------------------------------------
# Alert dataclass
# ---------------------------------------------------------------------------

class TestAlert:
    def test_construction_defaults(self):
        alert = Alert(
            title="Test",
            message="msg",
            severity=AlertSeverity.WARNING,
            source="test",
        )
        assert alert.model_name is None
        assert alert.metadata == {}
        assert isinstance(alert.timestamp, datetime)

    def test_to_dict_contains_required_keys(self):
        alert = Alert(
            title="Drift Alert",
            message="30% features drifted",
            severity=AlertSeverity.CRITICAL,
            source="drift_detector",
            model_name="fraud_v3",
        )
        d = alert.to_dict()
        assert d["title"] == "Drift Alert"
        assert d["severity"] == "critical"
        assert d["source"] == "drift_detector"
        assert d["model_name"] == "fraud_v3"

    def test_timestamp_serialized_as_iso(self):
        alert = Alert(
            title="t", message="m",
            severity=AlertSeverity.INFO, source="s"
        )
        d = alert.to_dict()
        datetime.fromisoformat(d["timestamp"])  # must parse without raising


# ---------------------------------------------------------------------------
# AlertResult
# ---------------------------------------------------------------------------

class TestAlertResult:
    def test_success_true_when_no_failures(self):
        alert = Alert("t", "m", AlertSeverity.WARNING, "s")
        result = AlertResult(
            alert=alert,
            channels_attempted=["slack"],
            channels_succeeded=["slack"],
        )
        assert result.success is True

    def test_success_false_when_failure(self):
        alert = Alert("t", "m", AlertSeverity.CRITICAL, "s")
        result = AlertResult(
            alert=alert,
            channels_attempted=["slack"],
            channels_failed=["slack"],
        )
        assert result.success is False

    def test_success_false_when_no_channels_succeeded(self):
        alert = Alert("t", "m", AlertSeverity.WARNING, "s")
        result = AlertResult(alert=alert)
        assert result.success is False


# ---------------------------------------------------------------------------
# AlertSeverity
# ---------------------------------------------------------------------------

class TestAlertSeverity:
    def test_values(self):
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.CRITICAL.value == "critical"


# ---------------------------------------------------------------------------
# AlertManager.__init__
# ---------------------------------------------------------------------------

class TestAlertManagerInit:
    def test_explicit_config(self):
        mgr = AlertManager(
            slack_webhook_url="https://hooks.slack.com/x",
            pagerduty_routing_key="abc123",
            email_recipients=["a@b.com"],
        )
        assert mgr._slack_url == "https://hooks.slack.com/x"
        assert mgr._pd_key == "abc123"
        assert mgr._email_recipients == ["a@b.com"]

    def test_reads_from_env_variables(self, monkeypatch):
        monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://slack/hook")
        monkeypatch.setenv("PAGERDUTY_ROUTING_KEY", "pd-key-xyz")
        monkeypatch.setenv("ALERT_EMAIL_RECIPIENTS", "a@x.com,b@x.com")
        mgr = AlertManager()
        assert mgr._slack_url == "https://slack/hook"
        assert mgr._pd_key == "pd-key-xyz"
        assert "a@x.com" in mgr._email_recipients
        assert "b@x.com" in mgr._email_recipients

    def test_empty_env_yields_empty_recipients(self, monkeypatch):
        monkeypatch.delenv("ALERT_EMAIL_RECIPIENTS", raising=False)
        monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
        monkeypatch.delenv("PAGERDUTY_ROUTING_KEY", raising=False)
        mgr = AlertManager()
        assert mgr._email_recipients == []
        assert mgr._slack_url is None
        assert mgr._pd_key is None

    def test_email_with_spaces_stripped(self, monkeypatch):
        monkeypatch.setenv("ALERT_EMAIL_RECIPIENTS", " a@b.com , c@d.com ")
        mgr = AlertManager()
        assert "a@b.com" in mgr._email_recipients
        assert "c@d.com" in mgr._email_recipients


# ---------------------------------------------------------------------------
# dispatch() — INFO severity (log only)
# ---------------------------------------------------------------------------

class TestDispatchInfo:
    def test_info_returns_result_with_no_channels(self):
        mgr = AlertManager(
            slack_webhook_url="https://hook",
            pagerduty_routing_key="key",
        )
        alert = Alert("Info", "msg", AlertSeverity.INFO, "src")
        result = mgr.dispatch(alert)
        assert result.channels_attempted == []
        assert result.channels_succeeded == []

    def test_info_does_not_call_slack(self):
        mgr = AlertManager(slack_webhook_url="https://hook")
        alert = Alert("t", "m", AlertSeverity.INFO, "s")
        with patch.object(mgr, "_send_slack") as mock_slack:
            mgr.dispatch(alert)
            mock_slack.assert_not_called()


# ---------------------------------------------------------------------------
# dispatch() — WARNING severity (Slack only)
# ---------------------------------------------------------------------------

class TestDispatchWarning:
    def test_warning_tries_slack_when_configured(self):
        mgr = AlertManager(slack_webhook_url="https://hook")
        alert = Alert("t", "m", AlertSeverity.WARNING, "s")
        with patch.object(mgr, "_send_slack") as mock_slack:
            mgr.dispatch(alert)
            mock_slack.assert_called_once()

    def test_warning_does_not_trigger_pagerduty(self):
        mgr = AlertManager(
            slack_webhook_url="https://hook",
            pagerduty_routing_key="key",
        )
        alert = Alert("t", "m", AlertSeverity.WARNING, "s")
        with patch.object(mgr, "_send_slack"), patch.object(mgr, "_send_pagerduty") as mock_pd:
            mgr.dispatch(alert)
            mock_pd.assert_not_called()

    def test_warning_slack_success_recorded(self):
        mgr = AlertManager(slack_webhook_url="https://hook")
        alert = Alert("t", "m", AlertSeverity.WARNING, "s")
        with patch.object(mgr, "_send_slack"):
            result = mgr.dispatch(alert)
            assert "slack" in result.channels_succeeded

    def test_warning_slack_failure_recorded(self):
        mgr = AlertManager(slack_webhook_url="https://hook")
        alert = Alert("t", "m", AlertSeverity.WARNING, "s")
        with patch.object(mgr, "_send_slack", side_effect=RuntimeError("timeout")):
            result = mgr.dispatch(alert)
            assert "slack" in result.channels_failed
            assert "slack" in result.errors


# ---------------------------------------------------------------------------
# dispatch() — CRITICAL severity (Slack + PagerDuty)
# ---------------------------------------------------------------------------

class TestDispatchCritical:
    def test_critical_triggers_both_slack_and_pd(self):
        mgr = AlertManager(
            slack_webhook_url="https://hook",
            pagerduty_routing_key="key",
        )
        alert = Alert("t", "m", AlertSeverity.CRITICAL, "s")
        with patch.object(mgr, "_send_slack") as mS, patch.object(mgr, "_send_pagerduty") as mP:
            mgr.dispatch(alert)
            mS.assert_called_once()
            mP.assert_called_once()

    def test_critical_no_pd_key_skips_pagerduty(self):
        mgr = AlertManager(slack_webhook_url="https://hook", pagerduty_routing_key=None)
        alert = Alert("t", "m", AlertSeverity.CRITICAL, "s")
        with patch.object(mgr, "_send_slack"), patch.object(mgr, "_send_pagerduty") as mP:
            mgr.dispatch(alert)
            mP.assert_not_called()

    def test_critical_pd_failure_recorded(self):
        mgr = AlertManager(
            slack_webhook_url="https://hook",
            pagerduty_routing_key="key",
        )
        alert = Alert("t", "m", AlertSeverity.CRITICAL, "s")
        with patch.object(mgr, "_send_slack"), \
             patch.object(mgr, "_send_pagerduty", side_effect=RuntimeError("PD down")):
            result = mgr.dispatch(alert)
            assert "pagerduty" in result.channels_failed


# ---------------------------------------------------------------------------
# dispatch() — email path
# ---------------------------------------------------------------------------

class TestDispatchEmail:
    def test_warning_sends_email_when_recipients_configured(self):
        mgr = AlertManager(email_recipients=["a@b.com"])
        alert = Alert("t", "m", AlertSeverity.WARNING, "s")
        with patch.object(mgr, "_send_email") as mock_email:
            result = mgr.dispatch(alert)
            mock_email.assert_called_once()
            assert "email" in result.channels_succeeded

    def test_no_recipients_skips_email(self):
        mgr = AlertManager(email_recipients=[])
        alert = Alert("t", "m", AlertSeverity.WARNING, "s")
        with patch.object(mgr, "_send_email") as mock_email:
            mgr.dispatch(alert)
            mock_email.assert_not_called()

    def test_email_failure_recorded(self):
        mgr = AlertManager(email_recipients=["a@b.com"])
        alert = Alert("t", "m", AlertSeverity.WARNING, "s")
        with patch.object(mgr, "_send_email", side_effect=ConnectionError("SMTP down")):
            result = mgr.dispatch(alert)
            assert "email" in result.channels_failed


# ---------------------------------------------------------------------------
# High-level alert builders
# ---------------------------------------------------------------------------

class TestAlertDrift:
    def _make_drift_report(self, ratio=0.25, retrain=True, drifted_features=None):
        report = MagicMock()
        report.recommend_retrain = retrain
        report.dataset_drift_ratio = ratio
        report.drifted_features = 3
        report.total_features = 5
        report.feature_results = [
            MagicMock(feature_name=f, drifted=True)
            for f in (drifted_features or ["feat_a", "feat_b", "feat_c"])
        ]
        return report

    def test_critical_when_retrain_recommended(self):
        mgr = AlertManager()
        report = self._make_drift_report(retrain=True)
        with patch.object(mgr, "dispatch", return_value=MagicMock()) as mock_d:
            mgr.alert_drift(report, "fraud_v3")
            dispatched_alert = mock_d.call_args[0][0]
            assert dispatched_alert.severity == AlertSeverity.CRITICAL

    def test_warning_when_no_retrain(self):
        mgr = AlertManager()
        report = self._make_drift_report(retrain=False, ratio=0.1)
        with patch.object(mgr, "dispatch", return_value=MagicMock()) as mock_d:
            mgr.alert_drift(report, "fraud_v3")
            dispatched_alert = mock_d.call_args[0][0]
            assert dispatched_alert.severity == AlertSeverity.WARNING

    def test_model_name_in_title(self):
        mgr = AlertManager()
        report = self._make_drift_report()
        with patch.object(mgr, "dispatch", return_value=MagicMock()) as mock_d:
            mgr.alert_drift(report, "credit_v2")
            alert = mock_d.call_args[0][0]
            assert "credit_v2" in alert.title

    def test_more_than_10_features_truncated(self):
        mgr = AlertManager()
        report = self._make_drift_report(
            drifted_features=[f"feat_{i}" for i in range(15)]
        )
        with patch.object(mgr, "dispatch", return_value=MagicMock()) as mock_d:
            mgr.alert_drift(report, "m")
            alert = mock_d.call_args[0][0]
            assert "..." in alert.message


class TestAlertDataQualityFailure:
    def _make_result(self, pct=65.0):
        r = MagicMock()
        r.success_percent = pct
        r.suite_name = "transactions_suite"
        r.successful_expectations = int(pct * 10 / 100)
        r.evaluated_expectations = 10
        r.failed_expectations = 10 - r.successful_expectations
        return r

    def test_critical_below_80_pct(self):
        mgr = AlertManager()
        with patch.object(mgr, "dispatch", return_value=MagicMock()) as mock_d:
            mgr.alert_data_quality_failure(self._make_result(65.0), "feature_pipeline")
            alert = mock_d.call_args[0][0]
            assert alert.severity == AlertSeverity.CRITICAL

    def test_warning_above_80_pct(self):
        mgr = AlertManager()
        with patch.object(mgr, "dispatch", return_value=MagicMock()) as mock_d:
            mgr.alert_data_quality_failure(self._make_result(90.0), "feature_pipeline")
            alert = mock_d.call_args[0][0]
            assert alert.severity == AlertSeverity.WARNING

    def test_pipeline_name_in_message(self):
        mgr = AlertManager()
        with patch.object(mgr, "dispatch", return_value=MagicMock()) as mock_d:
            mgr.alert_data_quality_failure(self._make_result(), "my_pipeline")
            alert = mock_d.call_args[0][0]
            assert "my_pipeline" in alert.message


class TestAlertModelRegistered:
    def test_info_severity(self):
        mgr = AlertManager()
        with patch.object(mgr, "dispatch", return_value=MagicMock()) as mock_d:
            mgr.alert_model_registered("fraud_v3", "42", "Production")
            alert = mock_d.call_args[0][0]
            assert alert.severity == AlertSeverity.INFO

    def test_model_name_in_title(self):
        mgr = AlertManager()
        with patch.object(mgr, "dispatch", return_value=MagicMock()) as mock_d:
            mgr.alert_model_registered("fraud_v3", "42", "Production")
            alert = mock_d.call_args[0][0]
            assert "fraud_v3" in alert.title


class TestAlertRetrain:
    def test_warning_severity(self):
        mgr = AlertManager()
        with patch.object(mgr, "dispatch", return_value=MagicMock()) as mock_d:
            mgr.alert_retrain_triggered("credit_v2", "Dataset drift ratio 0.35")
            alert = mock_d.call_args[0][0]
            assert alert.severity == AlertSeverity.WARNING

    def test_reason_in_message(self):
        mgr = AlertManager()
        with patch.object(mgr, "dispatch", return_value=MagicMock()) as mock_d:
            mgr.alert_retrain_triggered("m", "Feature distribution shift")
            alert = mock_d.call_args[0][0]
            assert "Feature distribution shift" in alert.message


# ---------------------------------------------------------------------------
# _post_json static method
# ---------------------------------------------------------------------------

class TestPostJson:
    def test_raises_on_bad_status(self):
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.status = 500

        with patch("urllib.request.urlopen", return_value=mock_resp):
            with pytest.raises(RuntimeError, match="HTTP 500"):
                AlertManager._post_json("https://example.com", {"key": "val"})

    def test_success_does_not_raise(self):
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.status = 200

        with patch("urllib.request.urlopen", return_value=mock_resp):
            AlertManager._post_json("https://example.com", {"key": "val"})  # no raise

    def test_accepted_202_does_not_raise(self):
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.status = 202

        with patch("urllib.request.urlopen", return_value=mock_resp):
            AlertManager._post_json("https://example.com", {})  # no raise


# ---------------------------------------------------------------------------
# _send_email
# ---------------------------------------------------------------------------

class TestSendEmail:
    def test_sends_email_via_smtp(self):
        mgr = AlertManager(email_recipients=["a@b.com"])
        alert = Alert("Subject", "Body text", AlertSeverity.WARNING, "test")
        mock_smtp = MagicMock()
        mock_smtp_inst = MagicMock()
        mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_smtp_inst)
        mock_smtp.return_value.__exit__ = MagicMock(return_value=False)

        with patch("smtplib.SMTP", mock_smtp):
            mgr._send_email(alert)
            mock_smtp_inst.sendmail.assert_called_once()

    def test_uses_tls_when_credentials_set(self, monkeypatch):
        monkeypatch.setenv("SMTP_USER", "user@x.com")
        monkeypatch.setenv("SMTP_PASSWORD", "secret")
        mgr = AlertManager(email_recipients=["a@b.com"])
        alert = Alert("t", "m", AlertSeverity.CRITICAL, "s")
        mock_smtp = MagicMock()
        mock_smtp_inst = MagicMock()
        mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_smtp_inst)
        mock_smtp.return_value.__exit__ = MagicMock(return_value=False)

        with patch("smtplib.SMTP", mock_smtp):
            mgr._send_email(alert)
            mock_smtp_inst.starttls.assert_called_once()
            mock_smtp_inst.login.assert_called_once_with("user@x.com", "secret")

