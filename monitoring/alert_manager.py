"""
Alert routing for drift and data quality events.

Supports:
  - Slack (webhook)
  - PagerDuty (Events API v2)
  - Email (SMTP)
  - Generic webhook

Alert severity levels:
  INFO    — informational, logged only
  WARNING — Slack notification
  CRITICAL — Slack + PagerDuty page
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    """A single alert event."""

    title: str
    message: str
    severity: AlertSeverity
    source: str  # e.g. "drift_detector", "data_quality"
    model_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "message": self.message,
            "severity": self.severity.value,
            "source": self.source,
            "model_name": self.model_name,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class AlertResult:
    """Result of dispatching an alert to one or more channels."""

    alert: Alert
    channels_attempted: list[str] = field(default_factory=list)
    channels_succeeded: list[str] = field(default_factory=list)
    channels_failed: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return len(self.channels_failed) == 0 and len(self.channels_succeeded) > 0


class AlertManager:
    """
    Routes alerts to configured notification channels.

    Configuration via environment variables (no secrets in code):
        SLACK_WEBHOOK_URL       — Slack incoming webhook URL
        PAGERDUTY_ROUTING_KEY   — PagerDuty Events API v2 routing key
        ALERT_EMAIL_RECIPIENTS  — comma-separated email addresses
        SMTP_HOST               — SMTP server hostname (default: localhost)
        SMTP_PORT               — SMTP port (default: 587)
        SMTP_USER               — SMTP username
        SMTP_PASSWORD           — SMTP password

    Example:
        manager = AlertManager()
        manager.alert_drift(report, model_name="fraud_v3")
        manager.alert_data_quality_failure(result, pipeline="feature_pipeline")
    """

    def __init__(
        self,
        slack_webhook_url: str | None = None,
        pagerduty_routing_key: str | None = None,
        email_recipients: list[str] | None = None,
    ) -> None:
        self._slack_url = slack_webhook_url or os.getenv("SLACK_WEBHOOK_URL")
        self._pd_key = pagerduty_routing_key or os.getenv("PAGERDUTY_ROUTING_KEY")
        self._email_recipients = email_recipients or [
            e.strip()
            for e in (os.getenv("ALERT_EMAIL_RECIPIENTS") or "").split(",")
            if e.strip()
        ]

    # ------------------------------------------------------------------
    # High-level alert builders
    # ------------------------------------------------------------------

    def alert_drift(self, report: Any, model_name: str) -> AlertResult:
        """Send drift alert based on a DriftReport."""
        severity = (
            AlertSeverity.CRITICAL
            if report.recommend_retrain
            else AlertSeverity.WARNING
        )
        pct = round(report.dataset_drift_ratio * 100, 1)
        drifted_features = [r.feature_name for r in report.feature_results if r.drifted]

        alert = Alert(
            title=f"Model Drift Detected — {model_name}",
            message=(
                f"{pct}% of features drifted ({report.drifted_features}/{report.total_features}). "
                f"Retrain recommended: {report.recommend_retrain}.\n"
                f"Drifted features: {', '.join(drifted_features[:10])}"
                + (" ..." if len(drifted_features) > 10 else "")
            ),
            severity=severity,
            source="drift_detector",
            model_name=model_name,
            metadata={
                "dataset_drift_ratio": report.dataset_drift_ratio,
                "drifted_features": drifted_features,
                "recommend_retrain": report.recommend_retrain,
            },
        )
        return self.dispatch(alert)

    def alert_data_quality_failure(self, result: Any, pipeline: str) -> AlertResult:
        """Send data quality alert based on a ValidationResult."""
        severity = (
            AlertSeverity.CRITICAL
            if result.success_percent < 80
            else AlertSeverity.WARNING
        )

        alert = Alert(
            title=f"Data Quality Failure — {result.suite_name}",
            message=(
                f"Pipeline: {pipeline}\n"
                f"Suite: {result.suite_name}\n"
                f"Pass rate: {result.success_percent:.1f}% "
                f"({result.successful_expectations}/{result.evaluated_expectations})\n"
                f"Failed checks: {result.failed_expectations}"
            ),
            severity=severity,
            source="data_quality",
            metadata={
                "suite_name": result.suite_name,
                "success_percent": result.success_percent,
                "failed_expectations": result.failed_expectations,
                "pipeline": pipeline,
            },
        )
        return self.dispatch(alert)

    def alert_model_registered(
        self, model_name: str, version: str, stage: str
    ) -> AlertResult:
        """Notify when a new model version is promoted to a registry stage."""
        alert = Alert(
            title=f"Model Registered — {model_name} v{version}",
            message=(
                f"Model {model_name} version {version} has been promoted to {stage}.\n"
                f"MLflow registry updated. Serving layer refresh may be required."
            ),
            severity=AlertSeverity.INFO,
            source="model_registry",
            model_name=model_name,
            metadata={"version": version, "stage": stage},
        )
        return self.dispatch(alert)

    def alert_retrain_triggered(self, model_name: str, reason: str) -> AlertResult:
        """Notify when automatic retraining is triggered."""
        alert = Alert(
            title=f"Retrain Triggered — {model_name}",
            message=f"Automatic retraining initiated for {model_name}.\nReason: {reason}",
            severity=AlertSeverity.WARNING,
            source="training_pipeline",
            model_name=model_name,
            metadata={"reason": reason},
        )
        return self.dispatch(alert)

    # ------------------------------------------------------------------
    # Core dispatch
    # ------------------------------------------------------------------

    def dispatch(self, alert: Alert) -> AlertResult:
        """Route an alert to all configured channels based on severity."""
        result = AlertResult(alert=alert)

        # Always log
        log_fn = (
            logger.critical
            if alert.severity == AlertSeverity.CRITICAL
            else (
                logger.warning
                if alert.severity == AlertSeverity.WARNING
                else logger.info
            )
        )
        log_fn(
            "[ALERT][%s] %s — %s",
            alert.severity.value.upper(),
            alert.title,
            alert.message,
        )

        # INFO — log only
        if alert.severity == AlertSeverity.INFO:
            return result

        # WARNING + CRITICAL — Slack
        if self._slack_url:
            result.channels_attempted.append("slack")
            try:
                self._send_slack(alert)
                result.channels_succeeded.append("slack")
            except Exception as exc:
                result.channels_failed.append("slack")
                result.errors["slack"] = str(exc)
                logger.error("Slack alert failed: %s", exc)

        # CRITICAL — PagerDuty
        if alert.severity == AlertSeverity.CRITICAL and self._pd_key:
            result.channels_attempted.append("pagerduty")
            try:
                self._send_pagerduty(alert)
                result.channels_succeeded.append("pagerduty")
            except Exception as exc:
                result.channels_failed.append("pagerduty")
                result.errors["pagerduty"] = str(exc)
                logger.error("PagerDuty alert failed: %s", exc)

        # Email for all non-INFO if configured
        if self._email_recipients:
            result.channels_attempted.append("email")
            try:
                self._send_email(alert)
                result.channels_succeeded.append("email")
            except Exception as exc:
                result.channels_failed.append("email")
                result.errors["email"] = str(exc)
                logger.error("Email alert failed: %s", exc)

        return result

    # ------------------------------------------------------------------
    # Channel implementations
    # ------------------------------------------------------------------

    def _send_slack(self, alert: Alert) -> None:
        emoji = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}[alert.severity.value]
        color = {"info": "#36a64f", "warning": "#ff9900", "critical": "#cc0000"}[
            alert.severity.value
        ]

        payload = {
            "attachments": [
                {
                    "color": color,
                    "title": f"{emoji} {alert.title}",
                    "text": alert.message,
                    "fields": [
                        {
                            "title": "Severity",
                            "value": alert.severity.value.upper(),
                            "short": True,
                        },
                        {"title": "Source", "value": alert.source, "short": True},
                        {
                            "title": "Model",
                            "value": alert.model_name or "N/A",
                            "short": True,
                        },
                        {
                            "title": "Time",
                            "value": alert.timestamp.strftime("%Y-%m-%d %H:%M UTC"),
                            "short": True,
                        },
                    ],
                    "footer": "MLOps Feature Platform",
                }
            ]
        }
        self._post_json(self._slack_url, payload)  # type: ignore[arg-type]

    def _send_pagerduty(self, alert: Alert) -> None:
        payload = {
            "routing_key": self._pd_key,
            "event_action": "trigger",
            "dedup_key": f"{alert.source}-{alert.model_name}-{alert.timestamp.date()}",
            "payload": {
                "summary": alert.title,
                "severity": "critical"
                if alert.severity == AlertSeverity.CRITICAL
                else "warning",
                "source": alert.source,
                "custom_details": alert.metadata,
                "timestamp": alert.timestamp.isoformat(),
            },
        }
        self._post_json("https://events.pagerduty.com/v2/enqueue", payload)

    def _send_email(self, alert: Alert) -> None:
        smtp_host = os.getenv("SMTP_HOST", "localhost")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        smtp_user = os.getenv("SMTP_USER", "")
        smtp_pass = os.getenv("SMTP_PASSWORD", "")
        from_addr = smtp_user or "mlops-alerts@noreply.local"

        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[{alert.severity.value.upper()}] {alert.title}"
        msg["From"] = from_addr
        msg["To"] = ", ".join(self._email_recipients)

        text_body = f"{alert.message}\n\nTimestamp: {alert.timestamp.isoformat()}\nSource: {alert.source}"
        msg.attach(MIMEText(text_body, "plain"))

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            if smtp_user and smtp_pass:
                server.starttls()
                server.login(smtp_user, smtp_pass)
            server.sendmail(from_addr, self._email_recipients, msg.as_string())

    @staticmethod
    def _post_json(url: str, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status not in (200, 201, 202):
                raise RuntimeError(f"HTTP {resp.status} from {url}")
