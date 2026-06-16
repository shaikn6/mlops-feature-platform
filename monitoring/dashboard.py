"""
Monitoring metrics API — aggregates drift, data quality, and model performance
metrics into a single queryable interface for dashboards.

Consumed by:
  - Grafana (via FastAPI /metrics endpoint)
  - Internal Slack digests (daily summary)
  - Airflow task sensors (block deployment on health failure)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ModelHealthSummary:
    """Point-in-time health snapshot for a single model."""

    model_name: str
    model_version: str
    evaluated_at: datetime

    # Drift
    drift_ratio: float = 0.0
    drifted_features: list[str] = field(default_factory=list)
    recommend_retrain: bool = False
    last_drift_check: datetime | None = None

    # Data quality
    dq_pass_rate: float = 100.0
    dq_suite_name: str = ""
    last_dq_check: datetime | None = None

    # Model performance (from production labels, optional)
    auc_roc: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None

    # Overall health
    health_status: str = "HEALTHY"  # HEALTHY | DEGRADED | CRITICAL

    def compute_health_status(self) -> str:
        """Derive health status from constituent signals."""
        if self.recommend_retrain or self.dq_pass_rate < 80:
            status = "CRITICAL"
        elif self.drift_ratio > 0.10 or self.dq_pass_rate < 95:
            status = "DEGRADED"
        else:
            status = "HEALTHY"
        self.health_status = status
        return status

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "evaluated_at": self.evaluated_at.isoformat(),
            "health_status": self.health_status,
            "drift": {
                "drift_ratio": round(self.drift_ratio, 4),
                "drifted_features": self.drifted_features,
                "recommend_retrain": self.recommend_retrain,
                "last_check": self.last_drift_check.isoformat()
                if self.last_drift_check
                else None,
            },
            "data_quality": {
                "pass_rate": round(self.dq_pass_rate, 2),
                "suite_name": self.dq_suite_name,
                "last_check": self.last_dq_check.isoformat()
                if self.last_dq_check
                else None,
            },
            "performance": {
                "auc_roc": self.auc_roc,
                "precision": self.precision,
                "recall": self.recall,
                "f1": self.f1,
            },
        }


@dataclass
class PlatformSummary:
    """Aggregate health across all monitored models."""

    generated_at: datetime = field(default_factory=datetime.utcnow)
    models: list[ModelHealthSummary] = field(default_factory=list)

    @property
    def healthy_count(self) -> int:
        return sum(1 for m in self.models if m.health_status == "HEALTHY")

    @property
    def degraded_count(self) -> int:
        return sum(1 for m in self.models if m.health_status == "DEGRADED")

    @property
    def critical_count(self) -> int:
        return sum(1 for m in self.models if m.health_status == "CRITICAL")

    @property
    def platform_status(self) -> str:
        if self.critical_count > 0:
            return "CRITICAL"
        if self.degraded_count > 0:
            return "DEGRADED"
        return "HEALTHY"

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "platform_status": self.platform_status,
            "summary": {
                "total_models": len(self.models),
                "healthy": self.healthy_count,
                "degraded": self.degraded_count,
                "critical": self.critical_count,
            },
            "models": [m.to_dict() for m in self.models],
        }


class MonitoringDashboard:
    """
    Aggregates monitoring signals from drift detector and data quality monitor.

    Maintains an in-memory registry of the latest health snapshots per model.
    In production, back this with Redis or PostgreSQL for durability.

    Example:
        dashboard = MonitoringDashboard()
        dashboard.record_drift(report)
        dashboard.record_dq(result, model_name="fraud_v3")
        summary = dashboard.get_platform_summary()
    """

    def __init__(self) -> None:
        # model_name -> ModelHealthSummary
        self._snapshots: dict[str, ModelHealthSummary] = {}

    # ------------------------------------------------------------------
    # Ingest signals
    # ------------------------------------------------------------------

    def record_drift(
        self,
        report: Any,
        model_version: str = "unknown",
    ) -> ModelHealthSummary:
        """Ingest a DriftReport and update the health snapshot."""
        snapshot = self._get_or_create(report.model_name, model_version)
        snapshot.drift_ratio = report.dataset_drift_ratio
        snapshot.drifted_features = [
            r.feature_name for r in report.feature_results if r.drifted
        ]
        snapshot.recommend_retrain = report.recommend_retrain
        snapshot.last_drift_check = datetime.utcnow()
        snapshot.compute_health_status()

        logger.info(
            "Dashboard updated: %s drift_ratio=%.3f status=%s",
            report.model_name,
            report.dataset_drift_ratio,
            snapshot.health_status,
        )
        return snapshot

    def record_dq(
        self,
        result: Any,
        model_name: str,
        model_version: str = "unknown",
    ) -> ModelHealthSummary:
        """Ingest a ValidationResult and update the health snapshot."""
        snapshot = self._get_or_create(model_name, model_version)
        snapshot.dq_pass_rate = result.success_percent
        snapshot.dq_suite_name = result.suite_name
        snapshot.last_dq_check = datetime.utcnow()
        snapshot.compute_health_status()

        logger.info(
            "Dashboard updated: %s dq_pass_rate=%.1f%% status=%s",
            model_name,
            result.success_percent,
            snapshot.health_status,
        )
        return snapshot

    def record_performance(
        self,
        model_name: str,
        auc_roc: float | None = None,
        precision: float | None = None,
        recall: float | None = None,
        f1: float | None = None,
        model_version: str = "unknown",
    ) -> ModelHealthSummary:
        """Record production model performance metrics."""
        snapshot = self._get_or_create(model_name, model_version)
        if auc_roc is not None:
            snapshot.auc_roc = auc_roc
        if precision is not None:
            snapshot.precision = precision
        if recall is not None:
            snapshot.recall = recall
        if f1 is not None:
            snapshot.f1 = f1
        snapshot.compute_health_status()
        return snapshot

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_model_health(self, model_name: str) -> ModelHealthSummary | None:
        """Return the latest health snapshot for a model."""
        return self._snapshots.get(model_name)

    def get_platform_summary(self) -> PlatformSummary:
        """Return aggregate health across all monitored models."""
        return PlatformSummary(models=list(self._snapshots.values()))

    def get_critical_models(self) -> list[ModelHealthSummary]:
        """Return models in CRITICAL state — used by alerting and Airflow sensors."""
        return [s for s in self._snapshots.values() if s.health_status == "CRITICAL"]

    def get_models_needing_retrain(self) -> list[str]:
        """Return model names where drift detector recommends retraining."""
        return [name for name, s in self._snapshots.items() if s.recommend_retrain]

    def prometheus_metrics(self) -> str:
        """
        Emit Prometheus text format metrics for scraping by Grafana.

        Exposes:
          mlops_model_drift_ratio{model="..."}
          mlops_model_dq_pass_rate{model="..."}
          mlops_model_health_status{model="...", status="..."} 1
          mlops_model_auc_roc{model="..."}
        """
        lines: list[str] = [
            "# HELP mlops_model_drift_ratio Fraction of features that drifted",
            "# TYPE mlops_model_drift_ratio gauge",
        ]
        for name, s in self._snapshots.items():
            lines.append(
                f'mlops_model_drift_ratio{{model="{name}"}} {s.drift_ratio:.6f}'
            )

        lines += [
            "# HELP mlops_model_dq_pass_rate Data quality expectation pass rate (0-100)",
            "# TYPE mlops_model_dq_pass_rate gauge",
        ]
        for name, s in self._snapshots.items():
            lines.append(
                f'mlops_model_dq_pass_rate{{model="{name}"}} {s.dq_pass_rate:.2f}'
            )

        lines += [
            "# HELP mlops_model_auc_roc Model AUC-ROC on production labels",
            "# TYPE mlops_model_auc_roc gauge",
        ]
        for name, s in self._snapshots.items():
            if s.auc_roc is not None:
                lines.append(f'mlops_model_auc_roc{{model="{name}"}} {s.auc_roc:.6f}')

        lines += [
            "# HELP mlops_model_health_status Model health status (1=true)",
            "# TYPE mlops_model_health_status gauge",
        ]
        for name, s in self._snapshots.items():
            for status in ("HEALTHY", "DEGRADED", "CRITICAL"):
                val = 1 if s.health_status == status else 0
                lines.append(
                    f'mlops_model_health_status{{model="{name}",status="{status}"}} {val}'
                )

        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_or_create(self, model_name: str, model_version: str) -> ModelHealthSummary:
        if model_name not in self._snapshots:
            self._snapshots[model_name] = ModelHealthSummary(
                model_name=model_name,
                model_version=model_version,
                evaluated_at=datetime.utcnow(),
            )
        snap = self._snapshots[model_name]
        snap.evaluated_at = datetime.utcnow()
        if model_version != "unknown":
            snap.model_version = model_version
        return snap
