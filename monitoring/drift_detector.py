"""
Model drift detector using Evidently AI.

Compares reference (training) distribution against current (production) data.
Returns per-feature drift scores, PSI values, and a retrain recommendation.

Supported drift tests:
  - Statistical tests: KS, chi-squared, Wasserstein, Jensen-Shannon
  - PSI (Population Stability Index) for credit/risk model compatibility
  - Dataset-level drift summary

Retrain threshold: dataset drift ratio > 0.2 (20% of features drifted).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Population Stability Index thresholds (industry standard for credit models)
PSI_LOW = 0.1  # No significant change
PSI_MEDIUM = 0.2  # Some change — monitor closely
PSI_HIGH = 0.25  # Significant change — retrain recommended

# Fraction of features that must drift before triggering retrain recommendation
DATASET_DRIFT_THRESHOLD = 0.20


@dataclass
class FeatureDriftResult:
    """Drift result for a single feature."""

    feature_name: str
    stattest_name: str
    drift_score: float  # p-value or distance metric depending on test
    drifted: bool  # True if drift detected at configured threshold
    psi: float | None = None  # PSI value (None for non-numeric features)
    psi_label: str | None = None  # "LOW" | "MEDIUM" | "HIGH"
    reference_mean: float | None = None
    current_mean: float | None = None
    reference_std: float | None = None
    current_std: float | None = None


@dataclass
class DriftReport:
    """Full drift report for a model snapshot comparison."""

    model_name: str
    reference_date: datetime
    current_date: datetime
    generated_at: datetime = field(default_factory=datetime.utcnow)

    feature_results: list[FeatureDriftResult] = field(default_factory=list)

    # Aggregate metrics
    total_features: int = 0
    drifted_features: int = 0
    dataset_drift_ratio: float = 0.0
    dataset_drifted: bool = False
    recommend_retrain: bool = False

    # Raw Evidently report (JSON-serializable dict)
    raw_report: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-safe dict."""
        return {
            "model_name": self.model_name,
            "reference_date": self.reference_date.isoformat(),
            "current_date": self.current_date.isoformat(),
            "generated_at": self.generated_at.isoformat(),
            "summary": {
                "total_features": self.total_features,
                "drifted_features": self.drifted_features,
                "dataset_drift_ratio": round(self.dataset_drift_ratio, 4),
                "dataset_drifted": self.dataset_drifted,
                "recommend_retrain": self.recommend_retrain,
            },
            "feature_results": [
                {
                    "feature_name": r.feature_name,
                    "stattest_name": r.stattest_name,
                    "drift_score": round(r.drift_score, 6),
                    "drifted": r.drifted,
                    "psi": round(r.psi, 4) if r.psi is not None else None,
                    "psi_label": r.psi_label,
                    "reference_mean": round(r.reference_mean, 4)
                    if r.reference_mean is not None
                    else None,
                    "current_mean": round(r.current_mean, 4)
                    if r.current_mean is not None
                    else None,
                    "reference_std": round(r.reference_std, 4)
                    if r.reference_std is not None
                    else None,
                    "current_std": round(r.current_std, 4)
                    if r.current_std is not None
                    else None,
                }
                for r in self.feature_results
            ],
        }


def _compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Population Stability Index (PSI).

    PSI = sum((current_pct - ref_pct) * ln(current_pct / ref_pct))

    Args:
        reference: 1-D array of reference distribution values.
        current:   1-D array of current distribution values.
        n_bins:    Number of equal-frequency bins.

    Returns:
        PSI value. Higher = more drift.
    """
    # Bin edges from the reference distribution (equal-frequency)
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(reference, quantiles)
    bin_edges = np.unique(bin_edges)  # collapse identical edges

    if len(bin_edges) < 2:
        return 0.0  # degenerate distribution

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)

    eps = 1e-8  # avoid log(0)
    ref_pct = (ref_counts + eps) / (reference.shape[0] + eps * n_bins)
    cur_pct = (cur_counts + eps) / (current.shape[0] + eps * n_bins)

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return max(psi, 0.0)


def _psi_label(psi: float) -> str:
    if psi < PSI_LOW:
        return "LOW"
    if psi < PSI_HIGH:
        return "MEDIUM"
    return "HIGH"


class DriftDetector:
    """
    Drift detector wrapping Evidently AI DataDriftReport.

    Supports numeric and categorical features.  PSI is computed for all
    numeric features as a supplementary signal regardless of test outcome.

    Example:
        detector = DriftDetector(model_name="fraud_v3")
        report = detector.detect(reference_df, current_df)
        if report.recommend_retrain:
            trigger_retrain_pipeline()
    """

    def __init__(
        self,
        model_name: str,
        drift_threshold: float = DATASET_DRIFT_THRESHOLD,
        stattest: str | None = None,
        num_stattest: str | None = None,
        cat_stattest: str | None = None,
        stattest_threshold: float | None = None,
        column_mapping: dict[str, Any] | None = None,
    ) -> None:
        """
        Args:
            model_name:          Human-readable model name for the report.
            drift_threshold:     Fraction of drifted features that triggers retrain.
            stattest:            Evidently stattest name (overrides num/cat split).
            num_stattest:        Stattest for numeric columns (default: ks).
            cat_stattest:        Stattest for categorical columns (default: chi2).
            stattest_threshold:  p-value / distance threshold for drift detection.
            column_mapping:      Evidently ColumnMapping dict.
        """
        self.model_name = model_name
        self.drift_threshold = drift_threshold
        self._stattest = stattest
        self._num_stattest = num_stattest or "ks"
        self._cat_stattest = cat_stattest or "chi2"
        self._stattest_threshold = stattest_threshold
        self._column_mapping = column_mapping

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        reference_date: datetime | None = None,
        current_date: datetime | None = None,
    ) -> DriftReport:
        """
        Run drift detection and return a DriftReport.

        Args:
            reference_df:   Training / baseline data (no index requirement).
            current_df:     Production window data to compare against reference.
            reference_date: Timestamp representing the reference period.
            current_date:   Timestamp representing the current period.

        Returns:
            DriftReport with per-feature results and retrain recommendation.
        """
        from evidently import ColumnMapping  # noqa: PLC0415
        from evidently.metrics import DataDriftTable, DatasetDriftMetric  # noqa: PLC0415
        from evidently.report import Report  # noqa: PLC0415

        ref_date = reference_date or datetime.utcnow()
        cur_date = current_date or datetime.utcnow()

        # Build Evidently ColumnMapping
        col_mapping = None
        if self._column_mapping:
            col_mapping = ColumnMapping(**self._column_mapping)

        # Configure stat tests per column type
        metric_kwargs: dict[str, Any] = {}
        if self._stattest:
            metric_kwargs["stattest"] = self._stattest
        if self._stattest_threshold:
            metric_kwargs["stattest_threshold"] = self._stattest_threshold

        report = Report(
            metrics=[
                DatasetDriftMetric(**metric_kwargs),
                DataDriftTable(**metric_kwargs),
            ]
        )
        report.run(
            reference_data=reference_df,
            current_data=current_df,
            column_mapping=col_mapping,
        )

        raw = report.as_dict()
        return self._parse_report(raw, ref_date, cur_date, reference_df, current_df)

    def detect_prediction_drift(
        self,
        reference_predictions: pd.Series,
        current_predictions: pd.Series,
    ) -> FeatureDriftResult:
        """
        Lightweight check on model prediction distribution shift.

        Uses PSI + KS test on the prediction scores.  Useful for catching
        silent failures where input features look stable but outputs shift.
        """
        from scipy import stats  # noqa: PLC0415

        ref = reference_predictions.dropna().to_numpy()
        cur = current_predictions.dropna().to_numpy()

        ks_stat, p_value = stats.ks_2samp(ref, cur)
        psi_val = _compute_psi(ref, cur)

        return FeatureDriftResult(
            feature_name="__prediction__",
            stattest_name="ks",
            drift_score=float(p_value),
            drifted=bool(p_value < 0.05),
            psi=psi_val,
            psi_label=_psi_label(psi_val),
            reference_mean=float(np.mean(ref)),
            current_mean=float(np.mean(cur)),
            reference_std=float(np.std(ref)),
            current_std=float(np.std(cur)),
        )

    def save_report(self, report: DriftReport, path: str) -> None:
        """Persist a DriftReport as JSON."""
        from pathlib import Path  # noqa: PLC0415

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(report.to_dict(), fh, indent=2)
        logger.info("Drift report saved to %s", path)

    # ------------------------------------------------------------------
    # Internal parsing
    # ------------------------------------------------------------------

    def _parse_report(
        self,
        raw: dict[str, Any],
        ref_date: datetime,
        cur_date: datetime,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
    ) -> DriftReport:
        """Convert raw Evidently dict into a DriftReport."""
        metrics = raw.get("metrics", [])

        # Parse dataset-level metric
        dataset_metric = next(
            (m for m in metrics if m.get("metric") == "DatasetDriftMetric"), {}
        )
        result = dataset_metric.get("result", {})
        dataset_drifted: bool = result.get("dataset_drift", False)
        drift_ratio: float = result.get("share_of_drifted_columns", 0.0)
        drifted_count: int = result.get("number_of_drifted_columns", 0)
        total_count: int = result.get("number_of_columns", len(reference_df.columns))

        # Parse per-feature table
        table_metric = next(
            (m for m in metrics if m.get("metric") == "DataDriftTable"), {}
        )
        column_results = table_metric.get("result", {}).get("drift_by_columns", {})

        feature_results: list[FeatureDriftResult] = []
        for col_name, col_data in column_results.items():
            ref_arr = (
                reference_df[col_name].dropna().to_numpy()
                if col_name in reference_df.columns
                else np.array([])
            )
            cur_arr = (
                current_df[col_name].dropna().to_numpy()
                if col_name in current_df.columns
                else np.array([])
            )

            psi_val: float | None = None
            psi_lbl: str | None = None
            if (
                np.issubdtype(ref_arr.dtype, np.number)
                and len(ref_arr) > 0
                and len(cur_arr) > 0
            ):
                psi_val = _compute_psi(ref_arr, cur_arr)
                psi_lbl = _psi_label(psi_val)

            feature_results.append(
                FeatureDriftResult(
                    feature_name=col_name,
                    stattest_name=col_data.get("stattest_name", "unknown"),
                    drift_score=float(col_data.get("drift_score", 0.0)),
                    drifted=bool(col_data.get("drift_detected", False)),
                    psi=psi_val,
                    psi_label=psi_lbl,
                    reference_mean=col_data.get("reference", {}).get("mean"),
                    current_mean=col_data.get("current", {}).get("mean"),
                    reference_std=col_data.get("reference", {}).get("std"),
                    current_std=col_data.get("current", {}).get("std"),
                )
            )

        recommend_retrain = drift_ratio >= self.drift_threshold or dataset_drifted

        logger.info(
            "Drift detection complete for %s: %d/%d features drifted (ratio=%.2f), retrain=%s",
            self.model_name,
            drifted_count,
            total_count,
            drift_ratio,
            recommend_retrain,
        )

        return DriftReport(
            model_name=self.model_name,
            reference_date=ref_date,
            current_date=cur_date,
            feature_results=feature_results,
            total_features=total_count,
            drifted_features=drifted_count,
            dataset_drift_ratio=drift_ratio,
            dataset_drifted=dataset_drifted,
            recommend_retrain=recommend_retrain,
            raw_report=raw,
        )
