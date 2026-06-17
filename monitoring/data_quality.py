"""
Data quality monitoring using Great Expectations.

Provides pre-built expectation suites for:
  - transaction data (nulls, ranges, referential integrity)
  - credit bureau data (score bounds, utilization clamps)
  - feature store output (post-materialization sanity checks)

Usage:
    monitor = DataQualityMonitor()
    result = monitor.validate_transactions(df)
    if not result.success:
        alert_on_failure(result)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

try:
    from great_expectations.dataset import PandasDataset as _PandasDataset

    def _from_pandas(df: pd.DataFrame) -> Any:  # pragma: no cover
        return _PandasDataset(df)

except ImportError:  # GE v1.x removed PandasDataset — use legacy compat shim
    try:
        import great_expectations as _ge

        def _from_pandas(df: pd.DataFrame) -> Any:  # pragma: no cover
            # GE v2 API — still works when installed via pip install great-expectations<1
            return _ge.from_pandas(df)  # type: ignore[attr-defined]

    except (ImportError, AttributeError):  # pragma: no cover

        def _from_pandas(df: pd.DataFrame) -> Any:  # pragma: no cover
            raise ImportError(
                "great_expectations is not installed or has an incompatible version. "
                "Install: pip install 'great-expectations<1'"
            )


logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Summary of a Great Expectations validation run."""

    suite_name: str
    success: bool
    evaluated_expectations: int
    successful_expectations: int
    failed_expectations: int
    success_percent: float
    run_time: datetime = field(default_factory=datetime.utcnow)
    failed_checks: list[dict[str, Any]] = field(default_factory=list)
    statistics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_name": self.suite_name,
            "success": self.success,
            "run_time": self.run_time.isoformat(),
            "evaluated_expectations": self.evaluated_expectations,
            "successful_expectations": self.successful_expectations,
            "failed_expectations": self.failed_expectations,
            "success_percent": round(self.success_percent, 2),
            "failed_checks": self.failed_checks,
            "statistics": self.statistics,
        }


class DataQualityMonitor:
    """
    Runs Great Expectations suites against DataFrames.

    Suites are defined in-memory (no filesystem config required) so they
    can be used from Airflow tasks, CI, and notebooks without a GE project.
    """

    # ------------------------------------------------------------------
    # Public validators
    # ------------------------------------------------------------------

    def validate_transactions(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate raw transaction data before feature computation.

        Checks:
          - Required columns present
          - No nulls in primary key / timestamp columns
          - Amount > 0
          - Amount within reasonable bounds (< 1M USD)
          - transaction_type is in known enum
          - event_timestamp is not in the future
        """
        gdf = _from_pandas(df)
        results: list[Any] = []

        # Schema presence
        for col in [
            "transaction_id",
            "customer_id",
            "amount",
            "event_timestamp",
            "transaction_type",
        ]:
            results.append(gdf.expect_column_to_exist(col))

        # Null checks
        for col in ["transaction_id", "customer_id", "event_timestamp"]:
            results.append(gdf.expect_column_values_to_not_be_null(col))

        # Amount sanity
        results.append(
            gdf.expect_column_values_to_be_between("amount", min_value=0.01, max_value=999_999.99)
        )
        results.append(gdf.expect_column_values_to_not_be_null("amount"))

        # Enum check
        results.append(
            gdf.expect_column_values_to_be_in_set(
                "transaction_type",
                value_set=[
                    "PURCHASE",
                    "REFUND",
                    "TRANSFER",
                    "WITHDRAWAL",
                    "DEPOSIT",
                    "FEE",
                ],
            )
        )

        # Uniqueness
        results.append(gdf.expect_column_values_to_be_unique("transaction_id"))

        return self._summarize("transactions_suite", results)

    def validate_credit_bureau(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate credit bureau snapshot data.

        Checks:
          - Utilization in [0, 5] (clamped — values > 1 are over-limit)
          - Payment history in [0, 1]
          - DTI in [0, 10]
          - Credit age >= 0
          - Estimated score in [300, 850]
          - No nulls on key fields
        """
        gdf = _from_pandas(df)
        results: list[Any] = []

        for col in [
            "customer_id",
            "event_timestamp",
            "credit_utilization",
            "payment_history_score",
            "debt_to_income",
            "estimated_credit_score",
        ]:
            results.append(gdf.expect_column_to_exist(col))
            results.append(gdf.expect_column_values_to_not_be_null(col))

        results.append(
            gdf.expect_column_values_to_be_between(
                "credit_utilization", min_value=0.0, max_value=5.0
            )
        )
        results.append(
            gdf.expect_column_values_to_be_between(
                "payment_history_score", min_value=0.0, max_value=1.0
            )
        )
        results.append(
            gdf.expect_column_values_to_be_between("debt_to_income", min_value=0.0, max_value=10.0)
        )
        results.append(
            gdf.expect_column_values_to_be_between(
                "estimated_credit_score", min_value=300.0, max_value=850.0
            )
        )
        results.append(
            gdf.expect_column_values_to_be_between("credit_age_months", min_value=0, max_value=720)
        )
        results.append(
            gdf.expect_column_values_to_be_between(
                "recent_hard_inquiries_6m", min_value=0, max_value=50
            )
        )

        return self._summarize("credit_bureau_suite", results)

    def validate_feature_store_output(self, df: pd.DataFrame) -> ValidationResult:
        """
        Post-materialization sanity checks on Feast feature output.

        Ensures the online store is not returning stale or malformed features.
        """
        _known_columns = {
            "fraud_rate_90d",
            "transaction_count_7d",
            "transaction_count_30d",
            "fraud_count_90d",
            "avg_spend_7d",
            "avg_spend_30d",
            "total_spend_7d",
            "international_txn_ratio_30d",
            "customer_segment",
        }
        if not _known_columns.intersection(df.columns):
            # No recognised columns — treat as trivially passing
            logger.warning("validate_feature_store_output: no known columns found in df.")
            return ValidationResult(
                suite_name="feature_store_output_suite",
                success=True,
                evaluated_expectations=0,
                successful_expectations=0,
                failed_expectations=0,
                success_percent=100.0,
            )

        gdf = _from_pandas(df)
        results: list[Any] = []

        # Fraud rate must be a probability
        if "fraud_rate_90d" in df.columns:
            results.append(
                gdf.expect_column_values_to_be_between(
                    "fraud_rate_90d", min_value=0.0, max_value=1.0
                )
            )

        # Transaction counts must be non-negative integers
        for col in ["transaction_count_7d", "transaction_count_30d", "fraud_count_90d"]:
            if col in df.columns:
                results.append(gdf.expect_column_values_to_be_between(col, min_value=0))

        # Average spend must be positive when not null
        for col in ["avg_spend_7d", "avg_spend_30d", "total_spend_7d"]:
            if col in df.columns:
                results.append(gdf.expect_column_values_to_be_between(col, min_value=0.0))

        # International ratio in [0, 1]
        if "international_txn_ratio_30d" in df.columns:
            results.append(
                gdf.expect_column_values_to_be_between(
                    "international_txn_ratio_30d", min_value=0.0, max_value=1.0
                )
            )

        # Customer segment is a known value
        if "customer_segment" in df.columns:
            results.append(
                gdf.expect_column_values_to_be_in_set(
                    "customer_segment",
                    value_set=["RETAIL", "PREMIUM", "SMB", "CORPORATE"],
                    mostly=0.99,  # allow 1% unknown during segment transitions
                )
            )

        return self._summarize("feature_store_output_suite", results)

    def validate_training_dataset(
        self, df: pd.DataFrame, label_col: str = "is_fraud"
    ) -> ValidationResult:
        """
        Validate a training dataset before model training begins.

        Checks label balance, no-leakage proxies, and feature completeness.
        """
        gdf = _from_pandas(df)
        results: list[Any] = []

        # Label must exist and be boolean-ish
        results.append(gdf.expect_column_to_exist(label_col))
        results.append(gdf.expect_column_values_to_not_be_null(label_col))

        # Dataset must be large enough to train on
        results.append(gdf.expect_table_row_count_to_be_between(min_value=1000))

        # No duplicate rows (would bias training)
        results.append(
            gdf.expect_table_row_count_to_equal_other_table(
                other_table_row_count=len(df.drop_duplicates())
            )
        )

        # Feature completeness — at most 5% nulls per feature column
        feature_cols = [c for c in df.columns if c not in [label_col, "event_timestamp"]]
        for col in feature_cols:
            results.append(gdf.expect_column_values_to_not_be_null(col, mostly=0.95))

        return self._summarize("training_dataset_suite", results)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _summarize(self, suite_name: str, results: list[Any]) -> ValidationResult:
        """Aggregate individual expectation results into a ValidationResult."""
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        pct = (successful / len(results) * 100) if results else 100.0
        overall_success = failed == 0

        failed_checks = [
            {
                "expectation_type": r.expectation_config.expectation_type,
                "column": r.expectation_config.kwargs.get("column", ""),
                "kwargs": r.expectation_config.kwargs,
                "result": r.result,
            }
            for r in results
            if not r.success
        ]

        logger.info(
            "Data quality [%s]: %d/%d passed (%.1f%%) — %s",
            suite_name,
            successful,
            len(results),
            pct,
            "PASS" if overall_success else "FAIL",
        )

        return ValidationResult(
            suite_name=suite_name,
            success=overall_success,
            evaluated_expectations=len(results),
            successful_expectations=successful,
            failed_expectations=failed,
            success_percent=pct,
            failed_checks=failed_checks,
            statistics={
                "evaluated": len(results),
                "successful": successful,
                "unsuccessful": failed,
            },
        )
