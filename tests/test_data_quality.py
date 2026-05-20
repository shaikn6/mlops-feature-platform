"""
Tests for monitoring.data_quality — DataQualityMonitor, ValidationResult
Covers: all four validators, _summarize, edge cases.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from monitoring.data_quality import DataQualityMonitor, ValidationResult


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------


class TestValidationResult:
    def test_to_dict_contains_all_keys(self):
        result = ValidationResult(
            suite_name="transactions_suite",
            success=True,
            evaluated_expectations=10,
            successful_expectations=10,
            failed_expectations=0,
            success_percent=100.0,
        )
        d = result.to_dict()
        assert d["suite_name"] == "transactions_suite"
        assert d["success"] is True
        assert d["evaluated_expectations"] == 10
        assert d["success_percent"] == 100.0

    def test_run_time_serialized(self):
        result = ValidationResult(
            suite_name="s",
            success=True,
            evaluated_expectations=0,
            successful_expectations=0,
            failed_expectations=0,
            success_percent=100.0,
        )
        d = result.to_dict()
        datetime.fromisoformat(d["run_time"])

    def test_failed_checks_defaults_to_empty_list(self):
        result = ValidationResult(
            suite_name="s",
            success=False,
            evaluated_expectations=5,
            successful_expectations=4,
            failed_expectations=1,
            success_percent=80.0,
        )
        assert result.failed_checks == []


# ---------------------------------------------------------------------------
# Helpers to build GE mock expectations
# ---------------------------------------------------------------------------


def _ge_result(
    success: bool, expectation_type: str = "expect_column_to_exist", column: str = "col"
):
    r = MagicMock()
    r.success = success
    r.expectation_config = MagicMock()
    r.expectation_config.expectation_type = expectation_type
    r.expectation_config.kwargs = {"column": column}
    r.result = {}
    return r


def _mock_ge_df(all_pass: bool = True, num_checks: int = 8):
    """Build a mock great_expectations dataset."""
    gdf = MagicMock()
    result = _ge_result(all_pass)
    # Every expectation method returns a passing/failing result
    for method_name in [
        "expect_column_to_exist",
        "expect_column_values_to_not_be_null",
        "expect_column_values_to_be_between",
        "expect_column_values_to_be_in_set",
        "expect_column_values_to_be_unique",
        "expect_table_row_count_to_be_between",
        "expect_table_row_count_to_equal_other_table",
    ]:
        getattr(gdf, method_name).return_value = result
    return gdf


# ---------------------------------------------------------------------------
# validate_transactions()
# ---------------------------------------------------------------------------


class TestValidateTransactions:
    def _valid_df(self):
        return pd.DataFrame(
            {
                "transaction_id": ["t1", "t2"],
                "customer_id": ["c1", "c2"],
                "amount": [100.0, 200.0],
                "event_timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
                "transaction_type": ["PURCHASE", "REFUND"],
            }
        )

    def test_returns_validation_result(self):
        monitor = DataQualityMonitor()
        mock_gdf = _mock_ge_df(all_pass=True)
        with patch("monitoring.data_quality.ge.from_pandas", return_value=mock_gdf):
            result = monitor.validate_transactions(self._valid_df())
        assert isinstance(result, ValidationResult)

    def test_suite_name_is_transactions_suite(self):
        monitor = DataQualityMonitor()
        mock_gdf = _mock_ge_df(all_pass=True)
        with patch("monitoring.data_quality.ge.from_pandas", return_value=mock_gdf):
            result = monitor.validate_transactions(self._valid_df())
        assert result.suite_name == "transactions_suite"

    def test_all_pass_means_success(self):
        monitor = DataQualityMonitor()
        mock_gdf = _mock_ge_df(all_pass=True)
        with patch("monitoring.data_quality.ge.from_pandas", return_value=mock_gdf):
            result = monitor.validate_transactions(self._valid_df())
        assert result.success is True
        assert result.failed_expectations == 0

    def test_failure_propagates(self):
        monitor = DataQualityMonitor()
        mock_gdf = _mock_ge_df(all_pass=False)
        with patch("monitoring.data_quality.ge.from_pandas", return_value=mock_gdf):
            result = monitor.validate_transactions(self._valid_df())
        assert result.success is False
        assert result.successful_expectations == 0

    def test_evaluates_multiple_expectations(self):
        monitor = DataQualityMonitor()
        mock_gdf = _mock_ge_df(all_pass=True)
        with patch("monitoring.data_quality.ge.from_pandas", return_value=mock_gdf):
            result = monitor.validate_transactions(self._valid_df())
        # At minimum: 5 column exists + 3 null + 2 amount + 1 enum + 1 unique
        assert result.evaluated_expectations >= 10


# ---------------------------------------------------------------------------
# validate_credit_bureau()
# ---------------------------------------------------------------------------


class TestValidateCreditBureau:
    def _valid_df(self):
        return pd.DataFrame(
            {
                "customer_id": ["c1"],
                "event_timestamp": [datetime(2024, 1, 1)],
                "credit_utilization": [0.35],
                "payment_history_score": [0.9],
                "debt_to_income": [0.4],
                "estimated_credit_score": [720.0],
                "credit_age_months": [48],
                "recent_hard_inquiries_6m": [1],
            }
        )

    def test_returns_validation_result(self):
        monitor = DataQualityMonitor()
        mock_gdf = _mock_ge_df(all_pass=True)
        with patch("monitoring.data_quality.ge.from_pandas", return_value=mock_gdf):
            result = monitor.validate_credit_bureau(self._valid_df())
        assert isinstance(result, ValidationResult)

    def test_suite_name(self):
        monitor = DataQualityMonitor()
        mock_gdf = _mock_ge_df(all_pass=True)
        with patch("monitoring.data_quality.ge.from_pandas", return_value=mock_gdf):
            result = monitor.validate_credit_bureau(self._valid_df())
        assert result.suite_name == "credit_bureau_suite"

    def test_success_on_all_pass(self):
        monitor = DataQualityMonitor()
        mock_gdf = _mock_ge_df(all_pass=True)
        with patch("monitoring.data_quality.ge.from_pandas", return_value=mock_gdf):
            result = monitor.validate_credit_bureau(self._valid_df())
        assert result.success is True

    def test_checks_many_columns(self):
        monitor = DataQualityMonitor()
        mock_gdf = _mock_ge_df(all_pass=True)
        with patch("monitoring.data_quality.ge.from_pandas", return_value=mock_gdf):
            result = monitor.validate_credit_bureau(self._valid_df())
        assert result.evaluated_expectations >= 8


# ---------------------------------------------------------------------------
# validate_feature_store_output()
# ---------------------------------------------------------------------------


class TestValidateFeatureStoreOutput:
    def test_returns_trivially_passing_on_empty_df(self):
        monitor = DataQualityMonitor()
        empty = pd.DataFrame()
        result = monitor.validate_feature_store_output(empty)
        assert result.success is True
        assert result.evaluated_expectations == 0

    def test_unknown_columns_treated_as_trivially_passing(self):
        monitor = DataQualityMonitor()
        df = pd.DataFrame({"unknown_col": [1, 2, 3]})
        result = monitor.validate_feature_store_output(df)
        assert result.success is True

    def test_fraud_rate_checked_when_present(self):
        monitor = DataQualityMonitor()
        df = pd.DataFrame({"fraud_rate_90d": [0.01, 0.05, 0.1]})
        mock_gdf = _mock_ge_df(all_pass=True)
        with patch("monitoring.data_quality.ge.from_pandas", return_value=mock_gdf):
            result = monitor.validate_feature_store_output(df)
        assert result.evaluated_expectations >= 1

    def test_transaction_counts_checked_when_present(self):
        monitor = DataQualityMonitor()
        df = pd.DataFrame(
            {
                "transaction_count_7d": [5, 10],
                "transaction_count_30d": [20, 30],
            }
        )
        mock_gdf = _mock_ge_df(all_pass=True)
        with patch("monitoring.data_quality.ge.from_pandas", return_value=mock_gdf):
            result = monitor.validate_feature_store_output(df)
        assert result.evaluated_expectations >= 2

    def test_customer_segment_in_known_set_checked(self):
        monitor = DataQualityMonitor()
        df = pd.DataFrame({"customer_segment": ["RETAIL", "PREMIUM"]})
        mock_gdf = _mock_ge_df(all_pass=True)
        with patch("monitoring.data_quality.ge.from_pandas", return_value=mock_gdf):
            result = monitor.validate_feature_store_output(df)
        assert result.evaluated_expectations >= 1

    def test_international_ratio_checked(self):
        monitor = DataQualityMonitor()
        df = pd.DataFrame({"international_txn_ratio_30d": [0.05, 0.1]})
        mock_gdf = _mock_ge_df(all_pass=True)
        with patch("monitoring.data_quality.ge.from_pandas", return_value=mock_gdf):
            result = monitor.validate_feature_store_output(df)
        assert result.evaluated_expectations >= 1


# ---------------------------------------------------------------------------
# validate_training_dataset()
# ---------------------------------------------------------------------------


class TestValidateTrainingDataset:
    def _valid_df(self, rows=2000):
        return pd.DataFrame(
            {
                "customer_id": [f"c{i}" for i in range(rows)],
                "event_timestamp": [datetime(2024, 1, 1)] * rows,
                "avg_spend_7d": [100.0] * rows,
                "is_fraud": [False] * rows,
            }
        )

    def test_returns_validation_result(self):
        monitor = DataQualityMonitor()
        mock_gdf = _mock_ge_df(all_pass=True)
        with patch("monitoring.data_quality.ge.from_pandas", return_value=mock_gdf):
            result = monitor.validate_training_dataset(self._valid_df())
        assert isinstance(result, ValidationResult)

    def test_suite_name(self):
        monitor = DataQualityMonitor()
        mock_gdf = _mock_ge_df(all_pass=True)
        with patch("monitoring.data_quality.ge.from_pandas", return_value=mock_gdf):
            result = monitor.validate_training_dataset(self._valid_df())
        assert result.suite_name == "training_dataset_suite"

    def test_custom_label_column(self):
        monitor = DataQualityMonitor()
        df = self._valid_df()
        df = df.rename(columns={"is_fraud": "churn_label"})
        mock_gdf = _mock_ge_df(all_pass=True)
        with patch("monitoring.data_quality.ge.from_pandas", return_value=mock_gdf):
            result = monitor.validate_training_dataset(df, label_col="churn_label")
        assert isinstance(result, ValidationResult)

    def test_at_least_4_expectations_checked(self):
        monitor = DataQualityMonitor()
        mock_gdf = _mock_ge_df(all_pass=True)
        with patch("monitoring.data_quality.ge.from_pandas", return_value=mock_gdf):
            result = monitor.validate_training_dataset(self._valid_df())
        assert result.evaluated_expectations >= 4

    def test_failure_propagates(self):
        monitor = DataQualityMonitor()
        mock_gdf = _mock_ge_df(all_pass=False)
        with patch("monitoring.data_quality.ge.from_pandas", return_value=mock_gdf):
            result = monitor.validate_training_dataset(self._valid_df())
        assert result.success is False


# ---------------------------------------------------------------------------
# _summarize()
# ---------------------------------------------------------------------------


class TestSummarize:
    def test_all_pass(self):
        monitor = DataQualityMonitor()
        results = [_ge_result(True) for _ in range(10)]
        summary = monitor._summarize("test_suite", results)
        assert summary.success is True
        assert summary.successful_expectations == 10
        assert summary.failed_expectations == 0
        assert summary.success_percent == 100.0

    def test_all_fail(self):
        monitor = DataQualityMonitor()
        results = [_ge_result(False) for _ in range(5)]
        summary = monitor._summarize("test_suite", results)
        assert summary.success is False
        assert summary.successful_expectations == 0
        assert summary.failed_expectations == 5
        assert summary.success_percent == 0.0

    def test_partial_pass(self):
        monitor = DataQualityMonitor()
        results = [_ge_result(True)] * 7 + [_ge_result(False)] * 3
        summary = monitor._summarize("test_suite", results)
        assert not summary.success
        assert summary.success_percent == pytest.approx(70.0)

    def test_empty_results_gives_100_percent(self):
        monitor = DataQualityMonitor()
        summary = monitor._summarize("empty_suite", [])
        assert summary.success_percent == 100.0
        assert summary.evaluated_expectations == 0

    def test_failed_checks_populated(self):
        monitor = DataQualityMonitor()
        results = [_ge_result(False, "expect_column_to_exist", "customer_id")]
        summary = monitor._summarize("s", results)
        assert len(summary.failed_checks) == 1
        assert summary.failed_checks[0]["column"] == "customer_id"

    def test_statistics_populated(self):
        monitor = DataQualityMonitor()
        results = [_ge_result(True)] * 4 + [_ge_result(False)] * 1
        summary = monitor._summarize("s", results)
        assert summary.statistics["evaluated"] == 5
        assert summary.statistics["successful"] == 4
        assert summary.statistics["unsuccessful"] == 1
