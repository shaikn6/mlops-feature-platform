"""
Tests for monitoring.drift_detector
Covers: _compute_psi, _psi_label, DriftDetector, DriftReport, FeatureDriftResult
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from monitoring.drift_detector import (
    DATASET_DRIFT_THRESHOLD,
    PSI_HIGH,
    PSI_LOW,
    DriftDetector,
    DriftReport,
    FeatureDriftResult,
    _compute_psi,
    _psi_label,
)


# ---------------------------------------------------------------------------
# _compute_psi
# ---------------------------------------------------------------------------


class TestComputePsi:
    def test_identical_distributions_near_zero(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 1000)
        psi = _compute_psi(data, data)
        assert psi < 0.01

    def test_very_different_distributions_high_psi(self):
        ref = np.random.default_rng(0).normal(0, 1, 1000)
        cur = np.random.default_rng(0).normal(10, 1, 1000)
        psi = _compute_psi(ref, cur)
        assert psi > PSI_HIGH

    def test_psi_is_non_negative(self):
        rng = np.random.default_rng(1)
        ref = rng.exponential(1, 500)
        cur = rng.exponential(2, 500)
        assert _compute_psi(ref, cur) >= 0.0

    def test_degenerate_single_value_returns_zero(self):
        ref = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        cur = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        psi = _compute_psi(ref, cur)
        assert psi == 0.0

    def test_custom_n_bins(self):
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 2000)
        cur = rng.normal(0.5, 1, 2000)
        psi_10 = _compute_psi(ref, cur, n_bins=10)
        psi_20 = _compute_psi(ref, cur, n_bins=20)
        assert psi_10 >= 0
        assert psi_20 >= 0

    def test_psi_moderate_shift_in_medium_range(self):
        ref = np.random.default_rng(10).normal(0, 1, 2000)
        cur = np.random.default_rng(10).normal(1.5, 1, 2000)
        psi = _compute_psi(ref, cur)
        assert psi > PSI_LOW


# ---------------------------------------------------------------------------
# _psi_label
# ---------------------------------------------------------------------------


class TestPsiLabel:
    def test_below_low_threshold_returns_low(self):
        assert _psi_label(PSI_LOW - 0.01) == "LOW"

    def test_at_low_threshold_returns_medium(self):
        assert _psi_label(PSI_LOW) == "MEDIUM"

    def test_between_low_and_high_returns_medium(self):
        assert _psi_label((PSI_LOW + PSI_HIGH) / 2) == "MEDIUM"

    def test_at_high_threshold_returns_high(self):
        assert _psi_label(PSI_HIGH) == "HIGH"

    def test_above_high_threshold_returns_high(self):
        assert _psi_label(1.0) == "HIGH"

    def test_zero_returns_low(self):
        assert _psi_label(0.0) == "LOW"


# ---------------------------------------------------------------------------
# FeatureDriftResult
# ---------------------------------------------------------------------------


class TestFeatureDriftResult:
    def test_basic_construction(self):
        result = FeatureDriftResult(
            feature_name="avg_spend_7d",
            stattest_name="ks",
            drift_score=0.03,
            drifted=True,
        )
        assert result.feature_name == "avg_spend_7d"
        assert result.drifted is True
        assert result.psi is None
        assert result.psi_label is None

    def test_with_all_fields(self):
        result = FeatureDriftResult(
            feature_name="credit_utilization",
            stattest_name="chi2",
            drift_score=0.01,
            drifted=False,
            psi=0.05,
            psi_label="LOW",
            reference_mean=0.5,
            current_mean=0.55,
            reference_std=0.1,
            current_std=0.12,
        )
        assert result.psi == 0.05
        assert result.psi_label == "LOW"
        assert result.reference_mean == 0.5


# ---------------------------------------------------------------------------
# DriftReport
# ---------------------------------------------------------------------------


class TestDriftReport:
    def _make_report(self, drifted=False, ratio=0.0, retrain=False):
        feat = FeatureDriftResult(
            feature_name="feat_a",
            stattest_name="ks",
            drift_score=0.01,
            drifted=drifted,
        )
        return DriftReport(
            model_name="fraud_v3",
            reference_date=datetime(2024, 1, 1),
            current_date=datetime(2024, 6, 1),
            feature_results=[feat],
            total_features=1,
            drifted_features=1 if drifted else 0,
            dataset_drift_ratio=ratio,
            dataset_drifted=drifted,
            recommend_retrain=retrain,
        )

    def test_to_dict_contains_expected_keys(self):
        report = self._make_report()
        d = report.to_dict()
        assert "model_name" in d
        assert "summary" in d
        assert "feature_results" in d

    def test_summary_contains_all_fields(self):
        report = self._make_report(drifted=True, ratio=0.25, retrain=True)
        summary = report.to_dict()["summary"]
        assert summary["recommend_retrain"] is True
        assert summary["dataset_drift_ratio"] == 0.25

    def test_reference_date_serialized_as_iso(self):
        report = self._make_report()
        d = report.to_dict()
        assert d["reference_date"] == "2024-01-01T00:00:00"

    def test_feature_results_are_serialized(self):
        report = self._make_report()
        d = report.to_dict()
        assert isinstance(d["feature_results"], list)
        assert d["feature_results"][0]["feature_name"] == "feat_a"

    def test_none_psi_serializes_as_null(self):
        report = self._make_report()
        d = report.to_dict()
        assert d["feature_results"][0]["psi"] is None


# ---------------------------------------------------------------------------
# DriftDetector.__init__
# ---------------------------------------------------------------------------


class TestDriftDetectorInit:
    def test_default_values(self):
        det = DriftDetector(model_name="test_model")
        assert det.model_name == "test_model"
        assert det.drift_threshold == DATASET_DRIFT_THRESHOLD
        assert det._num_stattest == "ks"
        assert det._cat_stattest == "chi2"

    def test_custom_values(self):
        det = DriftDetector(
            model_name="credit_v2",
            drift_threshold=0.15,
            num_stattest="wasserstein",
            cat_stattest="chi2",
            stattest_threshold=0.01,
        )
        assert det.drift_threshold == 0.15
        assert det._num_stattest == "wasserstein"
        assert det._stattest_threshold == 0.01

    def test_stattest_overrides_num_cat(self):
        det = DriftDetector(model_name="m", stattest="psi")
        assert det._stattest == "psi"


# ---------------------------------------------------------------------------
# DriftDetector.detect() — mocked Evidently
# ---------------------------------------------------------------------------


def _make_evidently_raw(drifted_count=1, total=5, drift_ratio=0.2, dataset_drift=False):
    return {
        "metrics": [
            {
                "metric": "DatasetDriftMetric",
                "result": {
                    "dataset_drift": dataset_drift,
                    "share_of_drifted_columns": drift_ratio,
                    "number_of_drifted_columns": drifted_count,
                    "number_of_columns": total,
                },
            },
            {
                "metric": "DataDriftTable",
                "result": {
                    "drift_by_columns": {
                        "avg_spend_7d": {
                            "stattest_name": "ks",
                            "drift_score": 0.01,
                            "drift_detected": True,
                            "reference": {"mean": 100.0, "std": 20.0},
                            "current": {"mean": 150.0, "std": 30.0},
                        },
                        "customer_segment": {
                            "stattest_name": "chi2",
                            "drift_score": 0.5,
                            "drift_detected": False,
                            "reference": {},
                            "current": {},
                        },
                    }
                },
            },
        ]
    }


@pytest.fixture
def mock_evidently():
    """Patch evidently imports used in detect()."""
    mock_report_cls = MagicMock()
    mock_report_inst = MagicMock()
    mock_report_cls.return_value = mock_report_inst
    mock_report_inst.as_dict.return_value = _make_evidently_raw()

    with patch.dict(
        "sys.modules",
        {
            "evidently": MagicMock(ColumnMapping=MagicMock()),
            "evidently.metrics": MagicMock(
                DataDriftTable=MagicMock(),
                DatasetDriftMetric=MagicMock(),
            ),
            "evidently.report": MagicMock(Report=mock_report_cls),
        },
    ):
        yield mock_report_cls, mock_report_inst


class TestDriftDetectorDetect:
    def _ref_df(self):
        rng = np.random.default_rng(0)
        return pd.DataFrame(
            {
                "avg_spend_7d": rng.normal(100, 20, 200),
                "customer_segment": np.random.choice(["RETAIL", "PREMIUM"], 200),
            }
        )

    def _cur_df(self):
        rng = np.random.default_rng(1)
        return pd.DataFrame(
            {
                "avg_spend_7d": rng.normal(150, 30, 200),
                "customer_segment": np.random.choice(["RETAIL", "PREMIUM"], 200),
            }
        )

    def test_returns_drift_report(self, mock_evidently):
        det = DriftDetector(model_name="fraud_v3")
        report = det.detect(self._ref_df(), self._cur_df())
        assert isinstance(report, DriftReport)

    def test_model_name_set_on_report(self, mock_evidently):
        det = DriftDetector(model_name="fraud_v3")
        report = det.detect(self._ref_df(), self._cur_df())
        assert report.model_name == "fraud_v3"

    def test_feature_results_parsed(self, mock_evidently):
        det = DriftDetector(model_name="fraud_v3")
        report = det.detect(self._ref_df(), self._cur_df())
        assert len(report.feature_results) == 2

    def test_numeric_feature_has_psi(self, mock_evidently):
        det = DriftDetector(model_name="fraud_v3")
        report = det.detect(self._ref_df(), self._cur_df())
        numeric_feat = next(
            r for r in report.feature_results if r.feature_name == "avg_spend_7d"
        )
        assert numeric_feat.psi is not None

    def test_categorical_feature_psi_none(self, mock_evidently):
        det = DriftDetector(model_name="fraud_v3")
        report = det.detect(self._ref_df(), self._cur_df())
        cat_feat = next(
            r for r in report.feature_results if r.feature_name == "customer_segment"
        )
        # categorical arrays are object dtype, psi should be None
        assert cat_feat.psi is None

    def test_recommend_retrain_when_high_ratio(self):
        raw = _make_evidently_raw(
            drifted_count=3, total=5, drift_ratio=0.6, dataset_drift=True
        )
        mock_report_cls = MagicMock()
        mock_report_inst = MagicMock()
        mock_report_cls.return_value = mock_report_inst
        mock_report_inst.as_dict.return_value = raw

        with patch.dict(
            "sys.modules",
            {
                "evidently": MagicMock(ColumnMapping=MagicMock()),
                "evidently.metrics": MagicMock(
                    DataDriftTable=MagicMock(), DatasetDriftMetric=MagicMock()
                ),
                "evidently.report": MagicMock(Report=mock_report_cls),
            },
        ):
            rng = np.random.default_rng(0)
            ref = pd.DataFrame({"avg_spend_7d": rng.normal(100, 20, 200)})
            cur = pd.DataFrame({"avg_spend_7d": rng.normal(200, 50, 200)})
            det = DriftDetector(model_name="fraud_v3")
            report = det.detect(ref, cur)
            assert report.recommend_retrain is True

    def test_no_retrain_when_low_drift(self):
        raw = _make_evidently_raw(
            drifted_count=0, total=5, drift_ratio=0.0, dataset_drift=False
        )
        mock_report_cls = MagicMock()
        mock_report_inst = MagicMock()
        mock_report_cls.return_value = mock_report_inst
        mock_report_inst.as_dict.return_value = raw

        with patch.dict(
            "sys.modules",
            {
                "evidently": MagicMock(ColumnMapping=MagicMock()),
                "evidently.metrics": MagicMock(
                    DataDriftTable=MagicMock(), DatasetDriftMetric=MagicMock()
                ),
                "evidently.report": MagicMock(Report=mock_report_cls),
            },
        ):
            rng = np.random.default_rng(0)
            ref = pd.DataFrame({"avg_spend_7d": rng.normal(100, 20, 200)})
            cur = pd.DataFrame({"avg_spend_7d": rng.normal(100, 20, 200)})
            det = DriftDetector(model_name="fraud_v3")
            report = det.detect(ref, cur)
            assert report.recommend_retrain is False

    def test_custom_dates_set_on_report(self, mock_evidently):
        det = DriftDetector(model_name="m")
        ref_date = datetime(2024, 1, 1)
        cur_date = datetime(2024, 6, 1)
        report = det.detect(
            self._ref_df(),
            self._cur_df(),
            reference_date=ref_date,
            current_date=cur_date,
        )
        assert report.reference_date == ref_date
        assert report.current_date == cur_date

    def test_column_mapping_used_when_provided(self, mock_evidently):
        det = DriftDetector(model_name="m", column_mapping={"target": "is_fraud"})
        # Should not raise — column_mapping is passed to ColumnMapping
        report = det.detect(self._ref_df(), self._cur_df())
        assert report is not None


# ---------------------------------------------------------------------------
# detect_prediction_drift()
# ---------------------------------------------------------------------------


class TestDetectPredictionDrift:
    def test_returns_feature_drift_result(self):
        det = DriftDetector(model_name="m")
        rng = np.random.default_rng(0)
        ref = pd.Series(rng.uniform(0, 1, 500))
        cur = pd.Series(rng.uniform(0.3, 1, 500))
        with patch("monitoring.drift_detector._compute_psi", return_value=0.05):
            result = det.detect_prediction_drift(ref, cur)
        assert isinstance(result, FeatureDriftResult)
        assert result.feature_name == "__prediction__"

    def test_feature_name_is_prediction(self):
        det = DriftDetector(model_name="m")
        rng = np.random.default_rng(0)
        ref = pd.Series(rng.normal(0.3, 0.1, 1000))
        cur = pd.Series(rng.normal(0.3, 0.1, 1000))
        result = det.detect_prediction_drift(ref, cur)
        assert result.feature_name == "__prediction__"

    def test_identical_predictions_not_drifted(self):
        det = DriftDetector(model_name="m")
        rng = np.random.default_rng(5)
        preds = pd.Series(rng.uniform(0, 1, 1000))
        result = det.detect_prediction_drift(preds, preds)
        assert result.drifted is False

    def test_very_different_predictions_drifted(self):
        det = DriftDetector(model_name="m")
        ref = pd.Series(np.zeros(1000))
        cur = pd.Series(np.ones(1000))
        result = det.detect_prediction_drift(ref, cur)
        assert result.drifted is True

    def test_result_has_mean_and_std(self):
        det = DriftDetector(model_name="m")
        ref = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5] * 100)
        cur = pd.Series([0.6, 0.7, 0.8, 0.9, 1.0] * 100)
        result = det.detect_prediction_drift(ref, cur)
        assert result.reference_mean is not None
        assert result.current_mean is not None
        assert result.reference_std is not None
        assert result.current_std is not None


# ---------------------------------------------------------------------------
# save_report()
# ---------------------------------------------------------------------------


class TestSaveReport:
    def test_creates_file(self, tmp_path):
        det = DriftDetector(model_name="test")
        report = DriftReport(
            model_name="test",
            reference_date=datetime(2024, 1, 1),
            current_date=datetime(2024, 6, 1),
        )
        out_path = str(tmp_path / "subdir" / "report.json")
        det.save_report(report, out_path)
        assert Path(out_path).exists()

    def test_file_is_valid_json(self, tmp_path):
        det = DriftDetector(model_name="test")
        report = DriftReport(
            model_name="test",
            reference_date=datetime(2024, 1, 1),
            current_date=datetime(2024, 6, 1),
        )
        out_path = str(tmp_path / "report.json")
        det.save_report(report, out_path)
        with open(out_path) as fh:
            data = json.load(fh)
        assert data["model_name"] == "test"

    def test_creates_parent_directories(self, tmp_path):
        det = DriftDetector(model_name="test")
        report = DriftReport(
            model_name="test",
            reference_date=datetime(2024, 1, 1),
            current_date=datetime(2024, 6, 1),
        )
        deep_path = str(tmp_path / "a" / "b" / "c" / "report.json")
        det.save_report(report, deep_path)
        assert Path(deep_path).exists()
