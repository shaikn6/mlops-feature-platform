"""
Tests for the pipelines package.
Covers orchestration task stubs, Airflow DAG simulation, and pipeline utilities.
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch, call
from datetime import datetime, timedelta

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Airflow mock stubs (Airflow not installed in CI)
# ---------------------------------------------------------------------------

def _make_airflow_stubs():
    """Build minimal Airflow mock modules."""
    airflow = ModuleType("airflow")
    airflow_models = ModuleType("airflow.models")
    airflow_operators = ModuleType("airflow.operators")
    airflow_operators_python = ModuleType("airflow.operators.python")
    airflow_utils = ModuleType("airflow.utils")
    airflow_utils_dates = ModuleType("airflow.utils.dates")

    class DAG:
        def __init__(self, dag_id, **kwargs):
            self.dag_id = dag_id
            self.kwargs = kwargs
            self.tasks = []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class PythonOperator:
        def __init__(self, task_id, python_callable, **kwargs):
            self.task_id = task_id
            self.python_callable = python_callable
            self.kwargs = kwargs

        def set_downstream(self, other):
            pass

        def __rshift__(self, other):
            return other

    airflow_models.DAG = DAG
    airflow_operators_python.PythonOperator = PythonOperator
    airflow.DAG = DAG

    for mod_name, mod in [
        ("airflow", airflow),
        ("airflow.models", airflow_models),
        ("airflow.operators", airflow_operators),
        ("airflow.operators.python", airflow_operators_python),
        ("airflow.utils", airflow_utils),
        ("airflow.utils.dates", airflow_utils_dates),
    ]:
        sys.modules[mod_name] = mod

    return DAG, PythonOperator


@pytest.fixture(autouse=True)
def airflow_stubs():
    dag_cls, op_cls = _make_airflow_stubs()
    yield dag_cls, op_cls
    for key in list(sys.modules.keys()):
        if key.startswith("airflow"):
            sys.modules.pop(key, None)


# ---------------------------------------------------------------------------
# Pipeline task function tests (inline implementations)
# ---------------------------------------------------------------------------

class TestFeaturePipelineTasks:
    """Tests for common pipeline task patterns used in Airflow DAGs."""

    def test_materialize_features_task_calls_registry(self):
        """Simulate an Airflow task that materializes features."""
        from feature_store.registry import FeatureRegistry

        mock_registry = MagicMock(spec=FeatureRegistry)

        def materialize_features_task(**context):
            """Typical Airflow task callable."""
            end_date = context.get("data_interval_end", datetime.utcnow())
            start_date = end_date - timedelta(days=7)
            mock_registry.materialize(start_date, end_date)

        materialize_features_task(data_interval_end=datetime(2024, 6, 1))
        mock_registry.materialize.assert_called_once()

    def test_validate_data_task_returns_result(self):
        """Simulate an Airflow task that validates data quality."""
        from monitoring.data_quality import DataQualityMonitor, ValidationResult

        monitor = MagicMock(spec=DataQualityMonitor)
        monitor.validate_transactions.return_value = ValidationResult(
            suite_name="transactions_suite",
            success=True,
            evaluated_expectations=10,
            successful_expectations=10,
            failed_expectations=0,
            success_percent=100.0,
        )

        def validate_data_task(df: pd.DataFrame, **context):
            result = monitor.validate_transactions(df)
            if not result.success:
                raise ValueError(f"Data quality failure: {result.suite_name}")
            return result

        df = pd.DataFrame({"transaction_id": ["t1"]})
        result = validate_data_task(df)
        assert result.success is True

    def test_validate_data_task_raises_on_failure(self):
        """Task should raise when data quality fails."""
        from monitoring.data_quality import DataQualityMonitor, ValidationResult

        monitor = MagicMock(spec=DataQualityMonitor)
        monitor.validate_transactions.return_value = ValidationResult(
            suite_name="transactions_suite",
            success=False,
            evaluated_expectations=10,
            successful_expectations=7,
            failed_expectations=3,
            success_percent=70.0,
        )

        def validate_data_task(df: pd.DataFrame):
            result = monitor.validate_transactions(df)
            if not result.success:
                raise ValueError(f"Data quality failure: {result.suite_name}")
            return result

        with pytest.raises(ValueError, match="Data quality failure"):
            validate_data_task(pd.DataFrame())

    def test_drift_check_task_triggers_alert_on_retrain(self):
        """Task should send alert when drift detected."""
        from monitoring.drift_detector import DriftDetector
        from monitoring.alert_manager import AlertManager

        mock_detector = MagicMock(spec=DriftDetector)
        mock_alert_mgr = MagicMock(spec=AlertManager)
        mock_report = MagicMock()
        mock_report.recommend_retrain = True
        mock_report.model_name = "fraud_v3"
        mock_detector.detect.return_value = mock_report

        def drift_check_task(ref_df, cur_df):
            report = mock_detector.detect(ref_df, cur_df)
            if report.recommend_retrain:
                mock_alert_mgr.alert_drift(report, report.model_name)
            return report

        ref = pd.DataFrame({"feat": [1.0, 2.0]})
        cur = pd.DataFrame({"feat": [5.0, 6.0]})
        drift_check_task(ref, cur)
        mock_alert_mgr.alert_drift.assert_called_once()

    def test_drift_check_task_no_alert_when_no_drift(self):
        """Task should not alert when drift is below threshold."""
        from monitoring.drift_detector import DriftDetector
        from monitoring.alert_manager import AlertManager

        mock_detector = MagicMock(spec=DriftDetector)
        mock_alert_mgr = MagicMock(spec=AlertManager)
        mock_report = MagicMock()
        mock_report.recommend_retrain = False
        mock_detector.detect.return_value = mock_report

        def drift_check_task(ref_df, cur_df):
            report = mock_detector.detect(ref_df, cur_df)
            if report.recommend_retrain:
                mock_alert_mgr.alert_drift(report, report.model_name)
            return report

        drift_check_task(pd.DataFrame(), pd.DataFrame())
        mock_alert_mgr.alert_drift.assert_not_called()


class TestPipelineOrchestration:
    """Integration-style tests of pipeline component interaction."""

    def test_full_feature_pipeline_sequence(self):
        """Validate registry → dq → drift → alert task chain."""
        from feature_store.registry import FeatureRegistry
        from monitoring.data_quality import DataQualityMonitor, ValidationResult
        from monitoring.drift_detector import DriftDetector
        from monitoring.alert_manager import AlertManager

        registry = MagicMock(spec=FeatureRegistry)
        monitor = MagicMock(spec=DataQualityMonitor)
        detector = MagicMock(spec=DriftDetector)
        alerter = MagicMock(spec=AlertManager)

        # Configure mocks
        monitor.validate_transactions.return_value = ValidationResult(
            suite_name="s", success=True, evaluated_expectations=5,
            successful_expectations=5, failed_expectations=0, success_percent=100.0,
        )
        mock_report = MagicMock(recommend_retrain=False)
        detector.detect.return_value = mock_report

        # Simulate pipeline
        def run_pipeline(raw_df, ref_df, cur_df):
            registry.materialize(datetime(2024, 1, 1), datetime(2024, 1, 31))
            dq_result = monitor.validate_transactions(raw_df)
            assert dq_result.success
            report = detector.detect(ref_df, cur_df)
            if report.recommend_retrain:
                alerter.alert_drift(report, "fraud_v3")

        run_pipeline(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

        registry.materialize.assert_called_once()
        monitor.validate_transactions.assert_called_once()
        detector.detect.assert_called_once()
        alerter.alert_drift.assert_not_called()

    def test_pipeline_aborts_on_dq_failure(self):
        """If DQ fails, downstream tasks should not run."""
        from monitoring.data_quality import DataQualityMonitor, ValidationResult
        from monitoring.drift_detector import DriftDetector

        monitor = MagicMock(spec=DataQualityMonitor)
        detector = MagicMock(spec=DriftDetector)

        monitor.validate_transactions.return_value = ValidationResult(
            suite_name="s", success=False, evaluated_expectations=5,
            successful_expectations=2, failed_expectations=3, success_percent=40.0,
        )

        def run_pipeline(raw_df, ref_df, cur_df):
            result = monitor.validate_transactions(raw_df)
            if not result.success:
                raise RuntimeError("Aborting pipeline: DQ check failed")
            detector.detect(ref_df, cur_df)

        with pytest.raises(RuntimeError, match="DQ check failed"):
            run_pipeline(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

        detector.detect.assert_not_called()


class TestPipelinesPackage:
    def test_pipelines_init_importable(self):
        import pipelines
        assert pipelines.__version__ == "1.0.0"

    def test_pipelines_tasks_init_importable(self):
        import pipelines.tasks
        assert hasattr(pipelines.tasks, "__file__")


class TestAirflowDAGSimulation:
    """Verify that a minimal DAG-style definition works with our stubs."""

    def test_dag_creation(self, airflow_stubs):
        DAG, PythonOperator = airflow_stubs
        dag = DAG(
            "feature_store_pipeline",
            schedule_interval="@daily",
            start_date=datetime(2024, 1, 1),
        )
        assert dag.dag_id == "feature_store_pipeline"

    def test_python_operator_stores_callable(self, airflow_stubs):
        _, PythonOperator = airflow_stubs

        def my_task():
            return "done"

        op = PythonOperator(task_id="my_task", python_callable=my_task)
        assert op.task_id == "my_task"
        assert op.python_callable() == "done"

    def test_task_rshift_chaining(self, airflow_stubs):
        _, PythonOperator = airflow_stubs

        op1 = PythonOperator(task_id="t1", python_callable=lambda: None)
        op2 = PythonOperator(task_id="t2", python_callable=lambda: None)
        result = op1 >> op2
        assert result is op2

    def test_tasks_callables_are_invocable(self, airflow_stubs):
        _, PythonOperator = airflow_stubs
        called = []

        def extract():
            called.append("extract")

        def transform():
            called.append("transform")

        t1 = PythonOperator(task_id="extract", python_callable=extract)
        t2 = PythonOperator(task_id="transform", python_callable=transform)

        t1.python_callable()
        t2.python_callable()
        assert called == ["extract", "transform"]


class TestPipelineRetryLogic:
    """Tests for retry and error handling patterns in pipeline tasks."""

    def test_task_retries_on_transient_error(self):
        """Simulate a task that retries on transient failures."""
        call_count = [0]

        def flaky_task():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError("Transient error")
            return "success"

        def with_retry(fn, max_retries=3):
            for attempt in range(max_retries):
                try:
                    return fn()
                except ConnectionError:
                    if attempt == max_retries - 1:
                        raise
            return None

        result = with_retry(flaky_task, max_retries=3)
        assert result == "success"
        assert call_count[0] == 3

    def test_task_fails_after_max_retries(self):
        def always_fails():
            raise ConnectionError("Always fails")

        def with_retry(fn, max_retries=2):
            for attempt in range(max_retries):
                try:
                    return fn()
                except ConnectionError:
                    if attempt == max_retries - 1:
                        raise
            return None

        with pytest.raises(ConnectionError):
            with_retry(always_fails, max_retries=2)

