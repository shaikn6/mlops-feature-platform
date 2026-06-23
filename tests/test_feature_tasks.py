"""Unit tests for Airflow feature-pipeline task callables.

All external side effects (spark-submit / feast / boto3 / S3) are mocked, so
these run with no cluster, no AWS, and no real subprocess.
"""

from __future__ import annotations

import json
import sys
import types
from unittest import mock

import pytest

from pipelines.tasks import feature_tasks


class _FakeTI:
    """Minimal Airflow TaskInstance stand-in for XCom push/pull."""

    def __init__(self, pulls: dict | None = None):
        self.pushed: dict = {}
        self._pulls = pulls or {}

    def xcom_push(self, key, value):
        self.pushed[key] = value

    def xcom_pull(self, task_ids=None, key=None):
        return self._pulls.get(key)


def _completed(returncode=0, stdout="", stderr=""):
    return types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


# ── compute_features ─────────────────────────────────────────────────────────


def test_compute_features_success_pushes_xcom():
    ti = _FakeTI()
    with mock.patch.object(
        feature_tasks.subprocess, "run", return_value=_completed(0)
    ) as run:
        out = feature_tasks.compute_features(
            s3_input_path="s3a://in/data/",
            feature_table="customer_features",
            execution_date="2026-01-15",
            task_instance=ti,
        )
    run.assert_called_once()
    assert out["feature_table"] == "customer_features"
    assert out["output_path"].endswith("customer_features/")
    assert out["exit_code"] == 0
    assert ti.pushed["feature_output"] == out


def test_compute_features_failure_raises():
    ti = _FakeTI()
    with mock.patch.object(
        feature_tasks.subprocess, "run", return_value=_completed(1, stderr="boom")
    ):
        with pytest.raises(RuntimeError, match="feature_computation"):
            feature_tasks.compute_features(
                s3_input_path="s3a://in/",
                feature_table="t",
                execution_date="2026-01-15",
                task_instance=ti,
            )
    assert "feature_output" not in ti.pushed


def test_compute_features_honours_env_overrides(monkeypatch):
    monkeypatch.setenv("SPARK_JOB_PATH", "s3a://custom/job.py")
    monkeypatch.setenv("SPARK_NAMESPACE", "ml")
    ti = _FakeTI()
    with mock.patch.object(
        feature_tasks.subprocess, "run", return_value=_completed(0)
    ) as run:
        feature_tasks.compute_features(
            s3_input_path="s3a://in/",
            feature_table="t",
            execution_date="2026-01-15",
            task_instance=ti,
        )
    cmd = run.call_args.args[0]
    assert "s3a://custom/job.py" in cmd
    assert "spark.kubernetes.namespace=ml" in " ".join(cmd)


# ── validate_features ────────────────────────────────────────────────────────


def test_validate_features_success():
    ti = _FakeTI()
    with mock.patch.object(feature_tasks.subprocess, "run", return_value=_completed(0)):
        out = feature_tasks.validate_features(
            feature_table="t",
            execution_date="2026-02-10",
            task_instance=ti,
        )
    assert out["status"] == "pass"
    assert out["report_path"].endswith("validation_2026-02-10.json")
    assert ti.pushed["validation_report_path"] == out["report_path"]


def test_validate_features_failure_raises():
    ti = _FakeTI()
    with mock.patch.object(
        feature_tasks.subprocess, "run", return_value=_completed(2, stderr="bad rows")
    ):
        with pytest.raises(RuntimeError, match="data_validation"):
            feature_tasks.validate_features(
                feature_table="t",
                execution_date="2026-02-10",
                task_instance=ti,
            )


def test_validate_features_reference_date_is_30_days_back():
    ti = _FakeTI()
    with mock.patch.object(
        feature_tasks.subprocess, "run", return_value=_completed(0)
    ) as run:
        feature_tasks.validate_features(
            feature_table="t",
            execution_date="2026-02-10",
            task_instance=ti,
        )
    cmd = run.call_args.args[0]
    assert "2026-01-11" in cmd  # 2026-02-10 minus 30 days


# ── materialize_to_feast ─────────────────────────────────────────────────────


def test_materialize_to_feast_all_views_success():
    ti = _FakeTI()
    with mock.patch("subprocess.run", return_value=_completed(0)) as run:
        out = feature_tasks.materialize_to_feast(
            feature_repo_path="/repo",
            feature_views=["fv_a", "fv_b"],
            start_date="2026-01-01",
            end_date="2026-01-31",
            task_instance=ti,
        )
    assert run.call_count == 2
    assert out["feature_views"] == {"fv_a": "success", "fv_b": "success"}
    assert ti.pushed["materialised_views"] == {"fv_a": "success", "fv_b": "success"}


def test_materialize_to_feast_failure_raises():
    ti = _FakeTI()
    with mock.patch("subprocess.run", return_value=_completed(1, stderr="redis down")):
        with pytest.raises(RuntimeError, match="fv_a"):
            feature_tasks.materialize_to_feast(
                feature_repo_path="/repo",
                feature_views=["fv_a"],
                start_date="2026-01-01",
                end_date="2026-01-31",
                task_instance=ti,
            )


# ── check_drift ──────────────────────────────────────────────────────────────


def _install_fake_boto3(report: dict | None, raise_on_get: bool = False):
    """Inject a fake boto3 module so check_drift's local import resolves."""
    fake = types.ModuleType("boto3")

    class _Body:
        def read(self_inner):
            return json.dumps(report).encode("utf-8")

    class _Client:
        def get_object(self_inner, Bucket, Key):
            if raise_on_get:
                raise RuntimeError("s3 unavailable")
            return {"Body": _Body()}

    fake.client = lambda name: _Client()
    return mock.patch.dict(sys.modules, {"boto3": fake})


def test_check_drift_no_report_path():
    ti = _FakeTI(pulls={"validation_report_path": None})
    with _install_fake_boto3(None):
        assert (
            feature_tasks.check_drift("t", "2026-03-01", task_instance=ti)
            == "no_retraining"
        )


def test_check_drift_triggers_on_drifted_columns():
    report = {
        "checks": {"drift": {"status": "fail", "drifted_columns": ["age", "income"]}}
    }
    ti = _FakeTI(pulls={"validation_report_path": "s3://b/k.json"})
    with _install_fake_boto3(report):
        assert (
            feature_tasks.check_drift("t", "2026-03-01", task_instance=ti)
            == "trigger_retraining"
        )


def test_check_drift_no_drift_returns_no_retraining():
    report = {"checks": {"drift": {"status": "pass", "drifted_columns": []}}}
    ti = _FakeTI(pulls={"validation_report_path": "s3://b/k.json"})
    with _install_fake_boto3(report):
        assert (
            feature_tasks.check_drift("t", "2026-03-01", task_instance=ti)
            == "no_retraining"
        )


def test_check_drift_s3_error_defaults_to_no_retraining():
    ti = _FakeTI(pulls={"validation_report_path": "s3://b/k.json"})
    with _install_fake_boto3(None, raise_on_get=True):
        assert (
            feature_tasks.check_drift("t", "2026-03-01", task_instance=ti)
            == "no_retraining"
        )
