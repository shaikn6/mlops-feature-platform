"""
Airflow task functions for the feature pipeline.
Each function is designed to be called by PythonOperator or BranchPythonOperator.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from datetime import datetime, timedelta
from typing import Any

log = logging.getLogger(__name__)

# ── compute_features ─────────────────────────────────────────────────────────

def compute_features(
    s3_input_path: str,
    feature_table: str,
    execution_date: str,
    spark_master: str = "k8s://https://kubernetes.default.svc",
    **context: Any,
) -> dict:
    """
    Submit the PySpark feature_computation job via spark-submit (or SparkKubernetesOperator).
    Returns job metadata pushed to XCom.
    """
    spark_job_path = os.environ.get(
        "SPARK_JOB_PATH",
        "s3a://mlops-spark-jobs/feature_computation.py",
    )
    output_path = f"s3a://mlops-feature-store/features/{feature_table}/"

    cmd = [
        "spark-submit",
        "--master", spark_master,
        "--deploy-mode", "cluster",
        "--name", f"feature-computation-{execution_date}",
        "--conf", "spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension",
        "--conf", "spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog",
        "--conf", f"spark.kubernetes.namespace={os.environ.get('SPARK_NAMESPACE', 'spark')}",
        "--conf", f"spark.kubernetes.container.image={os.environ.get('SPARK_IMAGE', 'ghcr.io/shaikn6/mlops-feature-platform/spark:3.5-delta')}",
        "--conf", "spark.executor.instances=5",
        "--conf", "spark.executor.memory=8g",
        "--conf", "spark.driver.memory=4g",
        spark_job_path,
        "--input", s3_input_path,
        "--output", output_path,
        "--execution-date", execution_date,
    ]

    log.info("Submitting Spark job: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)

    if result.returncode != 0:
        log.error("Spark job FAILED:\nSTDOUT: %s\nSTDERR: %s", result.stdout[-2000:], result.stderr[-2000:])
        raise RuntimeError(f"feature_computation Spark job failed with code {result.returncode}")

    log.info("Spark job completed successfully")
    output = {
        "feature_table": feature_table,
        "output_path": output_path,
        "execution_date": execution_date,
        "exit_code": result.returncode,
    }
    context["task_instance"].xcom_push(key="feature_output", value=output)
    return output


# ── validate_features ────────────────────────────────────────────────────────

def validate_features(
    feature_table: str,
    execution_date: str,
    max_null_rate: float = 0.02,
    report_s3_path: str = "s3://mlops-validation-reports/",
    **context: Any,
) -> dict:
    """
    Run the PySpark data_validation job and parse its JSON report.
    Raises ValueError if validation fails.
    """
    feature_path = f"s3a://mlops-feature-store/features/{feature_table}/"
    reference_date = (
        datetime.strptime(execution_date, "%Y-%m-%d") - timedelta(days=30)
    ).strftime("%Y-%m-%d")

    cmd = [
        "spark-submit",
        "--master", os.environ.get("SPARK_MASTER", "k8s://https://kubernetes.default.svc"),
        "--deploy-mode", "cluster",
        "--name", f"data-validation-{execution_date}",
        "--conf", "spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension",
        "--conf", "spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog",
        "--conf", "spark.executor.instances=3",
        "--conf", "spark.executor.memory=4g",
        os.environ.get("VALIDATION_JOB_PATH", "s3a://mlops-spark-jobs/data_validation.py"),
        "--feature-path",    feature_path,
        "--reference-date",  reference_date,
        "--current-date",    execution_date,
        "--report-path",     f"{report_s3_path.rstrip('/')}/{execution_date}/",
        "--max-null-rate",   str(max_null_rate),
        "--ks-threshold",    "0.05",
    ]

    log.info("Submitting validation job: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

    if result.returncode != 0:
        raise RuntimeError(
            f"data_validation job failed (code {result.returncode}).\n"
            f"STDERR: {result.stderr[-2000:]}"
        )

    report_location = f"{report_s3_path.rstrip('/')}/{execution_date}/validation_{execution_date}.json"
    log.info("Validation passed. Report at: %s", report_location)
    context["task_instance"].xcom_push(key="validation_report_path", value=report_location)
    return {"status": "pass", "report_path": report_location}


# ── materialize_to_feast ─────────────────────────────────────────────────────

def materialize_to_feast(
    feature_repo_path: str,
    feature_views: list[str],
    start_date: str,
    end_date: str,
    **context: Any,
) -> dict:
    """
    Materialise feature views from the offline store (S3) to the online store (Redis).
    Calls `feast materialize` for each feature view in the list.
    """
    import subprocess

    start_iso = f"{start_date}T00:00:00"
    end_iso   = f"{end_date}T23:59:59"

    results = {}
    for fv in feature_views:
        cmd = [
            "feast",
            "--feature-store-yaml", os.path.join(feature_repo_path, "feature_store.yaml"),
            "materialize",
            start_iso,
            end_iso,
        ]
        log.info("Materialising feature view %s: %s", fv, " ".join(cmd))
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800,
                           env={**os.environ, "FEAST_FEATURE_VIEW": fv})

        if r.returncode != 0:
            log.error("Feast materialize failed for %s:\n%s", fv, r.stderr[-1000:])
            raise RuntimeError(f"feast materialize failed for feature view: {fv}")

        log.info("Materialised %s successfully", fv)
        results[fv] = "success"

    context["task_instance"].xcom_push(key="materialised_views", value=results)
    return {"feature_views": results, "start_date": start_date, "end_date": end_date}


# ── check_drift ───────────────────────────────────────────────────────────────

def check_drift(
    feature_table: str,
    execution_date: str,
    drift_threshold: float = 0.05,
    reference_window_days: int = 30,
    **context: Any,
) -> str:
    """
    BranchPythonOperator callable.
    Reads the latest validation report and checks if distribution drift exceeded the threshold.
    Returns 'trigger_retraining' or 'no_retraining'.
    """
    import boto3

    ti = context["task_instance"]
    report_path: str = ti.xcom_pull(task_ids="validate_features", key="validation_report_path")

    if not report_path:
        log.warning("No validation report found; skipping drift check, defaulting to no_retraining.")
        return "no_retraining"

    # Download report from S3
    s3 = boto3.client("s3")
    # Parse s3://bucket/key
    path_parts = report_path.replace("s3://", "").split("/", 1)
    bucket, key = path_parts[0], path_parts[1]

    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        report = json.loads(obj["Body"].read().decode("utf-8"))
    except Exception as exc:
        log.error("Failed to read validation report from %s: %s", report_path, exc)
        return "no_retraining"

    drift_check = report.get("checks", {}).get("drift", {})
    drifted_cols: list[str] = drift_check.get("drifted_columns", [])
    overall_drift_status: str = drift_check.get("status", "pass")

    log.info(
        "Drift check: status=%s drifted_columns=%s threshold=%.4f",
        overall_drift_status, drifted_cols, float(drift_threshold),
    )

    if overall_drift_status == "fail" or len(drifted_cols) > 0:
        log.warning(
            "Drift detected in %d column(s): %s. Triggering retraining.",
            len(drifted_cols), drifted_cols,
        )
        return "trigger_retraining"

    log.info("No significant drift detected. Skipping retraining.")
    return "no_retraining"
