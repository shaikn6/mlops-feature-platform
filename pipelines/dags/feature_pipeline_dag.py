"""
Feature Pipeline DAG
Ingests raw events -> computes features -> materialises to Feast -> optionally triggers retraining.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.empty import EmptyOperator
from airflow.models import Variable
from airflow.utils.dates import days_ago

from pipelines.tasks.feature_tasks import (
    compute_features,
    validate_features,
    materialize_to_feast,
    check_drift,
)

log = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner": "mlops",
    "depends_on_past": False,
    "email": ["mlops-alerts@example.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=30),
    "execution_timeout": timedelta(hours=2),
    "sla": timedelta(hours=3),
}

with DAG(
    dag_id="feature_pipeline",
    default_args=DEFAULT_ARGS,
    description="End-to-end feature engineering pipeline: ingest -> compute -> materialise -> drift check",
    schedule_interval="0 */6 * * *",   # every 6 hours
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=["feature-platform", "feast", "spark"],
    params={
        "s3_input_path": "s3://mlops-raw-events/events/",
        "feature_table": "user_transaction_features",
        "drift_threshold": 0.05,
        "window_days": 30,
    },
) as dag:

    # ── Ingest raw data via Spark ────────────────────────────────────────────
    ingest = PythonOperator(
        task_id="ingest_raw_data",
        python_callable=compute_features,
        op_kwargs={
            "s3_input_path": "{{ params.s3_input_path }}",
            "feature_table": "{{ params.feature_table }}",
            "execution_date": "{{ ds }}",
            "spark_master": Variable.get("spark_master", default_var="k8s://https://kubernetes.default.svc"),
        },
        doc_md="Submit PySpark job to read raw events from S3/Delta and compute rolling aggregations.",
    )

    # ── Validate schema + null rates + distributions ─────────────────────────
    validate = PythonOperator(
        task_id="validate_features",
        python_callable=validate_features,
        op_kwargs={
            "feature_table": "{{ params.feature_table }}",
            "execution_date": "{{ ds }}",
            "max_null_rate": 0.02,
            "report_s3_path": "s3://mlops-validation-reports/{{ ds }}/",
        },
        doc_md="Schema validation, null checks, distribution report written to S3.",
    )

    # ── Materialise validated features into Feast online + offline store ─────
    materialise = PythonOperator(
        task_id="materialize_to_feast",
        python_callable=materialize_to_feast,
        op_kwargs={
            "feature_repo_path": Variable.get("feast_repo_path", default_var="/opt/airflow/feature_store"),
            "feature_views": ["user_transaction_features", "user_profile_features"],
            "start_date": "{{ macros.ds_add(ds, -1) }}",
            "end_date": "{{ ds }}",
        },
        doc_md="Run `feast materialize` for online store (Redis) and offline store (S3/Parquet).",
    )

    # ── Drift detection: returns branch task id ───────────────────────────────
    drift_check = BranchPythonOperator(
        task_id="check_drift",
        python_callable=check_drift,
        op_kwargs={
            "feature_table": "{{ params.feature_table }}",
            "execution_date": "{{ ds }}",
            "drift_threshold": "{{ params.drift_threshold }}",
            "reference_window_days": "{{ params.window_days }}",
        },
        doc_md="KS-test drift detection. Branches to trigger_retraining or no_retraining.",
    )

    # ── Branch: trigger model retraining DAG ─────────────────────────────────
    trigger_retraining = TriggerDagRunOperator(
        task_id="trigger_retraining",
        trigger_dag_id="model_retraining",
        conf={
            "triggered_by": "feature_pipeline",
            "execution_date": "{{ ds }}",
            "feature_table": "{{ params.feature_table }}",
        },
        wait_for_completion=False,
        doc_md="Trigger model_retraining DAG when drift is detected.",
    )

    # ── Branch: no retraining needed ─────────────────────────────────────────
    no_retraining = EmptyOperator(task_id="no_retraining")

    # ── Pipeline complete ─────────────────────────────────────────────────────
    done = EmptyOperator(
        task_id="pipeline_complete",
        trigger_rule="none_failed_min_one_success",
    )

    # ── Task dependencies ─────────────────────────────────────────────────────
    ingest >> validate >> materialise >> drift_check
    drift_check >> [trigger_retraining, no_retraining]
    [trigger_retraining, no_retraining] >> done
