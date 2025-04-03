"""
Model Retraining DAG
Pulls features from Feast -> trains model -> evaluates -> registers in MLflow -> promotes if above threshold.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.models import Variable
from airflow.utils.dates import days_ago

log = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner": "mlops",
    "depends_on_past": False,
    "email": ["mlops-alerts@example.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
    "execution_timeout": timedelta(hours=4),
    "sla": timedelta(hours=6),
}

# ── Task callables ─────────────────────────────────────────────────────────

def pull_training_dataset(**context) -> dict:
    """Retrieve historical features from Feast offline store for training."""
    import pandas as pd
    from feast import FeatureStore
    from feast.infra.offline_stores.contrib.spark_offline_store.spark_source import SparkSource

    feast_repo = Variable.get("feast_repo_path", default_var="/opt/airflow/feature_store")
    store = FeatureStore(repo_path=feast_repo)

    entity_df = pd.DataFrame({
        "user_id": list(range(1, 100_001)),
        "event_timestamp": [pd.Timestamp(context["ds"]) - pd.Timedelta(hours=i % 24)
                            for i in range(100_000)],
    })

    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "user_transaction_features:tx_count_7d",
            "user_transaction_features:tx_amount_sum_30d",
            "user_transaction_features:tx_amount_mean_90d",
            "user_transaction_features:tx_amount_stddev_30d",
            "user_transaction_features:is_high_value_customer",
            "user_profile_features:account_age_days",
            "user_profile_features:kyc_verified",
        ],
    ).to_df()

    output_path = f"/tmp/training_{context['ds']}.parquet"
    training_df.to_parquet(output_path, index=False)
    context["task_instance"].xcom_push(key="training_data_path", value=output_path)
    log.info("Training dataset pulled: %d rows, %d cols", *training_df.shape)
    return {"rows": len(training_df), "path": output_path}


def train_model(**context) -> dict:
    """Train XGBoost model on the pulled feature dataset."""
    import json
    import mlflow
    import mlflow.xgboost
    import pandas as pd
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, average_precision_score

    ti = context["task_instance"]
    data_path = ti.xcom_pull(task_ids="pull_training_dataset", key="training_data_path")

    df = pd.read_parquet(data_path)
    target_col = "label"  # binary churn label injected upstream
    feature_cols = [c for c in df.columns if c not in {target_col, "user_id", "event_timestamp"}]

    X = df[feature_cols].fillna(0)
    y = df[target_col]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    mlflow.set_tracking_uri(Variable.get("mlflow_tracking_uri", default_var="http://mlflow:5000"))
    mlflow.set_experiment("churn_prediction")

    params = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "auc",
        "use_label_encoder": False,
        "tree_method": "hist",
        "early_stopping_rounds": 20,
    }

    with mlflow.start_run(run_name=f"churn-{context['ds']}") as run:
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        y_prob = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_prob)
        pr_auc = average_precision_score(y_val, y_prob)

        mlflow.log_params(params)
        mlflow.log_metrics({"val_auc": auc, "val_pr_auc": pr_auc})
        mlflow.log_dict({"feature_cols": feature_cols}, "feature_cols.json")
        mlflow.xgboost.log_model(model, artifact_path="model", registered_model_name="churn_prediction")

        run_id = run.info.run_id
        log.info("Training complete | run_id=%s | AUC=%.4f | PR-AUC=%.4f", run_id, auc, pr_auc)

    ti.xcom_push(key="run_id", value=run_id)
    ti.xcom_push(key="val_auc", value=auc)
    return {"run_id": run_id, "val_auc": auc, "val_pr_auc": pr_auc}


def evaluate_and_branch(**context) -> str:
    """Return next task id based on whether AUC passes promotion threshold."""
    ti = context["task_instance"]
    auc = float(ti.xcom_pull(task_ids="train_model", key="val_auc"))
    threshold = float(Variable.get("promotion_auc_threshold", default_var="0.82"))
    log.info("Evaluation: AUC=%.4f  threshold=%.4f", auc, threshold)
    return "promote_model" if auc >= threshold else "reject_model"


def promote_model(**context) -> dict:
    """Transition model version to Production in MLflow registry."""
    from mlflow import MlflowClient

    ti = context["task_instance"]
    run_id = ti.xcom_pull(task_ids="train_model", key="run_id")

    client = MlflowClient(tracking_uri=Variable.get("mlflow_tracking_uri", default_var="http://mlflow:5000"))

    # Find the version registered in this run
    versions = client.search_model_versions(f"run_id='{run_id}'")
    if not versions:
        raise ValueError(f"No model version found for run_id={run_id}")

    version = versions[0].version
    client.transition_model_version_stage(
        name="churn_prediction",
        version=version,
        stage="Production",
        archive_existing_versions=True,
    )
    client.set_model_version_tag(name="churn_prediction", version=version,
                                  key="promoted_by", value="model_retraining_dag")
    log.info("Model version %s promoted to Production", version)
    return {"model_name": "churn_prediction", "version": version, "stage": "Production"}


def reject_model(**context):
    """Log rejection reason and archive the run."""
    from mlflow import MlflowClient

    ti = context["task_instance"]
    run_id = ti.xcom_pull(task_ids="train_model", key="run_id")
    auc = ti.xcom_pull(task_ids="train_model", key="val_auc")
    threshold = Variable.get("promotion_auc_threshold", default_var="0.82")

    client = MlflowClient(tracking_uri=Variable.get("mlflow_tracking_uri", default_var="http://mlflow:5000"))
    client.set_tag(run_id, "promotion_status", "rejected")
    client.set_tag(run_id, "rejection_reason", f"AUC {auc:.4f} < threshold {threshold}")
    log.warning("Model rejected: run_id=%s AUC=%.4f < %.4f", run_id, float(auc), float(threshold))


# ── DAG definition ──────────────────────────────────────────────────────────

with DAG(
    dag_id="model_retraining",
    default_args=DEFAULT_ARGS,
    description="Pull Feast features -> train -> evaluate -> promote via MLflow",
    schedule_interval=None,   # triggered by feature_pipeline DAG
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=["model-training", "mlflow", "feast"],
    params={
        "feature_table": "user_transaction_features",
    },
) as dag:

    pull_data = PythonOperator(
        task_id="pull_training_dataset",
        python_callable=pull_training_dataset,
    )

    train = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    evaluate = BranchPythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_and_branch,
    )

    promote = PythonOperator(
        task_id="promote_model",
        python_callable=promote_model,
    )

    reject = PythonOperator(
        task_id="reject_model",
        python_callable=reject_model,
    )

    done = EmptyOperator(
        task_id="retraining_complete",
        trigger_rule="none_failed_min_one_success",
    )

    pull_data >> train >> evaluate >> [promote, reject] >> done
