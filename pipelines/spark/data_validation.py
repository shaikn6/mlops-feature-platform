"""
PySpark Data Validation Job
Performs schema validation, null-rate checks, range checks, and KS-test distribution
drift detection against a reference window. Writes a JSON validation report to S3.

Usage:
    spark-submit data_validation.py \
        --feature-path s3://mlops-feature-store/features/user_transaction_features/ \
        --reference-date 2026-05-17 \
        --current-date 2026-06-17 \
        --report-path s3://mlops-validation-reports/ \
        --max-null-rate 0.02 \
        --ks-threshold 0.05
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timedelta
from typing import Any

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

NUMERIC_FEATURE_COLS = [
    "tx_count_7d", "tx_amount_sum_7d", "tx_amount_mean_7d", "tx_amount_max_7d",
    "tx_count_30d", "tx_amount_sum_30d", "tx_amount_mean_30d", "tx_amount_stddev_30d",
    "fraud_tx_count_30d", "distinct_merchant_categories_30d",
    "tx_count_90d", "tx_amount_sum_90d", "tx_amount_mean_90d", "tx_amount_stddev_90d",
    "tx_amount_7d_30d_ratio", "tx_count_7d_30d_ratio",
]

REQUIRED_COLUMNS = NUMERIC_FEATURE_COLS + ["user_id", "event_timestamp", "is_high_value_customer"]

RANGE_CHECKS = {
    "tx_count_7d":  (0, 10_000),
    "tx_count_30d": (0, 50_000),
    "tx_amount_sum_30d": (0, 10_000_000),
    "fraud_tx_count_30d": (0, 1_000),
}


# ── Schema validation ────────────────────────────────────────────────────────

def validate_schema(df: DataFrame, required_cols: list[str]) -> dict[str, Any]:
    actual = set(df.columns)
    missing = [c for c in required_cols if c not in actual]
    extra   = [c for c in actual if c not in required_cols and c != "date"]
    return {
        "status": "pass" if not missing else "fail",
        "missing_columns": missing,
        "extra_columns": extra,
        "total_columns": len(actual),
    }


# ── Null rate checks ─────────────────────────────────────────────────────────

def check_null_rates(df: DataFrame, max_null_rate: float, cols: list[str]) -> dict[str, Any]:
    total = df.count()
    if total == 0:
        return {"status": "fail", "reason": "empty_dataframe"}

    exprs = [F.mean(F.col(c).isNull().cast("int")).alias(c) for c in cols]
    null_rates = df.select(*exprs).collect()[0].asDict()

    violations = {col: rate for col, rate in null_rates.items() if rate > max_null_rate}
    return {
        "status": "fail" if violations else "pass",
        "total_rows": total,
        "max_allowed_null_rate": max_null_rate,
        "violations": violations,
        "null_rates": null_rates,
    }


# ── Range checks ─────────────────────────────────────────────────────────────

def check_ranges(df: DataFrame, range_checks: dict[str, tuple[float, float]]) -> dict[str, Any]:
    violations: dict[str, Any] = {}
    for col, (low, high) in range_checks.items():
        if col not in df.columns:
            continue
        out_of_range = df.filter(
            (F.col(col) < low) | (F.col(col) > high)
        ).count()
        if out_of_range > 0:
            violations[col] = {"out_of_range_count": out_of_range, "bounds": [low, high]}

    return {"status": "fail" if violations else "pass", "violations": violations}


# ── KS-test distribution drift detection ────────────────────────────────────

def compute_ks_statistic(df: DataFrame, col: str, n_quantiles: int = 100) -> tuple[float, float]:
    """
    Approximate two-sample KS test using empirical CDFs built from quantiles.
    df must have a 'partition' column ('reference' | 'current').
    Returns (ks_statistic, approximate_p_value).
    """
    from pyspark.ml.feature import QuantileDiscretizer
    import math

    ref_vals  = [r[col] for r in df.filter(F.col("partition") == "reference").select(col).dropna().collect()]
    curr_vals = [r[col] for r in df.filter(F.col("partition") == "current").select(col).dropna().collect()]

    if not ref_vals or not curr_vals:
        return 0.0, 1.0

    ref_vals.sort()
    curr_vals.sort()

    # Build empirical CDF for each sample
    def ecdf(values: list[float], x: float) -> float:
        lo, hi = 0, len(values)
        while lo < hi:
            mid = (lo + hi) // 2
            if values[mid] <= x:
                lo = mid + 1
            else:
                hi = mid
        return lo / len(values)

    all_vals = sorted(set(ref_vals + curr_vals))
    ks_stat = max(abs(ecdf(ref_vals, v) - ecdf(curr_vals, v)) for v in all_vals)

    # Kolmogorov-Smirnov asymptotic p-value approximation
    n1, n2 = len(ref_vals), len(curr_vals)
    en = math.sqrt(n1 * n2 / (n1 + n2))
    z = (en + 0.12 + 0.11 / en) * ks_stat
    # Asymptotic formula Q(z) = 2 * sum_{k=1}^{inf} (-1)^{k+1} exp(-2k^2 z^2)
    p_val = 2 * sum(
        ((-1) ** (k + 1)) * math.exp(-2 * k * k * z * z)
        for k in range(1, 20)
    )
    p_val = max(0.0, min(1.0, p_val))
    return ks_stat, p_val


def check_distribution_drift(
    ref_df: DataFrame,
    curr_df: DataFrame,
    cols: list[str],
    ks_threshold: float,
    spark: SparkSession,
) -> dict[str, Any]:
    ref_tagged  = ref_df.select(*cols).withColumn("partition", F.lit("reference"))
    curr_tagged = curr_df.select(*cols).withColumn("partition", F.lit("current"))
    combined = ref_tagged.union(curr_tagged).persist()

    drift_results: dict[str, Any] = {}
    drifted_cols: list[str] = []

    for col in cols:
        ks_stat, p_val = compute_ks_statistic(combined, col)
        drifted = ks_stat > ks_threshold
        drift_results[col] = {
            "ks_statistic": round(ks_stat, 6),
            "p_value": round(p_val, 6),
            "drifted": drifted,
        }
        if drifted:
            drifted_cols.append(col)

    combined.unpersist()
    return {
        "status": "fail" if drifted_cols else "pass",
        "ks_threshold": ks_threshold,
        "drifted_columns": drifted_cols,
        "column_results": drift_results,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def build_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("data_validation")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                "com.amazonaws.auth.WebIdentityTokenCredentialsProvider")
        .getOrCreate()
    )


def main(args: argparse.Namespace) -> None:
    spark = build_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    ref_path  = f"{args.feature_path.rstrip('/')}/date={args.reference_date}"
    curr_path = f"{args.feature_path.rstrip('/')}/date={args.current_date}"

    log.info("Loading reference features from %s", ref_path)
    ref_df  = spark.read.format("delta").load(ref_path)

    log.info("Loading current features from %s", curr_path)
    curr_df = spark.read.format("delta").load(curr_path)

    # Run all checks
    schema_result   = validate_schema(curr_df, REQUIRED_COLUMNS)
    null_result     = check_null_rates(curr_df, args.max_null_rate, NUMERIC_FEATURE_COLS)
    range_result    = check_ranges(curr_df, RANGE_CHECKS)
    drift_result    = check_distribution_drift(
                          ref_df, curr_df, NUMERIC_FEATURE_COLS, args.ks_threshold, spark)

    overall_status = "pass" if all(
        r["status"] == "pass"
        for r in [schema_result, null_result, range_result, drift_result]
    ) else "fail"

    report = {
        "execution_date": args.current_date,
        "reference_date": args.reference_date,
        "overall_status": overall_status,
        "checks": {
            "schema":     schema_result,
            "null_rates": null_result,
            "ranges":     range_result,
            "drift":      drift_result,
        },
    }

    # Write JSON report to S3
    report_path = f"{args.report_path.rstrip('/')}/validation_{args.current_date}.json"
    report_json = json.dumps(report, indent=2)
    log.info("Validation report:
%s", report_json)

    # Persist report as a single-partition DataFrame to S3
    spark.createDataFrame([{"report": report_json}])          .coalesce(1)          .write.mode("overwrite")          .text(report_path)

    if overall_status == "fail":
        log.error("Validation FAILED for date=%s", args.current_date)
        sys.exit(1)

    log.info("Validation PASSED for date=%s", args.current_date)
    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature data validation PySpark job")
    parser.add_argument("--feature-path",    required=True)
    parser.add_argument("--reference-date",  required=True, help="YYYY-MM-DD baseline date")
    parser.add_argument("--current-date",    required=True, help="YYYY-MM-DD current date")
    parser.add_argument("--report-path",     required=True, help="S3 path for validation report")
    parser.add_argument("--max-null-rate",   type=float, default=0.02)
    parser.add_argument("--ks-threshold",    type=float, default=0.05)
    main(parser.parse_args())
