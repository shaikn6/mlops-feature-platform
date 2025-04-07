"""
PySpark Feature Computation Job
Reads raw transaction events from S3/Delta Lake, computes rolling window aggregations,
and writes output as a Delta feature table.

Usage:
    spark-submit feature_computation.py \
        --input s3://mlops-raw-events/events/ \
        --output s3://mlops-feature-store/features/user_transaction_features/ \
        --execution-date 2026-06-17
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta

from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, LongType, TimestampType, BooleanType,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

# ── Schema definition for raw events ────────────────────────────────────────

RAW_EVENT_SCHEMA = StructType([
    StructField("event_id",       StringType(),    False),
    StructField("user_id",        StringType(),    False),
    StructField("event_timestamp", TimestampType(), False),
    StructField("transaction_amount", DoubleType(), True),
    StructField("transaction_type",   StringType(), True),
    StructField("merchant_category",  StringType(), True),
    StructField("country_code",       StringType(), True),
    StructField("is_fraud",           BooleanType(), True),
])

# ── Window specs ─────────────────────────────────────────────────────────────

def _days_window(partition_col: str, days: int) -> Window.WindowSpec:
    """Rolling window of `days` days ordered by event_timestamp."""
    return (
        Window
        .partitionBy(partition_col)
        .orderBy(F.col("event_timestamp").cast("long"))
        .rangeBetween(-days * 86_400, 0)
    )


# ── Core computation ──────────────────────────────────────────────────────────

def compute_rolling_features(df: DataFrame) -> DataFrame:
    """
    Compute 7d / 30d / 90d rolling aggregations per user_id.
    Returns one row per (user_id, event_timestamp) with feature columns appended.
    """
    w7  = _days_window("user_id", 7)
    w30 = _days_window("user_id", 30)
    w90 = _days_window("user_id", 90)

    return df.select(
        "user_id",
        "event_timestamp",
        # 7-day features
        F.count("event_id").over(w7).alias("tx_count_7d"),
        F.sum("transaction_amount").over(w7).alias("tx_amount_sum_7d"),
        F.mean("transaction_amount").over(w7).alias("tx_amount_mean_7d"),
        F.max("transaction_amount").over(w7).alias("tx_amount_max_7d"),
        # 30-day features
        F.count("event_id").over(w30).alias("tx_count_30d"),
        F.sum("transaction_amount").over(w30).alias("tx_amount_sum_30d"),
        F.mean("transaction_amount").over(w30).alias("tx_amount_mean_30d"),
        F.stddev("transaction_amount").over(w30).alias("tx_amount_stddev_30d"),
        F.sum(F.when(F.col("is_fraud"), 1).otherwise(0)).over(w30).alias("fraud_tx_count_30d"),
        F.countDistinct("merchant_category").over(w30).alias("distinct_merchant_categories_30d"),
        # 90-day features
        F.count("event_id").over(w90).alias("tx_count_90d"),
        F.sum("transaction_amount").over(w90).alias("tx_amount_sum_90d"),
        F.mean("transaction_amount").over(w90).alias("tx_amount_mean_90d"),
        F.stddev("transaction_amount").over(w90).alias("tx_amount_stddev_90d"),
        # Derived / ratio features
        (F.sum("transaction_amount").over(w7) /
         F.nullif(F.sum("transaction_amount").over(w30), 0)).alias("tx_amount_7d_30d_ratio"),
        (F.count("event_id").over(w7) /
         F.nullif(F.count("event_id").over(w30), 0)).alias("tx_count_7d_30d_ratio"),
        # High-value flag: 90d mean above $500
        (F.mean("transaction_amount").over(w90) > 500.0).cast("boolean").alias("is_high_value_customer"),
    )


def deduplicate_and_take_latest(df: DataFrame) -> DataFrame:
    """Keep only the most-recent row per user_id for online store materialisation."""
    w = Window.partitionBy("user_id").orderBy(F.col("event_timestamp").desc())
    return (
        df.withColumn("_rn", F.row_number().over(w))
          .filter(F.col("_rn") == 1)
          .drop("_rn")
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def build_spark_session(app_name: str = "feature_computation") -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.shuffle.partitions", "400")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                "com.amazonaws.auth.WebIdentityTokenCredentialsProvider")
        .getOrCreate()
    )


def main(args: argparse.Namespace) -> None:
    spark = build_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    execution_date = datetime.strptime(args.execution_date, "%Y-%m-%d")
    lookback_start = execution_date - timedelta(days=95)  # +5-day buffer beyond max 90d window

    log.info("Reading raw events from %s (since %s)", args.input, lookback_start.date())

    raw_df = (
        spark.read
        .schema(RAW_EVENT_SCHEMA)
        .parquet(args.input)          # or .format("delta").load(args.input) for Delta tables
        .filter(
            (F.col("event_timestamp") >= F.lit(lookback_start)) &
            (F.col("event_timestamp") <= F.lit(execution_date))
        )
        .filter(F.col("user_id").isNotNull())
        .filter(F.col("transaction_amount") > 0)
    )

    log.info("Computing rolling features ...")
    features_df = compute_rolling_features(raw_df)
    latest_df   = deduplicate_and_take_latest(features_df)

    partition_date = execution_date.strftime("%Y-%m-%d")
    output_path = f"{args.output.rstrip('/')}/date={partition_date}"

    log.info("Writing feature table to %s", output_path)
    (
        latest_df
        .repartition(50, "user_id")
        .write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .option("replaceWhere", f"date='{partition_date}'")
        .save(output_path)
    )

    count = latest_df.count()
    log.info("Feature table written: %d user rows for date=%s", count, partition_date)
    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature computation PySpark job")
    parser.add_argument("--input",          required=True, help="S3/Delta path to raw events")
    parser.add_argument("--output",         required=True, help="S3/Delta path for feature output")
    parser.add_argument("--execution-date", required=True, help="Execution date YYYY-MM-DD")
    main(parser.parse_args())
