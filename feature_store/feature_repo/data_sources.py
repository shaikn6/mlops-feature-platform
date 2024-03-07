"""
Data source configurations for Feast feature store.
Supports file-based (parquet) sources for local dev and BigQuery/Redshift for prod.
"""

from pathlib import Path

from feast import FileSource
from feast.data_format import ParquetFormat

# ---------------------------------------------------------------------------
# Base paths (overridden via environment in production)
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"


# ---------------------------------------------------------------------------
# Transaction data source
# ---------------------------------------------------------------------------
transactions_source = FileSource(
    name="transactions_source",
    path=str(DATA_DIR / "transactions.parquet"),
    file_format=ParquetFormat(),
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
    description=(
        "Raw customer transaction events — used to compute spend aggregates, "
        "transaction counts, and rolling fraud rates."
    ),
    tags={"domain": "finance", "team": "data-engineering"},
)

# ---------------------------------------------------------------------------
# Credit bureau data source
# ---------------------------------------------------------------------------
credit_bureau_source = FileSource(
    name="credit_bureau_source",
    path=str(DATA_DIR / "credit_bureau.parquet"),
    file_format=ParquetFormat(),
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
    description=(
        "Monthly credit bureau snapshot — credit utilization, payment history, "
        "open trade lines, and debt-to-income ratios."
    ),
    tags={"domain": "finance", "team": "risk"},
)

# ---------------------------------------------------------------------------
# Customer profile data source
# ---------------------------------------------------------------------------
customer_profile_source = FileSource(
    name="customer_profile_source",
    path=str(DATA_DIR / "customer_profiles.parquet"),
    file_format=ParquetFormat(),
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
    description="Static customer demographics and account-level attributes.",
    tags={"domain": "finance", "team": "data-engineering"},
)

# ---------------------------------------------------------------------------
# Fraud labels source (point-in-time safe ground truth)
# ---------------------------------------------------------------------------
fraud_labels_source = FileSource(
    name="fraud_labels_source",
    path=str(DATA_DIR / "fraud_labels.parquet"),
    file_format=ParquetFormat(),
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
    description=(
        "Confirmed fraud labels resolved 90 days post-transaction. "
        "Used only for offline training — never served online."
    ),
    tags={"domain": "finance", "team": "fraud", "sensitivity": "high"},
)
