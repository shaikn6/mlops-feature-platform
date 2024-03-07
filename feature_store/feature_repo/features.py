"""
Feast feature definitions for the finance ML platform.

Feature views:
  - customer_transaction_features: rolling spend/count/fraud aggregates
  - credit_features:               credit bureau snapshot features
  - customer_profile_features:     static customer demographics
  - fraud_label_features:          ground-truth fraud labels (offline only)

Usage:
    feast apply          # register all feature views
    feast materialize    # push offline → online store
"""

from datetime import timedelta

from feast import Entity, FeatureView, Field, ValueType
from feast.types import Bool, Float64, Int64, String

from .data_sources import (
    credit_bureau_source,
    customer_profile_source,
    fraud_labels_source,
    transactions_source,
)

# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------

customer = Entity(
    name="customer_id",
    value_type=ValueType.STRING,
    description="Unique customer identifier (UUID v4).",
    tags={"domain": "finance"},
)

transaction = Entity(
    name="transaction_id",
    value_type=ValueType.STRING,
    description="Unique transaction identifier.",
    tags={"domain": "finance"},
)

# ---------------------------------------------------------------------------
# Feature View 1 — Customer Transaction Features
# Rolling aggregates: 7d spend, 30d count, 90d fraud rate
# ---------------------------------------------------------------------------

customer_transaction_features = FeatureView(
    name="customer_transaction_features",
    entities=[customer],
    ttl=timedelta(days=7),
    schema=[
        Field(
            name="avg_spend_7d",
            dtype=Float64,
            description=(
                "Average transaction spend over the trailing 7 days (USD). "
                "Null if fewer than 1 transaction in the window."
            ),
            tags={"window": "7d", "aggregation": "mean"},
        ),
        Field(
            name="total_spend_7d",
            dtype=Float64,
            description="Total spend over trailing 7 days (USD).",
            tags={"window": "7d", "aggregation": "sum"},
        ),
        Field(
            name="transaction_count_7d",
            dtype=Int64,
            description="Number of transactions in the trailing 7 days.",
            tags={"window": "7d", "aggregation": "count"},
        ),
        Field(
            name="transaction_count_30d",
            dtype=Int64,
            description="Number of transactions in the trailing 30 days.",
            tags={"window": "30d", "aggregation": "count"},
        ),
        Field(
            name="avg_spend_30d",
            dtype=Float64,
            description="Average transaction spend over the trailing 30 days (USD).",
            tags={"window": "30d", "aggregation": "mean"},
        ),
        Field(
            name="std_spend_30d",
            dtype=Float64,
            description=(
                "Standard deviation of spend over trailing 30 days. "
                "High values signal spend volatility — a fraud signal."
            ),
            tags={"window": "30d", "aggregation": "std"},
        ),
        Field(
            name="fraud_rate_90d",
            dtype=Float64,
            description=(
                "Fraction of transactions confirmed fraudulent in the trailing 90 days. "
                "Range [0, 1]. Computed on resolved ground-truth labels."
            ),
            tags={"window": "90d", "aggregation": "mean", "sensitivity": "high"},
        ),
        Field(
            name="fraud_count_90d",
            dtype=Int64,
            description="Count of confirmed fraudulent transactions in trailing 90 days.",
            tags={"window": "90d", "aggregation": "count", "sensitivity": "high"},
        ),
        Field(
            name="unique_merchants_30d",
            dtype=Int64,
            description="Number of distinct merchants transacted with in trailing 30 days.",
            tags={"window": "30d", "aggregation": "nunique"},
        ),
        Field(
            name="max_single_spend_30d",
            dtype=Float64,
            description="Largest single transaction amount in trailing 30 days (USD).",
            tags={"window": "30d", "aggregation": "max"},
        ),
        Field(
            name="international_txn_ratio_30d",
            dtype=Float64,
            description=(
                "Fraction of transactions flagged as international in trailing 30 days. "
                "High ratios correlate with fraud on domestic-primary accounts."
            ),
            tags={"window": "30d", "aggregation": "mean"},
        ),
    ],
    online=True,
    source=transactions_source,
    tags={"domain": "finance", "team": "data-engineering", "model": "fraud,credit"},
)

# ---------------------------------------------------------------------------
# Feature View 2 — Credit Features
# Monthly credit bureau snapshot
# ---------------------------------------------------------------------------

credit_features = FeatureView(
    name="credit_features",
    entities=[customer],
    ttl=timedelta(days=35),  # bureau refreshes monthly
    schema=[
        Field(
            name="credit_utilization",
            dtype=Float64,
            description=(
                "Revolving credit utilization ratio = balance / limit. "
                "Range [0, 1+]. Values > 1 indicate over-limit accounts."
            ),
            tags={"bureau": "tradelines"},
        ),
        Field(
            name="payment_history_score",
            dtype=Float64,
            description=(
                "Normalized on-time payment history over 24 months. "
                "Range [0, 1] where 1 = perfect payment history."
            ),
            tags={"bureau": "payment"},
        ),
        Field(
            name="debt_to_income",
            dtype=Float64,
            description=(
                "Debt-to-income ratio = total monthly debt / gross monthly income. "
                "Key underwriting feature for credit scoring."
            ),
            tags={"bureau": "income"},
        ),
        Field(
            name="num_open_accounts",
            dtype=Int64,
            description="Total number of open credit accounts (revolving + installment).",
            tags={"bureau": "tradelines"},
        ),
        Field(
            name="num_derogatory_marks",
            dtype=Int64,
            description=(
                "Count of derogatory marks: collections, charge-offs, bankruptcies, "
                "and public records in the past 7 years."
            ),
            tags={"bureau": "derogatory"},
        ),
        Field(
            name="credit_age_months",
            dtype=Int64,
            description="Age of oldest open account in months.",
            tags={"bureau": "tradelines"},
        ),
        Field(
            name="recent_hard_inquiries_6m",
            dtype=Int64,
            description="Number of hard credit inquiries in the last 6 months.",
            tags={"bureau": "inquiries"},
        ),
        Field(
            name="installment_utilization",
            dtype=Float64,
            description=(
                "Installment loan utilization = remaining balance / original balance. "
                "Decreasing values indicate healthy repayment trajectory."
            ),
            tags={"bureau": "tradelines"},
        ),
        Field(
            name="months_since_last_delinquency",
            dtype=Int64,
            description=(
                "Months since the most recent delinquency event. "
                "Null-encoded as -1 when no delinquency exists."
            ),
            tags={"bureau": "derogatory"},
        ),
        Field(
            name="estimated_credit_score",
            dtype=Float64,
            description=(
                "Internal model-estimated credit score (300–850 scale). "
                "Not a FICO score — used as a derived feature only."
            ),
            tags={"bureau": "score", "derived": "true"},
        ),
    ],
    online=True,
    source=credit_bureau_source,
    tags={"domain": "finance", "team": "risk", "model": "credit,risk"},
)

# ---------------------------------------------------------------------------
# Feature View 3 — Customer Profile Features
# Static / slowly-changing customer attributes
# ---------------------------------------------------------------------------

customer_profile_features = FeatureView(
    name="customer_profile_features",
    entities=[customer],
    ttl=timedelta(days=90),
    schema=[
        Field(
            name="account_age_days",
            dtype=Int64,
            description="Days since account opening. Older accounts are lower churn/fraud risk.",
        ),
        Field(
            name="customer_segment",
            dtype=String,
            description=(
                "Business-assigned customer segment: RETAIL, PREMIUM, SMB, CORPORATE. "
                "Used to apply segment-specific risk thresholds."
            ),
        ),
        Field(
            name="kyc_verified",
            dtype=Bool,
            description="Whether the customer has completed KYC (Know Your Customer) verification.",
        ),
        Field(
            name="product_count",
            dtype=Int64,
            description="Number of distinct financial products held (checking, savings, loan, card).",
        ),
        Field(
            name="annual_income_band",
            dtype=String,
            description=(
                "Self-reported annual income band: <30K, 30-60K, 60-100K, 100-200K, >200K. "
                "Stored as ordinal string to avoid leakage of exact income."
            ),
        ),
        Field(
            name="state_code",
            dtype=String,
            description="US state code of primary account address (2-letter ISO 3166-2).",
        ),
        Field(
            name="channel",
            dtype=String,
            description="Acquisition channel: BRANCH, ONLINE, MOBILE, PARTNER, REFERRAL.",
        ),
    ],
    online=True,
    source=customer_profile_source,
    tags={
        "domain": "finance",
        "team": "data-engineering",
        "model": "fraud,credit,churn",
    },
)

# ---------------------------------------------------------------------------
# Feature View 4 — Fraud Labels (offline training only)
# ---------------------------------------------------------------------------

fraud_label_features = FeatureView(
    name="fraud_label_features",
    entities=[transaction],
    ttl=timedelta(days=365),
    schema=[
        Field(
            name="is_fraud",
            dtype=Bool,
            description=(
                "Ground-truth fraud label resolved 90 days post-transaction. "
                "True = confirmed fraudulent. Never served to the online store."
            ),
            tags={"sensitivity": "high", "label": "true"},
        ),
        Field(
            name="fraud_type",
            dtype=String,
            description=(
                "Fraud taxonomy label: CARD_NOT_PRESENT, ACCOUNT_TAKEOVER, "
                "SYNTHETIC_IDENTITY, FIRST_PARTY, FRIENDLY_FRAUD."
            ),
            tags={"sensitivity": "high", "label": "true"},
        ),
        Field(
            name="investigation_status",
            dtype=String,
            description="Case status: CONFIRMED, DISPUTED, CLEARED, PENDING.",
            tags={"sensitivity": "high"},
        ),
    ],
    online=False,  # never expose labels online
    source=fraud_labels_source,
    tags={"domain": "finance", "team": "fraud", "sensitivity": "high"},
)
