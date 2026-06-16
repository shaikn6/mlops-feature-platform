"""
Feature serving — online and batch retrieval from Feast.

Online path:  Redis → sub-millisecond p99 latency.
Batch path:   offline store → training datasets.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_REPO_PATH = Path(__file__).resolve().parent / "feature_repo"

# Default feature references grouped by model type
FRAUD_ONLINE_FEATURES = [
    "customer_transaction_features:avg_spend_7d",
    "customer_transaction_features:total_spend_7d",
    "customer_transaction_features:transaction_count_7d",
    "customer_transaction_features:transaction_count_30d",
    "customer_transaction_features:std_spend_30d",
    "customer_transaction_features:fraud_rate_90d",
    "customer_transaction_features:fraud_count_90d",
    "customer_transaction_features:unique_merchants_30d",
    "customer_transaction_features:international_txn_ratio_30d",
    "customer_profile_features:account_age_days",
    "customer_profile_features:customer_segment",
    "customer_profile_features:kyc_verified",
]

CREDIT_ONLINE_FEATURES = [
    "credit_features:credit_utilization",
    "credit_features:payment_history_score",
    "credit_features:debt_to_income",
    "credit_features:num_open_accounts",
    "credit_features:num_derogatory_marks",
    "credit_features:credit_age_months",
    "credit_features:recent_hard_inquiries_6m",
    "credit_features:months_since_last_delinquency",
    "credit_features:estimated_credit_score",
    "customer_transaction_features:avg_spend_30d",
    "customer_transaction_features:transaction_count_30d",
    "customer_profile_features:annual_income_band",
    "customer_profile_features:product_count",
]


class FeatureServer:
    """
    Serves features for real-time inference and batch training.

    Online retrieval hits Redis for p99 < 10ms.
    Batch retrieval performs point-in-time joins over the offline store.
    """

    def __init__(self, repo_path: str | Path | None = None) -> None:
        self._repo_path = Path(repo_path or FEATURE_REPO_PATH)
        self._store: Any | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_store(self) -> Any:
        if self._store is None:
            from feast import FeatureStore  # noqa: PLC0415

            self._store = FeatureStore(repo_path=str(self._repo_path))
        return self._store

    # ------------------------------------------------------------------
    # Online retrieval
    # ------------------------------------------------------------------

    def get_online_features(
        self,
        entity_rows: list[dict[str, Any]],
        features: list[str],
    ) -> list[dict[str, Any]]:
        """
        Retrieve features from the online store (Redis) for a list of entities.

        Args:
            entity_rows: List of dicts with entity key(s), e.g.
                         ``[{"customer_id": "c-001"}, ...]``
            features:    Feature references in ``view:feature`` format.

        Returns:
            List of dicts — one per entity row — with feature name → value mapping.
            Missing features are returned as ``None``.

        Raises:
            ValueError: If entity_rows is empty.
        """
        if not entity_rows:
            raise ValueError("entity_rows must not be empty.")

        store = self._get_store()
        response = store.get_online_features(
            features=features,
            entity_rows=entity_rows,
        )
        result_df = response.to_df()
        records = result_df.to_dict(orient="records")

        logger.info(
            "Online feature retrieval: %d entities × %d features",
            len(entity_rows),
            len(features),
        )
        return records

    def get_fraud_features_online(
        self,
        customer_ids: list[str],
    ) -> list[dict[str, Any]]:
        """Convenience wrapper — returns fraud model features for a batch of customer IDs."""
        entity_rows = [{"customer_id": cid} for cid in customer_ids]
        return self.get_online_features(
            entity_rows=entity_rows,
            features=FRAUD_ONLINE_FEATURES,
        )

    def get_credit_features_online(
        self,
        customer_ids: list[str],
    ) -> list[dict[str, Any]]:
        """Convenience wrapper — returns credit scoring features for a batch of customer IDs."""
        entity_rows = [{"customer_id": cid} for cid in customer_ids]
        return self.get_online_features(
            entity_rows=entity_rows,
            features=CREDIT_ONLINE_FEATURES,
        )

    # ------------------------------------------------------------------
    # Batch / historical retrieval
    # ------------------------------------------------------------------

    def get_training_dataset(
        self,
        entity_df: pd.DataFrame,
        features: list[str],
        label_column: str | None = None,
    ) -> pd.DataFrame:
        """
        Build a point-in-time correct training dataset.

        Args:
            entity_df:     DataFrame with entity keys and ``event_timestamp`` column.
            features:      Feature references in ``view:feature`` format.
            label_column:  Optional column name of the label already present in entity_df.

        Returns:
            Joined DataFrame with feature values aligned to each entity timestamp.
        """
        if "event_timestamp" not in entity_df.columns:
            raise ValueError("entity_df must contain an 'event_timestamp' column.")

        store = self._get_store()
        job = store.get_historical_features(
            entity_df=entity_df,
            features=features,
        )
        dataset = job.to_df()

        if label_column and label_column in entity_df.columns:
            # Re-attach label column (Feast strips non-entity, non-feature columns)
            dataset = dataset.merge(
                entity_df[["event_timestamp", label_column]],
                on="event_timestamp",
                how="left",
            )

        logger.info(
            "Training dataset built: %d rows, %d columns (label=%s)",
            len(dataset),
            len(dataset.columns),
            label_column,
        )
        return dataset

    def get_fraud_training_dataset(self, entity_df: pd.DataFrame) -> pd.DataFrame:
        """Return training dataset with all fraud model features."""
        features = FRAUD_ONLINE_FEATURES + [
            "fraud_label_features:is_fraud",
            "fraud_label_features:fraud_type",
        ]
        return self.get_training_dataset(
            entity_df=entity_df,
            features=features,
            label_column="is_fraud",
        )

    def get_credit_training_dataset(self, entity_df: pd.DataFrame) -> pd.DataFrame:
        """Return training dataset with all credit scoring features."""
        return self.get_training_dataset(
            entity_df=entity_df,
            features=CREDIT_ONLINE_FEATURES,
        )
