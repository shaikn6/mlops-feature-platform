"""
Feature Registry — wraps Feast FeatureStore for apply/materialize/list operations.

Provides a typed interface over the raw Feast SDK so application code
and pipeline tasks never import feast directly.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_REPO_PATH = Path(__file__).resolve().parent / "feature_repo"


class FeatureRegistry:
    """High-level wrapper around Feast FeatureStore registry operations."""

    def __init__(self, repo_path: str | Path | None = None) -> None:
        self._repo_path = Path(repo_path or FEATURE_REPO_PATH)
        self._store: Any | None = (
            None  # feast.FeatureStore, typed as Any to isolate import
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _get_store(self) -> Any:
        """Lazy-initialize the Feast FeatureStore (deferred import)."""
        if self._store is None:
            from feast import FeatureStore  # noqa: PLC0415

            self._store = FeatureStore(repo_path=str(self._repo_path))
        return self._store

    def apply(self) -> None:
        """Register all feature views, entities, and data sources with Feast."""
        from feature_store.feature_repo.features import (  # noqa: PLC0415
            credit_features,
            customer,
            customer_profile_features,
            customer_transaction_features,
            fraud_label_features,
            transaction,
        )

        store = self._get_store()
        store.apply(
            [
                customer,
                transaction,
                customer_transaction_features,
                credit_features,
                customer_profile_features,
                fraud_label_features,
            ]
        )
        logger.info("Feast apply completed — all feature views registered.")

    def materialize(
        self,
        start_date: datetime,
        end_date: datetime,
        feature_views: list[str] | None = None,
    ) -> None:
        """Materialize features from offline store → online store for a date range."""
        store = self._get_store()
        kwargs: dict[str, Any] = {
            "start_date": start_date,
            "end_date": end_date,
        }
        if feature_views:
            kwargs["feature_views"] = feature_views

        store.materialize(**kwargs)
        logger.info(
            "Materialization complete: %s → %s (views=%s)",
            start_date.isoformat(),
            end_date.isoformat(),
            feature_views or "all",
        )

    def materialize_incremental(self, end_date: datetime) -> None:
        """Incremental materialize from last materialization to end_date."""
        store = self._get_store()
        store.materialize_incremental(end_date=end_date)
        logger.info(
            "Incremental materialization complete up to %s.", end_date.isoformat()
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_feature_views(self) -> list[dict[str, Any]]:
        """Return metadata for all registered feature views."""
        store = self._get_store()
        views = store.list_feature_views()
        return [
            {
                "name": fv.name,
                "entities": [e for e in fv.entity_columns],
                "features": [f.name for f in fv.features],
                "ttl_seconds": int(fv.ttl.total_seconds()) if fv.ttl else None,
                "online": fv.online,
                "tags": fv.tags,
            }
            for fv in views
        ]

    def get_feature_view_schema(self, name: str) -> dict[str, Any]:
        """Return schema for a specific feature view by name."""
        store = self._get_store()
        fv = store.get_feature_view(name)
        return {
            "name": fv.name,
            "features": [
                {
                    "name": f.name,
                    "dtype": str(f.dtype),
                    "description": getattr(f, "description", ""),
                    "tags": getattr(f, "tags", {}),
                }
                for f in fv.features
            ],
            "ttl_seconds": int(fv.ttl.total_seconds()) if fv.ttl else None,
            "online": fv.online,
        }

    def get_historical_features(
        self,
        entity_df: pd.DataFrame,
        features: list[str],
    ) -> pd.DataFrame:
        """
        Point-in-time correct historical feature retrieval for model training.

        Args:
            entity_df: DataFrame with entity keys and ``event_timestamp`` column.
            features:  List of feature references in ``view:feature`` format.

        Returns:
            DataFrame with entity columns + requested feature columns.
        """
        store = self._get_store()
        job = store.get_historical_features(
            entity_df=entity_df,
            features=features,
        )
        result = job.to_df()
        logger.info(
            "Historical features retrieved: %d rows × %d feature cols",
            len(result),
            len(features),
        )
        return result
