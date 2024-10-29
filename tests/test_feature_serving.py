"""
Tests for feature_store.serving — FeatureServer
Covers: online retrieval, batch retrieval, convenience wrappers,
        training dataset construction, error paths.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from feature_store.serving import (
    CREDIT_ONLINE_FEATURES,
    FRAUD_ONLINE_FEATURES,
    FeatureServer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_store():
    store = MagicMock()
    response = MagicMock()
    response.to_df.return_value = pd.DataFrame({
        "customer_id": ["c1", "c2"],
        "avg_spend_7d": [100.0, 200.0],
    })
    store.get_online_features.return_value = response

    hist_job = MagicMock()
    hist_job.to_df.return_value = pd.DataFrame({
        "customer_id": ["c1"],
        "event_timestamp": [datetime(2024, 1, 1)],
        "avg_spend_7d": [150.0],
    })
    store.get_historical_features.return_value = hist_job
    return store


@pytest.fixture
def server(mock_store, tmp_path):
    server = FeatureServer(repo_path=str(tmp_path))
    server._store = mock_store
    return server, mock_store


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_fraud_online_features_is_list(self):
        assert isinstance(FRAUD_ONLINE_FEATURES, list)
        assert len(FRAUD_ONLINE_FEATURES) > 0

    def test_credit_online_features_is_list(self):
        assert isinstance(CREDIT_ONLINE_FEATURES, list)
        assert len(CREDIT_ONLINE_FEATURES) > 0

    def test_fraud_features_contain_customer_transaction_prefix(self):
        assert any("customer_transaction_features:" in f for f in FRAUD_ONLINE_FEATURES)

    def test_credit_features_contain_credit_prefix(self):
        assert any("credit_features:" in f for f in CREDIT_ONLINE_FEATURES)


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestFeatureServerInit:
    def test_default_repo_path(self):
        from feature_store.serving import FEATURE_REPO_PATH
        server = FeatureServer()
        assert server._repo_path == FEATURE_REPO_PATH

    def test_custom_repo_path(self, tmp_path):
        server = FeatureServer(repo_path=str(tmp_path))
        assert server._repo_path == tmp_path

    def test_store_starts_as_none(self, tmp_path):
        server = FeatureServer(repo_path=str(tmp_path))
        assert server._store is None


# ---------------------------------------------------------------------------
# _get_store
# ---------------------------------------------------------------------------

class TestGetStore:
    def test_lazy_init(self, tmp_path):
        mock_fs = MagicMock()
        with patch("feature_store.serving.FeatureStore", return_value=mock_fs) as MockFS:
            server = FeatureServer(repo_path=str(tmp_path))
            store = server._get_store()
            MockFS.assert_called_once_with(repo_path=str(tmp_path))
            assert store is mock_fs

    def test_reuses_existing_store(self, tmp_path):
        mock_fs = MagicMock()
        with patch("feature_store.serving.FeatureStore", return_value=mock_fs) as MockFS:
            server = FeatureServer(repo_path=str(tmp_path))
            server._get_store()
            server._get_store()
            assert MockFS.call_count == 1


# ---------------------------------------------------------------------------
# get_online_features()
# ---------------------------------------------------------------------------

class TestGetOnlineFeatures:
    def test_returns_list_of_dicts(self, server):
        srv, store = server
        rows = [{"customer_id": "c1"}, {"customer_id": "c2"}]
        result = srv.get_online_features(rows, ["view:feat"])
        assert isinstance(result, list)
        assert len(result) == 2

    def test_calls_store_with_correct_args(self, server):
        srv, store = server
        rows = [{"customer_id": "c1"}]
        features = ["customer_transaction_features:avg_spend_7d"]
        srv.get_online_features(rows, features)
        store.get_online_features.assert_called_once_with(
            features=features, entity_rows=rows
        )

    def test_empty_entity_rows_raises_value_error(self, server):
        srv, store = server
        with pytest.raises(ValueError, match="entity_rows must not be empty"):
            srv.get_online_features([], ["view:feat"])

    def test_propagates_store_error(self, server):
        srv, store = server
        store.get_online_features.side_effect = RuntimeError("Redis timeout")
        with pytest.raises(RuntimeError, match="Redis timeout"):
            srv.get_online_features([{"customer_id": "c1"}], ["view:feat"])

    def test_result_contains_expected_fields(self, server):
        srv, store = server
        result = srv.get_online_features(
            [{"customer_id": "c1"}, {"customer_id": "c2"}],
            ["avg_spend_7d"]
        )
        assert "customer_id" in result[0]


# ---------------------------------------------------------------------------
# get_fraud_features_online()
# ---------------------------------------------------------------------------

class TestGetFraudFeaturesOnline:
    def test_builds_entity_rows_from_ids(self, server):
        srv, store = server
        srv.get_fraud_features_online(["c1", "c2"])
        call_kwargs = store.get_online_features.call_args[1]
        assert call_kwargs["entity_rows"] == [
            {"customer_id": "c1"},
            {"customer_id": "c2"},
        ]

    def test_uses_fraud_feature_list(self, server):
        srv, store = server
        srv.get_fraud_features_online(["c1"])
        call_kwargs = store.get_online_features.call_args[1]
        assert call_kwargs["features"] == FRAUD_ONLINE_FEATURES

    def test_single_customer_id(self, server):
        srv, store = server
        result = srv.get_fraud_features_online(["c-only"])
        assert isinstance(result, list)

    def test_empty_list_propagates_value_error(self, server):
        srv, _ = server
        with pytest.raises(ValueError):
            srv.get_fraud_features_online([])


# ---------------------------------------------------------------------------
# get_credit_features_online()
# ---------------------------------------------------------------------------

class TestGetCreditFeaturesOnline:
    def test_builds_entity_rows(self, server):
        srv, store = server
        srv.get_credit_features_online(["c1", "c2", "c3"])
        call_kwargs = store.get_online_features.call_args[1]
        assert len(call_kwargs["entity_rows"]) == 3

    def test_uses_credit_feature_list(self, server):
        srv, store = server
        srv.get_credit_features_online(["c1"])
        call_kwargs = store.get_online_features.call_args[1]
        assert call_kwargs["features"] == CREDIT_ONLINE_FEATURES

    def test_empty_list_propagates_value_error(self, server):
        srv, _ = server
        with pytest.raises(ValueError):
            srv.get_credit_features_online([])


# ---------------------------------------------------------------------------
# get_training_dataset()
# ---------------------------------------------------------------------------

class TestGetTrainingDataset:
    def _entity_df(self):
        return pd.DataFrame({
            "customer_id": ["c1", "c2"],
            "event_timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
        })

    def test_returns_dataframe(self, server):
        srv, store = server
        result = srv.get_training_dataset(
            self._entity_df(), ["view:feat"]
        )
        assert isinstance(result, pd.DataFrame)

    def test_missing_event_timestamp_raises(self, server):
        srv, store = server
        bad_df = pd.DataFrame({"customer_id": ["c1"]})
        with pytest.raises(ValueError, match="event_timestamp"):
            srv.get_training_dataset(bad_df, ["view:feat"])

    def test_calls_historical_features(self, server):
        srv, store = server
        edf = self._entity_df()
        features = ["view:feat_a", "view:feat_b"]
        srv.get_training_dataset(edf, features)
        store.get_historical_features.assert_called_once_with(
            entity_df=edf, features=features
        )

    def test_label_column_merged_when_present(self, server):
        srv, store = server

        # dataset from store
        hist_df = pd.DataFrame({
            "customer_id": ["c1"],
            "event_timestamp": [datetime(2024, 1, 1)],
            "avg_spend_7d": [100.0],
        })
        job = MagicMock()
        job.to_df.return_value = hist_df
        store.get_historical_features.return_value = job

        entity_df = pd.DataFrame({
            "customer_id": ["c1"],
            "event_timestamp": [datetime(2024, 1, 1)],
            "is_fraud": [True],
        })
        result = srv.get_training_dataset(entity_df, ["view:feat"], label_column="is_fraud")
        assert "is_fraud" in result.columns

    def test_label_column_absent_from_entity_df_ignored(self, server):
        srv, store = server
        entity_df = self._entity_df()
        # label_col passed but not in entity_df — should not raise
        result = srv.get_training_dataset(entity_df, ["view:feat"], label_column="is_fraud")
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# get_fraud_training_dataset()
# ---------------------------------------------------------------------------

class TestGetFraudTrainingDataset:
    def test_includes_fraud_labels_in_features(self, server):
        srv, store = server
        entity_df = pd.DataFrame({
            "customer_id": ["c1"],
            "event_timestamp": [datetime(2024, 1, 1)],
            "is_fraud": [True],
        })
        srv.get_fraud_training_dataset(entity_df)
        call_kwargs = store.get_historical_features.call_args[1]
        features_used = call_kwargs["features"]
        assert "fraud_label_features:is_fraud" in features_used
        assert "fraud_label_features:fraud_type" in features_used

    def test_raises_without_event_timestamp(self, server):
        srv, _ = server
        bad_df = pd.DataFrame({"customer_id": ["c1"]})
        with pytest.raises(ValueError, match="event_timestamp"):
            srv.get_fraud_training_dataset(bad_df)


# ---------------------------------------------------------------------------
# get_credit_training_dataset()
# ---------------------------------------------------------------------------

class TestGetCreditTrainingDataset:
    def test_uses_credit_online_features(self, server):
        srv, store = server
        entity_df = pd.DataFrame({
            "customer_id": ["c1"],
            "event_timestamp": [datetime(2024, 1, 1)],
        })
        srv.get_credit_training_dataset(entity_df)
        call_kwargs = store.get_historical_features.call_args[1]
        assert call_kwargs["features"] == CREDIT_ONLINE_FEATURES

    def test_raises_without_event_timestamp(self, server):
        srv, _ = server
        bad_df = pd.DataFrame({"customer_id": ["c1"]})
        with pytest.raises(ValueError, match="event_timestamp"):
            srv.get_credit_training_dataset(bad_df)

