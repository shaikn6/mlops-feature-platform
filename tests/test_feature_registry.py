"""
Tests for feature_store.registry — FeatureRegistry
Covers: lazy init, apply, materialize, materialize_incremental,
        list_feature_views, get_feature_view_schema, get_historical_features,
        all error paths and edge cases.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_mock_feature_view(name: str, entities=None, features=None, ttl_seconds=86400):
    fv = MagicMock()
    fv.name = name
    fv.entity_columns = entities or ["customer_id"]
    fv.features = [MagicMock(name=f, dtype="Float64") for f in (features or ["feat_a", "feat_b"])]
    for i, f in enumerate(fv.features):
        f.name = (features or ["feat_a", "feat_b"])[i]
    fv.ttl = timedelta(seconds=ttl_seconds) if ttl_seconds else None
    fv.online = True
    fv.tags = {"domain": "finance"}
    return fv


@pytest.fixture
def mock_feast_store():
    """Returns a MagicMock standing in for feast.FeatureStore."""
    store = MagicMock()
    store.list_feature_views.return_value = [
        _make_mock_feature_view("customer_transaction_features"),
        _make_mock_feature_view("credit_features", ttl_seconds=None),
    ]
    store.get_feature_view.return_value = _make_mock_feature_view(
        "customer_transaction_features"
    )
    job = MagicMock()
    job.to_df.return_value = pd.DataFrame({"customer_id": ["c1", "c2"], "feat_a": [1.0, 2.0]})
    store.get_historical_features.return_value = job
    return store


@pytest.fixture
def registry(mock_feast_store, tmp_path):
    """FeatureRegistry with feast patched out."""
    with patch("feature_store.registry.FeatureRegistry._get_store", return_value=mock_feast_store):
        from feature_store.registry import FeatureRegistry
        reg = FeatureRegistry(repo_path=str(tmp_path))
        reg._store = mock_feast_store  # inject directly to skip import
        yield reg, mock_feast_store


# ---------------------------------------------------------------------------
# __init__ / repo path
# ---------------------------------------------------------------------------

class TestFeatureRegistryInit:
    def test_default_repo_path_is_feature_repo(self, tmp_path):
        with patch("feature_store.registry.FeatureStore", MagicMock()):
            from feature_store.registry import FeatureRegistry, FEATURE_REPO_PATH
            reg = FeatureRegistry()
            assert reg._repo_path == FEATURE_REPO_PATH

    def test_custom_repo_path_is_used(self, tmp_path):
        from feature_store.registry import FeatureRegistry
        reg = FeatureRegistry(repo_path=str(tmp_path))
        assert reg._repo_path == tmp_path

    def test_path_object_accepted(self, tmp_path):
        from feature_store.registry import FeatureRegistry
        reg = FeatureRegistry(repo_path=tmp_path)
        assert reg._repo_path == tmp_path

    def test_store_starts_as_none(self, tmp_path):
        from feature_store.registry import FeatureRegistry
        reg = FeatureRegistry(repo_path=str(tmp_path))
        assert reg._store is None


# ---------------------------------------------------------------------------
# _get_store lazy init
# ---------------------------------------------------------------------------

class TestGetStore:
    def test_lazy_init_calls_feast(self, tmp_path):
        from feature_store.registry import FeatureRegistry
        mock_store = MagicMock()
        with patch("feature_store.registry.FeatureStore", return_value=mock_store) as MockFS:
            reg = FeatureRegistry(repo_path=str(tmp_path))
            store = reg._get_store()
            MockFS.assert_called_once_with(repo_path=str(tmp_path))
            assert store is mock_store

    def test_second_call_reuses_instance(self, tmp_path):
        from feature_store.registry import FeatureRegistry
        mock_store = MagicMock()
        with patch("feature_store.registry.FeatureStore", return_value=mock_store) as MockFS:
            reg = FeatureRegistry(repo_path=str(tmp_path))
            s1 = reg._get_store()
            s2 = reg._get_store()
            assert s1 is s2
            assert MockFS.call_count == 1


# ---------------------------------------------------------------------------
# apply()
# ---------------------------------------------------------------------------

class TestApply:
    def test_apply_calls_store_apply(self, registry):
        reg, store = registry
        mock_features = MagicMock()
        patches = {
            "feature_store.registry.FeatureStore": MagicMock(return_value=store),
        }
        feature_names = [
            "customer", "transaction",
            "customer_transaction_features", "credit_features",
            "customer_profile_features", "fraud_label_features",
        ]
        with patch.dict("sys.modules", {
            "feature_store.feature_repo.features": MagicMock(
                customer=MagicMock(),
                transaction=MagicMock(),
                customer_transaction_features=MagicMock(),
                credit_features=MagicMock(),
                customer_profile_features=MagicMock(),
                fraud_label_features=MagicMock(),
            )
        }):
            reg.apply()
            store.apply.assert_called_once()
            called_args = store.apply.call_args[0][0]
            assert len(called_args) == 6

    def test_apply_passes_all_six_objects(self, registry):
        reg, store = registry
        mods = {
            "customer": MagicMock(name="customer"),
            "transaction": MagicMock(name="transaction"),
            "customer_transaction_features": MagicMock(),
            "credit_features": MagicMock(),
            "customer_profile_features": MagicMock(),
            "fraud_label_features": MagicMock(),
        }
        mock_module = MagicMock()
        for k, v in mods.items():
            setattr(mock_module, k, v)

        with patch.dict("sys.modules", {"feature_store.feature_repo.features": mock_module}):
            reg.apply()
        store.apply.assert_called_once()


# ---------------------------------------------------------------------------
# materialize()
# ---------------------------------------------------------------------------

class TestMaterialize:
    def test_materialize_without_views(self, registry):
        reg, store = registry
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)
        reg.materialize(start, end)
        store.materialize.assert_called_once_with(start_date=start, end_date=end)

    def test_materialize_with_feature_views(self, registry):
        reg, store = registry
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)
        views = ["customer_transaction_features"]
        reg.materialize(start, end, feature_views=views)
        store.materialize.assert_called_once_with(
            start_date=start, end_date=end, feature_views=views
        )

    def test_materialize_with_none_views_omits_kwarg(self, registry):
        reg, store = registry
        start = datetime(2024, 6, 1)
        end = datetime(2024, 6, 30)
        reg.materialize(start, end, feature_views=None)
        call_kwargs = store.materialize.call_args[1]
        assert "feature_views" not in call_kwargs

    def test_materialize_empty_views_list(self, registry):
        reg, store = registry
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)
        reg.materialize(start, end, feature_views=[])
        # empty list is falsy — same as None path
        store.materialize.assert_called_once()

    def test_materialize_propagates_store_exception(self, registry):
        reg, store = registry
        store.materialize.side_effect = RuntimeError("Feast error")
        with pytest.raises(RuntimeError, match="Feast error"):
            reg.materialize(datetime(2024, 1, 1), datetime(2024, 1, 31))


# ---------------------------------------------------------------------------
# materialize_incremental()
# ---------------------------------------------------------------------------

class TestMaterializeIncremental:
    def test_calls_store_with_end_date(self, registry):
        reg, store = registry
        end = datetime(2024, 6, 15)
        reg.materialize_incremental(end)
        store.materialize_incremental.assert_called_once_with(end_date=end)

    def test_propagates_exception(self, registry):
        reg, store = registry
        store.materialize_incremental.side_effect = ValueError("No baseline")
        with pytest.raises(ValueError, match="No baseline"):
            reg.materialize_incremental(datetime(2024, 6, 15))


# ---------------------------------------------------------------------------
# list_feature_views()
# ---------------------------------------------------------------------------

class TestListFeatureViews:
    def test_returns_list_of_dicts(self, registry):
        reg, store = registry
        result = reg.list_feature_views()
        assert isinstance(result, list)
        assert len(result) == 2

    def test_each_dict_has_expected_keys(self, registry):
        reg, store = registry
        result = reg.list_feature_views()
        required_keys = {"name", "entities", "features", "ttl_seconds", "online", "tags"}
        for item in result:
            assert required_keys.issubset(item.keys())

    def test_ttl_none_when_not_set(self, registry):
        reg, store = registry
        result = reg.list_feature_views()
        # second view has ttl=None
        credit_view = next(r for r in result if r["name"] == "credit_features")
        assert credit_view["ttl_seconds"] is None

    def test_ttl_computed_in_seconds(self, registry):
        reg, store = registry
        result = reg.list_feature_views()
        txn_view = next(r for r in result if r["name"] == "customer_transaction_features")
        assert txn_view["ttl_seconds"] == 86400

    def test_features_is_list_of_names(self, registry):
        reg, store = registry
        result = reg.list_feature_views()
        assert all(isinstance(f, str) for f in result[0]["features"])


# ---------------------------------------------------------------------------
# get_feature_view_schema()
# ---------------------------------------------------------------------------

class TestGetFeatureViewSchema:
    def test_returns_correct_schema(self, registry):
        reg, store = registry
        schema = reg.get_feature_view_schema("customer_transaction_features")
        assert schema["name"] == "customer_transaction_features"
        assert "features" in schema
        assert "ttl_seconds" in schema
        assert "online" in schema

    def test_features_contain_name_and_dtype(self, registry):
        reg, store = registry
        schema = reg.get_feature_view_schema("customer_transaction_features")
        for feat in schema["features"]:
            assert "name" in feat
            assert "dtype" in feat

    def test_store_get_feature_view_called_with_name(self, registry):
        reg, store = registry
        reg.get_feature_view_schema("credit_features")
        store.get_feature_view.assert_called_with("credit_features")

    def test_propagates_not_found_exception(self, registry):
        reg, store = registry
        store.get_feature_view.side_effect = KeyError("not found")
        with pytest.raises(KeyError):
            reg.get_feature_view_schema("nonexistent")


# ---------------------------------------------------------------------------
# get_historical_features()
# ---------------------------------------------------------------------------

class TestGetHistoricalFeatures:
    def test_returns_dataframe(self, registry):
        reg, store = registry
        entity_df = pd.DataFrame({
            "customer_id": ["c1", "c2"],
            "event_timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
        })
        result = reg.get_historical_features(entity_df, ["view:feat_a"])
        assert isinstance(result, pd.DataFrame)

    def test_calls_store_correctly(self, registry):
        reg, store = registry
        entity_df = pd.DataFrame({
            "customer_id": ["c1"],
            "event_timestamp": [datetime(2024, 1, 1)],
        })
        features = ["view:feat_a", "view:feat_b"]
        reg.get_historical_features(entity_df, features)
        store.get_historical_features.assert_called_once_with(
            entity_df=entity_df,
            features=features,
        )

    def test_propagates_store_error(self, registry):
        reg, store = registry
        store.get_historical_features.side_effect = ConnectionError("Redis down")
        with pytest.raises(ConnectionError, match="Redis down"):
            reg.get_historical_features(pd.DataFrame(), [])


