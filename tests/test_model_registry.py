"""
Tests for model_registry package.
Since experiment.py and deployment.py don't exist yet, this creates
stub modules so tests can cover the __init__.py import structure and
provide a comprehensive baseline for when those modules are added.

We also test the package-level __version__ and __all__.
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Setup stubs for missing sub-modules
# ---------------------------------------------------------------------------

def _make_experiment_stub():
    """Return a stub ExperimentManager class."""
    class ExperimentManager:
        def __init__(self, experiment_name: str, tracking_uri: str | None = None):
            self.experiment_name = experiment_name
            self._tracking_uri = tracking_uri
            self._client = None

        def _get_client(self):
            if self._client is None:
                import mlflow
                self._client = mlflow.tracking.MlflowClient(
                    tracking_uri=self._tracking_uri
                )
            return self._client

        def create_experiment(self) -> int:
            client = self._get_client()
            return client.create_experiment(self.experiment_name)

        def log_params(self, run_id: str, params: dict) -> None:
            client = self._get_client()
            for key, value in params.items():
                client.log_param(run_id, key, value)

        def log_metrics(self, run_id: str, metrics: dict, step: int | None = None) -> None:
            client = self._get_client()
            for key, value in metrics.items():
                client.log_metric(run_id, key, value, step=step)

        def log_model(self, run_id: str, model, artifact_path: str) -> str:
            import mlflow
            with mlflow.start_run(run_id=run_id):
                info = mlflow.sklearn.log_model(model, artifact_path)
            return info.model_uri

        def get_best_run(self, metric: str, ascending: bool = False) -> dict:
            import mlflow
            runs = mlflow.search_runs(
                experiment_names=[self.experiment_name],
                order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
                max_results=1,
            )
            if runs.empty:
                raise ValueError(f"No runs found in experiment {self.experiment_name}")
            return runs.iloc[0].to_dict()

    return ExperimentManager


def _make_deployment_stub():
    """Return a stub ModelDeployment class."""
    class ModelDeployment:
        def __init__(self, registry_uri: str | None = None):
            self._registry_uri = registry_uri
            self._client = None

        def _get_client(self):
            if self._client is None:
                import mlflow
                self._client = mlflow.tracking.MlflowClient(
                    registry_uri=self._registry_uri
                )
            return self._client

        def register_model(self, model_uri: str, name: str) -> str:
            import mlflow
            result = mlflow.register_model(model_uri, name)
            return result.version

        def promote(self, model_name: str, version: str, stage: str) -> None:
            allowed = {"Staging", "Production", "Archived"}
            if stage not in allowed:
                raise ValueError(f"Invalid stage: {stage}. Must be one of {allowed}")
            client = self._get_client()
            client.transition_model_version_stage(
                name=model_name, version=version, stage=stage
            )

        def get_production_model(self, model_name: str) -> dict:
            client = self._get_client()
            versions = client.get_latest_versions(model_name, stages=["Production"])
            if not versions:
                raise ValueError(f"No Production version found for {model_name}")
            v = versions[0]
            return {
                "name": v.name,
                "version": v.version,
                "stage": v.current_stage,
                "run_id": v.run_id,
                "source": v.source,
            }

        def list_model_versions(self, model_name: str) -> list:
            client = self._get_client()
            versions = client.search_model_versions(f"name='{model_name}'")
            return [
                {
                    "version": v.version,
                    "stage": v.current_stage,
                    "run_id": v.run_id,
                    "status": v.status,
                }
                for v in versions
            ]

        def archive_version(self, model_name: str, version: str) -> None:
            self.promote(model_name, version, "Archived")

    return ModelDeployment


@pytest.fixture(autouse=True)
def inject_stubs():
    """Inject stub modules into sys.modules before each test."""
    ExperimentManager = _make_experiment_stub()
    ModelDeployment = _make_deployment_stub()

    exp_mod = ModuleType("model_registry.experiment")
    exp_mod.ExperimentManager = ExperimentManager  # type: ignore[attr-defined]

    dep_mod = ModuleType("model_registry.deployment")
    dep_mod.ModelDeployment = ModelDeployment  # type: ignore[attr-defined]

    sys.modules["model_registry.experiment"] = exp_mod
    sys.modules["model_registry.deployment"] = dep_mod

    # Remove cached model_registry so it re-imports
    sys.modules.pop("model_registry", None)

    yield ExperimentManager, ModelDeployment

    sys.modules.pop("model_registry.experiment", None)
    sys.modules.pop("model_registry.deployment", None)
    sys.modules.pop("model_registry", None)


# ---------------------------------------------------------------------------
# Package-level imports
# ---------------------------------------------------------------------------

class TestPackageImports:
    def test_experiment_manager_importable(self, inject_stubs):
        import model_registry
        assert hasattr(model_registry, "ExperimentManager")

    def test_model_deployment_importable(self, inject_stubs):
        import model_registry
        assert hasattr(model_registry, "ModelDeployment")

    def test_all_contains_expected_names(self, inject_stubs):
        import model_registry
        assert "ExperimentManager" in model_registry.__all__
        assert "ModelDeployment" in model_registry.__all__

    def test_version_string(self, inject_stubs):
        import model_registry
        assert model_registry.__version__ == "1.0.0"


# ---------------------------------------------------------------------------
# ExperimentManager stub tests
# ---------------------------------------------------------------------------

class TestExperimentManagerStub:
    def test_init_stores_experiment_name(self, inject_stubs):
        ExperimentManager, _ = inject_stubs
        mgr = ExperimentManager("fraud_experiment")
        assert mgr.experiment_name == "fraud_experiment"

    def test_init_default_tracking_uri_is_none(self, inject_stubs):
        ExperimentManager, _ = inject_stubs
        mgr = ExperimentManager("exp")
        assert mgr._tracking_uri is None

    def test_init_custom_tracking_uri(self, inject_stubs):
        ExperimentManager, _ = inject_stubs
        mgr = ExperimentManager("exp", tracking_uri="http://mlflow:5000")
        assert mgr._tracking_uri == "http://mlflow:5000"

    def test_create_experiment_calls_client(self, inject_stubs):
        ExperimentManager, _ = inject_stubs
        mock_mlflow = MagicMock()
        mock_client = MagicMock()
        mock_mlflow.tracking.MlflowClient.return_value = mock_client
        mock_client.create_experiment.return_value = 42

        with patch.dict("sys.modules", {"mlflow": mock_mlflow,
                                         "mlflow.tracking": mock_mlflow.tracking}):
            mgr = ExperimentManager("exp")
            exp_id = mgr.create_experiment()
            mock_client.create_experiment.assert_called_once_with("exp")
            assert exp_id == 42

    def test_log_params_iterates_all_params(self, inject_stubs):
        ExperimentManager, _ = inject_stubs
        mock_mlflow = MagicMock()
        mock_client = MagicMock()
        mock_mlflow.tracking.MlflowClient.return_value = mock_client

        with patch.dict("sys.modules", {"mlflow": mock_mlflow,
                                         "mlflow.tracking": mock_mlflow.tracking}):
            mgr = ExperimentManager("exp")
            mgr.log_params("run-123", {"lr": 0.01, "epochs": 10})
            assert mock_client.log_param.call_count == 2

    def test_log_metrics_iterates_all_metrics(self, inject_stubs):
        ExperimentManager, _ = inject_stubs
        mock_mlflow = MagicMock()
        mock_client = MagicMock()
        mock_mlflow.tracking.MlflowClient.return_value = mock_client

        with patch.dict("sys.modules", {"mlflow": mock_mlflow,
                                         "mlflow.tracking": mock_mlflow.tracking}):
            mgr = ExperimentManager("exp")
            mgr.log_metrics("run-123", {"auc": 0.95, "f1": 0.87}, step=1)
            assert mock_client.log_metric.call_count == 2

    def test_get_best_run_raises_when_no_runs(self, inject_stubs):
        import pandas as pd
        ExperimentManager, _ = inject_stubs
        mock_mlflow = MagicMock()
        mock_mlflow.search_runs.return_value = pd.DataFrame()

        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            mgr = ExperimentManager("empty_exp")
            mgr._client = MagicMock()
            with pytest.raises(ValueError, match="No runs found"):
                mgr.get_best_run("auc_roc")

    def test_get_best_run_returns_first_result(self, inject_stubs):
        import pandas as pd
        ExperimentManager, _ = inject_stubs
        mock_mlflow = MagicMock()
        mock_mlflow.search_runs.return_value = pd.DataFrame(
            [{"run_id": "r1", "metrics.auc_roc": 0.95}]
        )

        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            mgr = ExperimentManager("exp")
            mgr._client = MagicMock()
            result = mgr.get_best_run("auc_roc")
            assert result["run_id"] == "r1"


# ---------------------------------------------------------------------------
# ModelDeployment stub tests
# ---------------------------------------------------------------------------

class TestModelDeploymentStub:
    def test_init_default_registry_uri(self, inject_stubs):
        _, ModelDeployment = inject_stubs
        dep = ModelDeployment()
        assert dep._registry_uri is None

    def test_register_model_calls_mlflow(self, inject_stubs):
        _, ModelDeployment = inject_stubs
        mock_mlflow = MagicMock()
        mock_result = MagicMock()
        mock_result.version = "3"
        mock_mlflow.register_model.return_value = mock_result

        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            dep = ModelDeployment()
            version = dep.register_model("models:/fraud_v3/1", "fraud")
            assert version == "3"
            mock_mlflow.register_model.assert_called_once()

    def test_promote_valid_stage(self, inject_stubs):
        _, ModelDeployment = inject_stubs
        mock_mlflow = MagicMock()
        mock_client = MagicMock()
        mock_mlflow.tracking.MlflowClient.return_value = mock_client

        with patch.dict("sys.modules", {"mlflow": mock_mlflow,
                                         "mlflow.tracking": mock_mlflow.tracking}):
            dep = ModelDeployment()
            dep.promote("fraud", "3", "Production")
            mock_client.transition_model_version_stage.assert_called_once_with(
                name="fraud", version="3", stage="Production"
            )

    def test_promote_invalid_stage_raises(self, inject_stubs):
        _, ModelDeployment = inject_stubs
        dep = ModelDeployment()
        dep._client = MagicMock()
        with pytest.raises(ValueError, match="Invalid stage"):
            dep.promote("fraud", "3", "InvalidStage")

    def test_get_production_model_returns_dict(self, inject_stubs):
        _, ModelDeployment = inject_stubs
        mock_mlflow = MagicMock()
        mock_client = MagicMock()
        mock_mlflow.tracking.MlflowClient.return_value = mock_client
        mv = MagicMock()
        mv.name = "fraud"
        mv.version = "3"
        mv.current_stage = "Production"
        mv.run_id = "run-42"
        mv.source = "s3://bucket/fraud/v3"
        mock_client.get_latest_versions.return_value = [mv]

        with patch.dict("sys.modules", {"mlflow": mock_mlflow,
                                         "mlflow.tracking": mock_mlflow.tracking}):
            dep = ModelDeployment()
            result = dep.get_production_model("fraud")
            assert result["version"] == "3"
            assert result["stage"] == "Production"

    def test_get_production_model_raises_when_none(self, inject_stubs):
        _, ModelDeployment = inject_stubs
        mock_mlflow = MagicMock()
        mock_client = MagicMock()
        mock_mlflow.tracking.MlflowClient.return_value = mock_client
        mock_client.get_latest_versions.return_value = []

        with patch.dict("sys.modules", {"mlflow": mock_mlflow,
                                         "mlflow.tracking": mock_mlflow.tracking}):
            dep = ModelDeployment()
            with pytest.raises(ValueError, match="No Production"):
                dep.get_production_model("unknown_model")

    def test_list_model_versions_returns_list(self, inject_stubs):
        _, ModelDeployment = inject_stubs
        mock_mlflow = MagicMock()
        mock_client = MagicMock()
        mock_mlflow.tracking.MlflowClient.return_value = mock_client
        mv = MagicMock()
        mv.version = "1"
        mv.current_stage = "Staging"
        mv.run_id = "run-1"
        mv.status = "READY"
        mock_client.search_model_versions.return_value = [mv]

        with patch.dict("sys.modules", {"mlflow": mock_mlflow,
                                         "mlflow.tracking": mock_mlflow.tracking}):
            dep = ModelDeployment()
            versions = dep.list_model_versions("fraud")
            assert len(versions) == 1
            assert versions[0]["version"] == "1"

    def test_archive_version_calls_promote(self, inject_stubs):
        _, ModelDeployment = inject_stubs
        dep = ModelDeployment()
        dep._client = MagicMock()
        dep._client.transition_model_version_stage = MagicMock()
        dep.archive_version("fraud", "3")
        dep._client.transition_model_version_stage.assert_called_once_with(
            name="fraud", version="3", stage="Archived"
        )

