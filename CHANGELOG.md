# Changelog

All notable changes to this project are documented here.

## [1.0.0] - 2026-06-16

### Added
- Feast feature store integration with point-in-time correct feature retrieval for training and serving
- MLflow model registry with staging/production promotion gates and artifact lineage tracking
- Airflow DAG templates for feature engineering, model training, validation, and deployment pipelines
- Evidently AI monitoring integration detecting feature drift and model performance degradation in production
- Financial ML example pipelines: credit scoring, fraud detection, and market signal generation
- Unified feature catalog with ownership metadata, freshness SLAs, and lineage graph visualization

### Changed
- Production-ready CI/CD with 95%+ test coverage enforcement

### Security
- Feature store access controlled via role-based policies; raw financial data never stored in feature tables
