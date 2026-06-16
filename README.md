<div align="center">

# MLOps Feature Platform

[![CI](https://github.com/shaikn6/mlops-feature-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/shaikn6/mlops-feature-platform/actions)
[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![Feast](https://img.shields.io/badge/Feast-feature--store-yellow)](https://feast.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker)](docker-compose.yml)

**Production MLOps platform — Feast feature store, drift detection, model registry, and Airflow pipelines for fintech ML**

</div>

## Architecture

```mermaid
graph TD
    A[Raw Data Sources] --> B[Airflow DAGs<br/>Feature Engineering]
    B --> C[Feast Feature Store]
    C --> D[Model Training]
    D --> E[Model Registry]
    E --> F[Serving API]
    F --> G[Monitoring]
    G --> H[Drift Detector]
    G --> I[Data Quality]
    H & I --> J[Alert Manager]
```

## Components

| Component | Tech | Purpose |
|-----------|------|---------|
| Feature Store | Feast | Offline + online feature serving |
| Pipelines | Airflow DAGs | Feature engineering, training |
| Model Registry | Custom | Versioning, metadata, lineage |
| Drift Detection | Statistical | PSI, KS test, population stability |
| Monitoring | Dashboard | Data quality, model health |
| API | FastAPI | Feature serving endpoint |

## Quick Start

```bash
git clone https://github.com/shaikn6/mlops-feature-platform
cd mlops-feature-platform && cp .env.example .env
docker compose up -d
# API: http://localhost:8000/docs
```

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v --cov=.
ruff check . --ignore E501
```

## License

MIT
