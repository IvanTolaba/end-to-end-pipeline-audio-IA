


---
## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### 🏗️ Architecture Overview

The project follows a modular layered architecture, separating responsibilities into orchestration, data engineering, machine learning, model serving, infrastructure and quality assurance.

| Layer | Directory | Responsibility |
|-------|-----------|----------------|
| Workflow Orchestration | `airflow/` | DAGs that orchestrate ETL, model training and evaluation. |
| Data Engineering | `etl/` | Data ingestion, preprocessing, segmentation and distributed MFCC extraction with PySpark. |
| Machine Learning | `ml/` | Model training, evaluation, reports and artifacts. |
| Model Serving | `api/` | REST API for real-time inference using FastAPI. |
| Infrastructure | `docker/` | Dockerfiles and containerization of all services. |
| Configuration | `config/` | Centralized configuration using YAML files. |
| Quality Assurance | `tests/` | Automated testing with Pytest. |