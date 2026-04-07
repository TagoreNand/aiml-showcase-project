# Architecture Notes

## System layers

### 1. Product layer
- API consumers
- Streamlit demo app
- future frontend / internal tooling

### 2. Application layer
- FastAPI endpoints
- orchestration and validation
- typed schemas

### 3. AI/ML inference layer
- classifier service
- entity extraction service
- retrieval engine
- QA synthesis logic

### 4. Data and feedback layer
- model artifacts
- corpus persistence
- feedback store
- future feature store / vector DB / warehouse

---

## Suggested production upgrades

- OCR service for scanned documents
- async task queue with Celery / Redis
- pgvector or Elasticsearch for retrieval
- MLflow for model registry
- monitoring with Prometheus + Grafana
- auth and tenant-aware storage
