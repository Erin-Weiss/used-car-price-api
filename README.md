
```
used-car-price-api/
│
├── app/
│   ├── __init__.py
│   ├── main.py               # FastAPI app and routes
│   ├── predict.py            # prediction orchestration
│   ├── pipeline.py           # copied from src/pipeline.py in Part 1
│   ├── model.py              # CatBoost loading / caching
│   ├── schemas.py            # Pydantic request/response models
│   └── config.py             # settings and paths
│
├── models/ (from Part 1)
│   ├── catboost_final.cbm
│   └── api_artifacts/
│       ├── model_metadata.json
│       ├── categorical_vocabularies.json
│       ├── models_by_manufacturer.json
│       └── numeric_medians.json
│
├── tests/
│   ├── __init__.py
│   ├── test_predict.py
│   └── test_health.py
│
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
│
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── .env.example
├── pyproject.toml
└── README.md

```