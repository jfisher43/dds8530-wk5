"""FastAPI application for county population-growth predictions."""
from __future__ import annotations

from functools import lru_cache
from time import perf_counter

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from api.schemas import CountyFeatures, PredictionResponse
from src.config import settings
from src.models.predict import load_model_bundle, predict_one
from src.monitoring.pipeline_metrics import PREDICTION_ERRORS, PREDICTION_LATENCY, PREDICTIONS

app = FastAPI(title="County Population Growth API", version="1.0.0")


@lru_cache(maxsize=1)
def get_bundle():
    return load_model_bundle(settings.model_path)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy", "model": "ready" if settings.model_path.exists() else "missing"}


@app.get("/model-info")
def model_info() -> dict:
    try:
        bundle = get_bundle()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"model_type": bundle["model_type"], "features": bundle["features"], "metrics": bundle["metrics"], "test_year": bundle["test_year"]}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CountyFeatures) -> PredictionResponse:
    started = perf_counter()
    try:
        bundle = get_bundle()
        result = predict_one(bundle, features.model_dump())
        PREDICTIONS.inc()
        return PredictionResponse(**result, model_type=bundle["model_type"])
    except FileNotFoundError as exc:
        PREDICTION_ERRORS.inc()
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        PREDICTION_ERRORS.inc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    finally:
        PREDICTION_LATENCY.observe(perf_counter() - started)


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
