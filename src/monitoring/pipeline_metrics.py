"""Prometheus metrics shared by the API."""
from prometheus_client import Counter, Histogram

PREDICTIONS = Counter("model_predictions_total", "Total model predictions")
PREDICTION_ERRORS = Counter("model_prediction_errors_total", "Total failed model predictions")
PREDICTION_LATENCY = Histogram("model_prediction_seconds", "Model prediction latency in seconds")
