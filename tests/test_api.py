from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_invalid_prediction_is_rejected():
    response = client.post("/predict", json={"population": -1})
    assert response.status_code == 422
