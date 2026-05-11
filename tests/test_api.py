from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from segmentation_mlops.api.main import app
from segmentation_mlops.config import get_settings


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_openapi_schema(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    assert r.json()["info"]["title"]


def test_metrics_endpoint(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert b"mlops_" in r.content or b"# HELP" in r.content


def test_predict_when_model_exists(client):
    mp = get_settings().resolved_model_path()
    if not Path(mp).exists():
        pytest.skip("model.joblib not present; run training first")
    body = {
        "days_since_last_order": 20,
        "sessions_last_month": 4,
        "avg_basket_size": 3.0,
        "category_diversity": 5,
        "discount_share": 0.15,
        "returns_rate": 0.03,
        "avg_session_minutes": 10,
        "device_type": "desktop",
        "platform": "web",
        "marketing_channel": "paid_search",
        "region": "US",
    }
    r = client.post("/api/v1/predict", json=body)
    assert r.status_code == 200
    data = r.json()
    assert "prediction" in data
    assert "probabilities" in data
