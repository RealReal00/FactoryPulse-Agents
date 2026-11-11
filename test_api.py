import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app as api


PAYLOAD = {"type": "L", "air_temperature_k": 300.0, "process_temperature_k": 310.0,
           "rotational_speed_rpm": 1500, "torque_nm": 40.0, "tool_wear_min": 100}


@pytest.fixture
def client():
    with TestClient(api.app) as client:
        yield client


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_predict_matches_pipeline(client):
    response = client.post("/predict", json=PAYLOAD)
    assert response.status_code == 200
    model = api.app.state.model
    row = pd.DataFrame([["L", 300.0, 310.0, 1500, 40.0, 100]], columns=model.feature_names_in_)
    assert response.json() == {"predicted_class": int(model.predict(row)[0]),
                               "failure_score": float(model.predict_proba(row)[0, list(model.classes_).index(1)])}


@pytest.mark.parametrize("changes", [{"type": "X"}, {"torque_nm": "40"}, {"torque_nm": True},
                                     {"torque_nm": None}, {"unexpected": 1}])
def test_invalid_input(client, changes):
    assert client.post("/predict", json={**PAYLOAD, **changes}).status_code == 422


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_input(client, value):
    body = '{"type":"L","air_temperature_k":300,"process_temperature_k":310,' \
           '"rotational_speed_rpm":1500,"torque_nm":' + value + ',"tool_wear_min":100}'
    assert client.post("/predict", content=body, headers={"content-type": "application/json"}).status_code == 422


def test_missing_field(client):
    payload = {k: v for k, v in PAYLOAD.items() if k != "type"}
    assert client.post("/predict", json=payload).status_code == 422


@pytest.mark.parametrize("field", ["air_temperature_k", "process_temperature_k",
                                   "rotational_speed_rpm", "torque_nm", "tool_wear_min"])
def test_extreme_finite_value(client, field):
    assert client.post("/predict", json={**PAYLOAD, field: 1e100}).status_code == 422


@pytest.mark.parametrize("field,value", [
    ("air_temperature_k", 249), ("process_temperature_k", 401),
    ("rotational_speed_rpm", -1), ("rotational_speed_rpm", 5001),
    ("torque_nm", -1), ("torque_nm", 151),
    ("tool_wear_min", -1), ("tool_wear_min", 501),
])
def test_outside_operating_limits(client, field, value):
    assert client.post("/predict", json={**PAYLOAD, field: value}).status_code == 422


@pytest.mark.parametrize("field,value", [
    ("air_temperature_k", 250), ("process_temperature_k", 400),
    ("rotational_speed_rpm", 5000), ("torque_nm", 0), ("tool_wear_min", 500),
])
def test_operating_limit_is_inclusive(client, field, value):
    assert client.post("/predict", json={**PAYLOAD, field: value}).status_code == 200


def test_model_load_failure(monkeypatch):
    def fail(_):
        raise FileNotFoundError("model missing")
    monkeypatch.setattr(api.joblib, "load", fail)
    with pytest.raises(FileNotFoundError), TestClient(api.app):
        pass
