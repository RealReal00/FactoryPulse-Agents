import json
import traceback
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


# Verifica che /health segnali lo stato "ok" e confermi che il modello è caricato.
def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


# Verifica che /predict la nostra API FastAPI restituisca esattamente gli stessi risultati 
# della Random Forest quando la chiamiamo direttamente da Python.
def test_predict_matches_pipeline(client):
    response = client.post("/predict", json=PAYLOAD)
    assert response.status_code == 200
    model = api.app.state.model
    row = pd.DataFrame([["L", 300.0, 310.0, 1500, 40.0, 100]], columns=model.feature_names_in_)
    assert response.json() == {"predicted_class": int(model.predict(row)[0]),
                               "failure_score": float(model.predict_proba(row)[0, list(model.classes_).index(1)])}


# Verifica che /predict risponda con HTTP 422 per valori non validi o campi extra.
@pytest.mark.parametrize("changes", [{"type": "X"}, {"torque_nm": "40"}, {"torque_nm": True},
                                     {"torque_nm": None}, {"unexpected": 1}])
def test_invalid_input(client, changes):
    assert client.post("/predict", json={**PAYLOAD, **changes}).status_code == 422


# Verifica che /predict rifiuti con HTTP 422 una coppia pari a NaN o a infinito.
@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_input(client, value):
    body = '{"type":"L","air_temperature_k":300,"process_temperature_k":310,' \
           '"rotational_speed_rpm":1500,"torque_nm":' + value + ',"tool_wear_min":100}'
    assert client.post("/predict", content=body, headers={"content-type": "application/json"}).status_code == 422


# Verifica che /predict risponda con HTTP 422 se manca il campo obbligatorio type.
def test_missing_field(client):
    payload = {k: v for k, v in PAYLOAD.items() if k != "type"}
    assert client.post("/predict", json=payload).status_code == 422


# Verifica che /predict rifiuti con HTTP 422 il valore finito 1e100 in ogni campo numerico.
@pytest.mark.parametrize("field", ["air_temperature_k", "process_temperature_k",
                                   "rotational_speed_rpm", "torque_nm", "tool_wear_min"])
def test_extreme_finite_value(client, field):
    assert client.post("/predict", json={**PAYLOAD, field: 1e100}).status_code == 422


# Verifica che /predict rifiuti con HTTP 422 i valori appena fuori dai limiti operativi.
@pytest.mark.parametrize("field,value", [
    ("air_temperature_k", 249), ("process_temperature_k", 401),
    ("rotational_speed_rpm", -1), ("rotational_speed_rpm", 5001),
    ("torque_nm", -1), ("torque_nm", 151),
    ("tool_wear_min", -1), ("tool_wear_min", 501),
])
def test_outside_operating_limits(client, field, value):
    assert client.post("/predict", json={**PAYLOAD, field: value}).status_code == 422


# Verifica che /predict accetti con HTTP 200 i valori esattamente sui limiti operativi.
@pytest.mark.parametrize("field,value", [
    ("air_temperature_k", 250), ("process_temperature_k", 400),
    ("rotational_speed_rpm", 5000), ("torque_nm", 0), ("tool_wear_min", 500),
])
def test_operating_limit_is_inclusive(client, field, value):
    assert client.post("/predict", json={**PAYLOAD, field: value}).status_code == 200


# Verifica che l'avvio sollevi RuntimeError se il caricamento del modello fallisce.
def test_model_load_failure(monkeypatch):
    def fail(_):
        raise FileNotFoundError("model missing")
    monkeypatch.setattr(api.joblib, "load", fail)
    with pytest.raises(RuntimeError, match="Unable to load model"), TestClient(api.app):
        pass


# Dopo una previsione riuscita, verifica gli eventi JSON di livello INFO su stderr
# e l'assenza del valore di velocità e del campo failure_score nei log.
def test_structured_logs_and_privacy(capsys, monkeypatch):
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.delenv("MODEL_PATH", raising=False)
    with TestClient(api.app) as client:
        assert client.post("/predict", json=PAYLOAD).status_code == 200
    lines = capsys.readouterr().err.splitlines()
    records = [json.loads(line) for line in lines]
    assert [r["event"] for r in records] == ["application_start", "model_loaded", "prediction_completed"]
    assert all(r["timestamp"] and r["level"] == "INFO" for r in records)
    assert "1500" not in " ".join(lines) and "failure_score" not in " ".join(lines)


# Verifica che l'app tenti di usare il percorso configurato, risponda con HTTP 200
# su /health e non scriva su stderr con LOG_LEVEL=ERROR durante l'avvio riuscito.
def test_configuration_model_path_and_level(monkeypatch, capsys):
    alternate = "alternate-controlled-model.joblib"
    monkeypatch.setenv("MODEL_PATH", alternate)
    monkeypatch.setenv("LOG_LEVEL", "ERROR")
    original_load = api.joblib.load
    seen = []
    def load(path):
        seen.append(str(path))
        return original_load(api.MODEL_PATH)
    monkeypatch.setattr(api.joblib, "load", load)
    with TestClient(api.app) as client:
        assert client.get("/health").status_code == 200
    assert seen == [alternate]
    assert capsys.readouterr().err == ""


# Verifica che un valore non valido di LOG_LEVEL provochi ValueError all'avvio.
def test_invalid_level(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "invalid")
    with pytest.raises(ValueError, match="Invalid LOG_LEVEL"), TestClient(api.app):
        pass


# Verifica che un modello mancante faccia fallire l'avvio con RuntimeError,
# registri model_load_failed e non riveli il percorso nel traceback.
def test_missing_model_logs_failure(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    with pytest.raises(RuntimeError, match="Unable to load model") as error:
        with TestClient(api.app):
            pass
    events = [json.loads(line)["event"] for line in capsys.readouterr().err.splitlines()]
    assert events == ["application_start", "model_load_failed"]
    rendered = "".join(traceback.format_exception(error.value))
    assert str(tmp_path) not in rendered and "missing.joblib" not in rendered


# Verifica che due avvii sovrapposti non duplichino i log e che il primo client
# continui a rispondere e a registrare le previsioni dopo la chiusura del secondo.
def test_overlapping_lifespans_keep_single_handler(monkeypatch, capsys):
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.delenv("MODEL_PATH", raising=False)
    with TestClient(api.app) as first:
        with TestClient(api.app) as second:
            assert second.post("/predict", json=PAYLOAD).status_code == 200
        assert first.post("/predict", json=PAYLOAD).status_code == 200
    records = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    assert [r["event"] for r in records].count("prediction_completed") == 2
    assert [r["event"] for r in records].count("model_loaded") == 2


# Verifica che una previsione fallita restituisca HTTP 500 con un messaggio generico.
# Il log segnala l'errore senza includere il dettaglio sensibile o la velocità inviata.
def test_prediction_failure_is_sanitized(monkeypatch, capsys):
    with TestClient(api.app) as client:
        def fail(_):
            raise RuntimeError("sensitive detail")
        monkeypatch.setattr(api.app.state.model, "predict", fail)
        response = client.post("/predict", json=PAYLOAD)
        assert response.status_code == 500
        assert response.json() == {"detail": "Prediction failed"}
    logs = capsys.readouterr().err
    assert "prediction_failed" in logs and "sensitive detail" not in logs and "1500" not in logs
