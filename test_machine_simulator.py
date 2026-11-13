import random

import httpx
import pytest

from app import MachineConditions
from machine_simulator import MACHINES, generate_reading, run, valid_api_url, main


def test_reproducible_scenarios_and_valid_payloads():
    def samples():
        rng = random.Random(42)
        return [generate_reading(rng, kind) for _ in range(100) for kind in MACHINES.values()]
    readings = samples()
    assert readings == samples()
    assert {scenario for scenario, _ in readings} == {"normale", "critico"}
    for scenario, payload in readings:
        MachineConditions.model_validate(payload)
        assert 295.3 <= payload["air_temperature_k"] <= 304.5
        assert 305.7 <= payload["process_temperature_k"] <= 313.8
        if scenario == "critico":
            assert payload["process_temperature_k"] - payload["air_temperature_k"] < 8.6
            assert payload["rotational_speed_rpm"] < 1380


def test_three_machines_and_http_requests(capsys):
    received = []
    def handler(request):
        import json
        received.append((str(request.url), json.loads(request.content)))
        return httpx.Response(200, json={"predicted_class": 1, "failure_score": 0.95})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert run(client, "http://example.test/", 0, 2, 42) == 0
    assert len(received) == 6
    assert all(url == "http://example.test/predict" for url, _ in received)
    assert [p["type"] for _, p in received] == list(MACHINES.values()) * 2
    output = capsys.readouterr().out
    assert all(output.count(machine) == 2 for machine in MACHINES)
    assert "classe=1" in output and "failure_score=0.95" in output


def test_http_errors_do_not_stop_other_machines(capsys):
    calls = []
    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ConnectError("offline", request=request)
        return httpx.Response(503) if len(calls) == 2 else httpx.Response(200, json={"predicted_class": 0, "failure_score": 0.1})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert run(client, "http://example.test", 0, 1, 42) == 1
    output = capsys.readouterr().out
    assert len(calls) == 3
    assert "M01" in output and "ConnectError" in output
    assert "M02" in output and "HTTPStatusError" in output
    assert "M03" in output and "classe=0" in output


@pytest.mark.parametrize("url", ["http://", "ftp://example.com", "localhost:8000", "http://bad host", "http://example.com:99999", "http://example.com?x=1"])
def test_invalid_urls(url):
    assert not valid_api_url(url)


@pytest.mark.parametrize("url", ["http://localhost:8000", "https://example.com/api", "http://[::1]:8000"])
def test_valid_urls(url):
    assert valid_api_url(url)


@pytest.mark.parametrize("response", [
    httpx.Response(200, content=b"not json"),
    httpx.Response(200, json={"predicted_class": 1}),
    httpx.Response(200, json={"predicted_class": True, "failure_score": 0.5}),
    httpx.Response(200, json={"predicted_class": 2, "failure_score": 0.5}),
    httpx.Response(200, json={"predicted_class": 1, "failure_score": "0.5"}),
    httpx.Response(200, json={"predicted_class": 1, "failure_score": 1.1}),
    httpx.Response(200, content=b'{"predicted_class":1,"failure_score":NaN}'),
])
def test_invalid_responses_fail_all_requests(response, capsys):
    with httpx.Client(transport=httpx.MockTransport(lambda request: response)) as client:
        assert run(client, "http://example.test", 0, 1, 1) == 1
    assert "riuscite=0 fallite=3" in capsys.readouterr().out


def test_timeout_partial_failure(capsys):
    calls = []
    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ReadTimeout("timeout", request=request)
        return httpx.Response(200, json={"predicted_class": 0, "failure_score": 0.0})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert run(client, "http://example.test", 0, 1, 1) == 1
    output = capsys.readouterr().out
    assert "ReadTimeout" in output and "riuscite=2 fallite=1" in output


def test_cli_rejects_invalid_url_before_client(monkeypatch):
    monkeypatch.setattr("sys.argv", ["machine_simulator.py", "--api-url", "http://"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
