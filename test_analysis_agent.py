import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Barrier
from types import SimpleNamespace

import httpx
import pytest

from analysis_agent import Trace, summarize
from machine_simulator import run
from simulation_store import atomic_json, read_run, start_run


def test_simulation_persistence_and_summary(tmp_path):
    snapshot = start_run(1, 42, tmp_path)
    assert read_run("latest", tmp_path)["status"] == "running"
    assert not (snapshot[0] / "agent_trace").exists()
    assert not (snapshot[0] / "report.md").exists()
    def handler(request):
        return httpx.Response(200, json={"predicted_class": 1, "failure_score": 0.8})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert run(client, "http://example.test", 0, 1, 42, snapshot) == 0
    data = read_run(snapshot[1]["run_id"], tmp_path)
    assert data["status"] == "completed" and len(data["readings"]) == 3
    assert all(row["result"]["predicted_class"] == 1 for row in summarize(data)["results"])


def test_failed_and_incomplete_runs_are_not_old_results(tmp_path, monkeypatch):
    old = start_run(1, 1, tmp_path)
    import simulation_store
    class NextSecond(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.now(tz) + timedelta(seconds=2)
    monkeypatch.setattr(simulation_store, "datetime", NextSecond)
    new = start_run(1, 2, tmp_path)
    assert read_run("latest", tmp_path)["run_id"] == new[1]["run_id"] != old[1]["run_id"]
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(503))) as client:
        assert run(client, "http://example.test", 0, 1, 2, new) == 1
    result = read_run("latest", tmp_path)
    assert result["status"] == "failed" and all(row["result"] is None for row in result["readings"])


def test_run_name_and_collision(tmp_path, monkeypatch):
    import simulation_store
    class FixedTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 11, 13, 13, 43, 7, tzinfo=timezone.utc).astimezone(tz)
    monkeypatch.setattr(simulation_store, "datetime", FixedTime)
    folder, data = start_run(1, 1, tmp_path)
    assert folder.name == data["run_id"] == "test_2026-11-13__14-43-07"
    with pytest.raises(FileExistsError):
        start_run(1, 1, tmp_path)
    assert read_run("latest", tmp_path)["run_id"] == data["run_id"]


def test_invalid_or_missing_files(tmp_path):
    with pytest.raises(ValueError):
        read_run("latest", tmp_path)
    folder, data = start_run(1, 1, tmp_path)
    with pytest.raises(ValueError):
        read_run("../outside", tmp_path)
    (folder / "simulation.json").write_text("{", encoding="utf-8")
    with pytest.raises(ValueError):
        read_run(data["run_id"], tmp_path)


def test_old_latest_requires_explicit_run_id(tmp_path):
    folder, data = start_run(1, 1, tmp_path)
    data["started_at"] = "2000-01-01T00:00:00+00:00"
    atomic_json(folder / "simulation.json", data)
    with pytest.raises(ValueError, match="older than 24 hours"):
        read_run("latest", tmp_path)
    assert read_run(data["run_id"], tmp_path)["run_id"] == data["run_id"]
    data["status"] = "completed"
    atomic_json(folder / "simulation.json", data)
    with pytest.raises(ValueError):
        read_run(data["run_id"], tmp_path)


def test_latest_interrupted_before_first_reading(tmp_path, monkeypatch):
    import analysis_agent
    import simulation_store

    current_time = datetime.now(timezone.utc)
    instants = iter([current_time - timedelta(seconds=2), current_time])

    class FixedTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return next(instants, current_time).astimezone(tz)

    monkeypatch.setattr(simulation_store, "datetime", FixedTime)
    _, old = start_run(1, 1, tmp_path)
    folder, newest = start_run(1, 2, tmp_path)

    class FakeAgent:
        def __init__(self, *, tools, **kwargs):
            self.tool = tools[0]

        def __call__(self, question):
            data = json.loads(self.tool())
            assert data["run_id"] == newest["run_id"]
            assert data["status"] == "running"
            assert data["results"] == []
            assert len(data["missing_readings"]) == 3
            return "Nessuna classificazione negativa."

    monkeypatch.setattr(analysis_agent, "Agent", FakeAgent)
    monkeypatch.setattr(analysis_agent, "OllamaModel", lambda **kwargs: None)
    answer = analysis_agent.analyze("Quali sono le classificazioni?", root=tmp_path)
    assert "dati sufficienti" in answer
    assert "Letture mancanti: 3" in answer
    assert answer in (folder / "analysis_001" / "report.md").read_text(encoding="utf-8")
    assert read_run(old["run_id"], tmp_path)["seed"] == 1
    assert not (tmp_path / "latest.json").exists()


def test_interrupted_after_partial_readings(tmp_path):
    snapshot = start_run(1, 42, tmp_path)
    responses = iter([httpx.Response(200, json={"predicted_class": 0, "failure_score": 0.1}),
                      httpx.Response(503)])

    def handler(request):
        response = next(responses, None)
        if response is None:
            raise KeyboardInterrupt
        return response

    with httpx.Client(transport=httpx.MockTransport(handler)) as client, pytest.raises(KeyboardInterrupt):
        run(client, "http://example.test", 0, 1, 42, snapshot)

    data = summarize(read_run("latest", tmp_path))
    assert data["status"] == "running"
    assert data["recorded_readings"] == 2
    assert data["valid_readings"] == 1
    assert data["results"][0]["result"]["predicted_class"] == 0
    assert data["results"][1]["result"] is None
    assert data["results"][1]["error"] == "HTTPStatusError"
    assert data["missing_readings"] == [{"cycle": 1, "machine_id": "M03"}]


def test_latest_uses_start_time_even_when_older_snapshot_is_published_last(tmp_path, monkeypatch):
    import simulation_store

    # When daylight saving ends, the later UTC instant has the earlier local name.
    older = datetime(2026, 10, 25, 0, 59, 59, 900000, tzinfo=timezone.utc)
    newer = datetime(2026, 10, 25, 1, 0, 0, 100000, tzinfo=timezone.utc)
    instants = iter([older, newer, newer])

    class FixedTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return next(instants).astimezone(tz)

    def interleave(path, data):
        if data["seed"] == 1:
            start_run(1, 2, tmp_path)
        atomic_json(path, data)

    monkeypatch.setattr(simulation_store, "datetime", FixedTime)
    monkeypatch.setattr(simulation_store, "atomic_json", interleave)
    _, old = start_run(1, 1, tmp_path)
    selected = read_run("latest", tmp_path)
    assert selected["seed"] == 2
    assert selected["run_id"] < old["run_id"]
    assert read_run(old["run_id"], tmp_path)["seed"] == 1


def test_latest_breaks_equal_timestamp_ties_by_run_id(tmp_path):
    started = datetime.now(timezone.utc)
    ids = ("test_2000-01-01__00-00-01", "test_2000-01-01__00-00-00")
    offsets = (timezone.utc, timezone(timedelta(hours=1)))
    for run_id, offset in zip(ids, offsets):
        folder = tmp_path / run_id
        folder.mkdir()
        atomic_json(folder / "simulation.json", {"schema_version": 1, "run_id": run_id,
                    "started_at": started.astimezone(offset).isoformat(), "status": "running",
                    "cycles": 1, "readings": []})
    assert read_run("latest", tmp_path)["run_id"] == max(ids)


def test_trace_hooks_record_order_and_contents(tmp_path, capsys):
    trace = Trace(tmp_path / "agent_trace")
    trace.emit("agent_started", question="Question")
    event = SimpleNamespace(tool_use={"name": "get_simulation_results", "input": {}})
    trace.before_tool(event)
    event.result = {"status": "success", "content": [{"text": '{"run_id":"example","results":[]}'}]}
    event.exception = None
    trace.after_tool(event)
    trace.emit("agent_completed")
    files = sorted(trace.directory.iterdir())
    assert [p.name for p in files] == ["001_agent_started.json", "002_tool_call.json", "003_tool_result.json", "004_agent_completed.json"]
    assert json.loads(files[1].read_text(encoding="utf-8"))["tool"] == "get_simulation_results"
    assert json.loads(files[2].read_text(encoding="utf-8"))["result"] == {"run_id": "example", "results": []}
    assert event.result["content"][0]["text"] == '{"run_id":"example","results":[]}'
    assert len(capsys.readouterr().out.splitlines()) == 4


def test_trace_preserves_non_json_text(tmp_path):
    trace = Trace(tmp_path / "trace")
    event = SimpleNamespace(tool_use={"name": "tool", "input": {}},
                            result={"status": "success", "content": [{"text": "plain text"}]}, exception=None)
    trace.after_tool(event)
    assert json.loads(next(trace.directory.iterdir()).read_text(encoding="utf-8"))["result"] == "plain text"


@pytest.mark.parametrize("first_fails", [False, True])
def test_repeated_analysis_preserves_reports_and_traces(tmp_path, monkeypatch, first_fails):
    import analysis_agent

    folder, data = start_run(1, 42, tmp_path)
    original_snapshot = (folder / "simulation.json").read_bytes()
    legacy_trace = folder / "agent_trace"
    legacy_trace.mkdir()
    legacy_event = legacy_trace / "001_agent_started.json"
    legacy_event.write_text("{}", encoding="utf-8")
    legacy_report = folder / "report.md"
    legacy_report.write_text("Legacy report", encoding="utf-8")

    class FakeAgent:
        def __init__(self, *, tools, **kwargs):
            self.tool = tools[0]

        def __call__(self, question):
            if first_fails and question == "Prima domanda":
                raise ConnectionError("Ollama unavailable")
            assert json.loads(self.tool())["run_id"] == data["run_id"]
            return question

    monkeypatch.setattr(analysis_agent, "Agent", FakeAgent)
    monkeypatch.setattr(analysis_agent, "OllamaModel", lambda **kwargs: None)

    first = folder / "analysis_001"
    if first_fails:
        with pytest.raises(ConnectionError):
            analysis_agent.analyze("Prima domanda", root=tmp_path)
        event = first / "agent_trace" / "002_agent_error.json"
        assert json.loads(event.read_text(encoding="utf-8"))["error"] == "ConnectionError"
        assert "Analisi fallita" in (first / "report.md").read_text(encoding="utf-8")
    else:
        answer = analysis_agent.analyze("Prima domanda", root=tmp_path)
        assert answer in (first / "report.md").read_text(encoding="utf-8")
        assert (first / "agent_trace" / "002_agent_completed.json").is_file()

    first_files = {path: path.read_bytes() for path in first.rglob("*") if path.is_file()}
    answer = analysis_agent.analyze("Seconda domanda", run_id=data["run_id"], root=tmp_path)
    second = folder / "analysis_002"
    assert answer in (second / "report.md").read_text(encoding="utf-8")
    started = second / "agent_trace" / "001_agent_started.json"
    assert json.loads(started.read_text(encoding="utf-8"))["question"] == "Seconda domanda"
    assert (second / "agent_trace" / "002_agent_completed.json").is_file()
    assert {path: path.read_bytes() for path in first.rglob("*") if path.is_file()} == first_files
    assert legacy_event.read_text(encoding="utf-8") == "{}"
    assert legacy_report.read_text(encoding="utf-8") == "Legacy report"
    assert (folder / "simulation.json").read_bytes() == original_snapshot


def test_concurrent_analyses_use_separate_directories(tmp_path, monkeypatch):
    import analysis_agent

    folder, data = start_run(1, 42, tmp_path)
    barrier = Barrier(2, timeout=10)

    class FakeAgent:
        def __init__(self, *, tools, **kwargs):
            self.tool = tools[0]

        def __call__(self, question):
            barrier.wait()
            assert json.loads(self.tool())["run_id"] == data["run_id"]
            return question

    monkeypatch.setattr(analysis_agent, "Agent", FakeAgent)
    monkeypatch.setattr(analysis_agent, "OllamaModel", lambda **kwargs: None)

    questions = {"Domanda A", "Domanda B"}
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = [executor.submit(analysis_agent.analyze, question, data["run_id"], tmp_path)
                   for question in questions]
        for result in results:
            assert result.result(timeout=15)

    recorded_questions = set()
    for name in ("analysis_001", "analysis_002"):
        analysis = folder / name
        started = analysis / "agent_trace" / "001_agent_started.json"
        recorded_questions.add(json.loads(started.read_text(encoding="utf-8"))["question"])
        assert (analysis / "agent_trace" / "002_agent_completed.json").is_file()
        assert (analysis / "report.md").is_file()
    assert recorded_questions == questions
