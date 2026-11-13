"""Local, atomic simulation snapshots and validation."""
import json
import math
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent / "simulation_results"
ID_PATTERN = re.compile(r"test_\d{4}-\d{2}-\d{2}__\d{2}-\d{2}-\d{2}")


def now():
    return datetime.now(timezone.utc).isoformat()

#Cosa succede se il programma si interrompe mentre sta scrivendo il file?
#L'idea è semplice: prima scriviamo un file temporaneo e soltanto quando è completo lo sostituiamo al file definitivo.
def atomic_json(path, value):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def start_run(cycles, seed, root=ROOT):
    started = datetime.now(timezone.utc)
    #ROOT ... path simulation_result
    root.mkdir(parents=True, exist_ok=True)
    run_id = started.astimezone(ZoneInfo("Europe/Rome")).strftime("test_%Y-%m-%d__%H-%M-%S")
    folder = root / run_id
    try:
        folder.mkdir()
    except FileExistsError:
        raise FileExistsError("Simulation ID already exists; retry in a different second") from None
    #inizializzo "data" all'avvio della run e il file simulation si aggiornerà nel corso della run
    data = {"schema_version": 1, "run_id": run_id, "started_at": started.isoformat(), "finished_at": None,
            "status": "running", "cycles": cycles, "seed": seed, "readings": []}
    atomic_json(folder / "simulation.json", data)
    return folder, data


def started_at(data):
    started = datetime.fromisoformat(data["started_at"])
    if started.utcoffset() is None:
        raise ValueError("Simulation timestamp must include timezone")
    return started


def latest_run(root):
    latest_key = None
    latest_data = None
    for folder in root.iterdir():
        if not ID_PATTERN.fullmatch(folder.name) or not folder.is_dir():
            continue
        try:
            raw = (folder / "simulation.json").read_text(encoding="utf-8")
        except FileNotFoundError:
            # The first snapshot has not been published yet.
            continue
        data = json.loads(raw)
        key = (started_at(data), folder.name)
        if latest_key is None or key > latest_key:
            latest_key, latest_data = key, data
    if latest_key is None:
        raise ValueError("No simulation snapshot available")
    return latest_key[1], latest_data


#read_run legge una simulazione già salvata sul disco (cioè il contenuto di simulation.json
# ) e verifica che i suoi dati siano validi.
def read_run(run_id="latest", root=ROOT):
    try:
        latest_requested = run_id == "latest"
        if latest_requested:
            run_id, data = latest_run(root)
        elif not isinstance(run_id, str) or not ID_PATTERN.fullmatch(run_id):
            raise ValueError("Invalid run ID")
        else:
            data = json.loads((root / run_id / "simulation.json").read_text(encoding="utf-8"))
        if (not isinstance(data, dict) or data.get("schema_version") != 1 or data.get("run_id") != run_id
                or data.get("status") not in ("running", "completed", "failed")
                or type(data.get("cycles")) is not int or data["cycles"] < 1
                or not isinstance(data.get("readings"), list)):
            raise ValueError("Invalid simulation schema")
        started = started_at(data)
        if latest_requested and (datetime.now(timezone.utc) - started).total_seconds() > 86400:
            raise ValueError("Latest simulation is older than 24 hours; select its run ID explicitly")
        seen = set()
        for row in data["readings"]:
            if not isinstance(row, dict) or row.get("machine_id") not in ("M01", "M02", "M03") or type(row.get("cycle")) is not int or not 1 <= row["cycle"] <= data["cycles"] or row.get("scenario") not in ("normale", "critico") or not isinstance(row.get("reading"), dict):
                raise ValueError("Invalid reading")
            key = (row["cycle"], row["machine_id"])
            if key in seen:
                raise ValueError("Duplicate reading")
            seen.add(key)
            result, error = row.get("result"), row.get("error")
            if result is not None:
                if not isinstance(result, dict) or type(result.get("predicted_class")) is not int or result["predicted_class"] not in (0, 1) or type(result.get("failure_score")) not in (int, float) or not math.isfinite(result["failure_score"]) or not 0 <= result["failure_score"] <= 1 or error is not None:
                    raise ValueError("Invalid prediction")
            elif not isinstance(error, str) or not error:
                raise ValueError("Missing result or error")
        expected = data["cycles"] * 3
        if len(seen) > expected or (data["status"] == "completed" and (len(seen) != expected or any(row["error"] for row in data["readings"]))):
            raise ValueError("Inconsistent simulation status")
        return data
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Simulation unavailable or invalid") from exc
