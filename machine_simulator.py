"""Generate synthetic machine readings and send them to the inference API."""

import argparse
import math
import random
import time
from urllib.parse import urlsplit

import httpx
from simulation_store import atomic_json, now, start_run


MACHINES = {"M01": "L", "M02": "M", "M03": "H"}


def generate_reading(rng, machine_type):
    critical = rng.random() < 0.2
    air = round(rng.uniform(299, 302.7), 1)
    if critical:
        process = round(air + rng.uniform(7.0, 8.4), 1)
        speed = rng.randint(1200, 1370)
        torque = round(rng.uniform(45, 65), 1)
        wear = rng.randint(180, 250)
    else:
        process = round(air + rng.uniform(9, 11), 1)
        speed = rng.randint(1450, 1900)
        torque = round(rng.uniform(25, 48), 1)
        wear = rng.randint(0, 160)
    return ("critico" if critical else "normale"), {
        "type": machine_type, "air_temperature_k": air, "process_temperature_k": process,
        "rotational_speed_rpm": speed, "torque_nm": torque, "tool_wear_min": wear,
    }


def run(client, api_url, interval, cycles, seed, snapshot=None):
    rng = random.Random(seed)
    url = api_url.rstrip("/") + "/predict"
    print("Letture sintetiche; failure_score non è una probabilità calibrata né una previsione temporale.")
    successes = failures = 0
    for cycle in range(cycles):
        for machine_id, machine_type in MACHINES.items():
            scenario, payload = generate_reading(rng, machine_type)
            row = {"cycle": cycle + 1, "machine_id": machine_id, "scenario": scenario,
                   "reading": payload, "result": None, "error": None}
            try:
                response = client.post(url, json=payload)
                response.raise_for_status()
                result = response.json()
                if not isinstance(result, dict):
                    raise ValueError("Invalid response")
                predicted_class = result.get("predicted_class")
                score = result.get("failure_score")
                if (type(predicted_class) is not int or predicted_class not in (0, 1)
                        or type(score) not in (int, float) or not math.isfinite(score)
                        or not 0 <= score <= 1):
                    raise ValueError("Invalid response")
                print(f"{machine_id} scenario={scenario} classe={result['predicted_class']} "
                      f"failure_score={result['failure_score']}")
                successes += 1
                row["result"] = {"predicted_class": predicted_class, "failure_score": score}
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
                print(f"{machine_id} scenario={scenario} errore={type(exc).__name__}")
                failures += 1
                row["error"] = type(exc).__name__
            if snapshot is not None:
                folder, data = snapshot
                data["readings"].append(row)
                atomic_json(folder / "simulation.json", data)
        if cycle + 1 < cycles:
            time.sleep(interval)
    print(f"Riepilogo: riuscite={successes} fallite={failures}")
    if snapshot is not None:
        data.update(status="failed" if failures else "completed", finished_at=now(),
                    succeeded=successes, failed=failures)
        atomic_json(folder / "simulation.json", data)
    return 1 if failures else 0


def valid_api_url(value):
    try:
        parsed = urlsplit(value)
        url = httpx.URL(value)
        return (parsed.scheme in ("http", "https") and bool(parsed.hostname)
                and bool(url.host) and parsed.username is None and parsed.password is None
                and not parsed.query and not parsed.fragment and parsed.port != 0
                and not any(c.isspace() for c in value))
    except (ValueError, httpx.InvalidURL):
        return False


def main():
    parser = argparse.ArgumentParser(description="Simulatore di letture sintetiche AI4I")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--interval", type=float, default=5.0, help="Secondi tra cicli")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not valid_api_url(args.api_url):
        parser.error("api-url deve essere un URL HTTP/HTTPS valido con host")
    if not math.isfinite(args.interval) or args.interval < 0 or args.cycles < 1:
        parser.error("interval deve essere finito e >= 0; cycles deve essere >= 1")
    with httpx.Client(timeout=10.0) as client:
        snapshot = start_run(args.cycles, args.seed)
        print(f"Simulazione: {snapshot[1]['run_id']} ({snapshot[0]})")
        return run(client, args.api_url, args.interval, args.cycles, args.seed, snapshot)


if __name__ == "__main__":
    raise SystemExit(main())
