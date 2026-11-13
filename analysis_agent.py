"""Analyze a specific local simulation with a Strands tool and Ollama."""
import argparse
import json
import threading
from pathlib import Path

from strands import Agent, tool
from strands.hooks import AfterToolCallEvent, BeforeToolCallEvent
from strands.models.ollama import OllamaModel

from machine_simulator import MACHINES
from simulation_store import ROOT, atomic_json, now, read_run


INSUFFICIENT_DATA = "Non ci sono dati sufficienti per l'analisi: nessuna lettura ha una classificazione valida."


def summarize(data):
    rows = data["readings"]
    recorded = {(row["cycle"], row["machine_id"]) for row in rows}
    valid_count = sum(row.get("result") is not None for row in rows)
    missing = [
        {"cycle": cycle, "machine_id": machine_id}
        for cycle in range(1, data["cycles"] + 1)
        for machine_id in MACHINES
        if (cycle, machine_id) not in recorded
    ]
    return {"run_id": data["run_id"], "started_at": data["started_at"], "status": data["status"],
            "expected_readings": data["cycles"] * len(MACHINES), "recorded_readings": len(rows),
            "valid_readings": valid_count, "missing_readings": missing,
            "data_sufficiency": INSUFFICIENT_DATA if valid_count == 0 else None,
            "results": [{"cycle": r["cycle"], "machine_id": r["machine_id"],
                         "scenario": r["scenario"], "result": r.get("result"), "error": r.get("error")}
                        for r in rows],
            "note": "Dati sintetici; lo scenario non è la classe ML. Score non necessariamente calibrato e non temporale."}

#classe che abbiamo creato appositamente per registrare cosa farà l'agente
class Trace:
    def __init__(self, directory):
        directory.mkdir(exist_ok=False)
        self.directory = directory
        self.count = 0
        self.lock = threading.Lock()

    #Ogni volta che chiamiamo trace.emit(), registriamo un evento in un file JSON.
    def emit(self, event, **details):
        with self.lock: #un solo thread alla volta possa eseguire quel blocco di codice usando lo stesso lock.
            self.count += 1 #soprattutto per creare nomi ordinati
            #001_agent_started.json
            #002_tool_call.json
            payload = {"timestamp": now(), "event": event, **details}
            atomic_json(self.directory / f"{self.count:03d}_{event}.json", payload)
            print(json.dumps(payload, ensure_ascii=False), flush=True)

    def register_hooks(self, registry, **kwargs):
        registry.add_callback(BeforeToolCallEvent, self.before_tool)
        registry.add_callback(AfterToolCallEvent, self.after_tool)

    def before_tool(self, event):
        self.emit("tool_call", tool=event.tool_use["name"], arguments=event.tool_use["input"])

    def after_tool(self, event):
        content = event.result.get("content")
        if isinstance(content, list):
            converted = []
            for block in content:
                if isinstance(block, dict) and set(block) == {"text"} and isinstance(block["text"], str):
                    try:
                        converted.append(json.loads(block["text"]))
                    except (ValueError, TypeError):
                        converted.append(block["text"])
                else:
                    converted.append(block)
            content = converted[0] if len(converted) == 1 else converted
        self.emit("tool_result", tool=event.tool_use["name"], status=event.result.get("status"),
                  result=content, error=type(event.exception).__name__ if event.exception else None)


def analyze(question, run_id="latest", root=ROOT):
    data = read_run(run_id, root)
    summary = summarize(data)
    selected_id = data["run_id"]
    folder = root / selected_id
    trace = Trace(folder / "agent_trace")
    trace.emit("agent_started", question=question, run_id=selected_id, simulation_status=data["status"])
    called = [] #Verificare che il nostro agente abbia effettivamente chiamato il tool per leggere i risultati.

    @tool
    def get_simulation_results() -> str:
        """Return recorded ML classifications and synthetic scenarios for the selected simulation."""
        called.append(True)
        return json.dumps(summary, ensure_ascii=False)

    try:
        agent = Agent(model=OllamaModel(host="http://localhost:11434", model_id="granite4.1:8b", temperature=0),
                      tools=[get_simulation_results], hooks=[trace], callback_handler=None,
                      system_prompt="Usa sempre il tool per leggere la simulazione selezionata. Non inventare dati. "
                      "La classe ML e lo scenario sintetico sono distinti; lo score non è calibrato né una previsione temporale. "
                      "Indica lo stato incompleto o fallito e le letture mancanti quando applicabile. "
                      "Le letture mancanti o con errori non hanno alcuna classificazione. "
                      "Se valid_readings è zero, dichiara che i dati sono insufficienti per l'analisi.")
        answer = str(agent(question))
        if not called:
            raise RuntimeError("Agent did not call the simulation tool")
        if summary["valid_readings"] == 0:
            answer = (f"Stato della simulazione: {summary['status']}. "
                      f"Letture registrate: {summary['recorded_readings']}/{summary['expected_readings']}. "
                      f"Letture mancanti: {len(summary['missing_readings'])}. {INSUFFICIENT_DATA}")
        (folder / "report.md").write_text(f"# Analysis {selected_id}\n\n{answer}\n", encoding="utf-8")
        trace.emit("agent_completed", run_id=selected_id)
        return answer
    except Exception as exc:
        trace.emit("agent_error", error=type(exc).__name__)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", help="Question about the selected simulation")
    parser.add_argument("--run-id", default="latest")
    args = parser.parse_args()
    print(analyze(args.question, args.run_id))


if __name__ == "__main__":
    main()
