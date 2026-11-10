from time import perf_counter

from strands import Agent, tool
from strands.models.ollama import OllamaModel


calls = []


@tool
def get_temperature(machine_id: str) -> str:
    """Restituisce la temperatura fittizia in gradi Celsius della macchina indicata."""
    value = "73"
    calls.append(("get_temperature", machine_id, value))
    print(f"\nTOOL: get_temperature(machine_id={machine_id!r}) -> {value} °C")
    return f"{value} °C"


@tool
def get_speed(machine_id: str) -> str:
    """Restituisce la velocità fittizia in giri al minuto della macchina indicata."""
    value = "1450"
    calls.append(("get_speed", machine_id, value))
    print(f"\nTOOL: get_speed(machine_id={machine_id!r}) -> {value} rpm")
    return f"{value} rpm"


def run_test(question: str, expected_tool: str, expected_value: str) -> bool:
    calls.clear()
    agent = Agent(
        model=OllamaModel(host="http://localhost:11434", model_id="granite4.1:8b", temperature=0),
        tools=[get_temperature, get_speed],
        system_prompt="Usa il tool appropriato per rispondere alle domande sulle macchine. Riporta il valore restituito dal tool.",
        callback_handler=None,
    )
    print(f"\nDOMANDA: {question}")
    start = perf_counter()
    result = agent(question)
    elapsed = perf_counter() - start
    response = str(result)
    tool_ok = calls == [(expected_tool, "M01", expected_value)]
    value_ok = expected_value in response
    print(f"RISPOSTA: {response}")
    print(f"LATENZA: {elapsed:.2f} s")
    print(f"VERIFICA TOOL E PARAMETRO: {'PASS' if tool_ok else 'FAIL'}; chiamate={calls!r}")
    print(f"VERIFICA VALORE NELLA RISPOSTA: {'PASS' if value_ok else 'FAIL'}")
    return tool_ok and value_ok


if __name__ == "__main__":
    results = [
        run_test("Qual è la temperatura della macchina M01?", "get_temperature", "73"),
        run_test("Qual è la velocità della macchina M01?", "get_speed", "1450"),
    ]
    raise SystemExit(0 if all(results) else 1)
