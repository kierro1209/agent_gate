import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from adapters.fixture import FixtureAdapter
from demo.run_before_after import fixture_score
from gate.criteria import load_criteria
from gate.gate import gate_memories
from gate.judge import FunctionJudge
from gate.models import TurnContext

CASES = [
    ("calendar", "What's on my calendar tomorrow afternoon?"),
    ("systems", "Explain CAP and Kafka."),
    ("nutrition", "What should I eat after the gym?"),
    ("mixed", "What's tomorrow's schedule, and explain CAP?"),
    ("empty", "No memories."),
]


def run(iterations: int) -> dict[str, object]:
    criteria, judge, store = load_criteria(), FunctionJudge(fixture_score), FixtureAdapter()
    store.seed_memories()
    samples, accepted = [], 0
    for iteration in range(iterations):
        for name, message in CASES:
            turn = TurnContext(f"{name}-{iteration}", criteria.agent_id, message)
            candidates = [] if name == "empty" else store.search_candidates(turn)
            started = time.perf_counter()
            result = gate_memories(turn, candidates, judge, criteria=criteria)
            samples.append((time.perf_counter() - started) * 1000)
            accepted += len(result.accepted)
    ordered = sorted(samples)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "iterations": iterations,
        "evaluations": iterations * len(CASES),
        "median_ms": statistics.median(samples),
        "p95_ms": ordered[max(0, int(len(ordered) * 0.95) - 1)],
        "accepted_total": accepted,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    payload = json.dumps(run(args.iterations), indent=2)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
