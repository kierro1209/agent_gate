import sys
from collections.abc import Callable

from adapters.mem0_adapter import MemoryStore
from connectors.google_workspace import GoogleWorkspace, SourceRecord, route_sources
from demo.run_before_after import runtime
from gate.criteria import Criteria, load_criteria
from gate.gate import ScoreCache, gate_memories
from gate.judge import Judge
from gate.models import GateResult, MemoryCandidate, TurnContext
from loop.agent import append_decisions, build_prompt


def print_retrieved(candidates: list[MemoryCandidate]) -> None:
    print("\nRetrieved memories:")
    if not candidates:
        print("  (none)")
    for candidate in candidates:
        score = candidate.metadata.get("score", "n/a")
        print(f"  [{candidate.id}] score={score}  {candidate.text}")


def print_decisions(result: GateResult) -> None:
    print("\nCMI decisions:")
    if not result.decisions:
        print("  (none)")
    for decision in result.decisions:
        label = "ACCEPTED" if decision.accepted else "REJECTED"
        print(f"  [{label}] {decision.candidate_id}")
        print(
            f"    s_no={decision.s_no:.3f}  s_with={decision.s_with:.3f}  "
            f"s_pert={decision.s_pert:.3f}"
        )
        print(f"    utility={decision.utility:.3f}  stability={decision.stability:.3f}")
        print(f"    {decision.reason}")


def answer_turn(
    turn: TurnContext,
    gate_enabled: bool,
    store: MemoryStore,
    judge: Judge,
    complete: Callable[[str], str],
    criteria: Criteria,
    cache: ScoreCache,
    workspace: GoogleWorkspace | None = None,
) -> str:
    candidates = store.search_candidates(turn, criteria.top_k)
    print_retrieved(candidates)
    selected = candidates
    if gate_enabled:
        result = gate_memories(turn, candidates, judge, criteria=criteria, cache=cache)
        print_decisions(result)
        append_decisions("logs/decisions.jsonl", turn, result.decisions)
        selected = result.accepted
    else:
        print("\nCMI gate: OFF (all retrieved memories bypassed)")

    print("\nInjected memories:")
    if not selected:
        print("  (none)")
    for candidate in selected:
        print(f"  [{candidate.id}] {candidate.text}")
    try:
        sources = route_sources(workspace, turn.user_message) if workspace else []
    except Exception as exc:
        print(f"\nGoogle source error: {exc}", file=sys.stderr)
        sources = []
    print("\nLive Google sources:")
    if not sources:
        print("  (none)")
    for source in sources:
        print(f"  [{source.source}:{source.id}] {source.text}")
    return complete(build_prompt(turn.user_message, selected, format_sources(sources)))


def format_sources(sources: list[SourceRecord]) -> str:
    return "\n".join(
        f"- source={item.source} id={item.id} observed_at={item.observed_at}: {item.text}"
        for item in sources
    )


def main() -> int:
    criteria = load_criteria()
    store, judge, complete, mode = runtime(criteria)
    store.seed_memories()
    gate_enabled = True
    cache: dict[tuple[str, str], tuple[float, float, float]] = {}
    workspace = GoogleWorkspace()
    turn_number = 0
    print(f"Memory Gate chat ({mode}); gate=ON")
    print("Commands: /gate on, /gate off, /quit")

    while True:
        try:
            message = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not message:
            continue
        if message == "/quit":
            return 0
        if message in {"/gate on", "/gate off"}:
            gate_enabled = message.endswith("on")
            print(f"CMI gate={'ON' if gate_enabled else 'OFF'}")
            continue
        if message.startswith("/"):
            print("Unknown command. Use /gate on, /gate off, or /quit.")
            continue

        turn_number += 1
        turn = TurnContext(f"chat-{turn_number}", criteria.agent_id, message)
        try:
            answer = answer_turn(
                turn, gate_enabled, store, judge, complete, criteria, cache, workspace
            )
        except Exception as exc:
            print(f"\nError: {exc}", file=sys.stderr)
            continue
        print(f"\nAgent: {answer}")


if __name__ == "__main__":
    sys.exit(main())
