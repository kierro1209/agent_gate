import os
import sys
from collections.abc import Callable

from dotenv import load_dotenv

from adapters.fixture import FixtureAdapter
from adapters.mem0_adapter import Mem0Adapter, MemoryStore
from gate.criteria import Criteria, load_criteria
from gate.formation import (
    FORMATION_JUDGE_PROMPT,
    FunctionMemoryExtractor,
    FunctionMemoryReconciler,
    MemoryExtractor,
    MemoryReconciler,
    OpenAIMemoryExtractor,
    OpenAIMemoryReconciler,
)
from gate.gate import gate_memories
from gate.judge import FunctionJudge, Judge, OpenAIJudge
from gate.models import TurnContext
from loop.agent import build_prompt

STICKY_MARKER = "CAP theorem"


def fixture_score(turn: TurnContext, memory: str) -> float:
    query, text = turn.user_message.lower(), memory.lower()
    if not memory:
        return 0.4
    systems = any(word in query for word in ("cap", "kafka", "distributed"))
    calendar = any(word in query for word in ("calendar", "schedule", "tomorrow"))
    relevant = (systems and any(x in text for x in ("cap", "kafka"))) or (
        calendar and any(x in text for x in ("appointment", "calendar"))
    )
    if relevant:
        return 0.8 if "unverified context" in text else 0.9
    return 0.35 if "unverified context" in text else 0.3


def fixture_formation_score(turn: TurnContext, memory: str) -> float:
    context = turn.user_message.lower()
    text = memory.lower()
    if not memory:
        return 0.3
    if "unverified context" in text:
        if any(word in context for word in ("prefer", "every", "project", "want")):
            return 0.55
        return 0.2
    if any(word in text for word in ("prefers", "every", "recurring", "project", "wants to")):
        return 0.9
    if any(word in text for word in ("tomorrow", "today")):
        return 0.2
    return 0.35


def live_complete(key: str, model: str) -> Callable[[str], str]:
    from openai import OpenAI

    client = OpenAI(api_key=key)

    def complete(prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content or ""

    return complete


def runtime(
    criteria: Criteria,
) -> tuple[
    MemoryStore,
    Judge,
    Callable[[str], str],
    str,
    MemoryExtractor,
    Judge,
    MemoryReconciler,
]:
    load_dotenv()
    mem0_key, openai_key = os.getenv("MEM0_API_KEY"), os.getenv("OPENAI_API_KEY")
    if bool(mem0_key) != bool(openai_key):
        raise RuntimeError("set both MEM0_API_KEY and OPENAI_API_KEY, or neither")
    if mem0_key and openai_key:
        return (
            Mem0Adapter(mem0_key),
            OpenAIJudge(openai_key, criteria.judge_model, criteria.judge_prompt),
            live_complete(openai_key, criteria.judge_model),
            "live",
            OpenAIMemoryExtractor(openai_key, criteria.judge_model),
            OpenAIJudge(openai_key, criteria.judge_model, FORMATION_JUDGE_PROMPT),
            OpenAIMemoryReconciler(openai_key, criteria.judge_model),
        )
    return (
        FixtureAdapter(),
        FunctionJudge(fixture_score),
        lambda prompt: prompt,
        "fixture",
        FunctionMemoryExtractor(),
        FunctionJudge(fixture_formation_score),
        FunctionMemoryReconciler(),
    )


def main() -> int:
    criteria = load_criteria()
    store, judge, complete, mode, _extractor, _formation_judge, _reconciler = runtime(criteria)
    store.seed_memories()
    turn = TurnContext(
        "calendar-demo", criteria.agent_id, "What's on my calendar tomorrow afternoon?"
    )
    candidates = store.search_candidates(turn, criteria.top_k)
    before = build_prompt(turn.user_message, candidates)
    result = gate_memories(turn, candidates, judge, criteria=criteria, cache={})
    after = build_prompt(turn.user_message, result.accepted)
    print(f"mode={mode}")
    print("seeded ids:", ", ".join(str(item.get("id", "")) for item in store.inspect()))
    print("search hits:", ", ".join(item.id for item in candidates))
    print(f"gate off sticky leak={'YES' if STICKY_MARKER in before else 'NO'}")
    print(f"gate on sticky leak={'YES' if STICKY_MARKER in after else 'NO'}")
    print("candidate | s_no | s_with | s_pert | utility | stability | accepted")
    for d in result.decisions:
        print(
            f"{d.candidate_id} | {d.s_no:.2f} | {d.s_with:.2f} | {d.s_pert:.2f} | "
            f"{d.utility:.2f} | {d.stability:.2f} | {d.accepted}"
        )
    print("\nBEFORE ANSWER\n", complete(before))
    print("\nAFTER ANSWER\n", complete(after))
    return int(STICKY_MARKER in after)


if __name__ == "__main__":
    sys.exit(main())
