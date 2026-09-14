import json
from pathlib import Path
from typing import Any

import pytest

from adapters.fixture import FixtureAdapter
from adapters.mem0_adapter import SCOPE, Mem0Adapter, normalize_hits
from demo.chat import answer_turn
from demo.run_before_after import STICKY_MARKER, fixture_score
from gate.criteria import load_criteria
from gate.gate import gate_memories
from gate.judge import FunctionJudge
from gate.models import MemoryCandidate, TurnContext
from loop.agent import build_prompt, handle_turn


@pytest.fixture
def setup_gate() -> tuple[Any, FixtureAdapter]:
    store = FixtureAdapter()
    store.seed_memories()
    return load_criteria(), store


def test_calendar_rejects_sticky_and_accepts_calendar(setup_gate: Any) -> None:
    criteria, store = setup_gate
    turn = TurnContext("calendar", criteria.agent_id, "What's tomorrow afternoon's schedule?")
    result = gate_memories(
        turn, store.search_candidates(turn), FunctionJudge(fixture_score), criteria=criteria
    )
    assert "sticky" in {x.id for x in result.rejected}
    assert "calendar" in {x.id for x in result.accepted}


def test_systems_accepts_sticky(setup_gate: Any) -> None:
    criteria, store = setup_gate
    turn = TurnContext("systems", criteria.agent_id, "Explain CAP and Kafka.")
    result = gate_memories(
        turn, store.search_candidates(turn), FunctionJudge(fixture_score), criteria=criteria
    )
    assert "sticky" in {x.id for x in result.accepted}


@pytest.mark.parametrize(
    "scores,accepted", [((0.5, 0.5, 0.5), False), ((0.5, 0.6, 0.6), True), ((0.5, 0.6, 0.7), False)]
)
def test_boundaries(scores: tuple[float, ...], accepted: bool) -> None:
    values = iter(scores)
    criteria = load_criteria()
    result = gate_memories(
        TurnContext("b", "a", "q"),
        [MemoryCandidate("m", "x")],
        FunctionJudge(lambda _t, _m: next(values)),
        criteria=criteria,
    )
    assert result.decisions[0].accepted is accepted


def test_cache_and_empty() -> None:
    criteria, calls, cache = load_criteria(), [], {}
    judge = FunctionJudge(lambda _t, m: calls.append(m) or 0.5)
    args = (TurnContext("c", "a", "q"), [MemoryCandidate("m", "x")], judge)
    gate_memories(*args, criteria=criteria, cache=cache)
    gate_memories(*args, criteria=criteria, cache=cache)
    assert len(calls) == 3
    assert gate_memories(args[0], [], judge, criteria=criteria).decisions == []


def test_normalize_preserves_hit() -> None:
    hit = {"id": "1", "memory": "fact", "score": 0.7}
    assert normalize_hits({"results": [hit]})[0].metadata == hit
    assert normalize_hits([hit])[0].text == "fact"


def test_mem0_scope_calls(tmp_path: Path) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    class Client:
        def add(self, text: str, **kw: Any) -> None:
            calls.append(("add", {"text": text, **kw}))

        def search(self, **kw: Any) -> dict[str, list[Any]]:
            calls.append(("search", kw))
            return {"results": []}

        def get_all(self, **kw: Any) -> dict[str, list[Any]]:
            calls.append(("get_all", kw))
            return {"results": []}

    adapter = object.__new__(Mem0Adapter)
    adapter._client = Client()
    seed = tmp_path / "seed.json"
    seed.write_text('[{"text":"fact","metadata":{}}]', encoding="utf-8")
    adapter.seed_memories(str(seed))
    adapter.search_candidates(TurnContext("t", "a", "q"))
    adapter.inspect()
    add_call = next(call for call in calls if call[0] == "add")
    assert add_call[1]["infer"] is False
    assert add_call[1]["user_id"] == SCOPE["user_id"]
    assert add_call[1]["agent_id"] == SCOPE["agent_id"]
    read_calls = [call for call in calls if call[0] in {"search", "get_all"}]
    assert all(call[1]["filters"] == SCOPE for call in read_calls)


def test_mem0_seed_skips_existing_exact_text(tmp_path: Path) -> None:
    added: list[str] = []

    class Client:
        def add(self, text: str, **_kw: Any) -> None:
            added.append(text)

        def get_all(self, **_kw: Any) -> dict[str, list[dict[str, str]]]:
            return {"results": [{"memory": "existing fact"}]}

    adapter = object.__new__(Mem0Adapter)
    adapter._client = Client()
    seed = tmp_path / "seed.json"
    seed.write_text(
        '[{"text":"existing fact","metadata":{}},{"text":"new fact","metadata":{}}]',
        encoding="utf-8",
    )
    adapter.seed_memories(str(seed))
    assert added == ["new fact"]


def test_injection_and_log(setup_gate: Any, tmp_path: Path) -> None:
    criteria, store = setup_gate
    captured = []
    handle_turn(
        TurnContext("i", criteria.agent_id, "What's on my calendar tomorrow?"),
        True,
        store,
        FunctionJudge(fixture_score),
        lambda p: captured.append(p) or "ok",
        criteria,
        log_path=tmp_path / "d.jsonl",
    )
    assert STICKY_MARKER not in captured[0]
    record = json.loads((tmp_path / "d.jsonl").read_text().splitlines()[0])
    assert {"utility", "stability", "criteriaVersion"} <= record.keys()
    assert STICKY_MARKER in build_prompt(
        "calendar", store.search_candidates(TurnContext("o", "a", "q"))
    )


def test_chat_shows_decisions_and_excludes_sticky(
    setup_gate: Any, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    criteria, store = setup_gate
    monkeypatch.setattr("demo.chat.append_decisions", lambda *_args: None)
    answer = answer_turn(
        TurnContext("chat", criteria.agent_id, "What's on my calendar tomorrow?"),
        True,
        store,
        FunctionJudge(fixture_score),
        lambda prompt: prompt,
        criteria,
        {},
    )
    output = capsys.readouterr().out
    assert "[REJECTED] sticky" in output
    assert "[ACCEPTED] calendar" in output
    assert STICKY_MARKER not in answer
