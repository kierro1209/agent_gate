import json
from pathlib import Path
from typing import Any, cast

import pytest

from adapters.fixture import FixtureAdapter
from adapters.mem0_adapter import SCOPE, Mem0Adapter, normalize_hits
from demo.chat import answer_turn, process_turn
from demo.multi_chat import answer_with_helper
from demo.run_before_after import STICKY_MARKER, fixture_score
from gate.criteria import load_criteria
from gate.formation import (
    FormationError,
    FunctionMemoryExtractor,
    FunctionMemoryReconciler,
    OpenAIMemoryExtractor,
    form_memories,
)
from gate.gate import gate_memories
from gate.judge import FunctionJudge
from gate.models import MemoryCandidate, TurnContext
from gate.perturb import perturb_variants
from loop.agent import build_prompt, handle_turn
from loop.multi_agent import consult_agent


@pytest.fixture
def setup_gate() -> tuple[Any, FixtureAdapter]:
    store = FixtureAdapter()
    store.seed_memories()
    return load_criteria(), store


def test_calendar_rejects_sticky_and_accepts_calendar(setup_gate: Any) -> None:
    criteria, store = setup_gate
    turn = TurnContext("calendar", criteria.agent_id, "What's tomorrow afternoon's schedule?")
    result = gate_memories(
        turn,
        store.search_candidates(turn, top_k=20),
        FunctionJudge(fixture_score),
        criteria=criteria,
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
    criteria = load_criteria()

    def score(_turn: TurnContext, memory: str) -> float:
        if not memory:
            return scores[0]
        if memory == "x":
            return scores[1]
        return scores[2]

    result = gate_memories(
        TurnContext("b", "a", "q"),
        [MemoryCandidate("m", "x")],
        FunctionJudge(score),
        criteria=criteria,
    )
    assert result.decisions[0].accepted is accepted


def test_cache_and_empty() -> None:
    criteria = load_criteria()
    calls: list[str] = []
    cache: dict[tuple[str, str], tuple[float, float, float]] = {}

    def score(_turn: TurnContext, memory: str) -> float:
        calls.append(memory)
        return 0.5

    judge = FunctionJudge(score)
    args = (TurnContext("c", "a", "q"), [MemoryCandidate("m", "x")], judge)
    gate_memories(*args, criteria=criteria, cache=cache)
    gate_memories(*args, criteria=criteria, cache=cache)
    assert len(calls) == 2 + len(perturb_variants("x", criteria.distractor))
    assert gate_memories(args[0], [], judge, criteria=criteria).decisions == []


def test_relevant_memories_have_positive_stability_with_fixture_scorer(setup_gate: Any) -> None:
    criteria, store = setup_gate
    turn = TurnContext(
        "calendar-stability",
        criteria.agent_id,
        "What's on my calendar tomorrow?",
    )
    result = gate_memories(
        turn, store.search_candidates(turn), FunctionJudge(fixture_score), criteria=criteria
    )
    calendar = next(
        decision for decision in result.decisions if decision.candidate_id == "calendar"
    )
    assert calendar.s_with > calendar.s_pert
    assert calendar.stability > 0


def test_normalize_preserves_hit() -> None:
    hit = {"id": "1", "memory": "fact", "score": 0.7}
    assert normalize_hits({"results": [hit]})[0].metadata == hit
    assert normalize_hits([hit])[0].text == "fact"


def test_mem0_scope_calls(tmp_path: Path) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    class Client:
        def add(self, text: str, **kw: Any) -> dict[str, str]:
            calls.append(("add", {"text": text, **kw}))
            return {"event_id": "evt-1"}

        def search(self, **kw: Any) -> dict[str, list[Any]]:
            calls.append(("search", kw))
            return {"results": []}

        def update(self, memory_id: str, **kw: Any) -> dict[str, Any]:
            calls.append(("update", {"memory_id": memory_id, **kw}))
            return {"id": memory_id, "memory": kw["text"], "metadata": kw.get("metadata", {})}

        def delete(self, memory_id: str) -> None:
            calls.append(("delete", {"memory_id": memory_id}))

        def get_all(self, **kw: Any) -> dict[str, list[Any]]:
            calls.append(("get_all", kw))
            return {"results": []}

    adapter = object.__new__(Mem0Adapter)
    adapter._client = Client()
    seed = tmp_path / "seed.json"
    seed.write_text('[{"text":"fact","metadata":{}}]', encoding="utf-8")
    adapter.seed_memories(str(seed))
    adapter.search_candidates(TurnContext("t", "a", "q"))
    adapter.find_related_memories("fact")
    adapter.add_memory("newer fact", {"kind": "project"})
    adapter.update_memory("memory-1", "updated fact", {"kind": "project"})
    adapter.delete_memory("memory-1")
    adapter.list_memories()
    adapter.inspect()
    add_call = next(call for call in calls if call[0] == "add")
    assert add_call[1]["infer"] is False
    assert add_call[1]["user_id"] == SCOPE["user_id"]
    assert "agent_id" not in add_call[1]
    read_calls = [call for call in calls if call[0] in {"search", "get_all"}]
    assert all(call[1]["filters"] == SCOPE for call in read_calls)
    assert any(call[0] == "update" for call in calls)
    assert any(call[0] == "delete" for call in calls)


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
    captured: list[str] = []

    def complete(prompt: str) -> str:
        captured.append(prompt)
        return "ok"

    handle_turn(
        TurnContext("i", criteria.agent_id, "What's on my calendar tomorrow?"),
        True,
        store,
        FunctionJudge(fixture_score),
        complete,
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
    monkeypatch.setattr("demo.chat.append_formation_decisions", lambda *_args: None)
    answer = answer_turn(
        TurnContext("chat", criteria.agent_id, "What's on my calendar tomorrow?"),
        True,
        store,
        FunctionJudge(fixture_score),
        lambda prompt: prompt,
        criteria,
        {},
        FunctionMemoryExtractor(),
        FunctionMemoryReconciler(),
    )
    output = capsys.readouterr().out
    assert "[REJECTED] sticky" in output
    assert "[ACCEPTED] calendar" in output
    assert STICKY_MARKER not in answer
    assert "Formed memory candidates:" in output


def test_formation_adds_new_memory_for_future_turn(setup_gate: Any) -> None:
    criteria, store = setup_gate
    turn = TurnContext("f1", criteria.agent_id, "I prefer tea over coffee.")
    result = form_memories(
        turn,
        "Understood.",
        store,
        FunctionMemoryExtractor(),
        FunctionMemoryReconciler(),
    )
    assert result.decisions[0].action == "added"
    hits = store.find_related_memories("tea preference", top_k=5)
    assert any("prefers tea over coffee" in memory.text.lower() for memory in hits)


def test_formation_rejects_ephemeral_memory(setup_gate: Any) -> None:
    criteria, store = setup_gate
    turn = TurnContext("f2", criteria.agent_id, "Today I had oatmeal for breakfast.")
    result = form_memories(
        turn,
        "Noted.",
        store,
        FunctionMemoryExtractor(),
        FunctionMemoryReconciler(),
    )
    assert result.decisions == []


def test_formation_skips_duplicate_memory(setup_gate: Any) -> None:
    criteria, store = setup_gate
    turn = TurnContext(
        "f3",
        criteria.agent_id,
        "I prefer calendar summaries grouped into morning, afternoon, and evening.",
    )
    result = form_memories(
        turn,
        "Understood.",
        store,
        FunctionMemoryExtractor(),
        FunctionMemoryReconciler(),
    )
    assert result.decisions[0].action == "skipped"


def test_formation_updates_conflicting_preference(setup_gate: Any) -> None:
    criteria, store = setup_gate
    store.add_memory("The user prefers coffee.", {"kind": "preference"})
    turn = TurnContext("f4", criteria.agent_id, "Actually, I prefer tea.")
    result = form_memories(
        turn,
        "Understood.",
        store,
        FunctionMemoryExtractor(),
        FunctionMemoryReconciler(),
    )
    assert result.decisions[0].action == "updated"
    assert any(memory.text == "The user prefers tea." for memory in store.list_memories())


def test_handle_turn_logs_formations(setup_gate: Any, tmp_path: Path) -> None:
    criteria, store = setup_gate
    handle_turn(
        TurnContext("f5", criteria.agent_id, "I prefer tea."),
        True,
        store,
        FunctionJudge(fixture_score),
        lambda prompt: "ok",
        criteria,
        log_path=tmp_path / "d.jsonl",
        formation_log_path=tmp_path / "f.jsonl",
        cache={},
        memory_extractor=FunctionMemoryExtractor(),
        formation_reconciler=FunctionMemoryReconciler(),
    )
    record = json.loads((tmp_path / "f.jsonl").read_text().splitlines()[0])
    assert record["action"] == "added"
    assert record["candidateText"] == "The user prefers tea."


def test_process_turn_prints_answer_when_formation_fails(
    setup_gate: Any, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    criteria, store = setup_gate
    monkeypatch.setattr("demo.chat.append_decisions", lambda *_args: None)

    def fail_formation(*_args: Any, **_kwargs: Any) -> None:
        raise FormationError("extractor", "response missing 'memories' field")

    monkeypatch.setattr("demo.chat.run_formation", fail_formation)
    process_turn(
        TurnContext("chat-fail", criteria.agent_id, "What's on my calendar tomorrow?"),
        True,
        store,
        FunctionJudge(fixture_score),
        lambda _prompt: "calendar answer",
        criteria,
        {},
        FunctionMemoryExtractor(),
        FunctionMemoryReconciler(),
    )
    captured = capsys.readouterr()
    assert "Agent: calendar answer" in captured.out
    assert "Memory formation error: extractor: response missing 'memories' field" in captured.err


def test_consult_agent_gates_shared_memories_and_logs(
    setup_gate: Any,
    tmp_path: Path,
) -> None:
    criteria, store = setup_gate
    captured: list[str] = []

    def complete(prompt: str) -> str:
        captured.append(prompt)
        return "helper note"

    result = consult_agent(
        criteria.agent_id,
        TurnContext("share-1", "helper", "What's on my calendar tomorrow?"),
        True,
        store,
        FunctionJudge(fixture_score),
        complete,
        criteria,
        log_path=tmp_path / "share.jsonl",
        cache={},
    )
    assert result.response == "helper note"
    assert STICKY_MARKER not in captured[0]
    assert any(memory.id == "calendar" for memory in result.shared)
    assert all(memory.id != "sticky" for memory in result.shared)
    record = json.loads((tmp_path / "share.jsonl").read_text().splitlines()[0])
    assert record["senderAgentId"] == criteria.agent_id
    assert record["recipientAgentId"] == "helper"
    assert {"utility", "stability", "criteriaVersion"} <= record.keys()


def test_multi_chat_prints_explicit_agent_transcript(
    setup_gate: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: Any,
) -> None:
    criteria, store = setup_gate
    monkeypatch.setattr("demo.multi_chat.append_decisions", lambda *_args: None)
    monkeypatch.setattr("demo.multi_chat.append_formation_decisions", lambda *_args: None)
    answer = answer_with_helper(
        TurnContext("multi-chat", criteria.agent_id, "What's on my calendar tomorrow?"),
        True,
        True,
        store,
        FunctionJudge(fixture_score),
        lambda prompt: prompt,
        criteria,
        {},
        FunctionMemoryExtractor(),
        FunctionMemoryReconciler(),
    )
    output = capsys.readouterr().out
    assert "=== USER -> SCHEDULING AGENT ===" in output
    assert "=== SCHEDULING AGENT -> HEALTH COACH ===" in output
    assert "=== HEALTH COACH -> SCHEDULING AGENT ===" in output
    assert "=== SCHEDULING AGENT FINAL ANSWER ===" in output
    assert "Scheduling agent task for health coach:" in output
    assert "Health coach (health-coach-agent):" in output
    assert STICKY_MARKER not in answer


def test_multi_chat_routes_health_requests_to_health_coach(
    setup_gate: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: Any,
) -> None:
    criteria, store = setup_gate
    monkeypatch.setattr("demo.multi_chat.append_decisions", lambda *_args: None)
    monkeypatch.setattr("demo.multi_chat.append_formation_decisions", lambda *_args: None)
    answer = answer_with_helper(
        TurnContext(
            "multi-health", criteria.agent_id, "Plan my gym sessions and protein target this week."
        ),
        True,
        True,
        store,
        FunctionJudge(fixture_score),
        lambda prompt: prompt,
        criteria,
        {},
        FunctionMemoryExtractor(),
        FunctionMemoryReconciler(),
    )
    output = capsys.readouterr().out
    assert "=== USER -> HEALTH COACH ===" in output
    assert "=== HEALTH COACH -> SCHEDULING AGENT ===" in output
    assert "=== SCHEDULING AGENT -> HEALTH COACH ===" in output
    assert "=== HEALTH COACH FINAL ANSWER ===" in output
    assert STICKY_MARKER not in answer


def test_openai_memory_extractor_requires_memories_field() -> None:
    extractor = object.__new__(OpenAIMemoryExtractor)
    extractor._model = "test-model"

    class Client:
        class Chat:
            class Completions:
                @staticmethod
                def create(**_kwargs: Any) -> Any:
                    class Message:
                        content = '{"items":[]}'

                    class Choice:
                        message = Message()

                    class Response:
                        choices = [Choice()]

                    return Response()

            completions = Completions()

        chat = Chat()

    extractor._client = cast(Any, Client())
    with pytest.raises(FormationError, match="extractor: response missing 'memories' field"):
        extractor.extract(TurnContext("f6", "agent", "I prefer tea."), "Understood.")
