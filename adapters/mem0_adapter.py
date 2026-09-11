import json
from pathlib import Path
from typing import Any, Protocol

from gate.models import MemoryCandidate, TurnContext

DEMO_USER, DEMO_AGENT = "kiersten-demo", "memory-gate-v0"
SCOPE = {"user_id": DEMO_USER, "agent_id": DEMO_AGENT}


class MemoryStore(Protocol):
    def seed_memories(self, path: str = "demo/seed_memories.json") -> None: ...
    def search_candidates(self, turn: TurnContext, top_k: int = 5) -> list[MemoryCandidate]: ...
    def inspect(self) -> list[dict[str, Any]]: ...


class Mem0Adapter:
    def __init__(self, api_key: str) -> None:
        from mem0 import MemoryClient  # type: ignore[import-untyped]

        self._client = MemoryClient(api_key=api_key)

    def seed_memories(self, path: str = "demo/seed_memories.json") -> None:
        seeds: list[dict[str, Any]] = json.loads(Path(path).read_text(encoding="utf-8"))
        for item in seeds:
            self._client.add(
                item["text"],
                user_id=DEMO_USER,
                agent_id=DEMO_AGENT,
                infer=False,
                metadata={**item.get("metadata", {}), "seed": True},
            )

    def search_candidates(self, turn: TurnContext, top_k: int = 5) -> list[MemoryCandidate]:
        return normalize_hits(
            self._client.search(
                query=turn.user_message,
                filters=SCOPE,
                top_k=top_k,
                threshold=0.1,
                rerank=False,
            )
        )

    def inspect(self) -> list[dict[str, Any]]:
        return extract_hits(self._client.get_all(filters=SCOPE, page=1, page_size=50))


def extract_hits(raw: Any) -> list[dict[str, Any]]:
    hits = raw.get("results", []) if isinstance(raw, dict) else raw
    if not isinstance(hits, list):
        raise ValueError("expected list or results envelope")
    return [hit for hit in hits if isinstance(hit, dict)]


def normalize_hits(raw: Any) -> list[MemoryCandidate]:
    return [
        MemoryCandidate(
            str(hit.get("id") or hit.get("memory_id") or ""),
            str(hit.get("memory") or hit.get("text") or ""),
            str(hit.get("created_at") or ""),
            dict(hit),
        )
        for hit in extract_hits(raw)
    ]
