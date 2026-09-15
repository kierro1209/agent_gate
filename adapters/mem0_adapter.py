import hashlib
import json
from pathlib import Path
from typing import Any, Protocol

from gate.models import MemoryCandidate, TurnContext

DEMO_USER, DEMO_AGENT = "kiersten-demo", "memory-gate-v0"
SCOPE = {"user_id": DEMO_USER}


class MemoryStore(Protocol):
    def seed_memories(self, path: str = "demo/seed_memories.json") -> None: ...
    def search_candidates(self, turn: TurnContext, top_k: int = 5) -> list[MemoryCandidate]: ...
    def find_related_memories(self, query: str, top_k: int = 5) -> list[MemoryCandidate]: ...
    def add_memory(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryCandidate: ...
    def update_memory(
        self,
        memory_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryCandidate: ...
    def delete_memory(self, memory_id: str) -> None: ...
    def list_memories(self) -> list[MemoryCandidate]: ...
    def inspect(self) -> list[dict[str, Any]]: ...


class Mem0Adapter:
    def __init__(self, api_key: str) -> None:
        from mem0 import MemoryClient  # type: ignore[import-untyped]

        self._client = MemoryClient(api_key=api_key)

    def seed_memories(self, path: str = "demo/seed_memories.json") -> None:
        seeds: list[dict[str, Any]] = json.loads(Path(path).read_text(encoding="utf-8"))
        existing_texts = {str(hit.get("memory") or hit.get("text") or "") for hit in self.inspect()}
        for item in seeds:
            if item["text"] in existing_texts:
                continue
            self._client.add(
                item["text"],
                user_id=DEMO_USER,
                infer=False,
                metadata={**item.get("metadata", {}), "seed": True},
            )
            existing_texts.add(item["text"])

    def search_candidates(self, turn: TurnContext, top_k: int = 5) -> list[MemoryCandidate]:
        return self.find_related_memories(turn.user_message, top_k)

    def find_related_memories(self, query: str, top_k: int = 5) -> list[MemoryCandidate]:
        return normalize_hits(
            self._client.search(
                query=query,
                filters=SCOPE,
                top_k=top_k,
                threshold=0.0,
                rerank=False,
            )
        )

    def add_memory(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryCandidate:
        response = self._client.add(
            text,
            user_id=DEMO_USER,
            infer=False,
            metadata={**(metadata or {}), "seed": False},
        )
        event_id = ""
        if isinstance(response, dict):
            event_id = str(response.get("event_id") or "")
        if not event_id:
            digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
            event_id = f"pending:{digest}"
        return MemoryCandidate(event_id, text, metadata=metadata or {})

    def update_memory(
        self,
        memory_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryCandidate:
        payload: dict[str, Any] = {"text": text}
        if metadata is not None:
            payload["metadata"] = metadata
        updated = self._client.update(memory_id=memory_id, **payload)
        if isinstance(updated, dict):
            return normalize_hits([updated])[0]
        return MemoryCandidate(memory_id, text, metadata=metadata or {})

    def delete_memory(self, memory_id: str) -> None:
        self._client.delete(memory_id=memory_id)

    def list_memories(self) -> list[MemoryCandidate]:
        return normalize_hits(self._client.get_all(filters=SCOPE, page=1, page_size=200))

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
