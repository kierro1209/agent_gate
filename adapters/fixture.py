import json
import re
from pathlib import Path
from typing import Any

from gate.models import MemoryCandidate, TurnContext


class FixtureAdapter:
    def __init__(self) -> None:
        self._memories: list[MemoryCandidate] = []
        self._next_id = 1

    def seed_memories(self, path: str = "demo/seed_memories.json") -> None:
        seeds: list[dict[str, Any]] = json.loads(Path(path).read_text(encoding="utf-8"))
        self._memories = [
            MemoryCandidate(
                str(item["metadata"]["kind"]),
                str(item["text"]),
                metadata={**item["metadata"], "seed": True, "score": 0.5},
            )
            for item in seeds
        ]
        self._next_id = len(self._memories) + 1

    def search_candidates(self, turn: TurnContext, top_k: int = 5) -> list[MemoryCandidate]:
        return self.find_related_memories(turn.user_message, top_k)

    def find_related_memories(self, query: str, top_k: int = 5) -> list[MemoryCandidate]:
        scored = sorted(
            enumerate(self._memories),
            key=lambda item: (-score_overlap(query, item[1].text), item[0]),
        )
        return [
            MemoryCandidate(
                memory.id,
                memory.text,
                memory.created_at,
                {**memory.metadata, "score": round(score_overlap(query, memory.text), 4)},
            )
            for _index, memory in scored[:top_k]
        ]

    def add_memory(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryCandidate:
        candidate = MemoryCandidate(
            f"formed-{self._next_id}",
            text,
            metadata={**(metadata or {}), "seed": False, "score": 1.0},
        )
        self._next_id += 1
        self._memories.append(candidate)
        return candidate

    def update_memory(
        self,
        memory_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryCandidate:
        for index, memory in enumerate(self._memories):
            if memory.id != memory_id:
                continue
            updated = MemoryCandidate(
                memory.id,
                text,
                memory.created_at,
                {**memory.metadata, **(metadata or {}), "score": memory.metadata.get("score", 1.0)},
            )
            self._memories[index] = updated
            return updated
        raise KeyError(f"unknown memory id: {memory_id}")

    def delete_memory(self, memory_id: str) -> None:
        self._memories = [memory for memory in self._memories if memory.id != memory_id]

    def list_memories(self) -> list[MemoryCandidate]:
        return [
            MemoryCandidate(memory.id, memory.text, memory.created_at, dict(memory.metadata))
            for memory in self._memories
        ]

    def inspect(self) -> list[dict[str, Any]]:
        return [
            {"id": memory.id, "memory": memory.text, "metadata": dict(memory.metadata)}
            for memory in self._memories
        ]


def tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in {"the", "and", "for", "with", "that", "this"}
    }


def score_overlap(query: str, memory: str) -> float:
    query_tokens = tokenize(query)
    memory_tokens = tokenize(memory)
    if not query_tokens and not memory_tokens:
        return 0.0
    overlap = len(query_tokens & memory_tokens)
    union = len(query_tokens | memory_tokens)
    return overlap / union if union else 0.0
