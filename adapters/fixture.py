import json
from pathlib import Path
from typing import Any

from gate.models import MemoryCandidate, TurnContext


class FixtureAdapter:
    def __init__(self) -> None:
        self._memories: list[MemoryCandidate] = []

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

    def search_candidates(self, turn: TurnContext, top_k: int = 5) -> list[MemoryCandidate]:
        del turn
        return self._memories[:top_k]

    def inspect(self) -> list[dict[str, Any]]:
        return [{"id": memory.id, "memory": memory.text} for memory in self._memories]
