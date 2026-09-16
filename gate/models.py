from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class MemoryCandidate:
    id: str
    text: str
    created_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TurnContext:
    turn_id: str
    agent_id: str
    user_message: str


@dataclass(frozen=True)
class GateDecision:
    candidate_id: str
    s_no: float
    s_with: float
    s_pert: float
    utility: float
    stability: float
    accepted: bool
    reason: str
    criteria_version: str = "cmi-v0"

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["candidateId"] = data.pop("candidate_id")
        data["criteriaVersion"] = data.pop("criteria_version")
        return data


@dataclass(frozen=True)
class GateResult:
    accepted: list[MemoryCandidate]
    rejected: list[MemoryCandidate]
    decisions: list[GateDecision]


@dataclass(frozen=True)
class MemoryDraft:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FormationDecision:
    candidate_text: str
    action: str
    reason: str
    target_memory_id: str = ""
    deleted_memory_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["candidateText"] = data.pop("candidate_text")
        data["targetMemoryId"] = data.pop("target_memory_id")
        data["deletedMemoryIds"] = data.pop("deleted_memory_ids")
        return data


@dataclass(frozen=True)
class FormationResult:
    drafts: list[MemoryDraft]
    decisions: list[FormationDecision]
