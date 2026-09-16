from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Protocol

from adapters.mem0_adapter import MemoryStore
from gate.models import (
    FormationDecision,
    FormationResult,
    MemoryCandidate,
    MemoryDraft,
    TurnContext,
)

EXTRACTION_PROMPT = """
Extract at most 3 concise memory statements that should persist for future turns.
Keep only user facts, preferences, recurring commitments, or ongoing projects.
Do not copy instructions, assistant plans, or one-off chatter unless it is clearly useful later.
Return only JSON in this shape:
{{"memories":[{{"text":"...","kind":"preference|schedule|project|profile|workflow"}}]}}
Conversation:
User: {user_message}
Assistant: {assistant_message}
""".strip()

RECONCILE_PROMPT = """
Decide whether a candidate memory should be added, skipped, or used to update one existing memory.
Choose "skip" when the same fact already exists or the candidate adds no meaningful information.
Choose "update" when one existing memory should become the canonical version because the candidate
corrects or supersedes it. Choose "add" when the candidate is a distinct new fact.
Return only JSON in this shape:
{{"action":"add|skip|update","target_id":"","delete_ids":[],"reason":"..."}}
Candidate:
{candidate}
Existing memories:
{existing}
""".strip()


class FormationError(RuntimeError):
    def __init__(self, stage: str, message: str) -> None:
        self.stage = stage
        self.detail = message
        super().__init__(f"{stage}: {message}")


class MemoryExtractor(Protocol):
    def extract(self, turn: TurnContext, assistant_message: str) -> list[MemoryDraft]: ...


class MemoryReconciler(Protocol):
    def reconcile(
        self,
        candidate: MemoryDraft,
        existing: Sequence[MemoryCandidate],
    ) -> tuple[str, str, list[str], str]: ...


class FunctionMemoryExtractor:
    def extract(self, turn: TurnContext, assistant_message: str) -> list[MemoryDraft]:
        del assistant_message
        message = " ".join(turn.user_message.split())
        lowered = message.lower()
        drafts: list[MemoryDraft] = []
        if match := re.search(r"\bi prefer ([^.!?]+)", message, re.IGNORECASE):
            drafts.append(
                MemoryDraft(
                    f"The user prefers {cleanup_clause(match.group(1))}.",
                    {"kind": "preference"},
                )
            )
        if match := re.search(r"\bi(?: am|'m) ([^.!?]+)", message, re.IGNORECASE):
            drafts.append(
                MemoryDraft(
                    f"The user is {cleanup_clause(match.group(1))}.",
                    {"kind": "profile"},
                )
            )
        recurring = re.search(r"\bi have ([^.!?]*\bevery\b[^.!?]+)", message, re.IGNORECASE)
        if recurring:
            drafts.append(
                MemoryDraft(
                    f"The user has {cleanup_clause(recurring.group(1))}.",
                    {"kind": "schedule"},
                )
            )
        if match := re.search(r"\bi want to ([^.!?]+)", message, re.IGNORECASE):
            drafts.append(
                MemoryDraft(
                    f"The user wants to {cleanup_clause(match.group(1))}.",
                    {"kind": "project"},
                )
            )
        if "idea for" in lowered:
            start = lowered.index("idea for")
            drafts.append(
                MemoryDraft(
                    f"The user has an {cleanup_clause(message[start:])}.",
                    {"kind": "project"},
                )
            )
        return dedupe_drafts(drafts)


class OpenAIMemoryExtractor:
    def __init__(self, api_key: str, model: str) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model

    def extract(self, turn: TurnContext, assistant_message: str) -> list[MemoryDraft]:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "user",
                    "content": EXTRACTION_PROMPT.format(
                        user_message=turn.user_message,
                        assistant_message=assistant_message,
                    ),
                }
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        content = response.choices[0].message.content
        if not content:
            return []
        payload = parse_json_object(content, "extractor")
        memories = require_list_field(payload, "memories", "extractor")
        drafts = [
            MemoryDraft(
                str(item.get("text") or "").strip(),
                {"kind": str(item.get("kind") or "other").strip() or "other"},
            )
            for item in memories
            if isinstance(item, dict) and str(item.get("text") or "").strip()
        ]
        return dedupe_drafts(drafts)


class FunctionMemoryReconciler:
    def reconcile(
        self,
        candidate: MemoryDraft,
        existing: Sequence[MemoryCandidate],
    ) -> tuple[str, str, list[str], str]:
        normalized = normalize_memory_text(candidate.text)
        for memory in existing:
            if normalize_memory_text(memory.text) == normalized:
                return "skip", memory.id, [], "duplicate of existing memory"
        if candidate.metadata.get("kind") == "preference":
            non_seed = [
                memory
                for memory in existing
                if memory.id
                and memory.metadata.get("kind") == "preference"
                and not memory.metadata.get("seed")
            ]
            if len(non_seed) == 1:
                return "update", non_seed[0].id, [], "preference updated with newer value"
        return "add", "", [], "new memory candidate"


class OpenAIMemoryReconciler:
    def __init__(self, api_key: str, model: str) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model

    def reconcile(
        self,
        candidate: MemoryDraft,
        existing: Sequence[MemoryCandidate],
    ) -> tuple[str, str, list[str], str]:
        if not existing:
            return "add", "", [], "no related memories found"
        serialized_existing = json.dumps(
            [
                {
                    "id": memory.id,
                    "text": memory.text,
                    "kind": memory.metadata.get("kind", "other"),
                }
                for memory in existing
            ],
            indent=2,
        )
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "user",
                    "content": RECONCILE_PROMPT.format(
                        candidate=json.dumps(
                            {
                                "text": candidate.text,
                                "kind": candidate.metadata.get("kind", "other"),
                            },
                            indent=2,
                        ),
                        existing=serialized_existing,
                    ),
                }
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        content = response.choices[0].message.content
        if not content:
            return "add", "", [], "reconciler returned no text"
        payload = parse_json_object(content, "reconciler")
        action = str(payload.get("action") or "add")
        target_id = str(payload.get("target_id") or "")
        delete_ids_value = payload.get("delete_ids", [])
        if not isinstance(delete_ids_value, list):
            raise FormationError("reconciler", "field 'delete_ids' must be a list")
        delete_ids = [
            str(item) for item in delete_ids_value if str(item) and str(item) != target_id
        ]
        allowed_ids = {memory.id for memory in existing if memory.id}
        if action == "update" and target_id not in allowed_ids:
            return "add", "", [], "invalid update target from reconciler"
        delete_ids = [memory_id for memory_id in delete_ids if memory_id in allowed_ids]
        if action not in {"add", "skip", "update"}:
            action = "add"
        return action, target_id, delete_ids, str(payload.get("reason") or "").strip()


def form_memories(
    turn: TurnContext,
    assistant_message: str,
    store: MemoryStore,
    extractor: MemoryExtractor,
    reconciler: MemoryReconciler,
) -> FormationResult:
    try:
        drafts = extractor.extract(turn, assistant_message)
    except FormationError:
        raise
    except Exception as exc:
        raise FormationError("extractor", f"failed to extract memory candidates: {exc}") from exc
    if not drafts:
        return FormationResult([], [])
    decisions: list[FormationDecision] = []
    try:
        existing = store.list_memories()
    except Exception as exc:
        raise FormationError("store:list", f"failed to list existing memories: {exc}") from exc
    for draft in drafts:
        exact = find_exact_duplicate(draft.text, existing)
        if exact is not None:
            decisions.append(
                FormationDecision(
                    draft.text,
                    "skipped",
                    "duplicate of existing memory",
                    target_memory_id=exact.id,
                )
            )
            continue
        try:
            related = store.find_related_memories(draft.text, top_k=5)
        except Exception as exc:
            raise FormationError(
                "store:search",
                f"failed to find related memories for {draft.text!r}: {exc}",
            ) from exc
        try:
            action, target_id, delete_ids, reason = reconciler.reconcile(draft, related)
        except FormationError:
            raise
        except Exception as exc:
            raise FormationError(
                "reconciler",
                f"failed to reconcile candidate {draft.text!r}: {exc}",
            ) from exc
        metadata = {**draft.metadata, "seed": False, "origin": "chat"}
        if action == "skip":
            decisions.append(
                FormationDecision(
                    draft.text,
                    "skipped",
                    reason or "reconciler skipped candidate",
                    target_memory_id=target_id,
                )
            )
            continue
        if action == "update":
            current = next((memory for memory in existing if memory.id == target_id), None)
            try:
                updated = store.update_memory(
                    target_id,
                    draft.text,
                    {**(current.metadata if current else {}), **metadata},
                )
            except Exception as exc:
                raise FormationError(
                    "store:update",
                    f"failed to update memory {target_id!r} for {draft.text!r}: {exc}",
                ) from exc
            existing = [
                updated if memory.id == target_id else memory
                for memory in existing
                if memory.id not in delete_ids
            ]
            for memory_id in delete_ids:
                try:
                    store.delete_memory(memory_id)
                except Exception as exc:
                    raise FormationError(
                        "store:delete",
                        f"failed to delete memory {memory_id!r}: {exc}",
                    ) from exc
            decisions.append(
                FormationDecision(
                    draft.text,
                    "updated",
                    reason or "updated conflicting memory",
                    target_memory_id=updated.id,
                    deleted_memory_ids=delete_ids,
                )
            )
            continue
        try:
            added = store.add_memory(draft.text, metadata)
        except Exception as exc:
            raise FormationError(
                "store:add",
                f"failed to add memory for {draft.text!r}: {exc}",
            ) from exc
        existing.append(added)
        decisions.append(
            FormationDecision(
                draft.text,
                "added",
                reason or "persisted new memory",
                target_memory_id=added.id,
            )
        )
    return FormationResult(drafts, decisions)


def dedupe_drafts(drafts: Sequence[MemoryDraft]) -> list[MemoryDraft]:
    seen: set[str] = set()
    result: list[MemoryDraft] = []
    for draft in drafts:
        normalized = normalize_memory_text(draft.text)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(MemoryDraft(sentence_case(draft.text.strip()), dict(draft.metadata)))
    return result


def find_exact_duplicate(text: str, existing: Sequence[MemoryCandidate]) -> MemoryCandidate | None:
    normalized = normalize_memory_text(text)
    return next(
        (memory for memory in existing if normalize_memory_text(memory.text) == normalized),
        None,
    )


def normalize_memory_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def cleanup_clause(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" .,!?:;")


def sentence_case(text: str) -> str:
    cleaned = cleanup_clause(text)
    if not cleaned:
        return ""
    return cleaned[0].upper() + cleaned[1:] + ("" if cleaned.endswith(".") else ".")


def parse_json_object(content: str, stage: str) -> dict[str, object]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise FormationError(stage, f"returned invalid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise FormationError(stage, "returned JSON that is not an object")
    return payload


def require_list_field(
    payload: dict[str, object],
    field_name: str,
    stage: str,
) -> list[object]:
    if field_name not in payload:
        raise FormationError(stage, f"response missing '{field_name}' field")
    value = payload[field_name]
    if not isinstance(value, list):
        raise FormationError(stage, f"field '{field_name}' must be a list")
    return value
