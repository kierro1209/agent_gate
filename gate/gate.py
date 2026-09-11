from collections.abc import MutableMapping

from gate.criteria import Criteria, load_criteria
from gate.judge import Judge
from gate.models import GateDecision, GateResult, MemoryCandidate, TurnContext
from gate.perturb import perturb

ScoreCache = MutableMapping[tuple[str, str], tuple[float, float, float]]


def gate_memories(
    turn: TurnContext,
    candidates: list[MemoryCandidate],
    judge: Judge,
    *,
    criteria: Criteria | None = None,
    criteria_path: str = "criteria/cmi_v0.yaml",
    top_k: int | None = None,
    cache: ScoreCache | None = None,
) -> GateResult:
    config = criteria or load_criteria(criteria_path)
    limit = config.top_k if top_k is None else top_k
    if limit < 0:
        raise ValueError("top_k cannot be negative")
    accepted: list[MemoryCandidate] = []
    rejected: list[MemoryCandidate] = []
    decisions: list[GateDecision] = []
    for candidate in candidates[:limit]:
        key = (turn.turn_id, candidate.id)
        scores = cache.get(key) if cache is not None else None
        if scores is None:
            scores = (
                judge.score(turn, ""),
                judge.score(turn, candidate.text),
                judge.score(turn, perturb(candidate.text, config.distractor)),
            )
            if cache is not None:
                cache[key] = scores
        s_no, s_with, s_pert = scores
        utility, stability = s_with - s_no, s_with - s_pert
        passes = utility > 0 and stability >= 0
        failed = []
        if utility <= 0:
            failed.append(f"Utility={utility:.3f}<=0")
        if stability < 0:
            failed.append(f"Stability={stability:.3f}<0")
        reason = (
            f"accepted: Utility={utility:.3f}>0 and Stability={stability:.3f}>=0"
            if passes
            else "rejected: " + " and ".join(failed)
        )
        decisions.append(
            GateDecision(
                candidate.id,
                s_no,
                s_with,
                s_pert,
                utility,
                stability,
                passes,
                reason,
                config.version,
            )
        )
        (accepted if passes else rejected).append(candidate)
    rejected.extend(candidates[limit:])
    return GateResult(accepted, rejected, decisions)
