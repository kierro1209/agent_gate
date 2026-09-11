from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Criteria:
    version: str
    top_k: int
    user_id: str
    agent_id: str
    judge_model: str
    judge_prompt: str
    score_min: float
    score_max: float
    distractor: str


def load_criteria(path: str | Path = "criteria/cmi_v0.yaml") -> Criteria:
    raw: Any = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    try:
        scale = raw["judge"]["scale"]
        result = Criteria(
            str(raw["criteriaVersion"]),
            int(raw["top_k"]),
            str(raw["scope"]["user_id"]),
            str(raw["scope"]["agent_id"]),
            str(raw["judge"]["model"]),
            str(raw["judge"]["prompt"]),
            float(scale["minimum"]),
            float(scale["maximum"]),
            str(raw["perturbation"]["distractor"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid criteria: {exc}") from exc
    if result.version != "cmi-v0" or result.top_k < 1 or result.score_min >= result.score_max:
        raise ValueError("invalid criteria version, top_k, or score scale")
    return result
