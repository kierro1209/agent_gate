def perturb(memory: str, distractor: str) -> str:
    normalized = " ".join(memory.split())
    paraphrase = f"In other words, {normalized[0].lower()}{normalized[1:]}" if normalized else ""
    return f"{paraphrase}\n{distractor}".strip()
