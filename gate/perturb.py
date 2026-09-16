def perturb_variants(memory: str, distractor: str) -> tuple[str, ...]:
    normalized = " ".join(memory.split())
    if not normalized:
        return ("",)
    paraphrase = f"In other words, {normalized[0].lower()}{normalized[1:]}"
    return (
        (
            f"{paraphrase}\n"
            f"{distractor}\n"
            "Possibly unrelated context is mixed into this memory block."
        ).strip(),
        (
            f"{paraphrase}\n"
            "Unverified context: parts of this memory may be incomplete, "
            "speculative, or outdated.\n"
            f"{distractor}"
        ).strip(),
        (
            f"{paraphrase}\n"
            "Unverified context: another note may partially conflict with this detail.\n"
            "Possibly unrelated planning details may also be mixed in."
        ).strip(),
    )
