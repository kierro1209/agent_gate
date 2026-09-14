# Memory Gate

A thin CMI §3.4 gate for Mem0 Platform retrieval. Each hit is judged with no memory,
the original memory, and a perturbed memory. It is injected only when
`s_with - s_no > 0` and `s_with - s_pert >= 0`.

## Setup and demo

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m demo.run_before_after
```

You can copy `.env.example` to `.env` and put both keys there instead of exporting
them. `.env` is ignored by Git.

## Terminal chat

```powershell
python -m demo.chat
```

The chat prints retrieved memories, retrieval scores, every CMI decision, Utility,
Stability, the exact injected memories, and the answer. Use `/gate off` to compare raw
retrieval injection, `/gate on` to restore CMI, and `/quit` to exit. Enabled-gate
decisions are also appended to `logs/decisions.jsonl`.

With neither API key set, the demo is deterministic and offline. For live Mem0 and OpenAI:

```powershell
$env:MEM0_API_KEY = "<Mem0 Platform key>"
$env:OPENAI_API_KEY = "<OpenAI API key>"
python -m demo.run_before_after
```

The live path uses Mem0 `MemoryClient`, exact `infer=False` seeds, and matching
`user_id` plus `agent_id` v3 filters. Adds can be asynchronous; if the initial search
misses a seed, wait briefly and rerun. Repeated live runs add duplicate demo seeds.

Output includes retrieved IDs, leak status, and CMI scores. Agent-loop decisions append
to `logs/decisions.jsonl`. Criteria, judge prompt, model, perturbation, scope, and top-k
are in `criteria/cmi_v0.yaml`.

## Validation

```powershell
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy
```

Part A intentionally contains no LangGraph, Agents SDK, multi-agent gate, or vendored
CMI research code.