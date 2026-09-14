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

### Google Calendar and Gmail

The chat optionally reads Google Calendar and Gmail as live sources. It does not save
events or messages to Mem0. In Google Cloud:

1. Enable the Google Calendar API and Gmail API.
2. Configure the OAuth consent screen and add your account as a test user if needed.
3. Create an OAuth client with application type **Desktop app**.
4. Download it as `google_credentials.json` in this repository.
5. Run `python -m demo.chat` and approve the browser prompt on the first calendar or
   email question.

The generated `google_token.json` is reused and ignored by Git. Access is read-only:
`calendar.readonly` and `gmail.readonly`. Calendar turns retrieve up to 10 events in
the next seven days; email turns retrieve metadata and snippets for up to 10 inbox
messages from the last 30 days. Full bodies and attachments are not requested.

With neither API key set, the demo is deterministic and offline. For live Mem0 and OpenAI:

```powershell
$env:MEM0_API_KEY = "<Mem0 Platform key>"
$env:OPENAI_API_KEY = "<OpenAI API key>"
python -m demo.run_before_after
```

The live path uses Mem0 `MemoryClient`, exact `infer=False` seeds, and a matching
`user_id=kiersten-demo` filter. Raw `infer=False` writes in the current Platform client
are user-scoped; `memory-gate-v0` remains the local agent and decision-log identity.
Before seeding, the adapter lists the user scope and skips exact seed texts that already
exist. Adds can be asynchronous; if a search misses a newly added seed, wait briefly
and rerun.

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