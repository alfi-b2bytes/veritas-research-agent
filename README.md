# Veritas — a research agent that brings its sources

> ODSC × Nebius Hackathon submission · Challenge: *Build a research agent that brings its sources*

**One-line pitch:** Veritas turns a research question into a sourced, verifiable answer for founders and analysts who need to trust what they read: every key claim carries a citation you can open, and the agent says out loud when the evidence is thin.

- **Demo video (2–3 min):** <link>
- **Team:** Diego Díaz (B2Bytes) — [@alfi-b2bytes](https://github.com/alfi-b2bytes)

## The problem

A founder doing market or technical research does not need another chatbot paragraph — they need to know *where each claim comes from* and *how solid it is*. Veritas is built for that user: it plans its own searches, gathers live web evidence, cross-checks its draft against the sources, and flags gaps and conflicts instead of inventing confidence.

## How it works

```
question → plan queries → Tavily search → evidence gate (loop) → Nebius synthesis with [n] citations → verification pass → sourced answer + evidence notes
```

- **Query planning:** a Token Factory model decomposes the question into 2–4 independent queries. After each round, an *evidence gate* (same model) judges whether the gathered sources are enough; if not, the planner proposes 1–3 follow-up queries targeting the missing angles. Max 3 rounds, and it stops early when the gate passes.
- **Evidence handling:** results are deduplicated by URL across rounds, trimmed to excerpts, and numbered. The synthesis prompt gets only these excerpts and must attach `[n]` citations to every key claim, prefer primary/recent sources, and present both sides when sources conflict.
- **Verification:** a second model pass audits the draft claim-by-claim, rewrites what does not hold up, and appends an **Evidence notes** section: verified claims, claims softened or removed, remaining gaps/conflicts, and an overall confidence level with reasoning.
- **Safety:** web text is untrusted. Source excerpts travel inside `<<< >>>` blocks, and the system prompt instructs the model to treat them as data, never as instructions. Tool failures (Tavily errors, empty results) are logged in the run trace on stderr and surfaced in the answer, not silently swallowed.
- **Trade-offs:** one 70B-class model keeps quality high at low cost per run (a few thousand tokens); the evidence gate costs one extra call per round but saves full rounds when the first search already suffices; excerpts are capped at 1.5 KB per source to bound latency and tokens.

## Required services

| Requirement | How we use it |
|---|---|
| Nebius Token Factory (inference) | Model: `meta-llama/Llama-3.3-70B-Instruct` (env-configurable) — runs all four roles: query planning, evidence gating, cited synthesis and the verification pass. |
| Tavily (real-time web search) | Every search round is a live Tavily `advanced` search; result titles, URLs and excerpts feed the model, and the URLs survive into the answer's numbered Sources section. |

## Setup

```bash
git clone https://github.com/alfi-b2bytes/veritas-research-agent.git
cd veritas-research-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your keys — never commit them
python src/agent.py "Your research question here"
```

Environment variables (see `.env.example`): `NEBIUS_API_KEY`, `TAVILY_API_KEY`, and optionally `NEBIUS_MODEL`, `NEBIUS_BASE_URL`, `MAX_ROUNDS`, `RESULTS_PER_QUERY`.

## Example

**Question:** *Is the Chilean startup ecosystem growing in 2026, and what is driving it?*

**Answer (excerpt):** see `docs/example-run.md` for a full run with citations, evidence notes, and the agent's trace (query plan, gate decisions, retrieval counts).

## Repo layout

```
src/agent.py   # the whole agent: planner, retriever, gate, synthesizer, verifier
docs/          # example run + trace
```
