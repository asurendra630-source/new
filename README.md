# OrbitDesk — Local-First Support Agent Network

An AI agent network that answers support questions about **OrbitDesk** (a fictional
workspace product) using **only** the supplied knowledge base and resolved cases.
Everything runs **locally** — the retrieval embedding model and the response LLM
are Hugging Face models loaded from disk. No remote language-model APIs are used.

This is the submission for the *AI Engineer Internship Assignment (Tantrabodh AI)*.

> **AI-assistance disclosure:** this implementation was built with the help of an
> AI coding assistant (Buffy / Freebuff), as permitted by the assignment rules.

---

## 1. What it does

A graph-based workflow classifies each incoming request, retrieves evidence from
the supplied documents, generates an answer with a local LLM, and verifies the
answer before returning it. Every run logs which nodes executed and which
conditional path was chosen.

```
START
 └─► triage
      ├─ answerable ────────────────► retrieve ──► generate ──► verify ──(pass)──► END
      │                                                    ▲            │
      │                                                    │  (fail, revisions<1)
      │                                                    └─── revise ──┘  (retry)
      │                                                                     └─(fail, revisions>=1)──► safe_failure ──► END
      ├─ requires_clarification ────► clarify (deterministic terminal) ──► END
      ├─ requires_escalation ───────► escalate (deterministic terminal) ──► END
      ├─ out_of_scope ──────────────► oos (deterministic terminal) ────────► END
      └─ safe_failure ──────────────► safe_failure (deterministic terminal) ► END
```

Graph diagram: `diagrams/graph_diagram.png` (regenerate with `python make_diagram.py`).

### Node responsibilities

| Node | Responsibility | Deterministic / Model |
|------|----------------|----------------------|
| **triage** | Classify request into `answerable / requires_clarification / requires_escalation / out_of_scope / safe_failure` using rules encoded from KB-001/002/006/008/010 | Deterministic |
| **retrieve** | Hybrid lexical + embedding retrieval over KB docs + resolved cases; KB takes precedence; superseded cases excluded from guidance | Deterministic + local HF embeddings |
| **generate** | Answer **only** from retrieved evidence, citing source IDs, via a local HF causal LM | Local model |
| **verify** | Schema conformance, source citations, grounding (lexical overlap), no unsupported claims, no superseded guidance | Deterministic |
| **revise** | Retry path: re-generate once with verifier feedback when verification fails | Local model |
| **terminals** | Deterministic answers for clarification / escalation / OOS / safe-failure routes (never improvised by the LLM) | Deterministic |

### Orchestration requirements coverage

- **Shared typed state** — `AgentState` dataclass threaded through every node (`orbitdesk/graph.py`).
- **Conditional routing** — edges with predicate conditions (`add_edge(src, dst, cond)`).
- **Retry / revision path** — `verify → revise → generate → verify` with a revision budget (`MAX_REVISIONS = 1`).
- **Clear separation** — triage/verify/terminals are pure deterministic code; only retrieval scoring and generation use models.
- **Node logs** — every node execution + route decision is appended to `state.node_log` (see `outputs/node_traces.json`).
- **Loop protection** — `Graph` enforces `max_steps=12` and `max_visits=3` per node, forcing `safe_failure` if exceeded.

---

## 2. Setup

Two run modes:

### 2a. Quick start on Kaggle (recommended — models download + GPU)

The models are **not bundled** in this repo; they download from Hugging Face on first
use. Kaggle provides internet during setup and a GPU, so the easiest path is the
included notebook:

1. Upload this repo as a Kaggle dataset named `orbitdesk-assignment` (or edit the
   notebook's clone URL).
2. Open `kaggle/orbitdesk_assignment.ipynb` and run all cells.
3. It installs deps, downloads both models, runs the 5 sample questions, writes
   `outputs/`, runs the test suite and renders the diagram.

### 2b. Local run (CPU)

Requires Python 3.10+. Tested on Python 3.11 (Windows, CPU-only).

```bash
# 1. Create a venv and install dependencies
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# CPU-only torch (recommended on machines without a big GPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 2. (One-time) download the local models — this is the only network step.
#    After this, set HF_HUB_OFFLINE=1 to run fully offline.
python -c "from orbitdesk.models import Embedder, LocalLLM; Embedder()._ensure(); LocalLLM()._ensure(); print('models cached')"

# 3. Zero-download smoke run (pure lexical retrieval + deterministic fallback):
python run_agent.py --lexical "question"
python run_demo.py --lexical
```

`--lexical` skips **both** models — nothing is downloaded; retrieval falls back to
lexical scoring and answers use a deterministic extractive fallback. Use it to
inspect triage/routing without any model downloads.

### Models used (local, via Hugging Face)

| Purpose | Model | Size (approx.) | Device |
|---------|-------|----------------|--------|
| Retrieval embeddings | `sentence-transformers/all-MiniLM-L6-v2` | ~90 MB | CPU |
| Response generation | `Qwen/Qwen2.5-0.5B-Instruct` | ~1 GB (fp32) | CPU |

- Override with env vars: `ORBITDESK_EMBEDDER_MODEL`, `ORBITDESK_LLM_MODEL`.
- **Exact model names, resolved HF revisions (commit hashes), approximate load
  times and generation latency are recorded in `outputs/run_report.json`** after
  a demo run (see §4), satisfying the brief's "state the exact model names and
  revisions used" requirement. Revisions are read from the loaded model configs
  (`config._commit_hash`).
- **Hardware used for the reference run:** 12th Gen Intel Core i5-1235U (10 cores),
  23.7 GB RAM, no GPU used (CPU inference; NVIDIA MX550 2 GB present but too small
  for the LLM). The 0.5B LLM is a deliberate hardware-aware trade-off: CPU-compatible,
  ~1 GB, seconds per answer, while keeping answer quality acceptable for a demo.

---

## 3. Usage

```bash
# Single question → JSON (schema-compliant)
python run_agent.py "Our daily exports stopped after a timezone change. What should we check?"

# Pretty console output
python run_agent.py --human-readable "Can a read-only Viewer create an API credential?"

# Chat loop
python run_agent.py --interactive

# Run the 5 sample questions from the package
python run_agent.py --samples

# Fully offline (after models are cached)
HF_HUB_OFFLINE=1 python run_agent.py "question"
```

## 4. Demo + sample outputs

```bash
python run_demo.py            # runs the 5 sample questions
```

Writes:

- `outputs/responses.json` — schema-compliant payloads for the 5 sample questions
- `outputs/node_traces.json` — which nodes ran per question + chosen route
- `outputs/run_report.json` — model names, load times, generation latency, counts

> **Note on the committed `outputs/`:** the files in this repo were produced by
> the full model run (Kaggle notebook, real MiniLM + Qwen metrics in
> `run_report.json`). Re-running the Kaggle notebook (or `python run_demo.py`
> with the models cached) regenerates `outputs/`. The zero-download
> `--lexical` smoke run instead records `null` model fields.

## 5. Tests

```bash
python -m pytest tests/ -v
```

The suite verifies **graph routing without depending on the exact wording
produced by the model** — the embedder and LLM are stubbed with deterministic
fakes. Coverage includes the five required test cases:

1. **Directly answerable** question (timezone + missed export → answerable, cited, grounded)
2. **Two-document** question (retrieval must surface KB-003 *and* KB-004)
3. **Ambiguous** question requiring clarification ("sync is not working" → `requires_clarification` + clarification question)
4. **Out-of-scope / unsafe** request (refund + legal + instruction-override → `safe_failure`, no LLM involved)
5. **Verification-failure → revision** (a bad first answer is re-generated once; a second failure routes to `safe_failure`)

Plus: triage routing table, superseded-case exclusion (CASE-0914), loop-guard
termination, and output-schema conformance for all five sample questions.

## 6. Design trade-offs & limitations

- **Deterministic terminals instead of LLM-generated policy for clarify/escalate/OOS/safe routes.**
  These routes carry safety/policy weight; a 0.5B model can improvise bad policy.
  Deterministic answers grounded in KB-006/008/010 are safer and testable. The LLM is
  used only where it adds value (evidence-grounded answers).
- **Revision budget = 1.** Bounded retries keep latency low and prevent the model
  from looping; a second failure degrades to a safe failure. With more time, a
  second revision with stricter prompts could be added.
- **Lexical grounding proxy.** Verification uses token overlap between the answer
  and evidence as a hallucination proxy; an embedding-based grounding score would
  be stronger but adds a second embedding pass.
- **Small LLM quality.** Qwen-0.5B keeps the runnable footprint small but writes
  plainer prose than a larger model. With more time, `Qwen2.5-1.5B-Instruct`
  (≈3 GB) would improve phrasing at the same integration cost.

## 7. Repository layout

```
orbitdesk_assignment/
├── README.md
├── requirements.txt
├── run_agent.py            # CLI (single question / interactive / samples)
├── run_demo.py             # demo runner → outputs/
├── make_diagram.py         # graph diagram generator (Pillow)
├── orbitdesk/              # package
│   ├── graph.py            # typed state + minimal graph runtime (loop guard)
│   ├── knowledge.py        # KB + cases ingestion
│   ├── triage.py           # deterministic triage rules
│   ├── retrieval.py        # hybrid lexical + embedding retrieval
│   ├── generate.py         # local-LLM evidence-only generation
│   ├── verify.py           # verification + terminal handlers
│   ├── schema.py           # output_schema.json validation
│   ├── models.py           # lazy HF model loaders with latency tracking
│   └── pipeline.py         # graph assembly + payload building
├── tests/test_agent.py     # automated tests (stubbed models)
├── data/                   # supplied assignment materials (source of truth)
├── outputs/                # sample responses + run report + node traces
└── diagrams/graph_diagram.png
```
