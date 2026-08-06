"""Demo runner: executes the 5 sample questions, records latency and node
traces, and writes results + a run report into outputs/.

Usage: python run_demo.py [--no-llm] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from orbitdesk.models import EMBEDDER_MODEL, LLM_MODEL, Embedder, LocalLLM  # noqa: E402
from orbitdesk.pipeline import build_graph, run_question, state_to_payload  # noqa: E402
from orbitdesk.schema import validate  # noqa: E402

KB_DIR = ROOT / "data" / "knowledge_base"
CASES = ROOT / "data" / "resolved_cases.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--lexical", action="store_true",
                    help="skip the embedding model AND the LLM (pure lexical retrieval; zero model downloads)")
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    emb = None if args.lexical else Embedder()
    # --lexical implies --no-llm: it promises zero model downloads, so the
    # LLM must not be constructed either (reviewer finding).
    gen = None if (args.no_llm or args.lexical) else LocalLLM()
    t0 = time.perf_counter()
    graph, passages = build_graph(KB_DIR, CASES, emb, gen)
    build_s = time.perf_counter() - t0
    if args.lexical:
        print("[run] lexical-only mode (no embedding model, no LLM)")

    samples = json.loads((ROOT / "data" / "sample_questions.json").read_text(encoding="utf-8"))["questions"]

    results, node_logs = [], {}
    total = 0.0
    for s in samples:
        t_start = time.perf_counter()
        state = run_question(s["question"], graph, KB_DIR, CASES, emb, gen)
        dt = time.perf_counter() - t_start
        total += dt
        # Keep responses.json schema-clean (output_schema.json has
        # additionalProperties=false); metadata goes to a separate file.
        payload = state_to_payload(state)
        violations = validate(payload)
        if violations:
            payload["warnings"] = payload.get("warnings", []) + [f"schema: {v}" for v in violations]
        results.append(payload)
        node_logs[s["question_id"]] = {
            "question": s["question"],
            "classification": state.classification,
            "latency_s": round(dt, 2),
            "top_sim": round(state.top_sim, 3),
            "revisions": state.revision_count,
            "node_log": state.node_log,
        }
        print(f"[{s['question_id']}] {state.classification}  ({dt:.1f}s)  "
              f"top_sim={state.top_sim:.3f}  revisions={state.revision_count}")

    (out / "responses.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "node_traces.json").write_text(
        json.dumps(node_logs, indent=2, ensure_ascii=False), encoding="utf-8")

    report = {
        "product": "OrbitDesk",
        "embedder_model": EMBEDDER_MODEL if emb else None,
        "embedder_revision": emb.revision if emb else None,
        "embedder_load_s": round(emb.load_time_s, 2) if emb else None,
        "llm_model": LLM_MODEL if gen else None,
        "llm_revision": gen.revision if gen else None,
        "llm_load_s": round(gen.load_time_s, 2) if gen else None,
        "llm_total_generation_s": round(gen.total_latency_s, 2) if gen else None,
        "llm_calls": gen.calls if gen else 0,
        "kb_passages": len(passages),
        "graph_build_s": round(build_s, 2),
        "questions_run": len(results),
        "total_run_s": round(total, 2),
        "avg_question_s": round(total / len(results), 2),
    }
    (out / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nrun report:", json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
