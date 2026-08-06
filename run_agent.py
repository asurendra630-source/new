"""CLI entry point for the OrbitDesk support agent.

Usage:
  python run_agent.py "question text"          # single question, JSON output
  python run_agent.py --interactive            # chat loop
  python run_agent.py --samples                # run the 5 sample questions
  python run_agent.py --human-readable "...?"  # pretty console output

Set HF_HUB_OFFLINE=1 after models are cached to run with network disabled.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from orbitdesk.models import Embedder, LocalLLM  # noqa: E402
from orbitdesk.pipeline import build_graph, run_question, state_to_payload  # noqa: E402
from orbitdesk.schema import validate  # noqa: E402

KB_DIR = ROOT / "data" / "knowledge_base"
CASES = ROOT / "data" / "resolved_cases.json"


def pretty_print(payload: dict) -> None:
    print("\n" + "=" * 70)
    print(f"QUESTION: {payload.get('_question', '')}")
    print(f"CLASSIFICATION: {payload['classification']}  (confidence {payload['confidence']:.2f}, human={payload['requires_human']})")
    print(f"ANSWER: {payload['answer']}")
    if payload.get("clarification_question"):
        print(f"CLARIFICATION: {payload['clarification_question']}")
    if payload.get("sources"):
        print("SOURCES:")
        for s in payload["sources"]:
            print(f"  - {s['source_id']}: {s['passage'][:110]}...")
    if payload.get("warnings"):
        print("WARNINGS:", "; ".join(payload["warnings"]))
    print("REASON:", payload["reason"])
    print("=" * 70)


def main() -> None:
    ap = argparse.ArgumentParser(description="OrbitDesk local support agent")
    ap.add_argument("question", nargs="?", help="question to answer")
    ap.add_argument("--interactive", action="store_true", help="chat loop")
    ap.add_argument("--samples", action="store_true", help="run the 5 sample questions")
    ap.add_argument("--no-llm", action="store_true", help="skip LLM (triage/retrieval only)")
    ap.add_argument("--lexical", action="store_true",
                    help="zero model downloads: skip LLM and embedding model (pure lexical retrieval)")
    ap.add_argument("--human-readable", action="store_true", help="pretty console output")
    args = ap.parse_args()

    emb = None if args.lexical else Embedder()
    gen = None if (args.no_llm or args.lexical) else LocalLLM()
    # Model status goes to stderr so stdout stays pure (schema-compliant) JSON.
    if emb is not None:
        emb._ensure()
        print(f"[models] embedder {emb.model_name} loaded in {emb.load_time_s:.1f}s", file=sys.stderr)
    if gen is not None:
        gen._ensure()
        print(f"[models] LLM {gen.model_name} loaded in {gen.load_time_s:.1f}s", file=sys.stderr)
    graph, _ = build_graph(KB_DIR, CASES, emb, gen)

    def handle(q: str) -> tuple[dict, list]:
        state = run_question(q, graph, KB_DIR, CASES, emb, gen)
        payload = state_to_payload(state)
        # Validate the schema-clean payload: output_schema.json has
        # additionalProperties=false, so metadata must NOT be mixed in.
        violations = validate(payload)
        if violations:
            payload["warnings"] = payload.get("warnings", []) + [f"schema: {v}" for v in violations]
        return payload, state.node_log

    if args.samples:
        samples = json.loads((ROOT / "data" / "sample_questions.json").read_text(encoding="utf-8"))["questions"]
        for s in samples:
            payload, node_log = handle(s["question"])
            payload_disp = dict(payload, _question=s["question"], _node_log=node_log)
            pretty_print(payload_disp)
            # The printed payload carries the schema keys only; the node trace
            # is printed to stderr so stdout stays schema-compliant JSON.
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            print(f"# node trace for {s['question_id']}:", file=sys.stderr)
            for line in node_log:
                print("#   " + line, file=sys.stderr)
    elif args.interactive:
        print("OrbitDesk agent (Ctrl-D to exit)")
        while True:
            try:
                q = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                continue
            payload, node_log = handle(q)
            pretty_print(dict(payload, _question=q, _node_log=node_log))
            for line in node_log:
                print("# " + line, file=sys.stderr)
    elif args.question:
        payload, node_log = handle(args.question)
        if args.human_readable:
            pretty_print(dict(payload, _question=args.question, _node_log=node_log))
        else:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        for line in node_log:
            print("# " + line, file=sys.stderr)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
