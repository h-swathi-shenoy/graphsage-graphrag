"""Step 3 - Ask questions.

  python 03_ask.py                          # run the demo question set
  python 03_ask.py "why did checkout break?"
  python 03_ask.py --compare "..."          # GraphRAG vs plain vector RAG
  python 03_ask.py --trace "..."            # show the retrieval internals
"""
import sys

from retriever import GraphRAG

DEMO_QUESTIONS = [
    # multi-hop: symptom -> aircraft -> shared ramp scheduler -> staffing policy -> SP
    "Why did Flight 202 depart late during INC-2209, and what was the real root cause?",
    # requires joining team ownership with the incident chain
    "Which team should be paged for INC-2209 and which team actually caused it?",
    # requires policy + change + incident join
    "Was any FAA safety policy violated in the INC-2209 chain of events?",
    # requires runbook + ordering constraint
    "If ramp-scheduler runs out of turnaround slots again, what exactly should I do first?",
    # requires CAPA + team coordination
    "What steps should Team Airport Operations take to resolve a gate turnaround backlog at ORD?",
]


def sep(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def show_trace(r):
    print("\n-- retrieval trace --")
    print("query attached to nodes :", ", ".join(r["attached_nodes"]))
    print("GraphSAGE seed entities :")
    for k, v in sorted(r["seed_entities"].items(), key=lambda x: -x[1]):
        print(f"    {v:+.3f}  {k}")
    print("vector-index chunk hits :",
          ", ".join(f"{h['id']}({h['score']:.3f})" for h in r["vector_hits"]))
    print(f"expanded triples        : {len(r['triples'])}")
    print("context chunks          :", ", ".join(r["chunks"].keys()))


def main():
    args = [a for a in sys.argv[1:]]
    compare = "--compare" in args
    trace = "--trace" in args
    args = [a for a in args if not a.startswith("--")]
    questions = args if args else DEMO_QUESTIONS

    rag = GraphRAG()
    try:
        for q in questions:
            sep(f"Q: {q}")
            if compare:
                base, hits = rag.baseline_answer(q)
                print("\n### Plain vector RAG (top-4 chunks)")
                print("retrieved:", ", ".join(h["id"] for h in hits))
                print(base)
            ans, r = rag.answer(q)
            print("\n### GraphRAG + GraphSAGE")
            if trace or compare:
                show_trace(r)
            print()
            print(ans)
    finally:
        rag.close()


if __name__ == "__main__":
    main()
