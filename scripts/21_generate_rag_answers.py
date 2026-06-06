from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geoai_flood.rag_answer import generate_cited_answer


def load_queries(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="rag_config.json")
    parser.add_argument("--queries", default="test_queries.json")
    parser.add_argument("--out", default="outputs/rag_generated_answers.json")
    parser.add_argument("--max-evidence", type=int, default=4)
    args = parser.parse_args()

    rows = []
    for item in load_queries(args.queries):
        generated = generate_cited_answer(
            item["question"],
            config_path=args.config,
            max_evidence=args.max_evidence,
        )
        row = {
            "id": item["id"],
            "segment": item["segment"],
            "query_type": item["query_type"],
            "question": item["question"],
            "system_prompt": generated["system_prompt"],
            "answer_policy": generated["answer_policy"],
            "answer": generated["answer"],
            "contexts": generated["contexts"],
            "ground_truth": item["ground_truth_answer"],
            "ground_truth_answer": item["ground_truth_answer"],
            "relevant_doc_source": item["relevant_doc_source"],
            "citations": generated["citations"],
            "evidence": generated["evidence"],
        }
        rows.append(row)

        print(f"\n{item['id']} [{item['segment']}]")
        print(item["question"])
        for cite in generated["citations"]:
            print(f"  - {cite}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    with tmp.open("r", encoding="utf-8") as f:
        parsed = json.load(f)
    if len(parsed) != len(rows):
        raise RuntimeError(f"Generated row count mismatch: wrote {len(rows)} but parsed {len(parsed)}")
    os.replace(tmp, out)
    print(f"\nSaved generated RAG answers: {out}")


if __name__ == "__main__":
    main()
