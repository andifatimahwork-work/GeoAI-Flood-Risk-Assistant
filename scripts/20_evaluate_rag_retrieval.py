from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geoai_flood.rag_retriever import search_regulation


def load_queries(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_one(item: dict, config_path: str) -> dict:
    hits = search_regulation(item["question"], config_path)
    retrieved_sources = [hit.get("source") for hit in hits if hit.get("source")]
    expected_sources = item["relevant_doc_source"]

    retrieved_set = set(retrieved_sources)
    expected_set = set(expected_sources)
    matched = retrieved_set & expected_set

    source_precision = len([src for src in retrieved_sources if src in expected_set]) / max(len(retrieved_sources), 1)
    source_recall = len(matched) / max(len(expected_set), 1)
    top1_match = bool(retrieved_sources and retrieved_sources[0] in expected_set)
    all_required_found = expected_set.issubset(retrieved_set)

    return {
        "id": item["id"],
        "segment": item["segment"],
        "query_type": item["query_type"],
        "question": item["question"],
        "expected_sources": expected_sources,
        "retrieved_sources": retrieved_sources,
        "top1_match": top1_match,
        "all_required_found": all_required_found,
        "source_precision": round(source_precision, 4),
        "source_recall": round(source_recall, 4),
        "hits": hits,
    }


def summarize(results: list[dict]) -> dict:
    by_segment = defaultdict(list)
    by_query_type = defaultdict(list)
    for result in results:
        by_segment[result["segment"]].append(result)
        by_query_type[result["query_type"]].append(result)

    def agg(rows: list[dict]) -> dict:
        return {
            "queries": len(rows),
            "top1_accuracy": round(sum(r["top1_match"] for r in rows) / max(len(rows), 1), 4),
            "all_required_found_rate": round(sum(r["all_required_found"] for r in rows) / max(len(rows), 1), 4),
            "mean_source_precision": round(sum(r["source_precision"] for r in rows) / max(len(rows), 1), 4),
            "mean_source_recall": round(sum(r["source_recall"] for r in rows) / max(len(rows), 1), 4),
        }

    return {
        "overall": agg(results),
        "by_segment": {key: agg(rows) for key, rows in sorted(by_segment.items())},
        "by_query_type": {key: agg(rows) for key, rows in sorted(by_query_type.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="rag_config.json")
    parser.add_argument("--queries", default="test_queries.json")
    parser.add_argument("--out", default="outputs/rag_retrieval_eval_report.json")
    args = parser.parse_args()

    queries = load_queries(args.queries)
    results = [evaluate_one(item, args.config) for item in queries]
    summary = summarize(results)

    report = {
        "summary": summary,
        "results": results,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    failures = [r for r in results if not r["top1_match"] or not r["all_required_found"]]
    if failures:
        print("\nQueries to inspect:")
        for row in failures:
            print(f"- {row['id']}: top1={row['top1_match']} all_required={row['all_required_found']}")
            print(f"  expected: {row['expected_sources']}")
            print(f"  retrieved: {row['retrieved_sources']}")
    else:
        print("\nAll queries retrieved the expected source set.")
    print(f"\nSaved retrieval evaluation report: {out}")


if __name__ == "__main__":
    main()
