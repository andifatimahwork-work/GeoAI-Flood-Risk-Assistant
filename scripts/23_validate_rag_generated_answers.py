from __future__ import annotations

import argparse
import json
from pathlib import Path


SENSITIVE_CLAIMS = {
    "greenwashing": ("greenwashing",),
    "Laporan Keberlanjutan": ("laporan keberlanjutan", "sustainability report"),
    "Rencana Aksi Keuangan Berkelanjutan": ("rencana aksi keuangan berkelanjutan",),
    "scenario analysis": ("scenario analysis", "analisis skenario"),
    "metrics and targets": ("metrics and targets", "metrik", "target"),
}


REQUIRED_FIELDS = {
    "id",
    "segment",
    "query_type",
    "question",
    "answer",
    "contexts",
    "ground_truth",
    "relevant_doc_source",
    "citations",
    "evidence",
}


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError("RAG generated answers must be a JSON list.")
    return rows


def validate_row(row: dict) -> list[str]:
    issues = []
    missing = REQUIRED_FIELDS - set(row)
    if missing:
        issues.append(f"missing_fields={sorted(missing)}")

    contexts = row.get("contexts", [])
    citations = row.get("citations", [])
    evidence = row.get("evidence", [])
    answer = str(row.get("answer", ""))
    context_text = " ".join(str(ctx) for ctx in contexts).lower()

    if not contexts:
        issues.append("empty_contexts")
    if not citations:
        issues.append("empty_citations")
    if len(contexts) != len(citations):
        issues.append(f"context_citation_count_mismatch={len(contexts)}!={len(citations)}")
    if len(evidence) != len(contexts):
        issues.append(f"evidence_context_count_mismatch={len(evidence)}!={len(contexts)}")

    answer_lower = answer.lower()
    for claim, support_terms in SENSITIVE_CLAIMS.items():
        if claim.lower() in answer_lower and not any(term in context_text for term in support_terms):
            issues.append(f"unsupported_sensitive_claim={claim}")

    if "tidak menemukan" not in answer_lower and "did not find" not in answer_lower:
        if not any(str(cite) in answer for cite in citations):
            issues.append("answer_missing_explicit_citation")

    return issues


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/rag_generated_answers.json")
    parser.add_argument("--out", default="outputs/rag_generated_answers_validation.json")
    args = parser.parse_args()

    rows = load_rows(Path(args.input))
    results = []
    for row in rows:
        issues = validate_row(row)
        results.append(
            {
                "id": row.get("id"),
                "issues": issues,
                "passed": not issues,
            }
        )

    report = {
        "input": args.input,
        "rows": len(rows),
        "passed_rows": sum(item["passed"] for item in results),
        "failed_rows": sum(not item["passed"] for item in results),
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({k: report[k] for k in ("rows", "passed_rows", "failed_rows")}, indent=2))
    failed = [item for item in results if not item["passed"]]
    if failed:
        print("\nRows to inspect:")
        for item in failed:
            print(f"- {item['id']}: {item['issues']}")
    else:
        print("\nAll generated answers passed structural and grounding checks.")
    print(f"\nSaved validation report: {out}")


if __name__ == "__main__":
    main()
