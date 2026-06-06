from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def load_rows(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def jsonable(value):
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, float) and np.isnan(value):
            return None
    except Exception:
        pass
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    return value


def collect_nan_metrics(summary: dict, scores: list[dict]) -> list[str]:
    nan_metrics = []
    for key, value in summary.items():
        if value is None:
            nan_metrics.append(str(key))
    for row in scores:
        for key, value in row.items():
            if value is None and key not in {"user_input", "response", "reference"}:
                nan_metrics.append(str(key))
    return sorted(set(nan_metrics))


def build_evaluator_llm(provider: str, google_model: str, groq_model: str):
    provider = provider.lower()
    if provider == "auto":
        if os.getenv("GOOGLE_API_KEY"):
            provider = "google"
        elif os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        elif os.getenv("GROQ_API_KEY"):
            provider = "groq"

    if provider == "google":
        if not os.getenv("GOOGLE_API_KEY"):
            raise RuntimeError("Set GOOGLE_API_KEY before using --provider google.")
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from ragas.llms import LangchainLLMWrapper
        except ImportError as exc:
            raise RuntimeError(
                "Google AI Studio support needs langchain-google-genai. "
                "Run: pip install langchain-google-genai"
            ) from exc
        langchain_llm = ChatGoogleGenerativeAI(
            model=google_model,
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
        return (LangchainLLMWrapper(langchain_llm), "google", google_model)

    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("Set OPENAI_API_KEY before using --provider openai.")
        return None, "openai", "ragas_default_openai"

    if provider == "groq":
        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError("Set GROQ_API_KEY before using --provider groq.")
        try:
            from langchain_groq import ChatGroq
            from ragas.llms import LangchainLLMWrapper
        except ImportError as exc:
            raise RuntimeError(
                "Groq support needs langchain-groq. "
                "Run: pip install langchain-groq"
            ) from exc
        langchain_llm = ChatGroq(
            model=groq_model,
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY"),
        )
        return (LangchainLLMWrapper(langchain_llm), "groq", groq_model)

    raise RuntimeError("No supported evaluator API key found. Set GOOGLE_API_KEY, GROQ_API_KEY, or OPENAI_API_KEY.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/rag_generated_answers.json")
    parser.add_argument("--out", default="outputs/ragas_report.json")
    parser.add_argument("--provider", default="auto", choices=["auto", "google", "openai", "groq"])
    parser.add_argument("--google-model", default="gemini-2.5-flash")
    parser.add_argument("--groq-model", default="llama-3.3-70b-versatile")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N rows for a quick smoke test.")
    parser.add_argument("--timeout", type=int, default=420, help="Per-call RAGAS timeout in seconds.")
    parser.add_argument("--max-workers", type=int, default=1, help="Concurrent evaluator workers. Keep 1 for free-tier APIs.")
    parser.add_argument(
        "--no-raise-exceptions",
        action="store_true",
        help="Let RAGAS return nan instead of stopping on evaluator errors.",
    )
    parser.add_argument(
        "--allow-no-api-key",
        action="store_true",
        help="Write a readiness report instead of failing when no evaluator API key is configured.",
    )
    args = parser.parse_args()

    rows = load_rows(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]
    has_api_key = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY"))
    if not has_api_key:
        message = (
            "RAGAS faithfulness/context precision needs an evaluator LLM. "
            "Set GOOGLE_API_KEY, GROQ_API_KEY, or OPENAI_API_KEY before running this script."
        )
        if not args.allow_no_api_key:
            raise RuntimeError(message)

        report = {
            "status": "skipped_no_evaluator_api_key",
            "message": message,
            "input": args.input,
            "queries": len(rows),
            "available_fields": ["user_input", "response", "retrieved_contexts", "reference"],
            "next_command_after_key": "python scripts/22_run_ragas_eval.py --input outputs/rag_generated_answers.json --provider groq",
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\nSaved readiness report: {out}")
        return

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference
        from ragas.run_config import RunConfig
    except ImportError as exc:
        raise RuntimeError("Install RAGAS dependencies first: pip install ragas datasets") from exc

    llm, evaluator_provider, evaluator_model = build_evaluator_llm(args.provider, args.google_model, args.groq_model)
    metrics = [
        LLMContextPrecisionWithReference(llm=llm),
        Faithfulness(llm=llm),
    ]

    dataset = Dataset.from_list(
        [
            {
                "user_input": row["question"],
                "response": row["answer"],
                "retrieved_contexts": row["contexts"],
                "reference": row["ground_truth"],
            }
            for row in rows
        ]
    )

    result = evaluate(
        dataset,
        metrics=metrics,
        llm=llm,
        run_config=RunConfig(
            timeout=int(args.timeout),
            max_retries=3,
            max_wait=30,
            max_workers=int(args.max_workers),
        ),
        batch_size=1,
        raise_exceptions=not args.no_raise_exceptions,
    )
    try:
        scores = jsonable(result.to_pandas().to_dict(orient="records"))
        summary = jsonable(dict(result))
    except Exception:
        scores = []
        summary = {"raw_result": str(result)}

    report = {
        "status": "completed_with_nan_metric" if collect_nan_metrics(summary, scores) else "completed",
        "evaluator_provider": evaluator_provider,
        "evaluator_model": evaluator_model,
        "summary": summary,
        "scores": scores,
        "nan_metrics": collect_nan_metrics(summary, scores),
        "input": args.input,
        "queries": len(rows),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved RAGAS report: {out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
