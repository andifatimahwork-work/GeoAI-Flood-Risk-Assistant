from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geoai_flood.query_pipeline import run_query_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--rag-config", default="rag_config.json")
    parser.add_argument("--mode", choices=["auto", "live", "local"], default="local")
    parser.add_argument("--provider", choices=["groq", "ollama"], default=None)
    parser.add_argument("--dry-run", action="store_true", help="Prepare FSI/RAG context without calling the LLM.")
    parser.add_argument("--out", default="outputs/agent_query_result.json")
    args = parser.parse_args()

    payload = run_query_pipeline(
        query=args.query,
        config_path=args.config,
        rag_config_path=args.rag_config,
        mode=args.mode,
        provider=args.provider,
        dry_run=args.dry_run,
    )

    for result in payload["flood_score_results"]:
        print(f"{result['name']}: FSI={result['fsi_score']} category={result['category']}")

    print("\nRAG citations:")
    for idx, item in enumerate(payload["rag_results"], start=1):
        print(f"{idx}. {item['source']} p.{item['page']} [{item['doc_type']}]")

    if payload["agent_status"]["status"] == "ok":
        print("\nAgent response:")
        print(payload["agent_response"])
    elif payload["agent_status"]["status"] == "error":
        print("\nAgent response ERROR:")
        print(payload["agent_status"]["error"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved agent query result: {out}")


if __name__ == "__main__":
    main()
