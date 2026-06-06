from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geoai_flood.agent import synthesize_business_report
from geoai_flood.geoai_tools import get_flood_score
from geoai_flood.rag_retriever import search_regulation


POINTS = [
    {"name": "Jakarta Utara", "lat": -6.1214, "lon": 106.7741},
    {"name": "Bogor", "lat": -6.5950, "lon": 106.8167},
    {"name": "Depok", "lat": -6.4025, "lon": 106.7942},
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--rag-config", default="rag_config.json")
    parser.add_argument("--mode", choices=["auto", "live", "local"], default="local")
    parser.add_argument("--run-agent", action="store_true")
    parser.add_argument("--out", default="outputs/agent_tool_smoke_test.json")
    args = parser.parse_args()

    os.environ["GEOAI_CONFIG_PATH"] = args.config
    os.environ["RAG_CONFIG_PATH"] = args.rag_config
    os.environ["FSI_COMPUTE_MODE"] = args.mode

    flood_results = []
    for point in POINTS:
        result = get_flood_score(point["lat"], point["lon"], args.config, mode=args.mode)
        flood_results.append({"name": point["name"], **result})
        print(f"{point['name']}: FSI={result['fsi_score']} category={result['category']} mode={result['computation_mode']}")

    rag_query = "Bagaimana developer properti dapat menggunakan flood susceptibility score untuk due diligence lokasi proyek?"
    rag_results = search_regulation(rag_query, args.rag_config)
    print("\nRAG citations:")
    for idx, item in enumerate(rag_results[:5], start=1):
        print(f"{idx}. {item['source']} p.{item['page']} [{item['doc_type']}]")

    agent_result = {
        "status": "skipped",
        "reason": "Run with --run-agent and set GROQ_API_KEY to test LLM synthesis.",
        "response": None,
        "error": None,
    }
    if args.run_agent:
        try:
            response = synthesize_business_report(
                "Bandingkan semua lokasi hasil scoring dan berikan rekomendasi due diligence untuk developer properti.",
                flood_results=flood_results,
                rag_results=rag_results,
            )
            agent_result = {
                "status": "ok",
                "reason": None,
                "response": response,
                "error": None,
            }
            print("\nAgent response:")
            print(response)
        except Exception as exc:
            agent_result = {
                "status": "error",
                "reason": "Agent LLM call failed. Check GROQ_API_KEY, Groq quota, or Ollama fallback.",
                "response": None,
                "error": str(exc),
            }
            print("\nAgent response ERROR:")
            print(exc)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "flood_score_results": flood_results,
                "rag_query": rag_query,
                "rag_results": rag_results,
                "agent_response": agent_result["response"],
                "agent_status": agent_result,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved agent/tool smoke test: {out}")


if __name__ == "__main__":
    main()
