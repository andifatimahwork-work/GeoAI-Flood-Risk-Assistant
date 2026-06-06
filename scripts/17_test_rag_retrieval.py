from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geoai_flood.rag_retriever import search_regulation


QUERIES = [
    "Apa kewajiban lembaga jasa keuangan terkait penerapan keuangan berkelanjutan menurut POJK 51/2017?",
    "How does TCFD recommend disclosing physical climate risk?",
    "Bagaimana risiko banjir dan bencana hidrometeorologi dijelaskan dalam dokumen BNPB?",
    "Apa hubungan Taksonomi Hijau Indonesia dengan pembiayaan aktivitas berkelanjutan?",
    "What does IPCC AR6 say about climate risk and adaptation in Asia?",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="rag_config.json")
    parser.add_argument("--out", default="outputs/rag_retrieval_smoke_test.json")
    args = parser.parse_args()

    results = []
    for query in QUERIES:
        hits = search_regulation(query, args.config)
        results.append({"query": query, "hits": hits})
        print("\nQUERY:", query)
        for i, hit in enumerate(hits, start=1):
            snippet = " ".join(hit["content"].split())[:260]
            print(f"{i}. {hit['source']} p.{hit['page']} [{hit['language']}/{hit['doc_type']}]")
            print("   ", snippet)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved retrieval smoke test: {out}")


if __name__ == "__main__":
    main()
