from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geoai_flood.rag_retriever import search_regulation


DEFAULT_QUERIES = [
    "Apa kewajiban lembaga jasa keuangan terkait penerapan keuangan berkelanjutan menurut POJK 51/2017?",
    "How does TCFD recommend disclosing physical climate risk?",
    "Bagaimana risiko banjir dan bencana hidrometeorologi dijelaskan dalam dokumen BNPB?",
    "Apa hubungan Taksonomi Hijau Indonesia dengan pembiayaan aktivitas berkelanjutan?",
    "What does IPCC AR6 say about climate risk and adaptation in Asia?",
]


def clean_snippet(text: str, max_chars: int = 900) -> str:
    snippet = " ".join(text.split())
    if len(snippet) <= max_chars:
        return snippet
    return snippet[: max_chars - 3].rstrip() + "..."


def citation(hit: dict) -> str:
    source = hit.get("source") or "unknown_source"
    page = hit.get("page")
    return f"{source}, p.{page}" if page else source


def build_context_pack(query: str, config_path: str) -> dict:
    hits = search_regulation(query, config_path)
    evidence = []
    for i, hit in enumerate(hits, start=1):
        evidence.append(
            {
                "rank": i,
                "citation": citation(hit),
                "source": hit.get("source"),
                "doc_name": hit.get("doc_name"),
                "page": hit.get("page"),
                "doc_type": hit.get("doc_type"),
                "language": hit.get("language"),
                "applied_filter": hit.get("applied_filter"),
                "snippet": clean_snippet(hit.get("content", "")),
            }
        )
    return {"query": query, "evidence": evidence}


def render_markdown(packs: list[dict]) -> str:
    lines = [
        "# RAG Context Pack",
        "",
        "Use only the evidence below when drafting answers. If the evidence is insufficient, say that the retrieved context is insufficient.",
        "",
    ]
    for pack in packs:
        lines.append(f"## Query: {pack['query']}")
        lines.append("")
        for item in pack["evidence"]:
            lines.append(f"### Evidence {item['rank']}: {item['citation']}")
            lines.append("")
            lines.append(f"- Document type: `{item.get('doc_type')}`")
            lines.append(f"- Language: `{item.get('language')}`")
            if item.get("applied_filter"):
                lines.append(f"- Applied filter: `{item['applied_filter']}`")
            lines.append("")
            lines.append(item["snippet"])
            lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="rag_config.json")
    parser.add_argument("--query", default=None, help="Single query. If omitted, demo queries are used.")
    parser.add_argument("--out-json", default="outputs/rag_context_pack.json")
    parser.add_argument("--out-md", default="outputs/rag_context_pack.md")
    args = parser.parse_args()

    queries = [args.query] if args.query else DEFAULT_QUERIES
    packs = [build_context_pack(query, args.config) for query in queries]

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(packs, indent=2, ensure_ascii=False), encoding="utf-8")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_markdown(packs), encoding="utf-8")

    for pack in packs:
        print("\nQUERY:", pack["query"])
        for item in pack["evidence"]:
            print(f"{item['rank']}. {item['citation']} [{item.get('language')}/{item.get('doc_type')}]")
    print(f"\nSaved context JSON: {out_json}")
    print(f"Saved context Markdown: {out_md}")


if __name__ == "__main__":
    main()
