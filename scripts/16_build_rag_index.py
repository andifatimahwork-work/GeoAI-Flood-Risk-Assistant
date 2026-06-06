from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from geoai_flood.rag_retriever import build_embeddings, load_rag_config


NOISY_MARKERS = (
    "daftar pustaka",
    "references",
    "bibliography",
    "appendices",
    "table of contents",
)


def is_noisy_chunk(text: str) -> bool:
    compact = " ".join(text.split())
    lower = compact.lower()
    if len(compact) < 120:
        return True

    bad_chars = compact.count("�") + compact.count("ï¿½") + compact.count("â€") + compact.count("Â")
    if bad_chars / max(len(compact), 1) > 0.015:
        return True

    marker_hits = sum(marker in lower for marker in NOISY_MARKERS)
    if marker_hits and len(compact) < 1400:
        return True

    digit_ratio = sum(ch.isdigit() for ch in compact) / max(len(compact), 1)
    if digit_ratio > 0.25:
        return True

    return False


def load_documents(cfg: dict) -> list:
    docs_dir = Path(cfg["docs_dir"])
    loaded = []
    for pdf_path in sorted(docs_dir.glob("*.pdf")):
        doc_cfg = cfg["documents"].get(pdf_path.name, {})
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        for page in pages:
            page.metadata.update(
                {
                    "source": pdf_path.name,
                    "doc_name": pdf_path.stem,
                    "language": doc_cfg.get("language", "unknown"),
                    "doc_type": doc_cfg.get("doc_type", "unknown"),
                    "relevance": doc_cfg.get("relevance", "unknown"),
                    # PyPDFLoader page is zero-based; expose human-friendly page number.
                    "page": int(page.metadata.get("page", 0)) + 1,
                }
            )
        loaded.extend(pages)
    if not loaded:
        raise RuntimeError(f"No PDF pages loaded from {docs_dir}")
    return loaded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="rag_config.json")
    parser.add_argument("--reset", action="store_true", help="Delete existing Chroma DB before indexing.")
    args = parser.parse_args()

    cfg = load_rag_config(args.config)
    persist_dir = Path(cfg["persist_directory"])
    if args.reset and persist_dir.exists():
        shutil.rmtree(persist_dir)

    pages = load_documents(cfg)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(cfg["chunk_size"]),
        chunk_overlap=int(cfg["chunk_overlap"]),
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    raw_chunk_count = len(chunks)
    chunks = [chunk for chunk in chunks if not is_noisy_chunk(chunk.page_content)]
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    embeddings = build_embeddings(cfg["embedding_model"])
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_dir),
    )
    count = vectorstore._collection.count()
    summary = {
        "pages_loaded": len(pages),
        "raw_chunks": raw_chunk_count,
        "chunks_indexed": count,
        "chunks_removed_as_noise": raw_chunk_count - len(chunks),
        "persist_directory": str(persist_dir),
        "embedding_model": cfg["embedding_model"],
        "chunk_size": cfg["chunk_size"],
        "chunk_overlap": cfg["chunk_overlap"],
        "retriever": cfg["retriever"],
    }
    out = Path("outputs/rag_index_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
