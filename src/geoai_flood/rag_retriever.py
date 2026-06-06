from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from .env import load_project_env

load_project_env()


def load_rag_config(path: str | Path = "rag_config.json") -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    chroma_path = os.getenv("CHROMA_PATH")
    if chroma_path:
        cfg["persist_directory"] = chroma_path
    return cfg


def build_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


ROUTE_RULES = [
    (("bnpb", "risiko bencana indonesia"), "BNPB_Risiko_Bencana_Indonesia_2024"),
    (("tcfd", "task force on climate-related financial disclosures"), "TCFD_Recommendations_Report"),
    (("pojk 51", "pojk no.51", "pojk nomor 51", "pojk.03/2017"), "OJK_POJK_No.51_2017"),
    (("taksonomi hijau", "green taxonomy"), "OJK_Taksonomi_Hijau_Indonesia"),
    (("ipcc", "ar6"), "IPCC_AR6_Chapter_10_Asia"),
    (("gri 201", "kinerja ekonomi", "economic impacts"), "GRI_201_Economic_Impacts"),
]


QUERY_EXPANSIONS = {
    "BNPB_Risiko_Bencana_Indonesia_2024": "kajian risiko bencana penanggulangan bencana banjir hidrometeorologi risiko kerugian",
    "TCFD_Recommendations_Report": "TCFD climate-related financial disclosure physical climate risk acute risk chronic risk flooding extreme weather water availability scenario analysis governance strategy risk management metrics targets financial impact exposure resilience assets locations",
    "OJK_POJK_No.51_2017": "POJK 51 POJK.03/2017 penerapan keuangan berkelanjutan prinsip keuangan berkelanjutan Rencana Aksi Keuangan Berkelanjutan RAKB Laporan Keberlanjutan Sustainability Report wajib menyampaikan laporan keberlanjutan kepada OJK lembaga jasa keuangan emiten perusahaan publik pasal 2 pasal 4 pasal 10",
    "OJK_Taksonomi_Hijau_Indonesia": "Taksonomi Hijau aktivitas hijau pembiayaan investasi hijau greenwashing klasifikasi monitoring manajemen risiko",
    "IPCC_AR6_Chapter_10_Asia": "Asia climate risk adaptation flood flooding extreme precipitation hazard exposure vulnerability",
    "GRI_201_Economic_Impacts": "GRI 201-2 Disclosure 201-2 financial implications risks opportunities due to climate change economic performance costs assets revenue operations",
}


TRANSLATION_EXPANSIONS = {
    "risiko fisik": "physical risk",
    "risiko iklim": "climate risk",
    "risiko banjir": "flood risk flooding physical climate risk",
    "banjir": "flood flooding",
    "pengungkapan": "disclosure",
    "pelaporan": "reporting disclosure",
    "dampak finansial": "financial impact financial implications",
    "dampak ekonomi": "economic impact financial implications",
    "aset": "assets",
    "properti": "property assets locations",
    "skenario": "scenario analysis",
    "metrik": "metrics targets",
    "manajemen risiko": "risk management",
}


def infer_doc_filters(query: str) -> list[dict[str, str]]:
    q = query.lower()
    filters = []
    seen = set()

    def add_doc(doc_name: str) -> None:
        if doc_name not in seen:
            filters.append({"doc_name": doc_name})
            seen.add(doc_name)

    def has_any(words: tuple[str, ...]) -> bool:
        return any(word in q for word in words)

    for triggers, doc_name in ROUTE_RULES:
        if any(trigger in q for trigger in triggers):
            add_doc(doc_name)

    explicit_doc_filters = list(filters)
    explicit_cross_doc = has_any(
        (
            "cross-document",
            "menggabungkan",
            "sama-sama",
            "sekaligus",
        )
    )
    if explicit_doc_filters and not explicit_cross_doc:
        return explicit_doc_filters
    if len(explicit_doc_filters) > 1 and explicit_cross_doc:
        return explicit_doc_filters

    flood_or_physical_risk = has_any(
        (
            "risiko banjir",
            "banjir",
            "flood",
            "flooding",
            "flood susceptibility",
            "fsi",
            "physical risk",
            "risiko fisik",
            "climate risk",
            "risiko iklim",
        )
    )
    green_finance = has_any(
        (
            "proyek hijau",
            "aktivitas hijau",
            "pembiayaan hijau",
            "pembiayaan proyek",
            "pembiayaan aktivitas",
            "green finance",
            "green project",
            "berkelanjutan",
            "taksonomi",
        )
    )
    financial_disclosure = has_any(
        (
            "pengungkapan",
            "disclosure",
            "pelaporan",
            "portofolio",
            "kredit",
            "aset",
            "finansial",
            "financial",
            "investasi",
        )
    )
    disaster_planning = has_any(
        (
            "pemerintah daerah",
            "mitigasi",
            "penanggulangan",
            "kajian risiko",
            "due diligence",
            "lokasi proyek",
            "kawasan hunian",
            "developer",
            "developer properti",
            "properti",
            "pemerintah",
            "daerah",
            "kebijakan daerah",
        )
    )
    policy_reporting = has_any(
        (
            "dokumen regulasi",
            "regulasi",
            "standar pelaporan",
            "pelaporan",
            "kebijakan daerah",
            "kebijakan",
            "pemangku kepentingan",
        )
    )
    developer_site_or_asset = has_any(
        (
            "developer",
            "developer properti",
            "properti",
            "lokasi proyek",
            "site selection",
            "due diligence",
            "kawasan hunian",
            "perumahan",
        )
    )

    # Intent-level routing for cross-document questions that do not explicitly
    # name every expected document.
    if green_finance:
        add_doc("OJK_Taksonomi_Hijau_Indonesia")
    if flood_or_physical_risk and (green_finance or financial_disclosure) and not has_any(("due diligence", "lokasi proyek", "developer properti")):
        add_doc("TCFD_Recommendations_Report")
    if flood_or_physical_risk and disaster_planning:
        add_doc("BNPB_Risiko_Bencana_Indonesia_2024")
        add_doc("IPCC_AR6_Chapter_10_Asia")
    if flood_or_physical_risk and developer_site_or_asset:
        add_doc("OJK_Taksonomi_Hijau_Indonesia")
    if flood_or_physical_risk and has_any(("adaptasi", "asia", "curah hujan", "extreme precipitation", "hazard", "vulnerability")):
        add_doc("IPCC_AR6_Chapter_10_Asia")
    if financial_disclosure and has_any(("dampak ekonomi", "implikasi ekonomi", "biaya", "pendapatan", "financial implications")):
        add_doc("GRI_201_Economic_Impacts")
    if has_any(("lembaga jasa keuangan", "bank", "portofolio kredit", "keuangan berkelanjutan")):
        add_doc("OJK_POJK_No.51_2017")
    if policy_reporting and (flood_or_physical_risk or disaster_planning or financial_disclosure):
        add_doc("OJK_POJK_No.51_2017")
        add_doc("TCFD_Recommendations_Report")
    if policy_reporting and disaster_planning:
        add_doc("BNPB_Risiko_Bencana_Indonesia_2024")

    if "cross-document" in q or "menggabungkan" in q or "sama-sama" in q:
        for keyword, doc_name in [
            ("risiko banjir", "BNPB_Risiko_Bencana_Indonesia_2024"),
            ("flood", "TCFD_Recommendations_Report"),
            ("physical", "TCFD_Recommendations_Report"),
            ("climate", "TCFD_Recommendations_Report"),
            ("keuangan berkelanjutan", "OJK_POJK_No.51_2017"),
            ("finansial", "GRI_201_Economic_Impacts"),
            ("ekonomi", "GRI_201_Economic_Impacts"),
        ]:
            if keyword in q and doc_name not in seen:
                add_doc(doc_name)
    return filters


def infer_doc_filter(query: str) -> dict[str, str] | None:
    filters = infer_doc_filters(query)
    return filters[0] if filters else None


def load_vectorstore(config_path: str | Path = "rag_config.json") -> Chroma:
    cfg = load_rag_config(config_path)
    embeddings = build_embeddings(cfg["embedding_model"])
    return Chroma(
        persist_directory=cfg["persist_directory"],
        embedding_function=embeddings,
    )


def _expanded_query(query: str, doc_filter: dict[str, str] | None) -> str:
    translation_terms = [
        expansion
        for keyword, expansion in TRANSLATION_EXPANSIONS.items()
        if keyword in query.lower()
    ]
    if not doc_filter:
        return f"{query} {' '.join(translation_terms)}".strip()
    doc_name = doc_filter.get("doc_name")
    expansion = QUERY_EXPANSIONS.get(doc_name, "")
    return f"{query} {' '.join(translation_terms)} {expansion}".strip()


def _retrieve_docs(vectorstore: Chroma, retriever_cfg: dict[str, Any], query: str, doc_filter: dict[str, str] | None, k: int) -> list[Any]:
    search_kwargs = {
        "k": int(k),
        "fetch_k": max(int(retriever_cfg.get("fetch_k", 20)), int(k) * 4),
        "lambda_mult": float(retriever_cfg.get("lambda_mult", 0.7)),
    }
    if doc_filter:
        search_kwargs["filter"] = doc_filter
    retriever = vectorstore.as_retriever(
        search_type=retriever_cfg.get("search_type", "mmr"),
        search_kwargs=search_kwargs,
    )
    return retriever.invoke(_expanded_query(query, doc_filter))


def _is_low_value_context(text: str) -> bool:
    lower = text.lower()
    if "references" in lower[:120] or "bibliography" in lower[:120] or "daftar pustaka" in lower[:160]:
        return True
    url_hits = lower.count("http://") + lower.count("https://") + lower.count("www.")
    citation_hits = lower.count(" et al") + lower.count("accessed on") + lower.count("accessed ")
    if url_hits >= 2 and citation_hits >= 1:
        return True
    if url_hits >= 4:
        return True
    doi_hits = lower.count("doi:")
    year_reference_hits = len(re.findall(r"\b20\d{2}:", lower[:900]))
    journal_markers = ("int. j.", "journal", "springer", "elsevier", "cambridge university press", " in: climate change")
    if doi_hits >= 1 and (year_reference_hits >= 1 or any(marker in lower for marker in journal_markers)):
        return True
    return False


def search_regulation(query: str, config_path: str | Path = "rag_config.json") -> list[dict[str, Any]]:
    cfg = load_rag_config(config_path)
    vectorstore = load_vectorstore(config_path)
    retriever_cfg = cfg["retriever"]
    k = int(retriever_cfg.get("k", 5))
    applied_filters = []
    docs = []
    if retriever_cfg.get("enable_query_routing", True):
        applied_filters = infer_doc_filters(query)

    if len(applied_filters) > 1:
        per_doc_k = max(2, (k + len(applied_filters) - 1) // len(applied_filters))
        grouped_docs = []
        for doc_filter in applied_filters:
            grouped_docs.append(_retrieve_docs(vectorstore, retriever_cfg, query, doc_filter, per_doc_k))
        for group in grouped_docs:
            if group:
                docs.append(group[0])
        for group in grouped_docs:
            docs.extend(group[1:])
    elif len(applied_filters) == 1:
        docs = _retrieve_docs(vectorstore, retriever_cfg, query, applied_filters[0], k)
        if not docs:
            docs = _retrieve_docs(vectorstore, retriever_cfg, query, None, k)
            applied_filters = []
    else:
        docs = _retrieve_docs(vectorstore, retriever_cfg, query, None, k)

    results = []
    seen = set()
    for doc in docs:
        meta = dict(doc.metadata)
        key = (meta.get("source"), meta.get("page"), meta.get("chunk_id"))
        if key in seen:
            continue
        if _is_low_value_context(doc.page_content):
            continue
        seen.add(key)
        results.append(
            {
                "content": doc.page_content,
                "source": meta.get("source"),
                "doc_name": meta.get("doc_name"),
                "page": meta.get("page"),
                "language": meta.get("language"),
                "doc_type": meta.get("doc_type"),
                "relevance": meta.get("relevance"),
                "applied_filter": {"doc_name": meta.get("doc_name")} if applied_filters else None,
            }
        )
        if len(results) >= k:
            break
    return results
