from __future__ import annotations

import os
import re
from typing import Any

from .agent import synthesize_business_report
from .geoai_tools import get_flood_score
from .rag_retriever import search_regulation


COORD_RE = re.compile(r"(?P<lat>-?\d{1,2}(?:\.\d+)?)\s*,\s*(?P<lon>1\d{2}(?:\.\d+)?)")


def _clean_name(text: str) -> str:
    text = re.sub(r"[\(\[\{:=,-]+$", "", text.strip())
    parts = re.split(r"[:;,.]|\bvs\b|\bdan\b|\batau\b", text, flags=re.IGNORECASE)
    candidate = parts[-1].strip()
    words = candidate.split()
    if len(words) > 4:
        candidate = " ".join(words[-4:])
    if re.search(r"\bdi\s+", candidate, flags=re.IGNORECASE):
        candidate = re.split(r"\bdi\s+", candidate, flags=re.IGNORECASE)[-1]
    candidate = re.sub(r"^(koordinat|lokasi|di|untuk|rumah|gudang|lahan)\s+", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\s+(koordinat|coordinate|coordinates)$", "", candidate, flags=re.IGNORECASE)
    return candidate.strip(" -") or "Lokasi"


def extract_coordinates(query: str) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    seen: set[tuple[float, float]] = set()
    for idx, match in enumerate(COORD_RE.finditer(query), start=1):
        lat = float(match.group("lat"))
        lon = float(match.group("lon"))
        key = (round(lat, 6), round(lon, 6))
        if key in seen:
            continue
        seen.add(key)
        prefix = query[max(0, match.start() - 80) : match.start()]
        points.append(
            {
                "name": _clean_name(prefix) if prefix.strip() else f"Lokasi {idx}",
                "lat": lat,
                "lon": lon,
            }
        )
    return points


def infer_segment(query: str, point_count: int) -> str:
    q = query.lower()
    if any(word in q for word in ("collateral", "agunan", "kpr", "kredit", "bank", "nasabah")):
        return "perbankan_collateral"
    if any(word in q for word in ("tcfd", "esg", "disclosure", "laporan", "portofolio", "korporasi")):
        return "korporasi_esg_tcfd"
    if any(word in q for word in ("developer", "site selection", "lahan", "perumahan", "proyek", "dikembangkan")):
        return "developer_site_selection"
    if point_count > 1:
        return "multi_site_screening"
    return "general_location_screening"


def build_retrieval_query(query: str, segment: str) -> str:
    additions = {
        "perbankan_collateral": "POJK 51 Taksonomi Hijau physical climate risk collateral kredit agunan due diligence risiko banjir",
        "korporasi_esg_tcfd": "TCFD physical climate risk disclosure assets locations acute flooding ESG reporting",
        "developer_site_selection": "Taksonomi Hijau developer green financing due diligence lokasi proyek risiko banjir BNPB IPCC",
        "multi_site_screening": "physical climate risk site selection due diligence risiko banjir BNPB IPCC Taksonomi Hijau",
        "general_location_screening": "risiko banjir physical climate risk due diligence BNPB IPCC Taksonomi Hijau",
    }
    return f"{query} {additions.get(segment, '')}".strip()


def is_masked_or_water(result: dict[str, Any]) -> bool:
    lulc = result.get("lulc") or {}
    return result.get("category") == "NoData/Masked" or lulc.get("class_name") == "Air"


def masked_location_response(flood_results: list[dict[str, Any]]) -> str:
    names = ", ".join(str(item.get("name", "Lokasi")) for item in flood_results)
    return (
        f"Lokasi berikut terdeteksi sebagai badan air atau area yang dimask: {names}.\n\n"
        "Titik terdeteksi sebagai badan air. Untuk analisis aset, pilih koordinat pada bangunan "
        "atau lahan darat terdekat.\n\n"
        "Sistem ini adalah first-pass screening tool; due diligence lapangan tetap diperlukan."
    )


def run_query_pipeline(
    query: str,
    config_path: str = "config/config.yaml",
    rag_config_path: str = "rag_config.json",
    mode: str = "local",
    provider: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    os.environ["GEOAI_CONFIG_PATH"] = config_path
    os.environ["RAG_CONFIG_PATH"] = rag_config_path
    os.environ["FSI_COMPUTE_MODE"] = mode

    points = extract_coordinates(query)
    if not points:
        raise ValueError("Tidak ada koordinat valid yang ditemukan. Gunakan format lat, lon, contoh: -6.23, 107.01")

    segment = infer_segment(query, len(points))
    retrieval_query = build_retrieval_query(query, segment)

    flood_results = []
    for point in points:
        result = get_flood_score(point["lat"], point["lon"], config_path, mode=mode)
        flood_results.append({"name": point["name"], **result})

    all_masked = all(is_masked_or_water(result) for result in flood_results)
    if all_masked:
        response = masked_location_response(flood_results)
        return {
            "query": query,
            "segment": segment,
            "mode": mode,
            "points": points,
            "flood_score_results": flood_results,
            "retrieval_query": retrieval_query,
            "rag_results": [],
            "agent_response": response,
            "agent_status": {
                "status": "masked_location",
                "reason": "All queried coordinates were detected as water or NoData/Masked; RAG/LLM recommendation skipped.",
                "error": None,
            },
        }

    rag_results = search_regulation(retrieval_query, rag_config_path)

    response = None
    status = {"status": "skipped", "reason": "dry-run enabled", "error": None}
    if not dry_run:
        try:
            response = synthesize_business_report(
                question=f"Segment: {segment}\n\n{query}",
                flood_results=flood_results,
                rag_results=rag_results,
                provider=provider,
            )
            status = {"status": "ok", "reason": None, "error": None}
        except Exception as exc:
            status = {
                "status": "error",
                "reason": "LLM synthesis failed. Check API key, quota, provider, or fallback.",
                "error": str(exc),
            }

    return {
        "query": query,
        "segment": segment,
        "mode": mode,
        "points": points,
        "flood_score_results": flood_results,
        "retrieval_query": retrieval_query,
        "rag_results": rag_results,
        "agent_response": response,
        "agent_status": status,
    }
