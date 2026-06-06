from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from .env import load_project_env
from .geoai_tools import get_flood_score
from .rag_retriever import search_regulation

load_project_env()


AGENT_SYSTEM_PROMPT = """Anda adalah GeoAI Flood Risk Assistant untuk Jabodetabek.

Jawab hanya berdasarkan hasil tool dan konteks dokumen yang tersedia. Jangan membuat angka, sitasi,
atau klaim regulasi di luar evidence. Jika evidence tidak cukup, katakan bagian mana yang tidak
ditemukan.

Aturan grounding yang wajib:
- Hanya kutip sumber yang muncul di output search_regulation_tool.
- Jangan pernah membuat nomor halaman, nama dokumen, peraturan, threshold bisnis, LTV, haircut,
  atau kewajiban asuransi jika tidak muncul di konteks tool.
- Gunakan format sitasi: [S1: Nama_Dokumen.pdf, p.X]. Citation id S1/S2 harus sama persis dengan
  field citation_id dari tool output.
- Jika tool mengambil BNPB, IPCC, dan OJK Taksonomi, gunakan ketiganya bila relevan; jangan hanya
  memilih satu dokumen.
- Jika ada gap evidence, tulis sebagai catatan, bukan mengisi dengan asumsi.

Format jawaban default:
1. Ringkasan risiko lokasi: sebutkan FSI score, confidence interval, kategori, LULC, dan mode komputasi.
2. Faktor dominan: sebutkan 2 faktor terbesar dari top_dominant_factors atau weighted_breakdown.
3. Relevansi dokumen: hubungkan hasil lokasi dengan konteks regulasi/risiko dan sertakan sitasi valid.
4. Rekomendasi bisnis: sesuaikan dengan segmen user.

Segmentasi rekomendasi:
- Perbankan/collateral/KPR: beri rekomendasi proses kredit berbasis risiko, kebutuhan due diligence,
  mitigasi, dan dokumen pendukung. Jangan menyebut angka LTV/haircut kecuali ada evidence.
- Korporasi/ESG/TCFD: fokus pada physical climate risk disclosure, prioritasi aset, dan narasi laporan.
- Developer/site selection: fokus pada kelayakan lokasi, mitigasi desain, green financing, dan due diligence.

Aturan domain:
- Flood Susceptibility Index (FSI) threshold validasi utama adalah 0.65.
- Kategori FSI: <0.65 Low, 0.65-0.73 Medium-High, 0.73-0.76 High, >0.76 Very High.
- FSI memakai log-transform elevation sehingga area elevasi rendah mendapat penalti risiko lebih kuat;
  setelah sekitar 50 m, kontribusi elevation risk turun lebih cepat daripada normalisasi linear.
- Elevation score tinggi berarti elevasi rendah dan flood-risk contribution tinggi. Jangan
  interpretasikan elevation score sebagai nilai absolut elevasi yang tinggi.
- Slope score tinggi berarti wilayah datar/landai sehingga air lebih mudah tertahan dan menggenang.
  Jangan interpretasikan slope score tinggi sebagai kemiringan tanah yang curam.
- Model LULC U-Net memakai 5 kelas: Impervious, Vegetasi, Sawah, Air, dan Lahan_Terbuka.
- Kelas Informal tidak dilatih karena tidak tersedia eksplisit di ESA WorldCover; WorldPop dipakai
  sebagai proxy kepadatan penduduk dalam formula FSI, bukan input U-Net.
- FSI raster Minggu 1 adalah artifact validasi, visualisasi, dan credibility artifact. Untuk input
  koordinat user, gunakan on-the-fly computation jika GEE tersedia.
- Saat menjawab risiko lokasi, jelaskan FSI, kategori, confidence interval, LULC, dan faktor dominan.
- Saat menjawab regulasi, selalu sertakan sitasi valid dari output tool, bukan dari memori model.
"""


def _json_default(value: Any) -> str:
    return str(value)


@tool
def get_flood_score_tool(lat: float, lon: float) -> str:
    """Compute a point-level flood susceptibility score for a latitude and longitude in Jabodetabek."""

    config_path = os.getenv("GEOAI_CONFIG_PATH", "config/config.yaml")
    mode = os.getenv("FSI_COMPUTE_MODE", "auto")
    result = get_flood_score(lat, lon, config_path=config_path, mode=mode)
    weighted = result.get("weighted_breakdown") or {}
    top_factors = [
        {"factor": key, "weighted_contribution": value}
        for key, value in sorted(
            ((k, v) for k, v in weighted.items() if isinstance(v, (int, float))),
            key=lambda item: item[1],
            reverse=True,
        )[:2]
    ]
    keep = {
        "lat": result.get("lat"),
        "lon": result.get("lon"),
        "fsi_score": result.get("fsi_score"),
        "category": result.get("category"),
        "threshold": result.get("threshold"),
        "ci_low": result.get("ci_low"),
        "ci_high": result.get("ci_high"),
        "lulc": result.get("lulc"),
        "breakdown": result.get("breakdown"),
        "weighted_breakdown": result.get("weighted_breakdown"),
        "top_dominant_factors": top_factors,
        "computation_mode": result.get("computation_mode"),
        "cached": result.get("cached"),
        "warnings": result.get("warnings"),
        "notes": result.get("notes"),
    }
    return json.dumps(keep, ensure_ascii=False, indent=2, default=_json_default)


@tool
def search_regulation_tool(query: str) -> str:
    """Search regulatory, climate-risk, disaster-risk, and ESG documents with citations."""

    config_path = os.getenv("RAG_CONFIG_PATH", "rag_config.json")
    results = search_regulation(query, config_path=config_path)
    compact = [
        {
            "citation_id": f"S{idx}",
            "allowed_citation": f"[S{idx}: {item.get('source')}, p.{item.get('page')}]",
            "source": item.get("source"),
            "doc_name": item.get("doc_name"),
            "page": item.get("page"),
            "language": item.get("language"),
            "doc_type": item.get("doc_type"),
            "content": item.get("content"),
        }
        for idx, item in enumerate(results, start=1)
    ]
    payload = {
        "instruction": "Only cite sources listed in allowed_citations. Never invent document names or page numbers.",
        "allowed_citations": [item["allowed_citation"] for item in compact],
        "results": compact,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)


GROUNDED_SYNTHESIS_PROMPT = """Anda adalah GeoAI Flood Risk Assistant.

Tugas Anda hanya menyusun business recommendation report dari DATA_FSI dan KONTEKS_DOKUMEN yang
diberikan user. Jangan memanggil sumber lain dan jangan memakai memori model untuk nomor halaman.

Aturan wajib:
- Sistem mendeteksi MULTIPLE LOKASI jika DATA_FSI berisi lebih dari satu lokasi. Anda WAJIB membuat
  ringkasan terpisah untuk setiap lokasi tanpa terkecuali.
- Wajib gunakan label [SUMBER: ..., HALAMAN: ...] yang tertera tepat di atas teks dokumen sebagai
  basis sitasi. Jangan mengarang nomor halaman sendiri.
- Format sitasi final harus memakai allowed citation, misalnya [S1: Dokumen.pdf, p.8].
- Jangan menyebut LTV, haircut, kewajiban asuransi, atau angka bisnis lain jika tidak ada di konteks.
- Semua nilai breakdown dan weighted contribution adalah risk score/kontribusi risiko, bukan nilai
  fisik mentah. Elevation score tinggi berarti elevasi rendah (high flood risk), bukan elevasi tinggi.
- Slope score tinggi berarti slope datar/landai (high flood risk), bukan lereng curam.
- Jangan menyebut computation_mode, local_raster, cache, fallback, atau keterbatasan teknis pipeline
  di output bisnis.
- Jangan menulis "belum ada data yang cukup" jika konteks sudah cukup untuk first-pass screening.
  Gunakan kalimat: "Sistem ini adalah first-pass screening tool; due diligence lapangan tetap diperlukan."
- Jika kategori lokasi adalah NoData/Masked atau LULC adalah Air, jangan beri rekomendasi aset seolah-olah
  titik tersebut lahan darat. Tampilkan pesan persis: "Titik terdeteksi sebagai badan air. Untuk analisis
  aset, pilih koordinat pada bangunan atau lahan darat terdekat."
- Gunakan Bahasa Indonesia profesional dan ringkas.

Struktur output:
1. Ringkasan lokasi: tabel/bullet berisi nama, FSI, CI, kategori, LULC, dan top 2 faktor dominan.
2. Interpretasi risiko: jelaskan lokasi tertinggi dan alasan dari breakdown.
3. Relevansi dokumen: gunakan minimal dua sitasi valid jika tersedia; untuk developer, sertakan
   Taksonomi Hijau jika tersedia.
4. Rekomendasi bisnis: actionable, tetapi tetap berbasis evidence.
5. Catatan akhir: "Sistem ini adalah first-pass screening tool; due diligence lapangan tetap diperlukan."

Segmentasi rekomendasi:
- Jika pertanyaan menyebut collateral, KPR, kredit, bank, atau agunan, jawab sebagai perbankan:
  fokus pada screening agunan, mitigasi kredit, dokumen pendukung, dan due diligence.
- Jika pertanyaan menyebut TCFD, ESG, disclosure, portofolio, gudang/aset korporasi, jawab sebagai
  korporasi: fokus pada prioritas aset, physical climate risk disclosure, dan narasi laporan.
- Jika pertanyaan menyebut developer, site selection, lahan, perumahan, atau proyek, jawab sebagai
  developer: fokus pada perbandingan lokasi, mitigasi desain, green financing, dan due diligence.
"""


def _top_factor_text(result: dict[str, Any]) -> str:
    label_map = {
        "elevation": "low_elevation_risk",
        "slope": "flat_slope_risk",
        "rainfall": "rainfall_risk",
        "river_distance": "near_river_risk",
        "lulc": "lulc_risk",
        "population": "population_density_risk",
    }
    factors = result.get("top_dominant_factors")
    if not factors:
        weighted = result.get("weighted_breakdown") or {}
        factors = [
            {"factor": key, "weighted_contribution": value}
            for key, value in sorted(
                ((k, v) for k, v in weighted.items() if isinstance(v, (int, float))),
                key=lambda item: item[1],
                reverse=True,
            )[:2]
        ]
    return ", ".join(
        f"{label_map.get(item['factor'], item['factor'])}={item['weighted_contribution']}" for item in factors
    ) or "tidak tersedia"


def format_flood_context(flood_results: list[dict[str, Any]]) -> str:
    lines = []
    for idx, result in enumerate(flood_results, start=1):
        lulc = result.get("lulc") or {}
        water_message = ""
        if result.get("category") == "NoData/Masked" or lulc.get("class_name") == "Air":
            water_message = (
                "\n- Pesan wajib: Titik terdeteksi sebagai badan air. Untuk analisis aset, pilih koordinat "
                "pada bangunan atau lahan darat terdekat."
            )
        lines.append(
            "\n".join(
                [
                    f"LOKASI {idx}: {result.get('name', 'Tanpa nama')}",
                    f"- Koordinat: lat={result.get('lat')}, lon={result.get('lon')}",
                    f"- FSI: {result.get('fsi_score')} ({result.get('category')})",
                    f"- Confidence interval: {result.get('ci_low')} - {result.get('ci_high')}",
                    f"- LULC: {lulc.get('class_name')}",
                    f"- Top 2 faktor dominan: {_top_factor_text(result)}",
                    f"- Breakdown: {json.dumps(result.get('breakdown'), ensure_ascii=False)}{water_message}",
                ]
            )
        )
    return "\n\n".join(lines)


def format_document_context(rag_results: list[dict[str, Any]]) -> str:
    blocks = []
    for idx, doc in enumerate(rag_results, start=1):
        source = doc.get("source")
        page = doc.get("page")
        content = str(doc.get("content") or "").strip()
        blocks.append(
            "\n".join(
                [
                    f"[S{idx}]",
                    f"[SUMBER: {source}, HALAMAN: {page}]",
                    f"Allowed citation: [S{idx}: {source}, p.{page}]",
                    f"Konteks: {content}",
                ]
            )
        )
    return "\n\n".join(blocks)


def synthesize_business_report(
    question: str,
    flood_results: list[dict[str, Any]],
    rag_results: list[dict[str, Any]],
    provider: str | None = None,
) -> str:
    provider = provider or os.getenv("AGENT_LLM_PROVIDER", "groq")
    llm = _build_llm(provider)
    prompt = "\n\n".join(
        [
            f"PERTANYAAN_USER:\n{question}",
            f"DATA_FSI:\n{format_flood_context(flood_results)}",
            f"KONTEKS_DOKUMEN:\n{format_document_context(rag_results)}",
        ]
    )
    messages = [SystemMessage(content=GROUNDED_SYNTHESIS_PROMPT), HumanMessage(content=prompt)]
    try:
        return llm.invoke(messages).content
    except Exception as exc:
        if provider.lower() != "ollama" and _should_fallback_to_ollama(exc):
            fallback_llm = _build_llm("ollama")
            return fallback_llm.invoke(messages).content
        raise


def _build_llm(provider: str | None = None):
    provider = (provider or os.getenv("AGENT_LLM_PROVIDER", "groq")).lower()

    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY belum diset. Isi .env atau environment variable dulu.")
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=float(os.getenv("AGENT_TEMPERATURE", "0")),
            max_retries=int(os.getenv("AGENT_MAX_RETRIES", "2")),
        )

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise RuntimeError("Install langchain-ollama dulu untuk fallback Ollama.") from exc

        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
            base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            temperature=float(os.getenv("AGENT_TEMPERATURE", "0")),
        )

    raise ValueError("AGENT_LLM_PROVIDER harus 'groq' atau 'ollama'.")


def build_geoai_agent(provider: str | None = None):
    llm = _build_llm(provider)
    return create_agent(
        model=llm,
        tools=[get_flood_score_tool, search_regulation_tool],
        system_prompt=AGENT_SYSTEM_PROMPT,
    )


def _should_fallback_to_ollama(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(token in message for token in ("429", "rate limit", "resource_exhausted", "quota"))


def run_agent(question: str, lat: float | None = None, lon: float | None = None, provider: str | None = None) -> str:
    provider = provider or os.getenv("AGENT_LLM_PROVIDER", "groq")
    agent = build_geoai_agent(provider)
    user_message = question
    if lat is not None and lon is not None:
        user_message = f"Koordinat lokasi: lat={lat}, lon={lon}.\n\nPertanyaan: {question}"
    try:
        response = agent.invoke({"messages": [{"role": "user", "content": user_message}]})
    except Exception as exc:
        if provider.lower() != "ollama" and _should_fallback_to_ollama(exc):
            fallback_agent = build_geoai_agent("ollama")
            response = fallback_agent.invoke({"messages": [{"role": "user", "content": user_message}]})
        else:
            raise
    messages = response.get("messages", [])
    if not messages:
        return ""
    return messages[-1].content


def save_agent_response(path: str | Path, response: str, metadata: dict[str, Any] | None = None) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"response": response, "metadata": metadata or {}}
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
