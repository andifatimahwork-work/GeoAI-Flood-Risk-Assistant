from __future__ import annotations

import json
import os
import sys
from html import escape
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from geoai_flood.query_pipeline import extract_coordinates, run_query_pipeline
from geoai_flood.env import load_project_env
from geoai_flood.geoai_tools import gee_live_status

ENV_PATH = load_project_env()


EXAMPLES = {
    "Perbankan - collateral": "Gudang logistik kami di Bekasi koordinat -6.23, 107.01 apakah layak dijadikan collateral kredit?",
    "Korporasi - ESG/TCFD": "Kami punya 3 gudang: Cakung (-6.1831, 106.9370), Cilincing (-6.1077, 106.9238), Marunda (-6.0993, 106.9720). Mana yang wajib masuk laporan TCFD kami tahun ini?",
    "Developer - site selection": "Serpong (-6.3194, 106.6641) vs Bekasi Timur (-6.2641, 107.0101), mana yang lebih aman dikembangkan untuk proyek perumahan?",
}


def score_color(category: str | None) -> str:
    return {
        "Low": "#10B981",
        "Medium-High": "#F59E0B",
        "High": "#EF4444",
        "Very High": "#DC2626",
        "NoData/Masked": "#64748B",
    }.get(category or "", "#495057")


def score_rgb(category: str | None) -> list[int]:
    return {
        "Low": [16, 185, 129, 210],
        "Medium-High": [245, 158, 11, 220],
        "High": [239, 68, 68, 225],
        "Very High": [220, 38, 38, 230],
        "NoData/Masked": [100, 116, 139, 210],
    }.get(category or "", [75, 85, 99, 210])


def factor_label(name: str) -> str:
    return {
        "elevation": "Elevasi rendah",
        "slope": "Dataran landai",
        "rainfall": "Curah hujan",
        "river_distance": "Dekat sungai",
        "lulc": "Tutupan lahan",
        "population": "Kepadatan penduduk",
    }.get(name, name.replace("_", " ").title())


def results_table(flood_results: list[dict]) -> pd.DataFrame:
    rows = []
    for item in flood_results:
        top = item.get("top_dominant_factors") or []
        top_text = ", ".join(f"{factor_label(x['factor'])} ({x['weighted_contribution']})" for x in top)
        rows.append(
            {
                "Lokasi": item.get("name"),
                "FSI": item.get("fsi_score"),
                "Kategori": item.get("category"),
                "CI": f"{item.get('ci_low')} - {item.get('ci_high')}",
                "LULC": (item.get("lulc") or {}).get("class_name"),
                "Faktor dominan": top_text,
                "lat": item.get("lat"),
                "lon": item.get("lon"),
            }
        )
    return pd.DataFrame(rows)


def map_dataframe(flood_results: list[dict]) -> pd.DataFrame:
    rows = []
    for item in flood_results:
        rows.append(
            {
                "lat": item.get("lat"),
                "lon": item.get("lon"),
                "name": item.get("name"),
                "fsi_score": item.get("fsi_score"),
                "category": item.get("category"),
                "color": score_rgb(item.get("category")),
            }
        )
    return pd.DataFrame(rows)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #F8F9FA;
            color: #111827;
        }
        [data-testid="stSidebar"] {
            background: #FFFFFF;
            border-right: 1px solid #E5E7EB;
        }
        h1, h2, h3 {
            color: #0F172A;
            letter-spacing: 0;
        }
        .subtitle {
            color: #475569;
            font-size: 0.98rem;
            margin-top: -0.35rem;
            margin-bottom: 1rem;
        }
        .hero-band {
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-left: 5px solid #0F766E;
            border-radius: 8px;
            padding: 16px 18px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
            margin-bottom: 14px;
        }
        .kpi-card {
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-top: 4px solid var(--risk-color);
            border-radius: 8px;
            padding: 14px 16px;
            margin-bottom: 12px;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
        }
        .kpi-location {
            color: #111827;
            font-size: 0.92rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0;
        }
        .kpi-score {
            color: #111827;
            font-size: 2.15rem;
            line-height: 1.05;
            font-weight: 800;
            margin: 8px 0 2px;
        }
        .risk-pill {
            display: inline-block;
            background: var(--risk-color);
            color: #FFFFFF;
            border-radius: 999px;
            padding: 3px 10px;
            font-size: 0.78rem;
            font-weight: 700;
        }
        .kpi-meta {
            color: #475569;
            font-size: 0.86rem;
            line-height: 1.45;
            margin-top: 8px;
        }
        .agent-panel {
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 8px;
            padding: 16px 18px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        }
        div.stButton > button[kind="primary"] {
            background: #0F766E;
            border-color: #0F766E;
            color: white;
        }
        div.stButton > button[kind="primary"]:hover {
            background: #115E59;
            border-color: #115E59;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(item: dict) -> None:
    top = item.get("top_dominant_factors") or []
    factors = ", ".join(factor_label(x["factor"]) for x in top) or "Tidak tersedia"
    lulc = (item.get("lulc") or {}).get("class_name") or "Tidak tersedia"
    category = item.get("category") or "NoData"
    color = score_color(category)
    water_note = ""
    if category == "NoData/Masked" or lulc == "Air":
        water_note = (
            "<br><strong>Catatan:</strong> Titik terdeteksi sebagai badan air. "
            "Untuk analisis aset, pilih koordinat pada bangunan atau lahan darat terdekat."
        )
    html = (
        f'<div class="kpi-card" style="--risk-color:{color}">'
        f'<div class="kpi-location">{escape(str(item.get("name", "Lokasi")))}</div>'
        f'<div class="kpi-score">{escape(str(item.get("fsi_score")))}</div>'
        f'<span class="risk-pill">{escape(category)} Risk</span>'
        '<div class="kpi-meta">'
        f'<strong>CI:</strong> {escape(str(item.get("ci_low")))} - {escape(str(item.get("ci_high")))}<br>'
        f'<strong>Karakteristik tanah:</strong> {escape(str(lulc))}<br>'
        f'<strong>Faktor dominan:</strong> {escape(factors)}'
        f"{water_note}"
        "</div>"
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def render_risk_map(flood_results: list[dict]) -> None:
    map_df = map_dataframe(flood_results)
    if map_df.empty:
        st.info("Belum ada lokasi untuk ditampilkan di peta.")
        return

    center_lat = float(map_df["lat"].mean())
    center_lon = float(map_df["lon"].mean())
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position="[lon, lat]",
        get_fill_color="color",
        get_radius=900,
        pickable=True,
        stroked=True,
        get_line_color=[255, 255, 255, 230],
        line_width_min_pixels=2,
    )
    deck = pdk.Deck(
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=9.5, pitch=0),
        layers=[layer],
        tooltip={
            "html": "<b>{name}</b><br/>FSI: {fsi_score}<br/>Risk: {category}",
            "style": {"backgroundColor": "#0F172A", "color": "white"},
        },
    )
    st.pydeck_chart(deck, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="GeoAI Flood Risk Assistant", layout="wide")
    inject_css()

    st.title("GeoAI Flood Risk Assistant")
    st.markdown(
        '<p class="subtitle">First-pass flood risk screening untuk Jabodetabek berbasis FSI, GeoAI, dan RAG regulasi.</p>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Settings")
        config_path = st.text_input("Config path", "config/config.yaml")
        rag_config_path = st.text_input("RAG config path", "rag_config.json")
        mode = st.selectbox("FSI mode", ["local", "auto", "live"], index=0)
        provider = st.selectbox("LLM provider", ["groq", "ollama"], index=0)
        dry_run = st.checkbox("Dry run tanpa LLM", value=False)
        st.divider()
        st.caption("API status")
        gee_status = gee_live_status(config_path)
        st.write(".env path:", str(ENV_PATH))
        st.write("GROQ_API_KEY:", "tersedia" if os.getenv("GROQ_API_KEY") else "belum diset")
        st.write("GEE local live:", "siap" if gee_status["ready_for_local_live"] else "belum siap")
        st.write("GEE service key:", "tersedia" if gee_status["service_account_key_exists"] else "belum diset")

    if "query_text" not in st.session_state:
        st.session_state.query_text = EXAMPLES["Perbankan - collateral"]

    st.markdown('<div class="hero-band">', unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)
    if col_a.button("Perbankan"):
        st.session_state.query_text = EXAMPLES["Perbankan - collateral"]
    if col_b.button("ESG / TCFD"):
        st.session_state.query_text = EXAMPLES["Korporasi - ESG/TCFD"]
    if col_c.button("Developer"):
        st.session_state.query_text = EXAMPLES["Developer - site selection"]

    query = st.text_area("Pertanyaan user", key="query_text", height=110)
    preview_col, action_col = st.columns([3, 1])
    points_preview = extract_coordinates(query)
    with preview_col:
        if points_preview:
            st.caption(f"Terdeteksi {len(points_preview)} koordinat: " + ", ".join(p["name"] for p in points_preview))
        else:
            st.warning("Belum ada koordinat valid. Gunakan format `lat, lon`, contoh `-6.23, 107.01`.")
    with action_col:
        run = st.button("Run Analysis", type="primary", disabled=not bool(points_preview), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if run:
        if mode == "live" and not gee_live_status(config_path)["ready_for_local_live"]:
            st.error("Mode live membutuhkan Earth Engine auth. Set EARTHENGINE_PROJECT dan jalankan earthengine authenticate, atau isi GEE_SERVICE_ACCOUNT_KEY.")
            return
        with st.spinner("Menghitung FSI, mengambil konteks RAG, dan menyusun rekomendasi..."):
            try:
                result = run_query_pipeline(
                    query=query,
                    config_path=config_path,
                    rag_config_path=rag_config_path,
                    mode=mode,
                    provider=provider,
                    dry_run=dry_run,
                )
            except Exception as exc:
                st.error(f"Pipeline gagal: {exc}")
                return
        st.session_state.result = result

    result = st.session_state.get("result")
    if not result:
        return

    st.divider()

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("Spatial Risk Overview")
        render_risk_map(result["flood_score_results"])

        st.subheader("Executive KPI Cards")
        for item in result["flood_score_results"]:
            render_kpi_card(item)

        with st.expander("Tabel Detail FSI"):
            df = results_table(result["flood_score_results"])
            st.dataframe(df.drop(columns=["lat", "lon"]), use_container_width=True)

    with right:
        st.subheader("AI Agent & Insights")
        st.markdown('<div class="agent-panel">', unsafe_allow_html=True)
        if result["agent_status"]["status"] == "ok":
            st.markdown(result["agent_response"])
        elif result["agent_status"]["status"] == "masked_location":
            st.warning(result["agent_response"])
        elif result["agent_status"]["status"] == "skipped":
            st.info("Dry run aktif. Agent response belum dibuat.")
        else:
            st.error(result["agent_status"]["error"])
        st.markdown("</div>", unsafe_allow_html=True)

    if result["rag_results"]:
        with st.expander("Lihat 5 Dokumen Regulasi & Referensi Ilmiah Terkait"):
            for idx, item in enumerate(result["rag_results"], start=1):
                st.markdown(f"**S{idx}. {item['source']} p.{item['page']}** [{item['doc_type']}]")
                st.write(item["content"][:1200])
    else:
        st.info("RAG tidak dijalankan karena seluruh titik terdeteksi sebagai badan air/NoData.")

    with st.expander("Raw JSON"):
        st.json(result)

    st.download_button(
        "Download JSON",
        data=json.dumps(result, indent=2, ensure_ascii=False),
        file_name="geoai_flood_agent_result.json",
        mime="application/json",
    )
    if result.get("agent_response"):
        st.download_button(
            "Download Report Markdown",
            data=result["agent_response"],
            file_name="geoai_flood_business_report.md",
            mime="text/markdown",
        )


if __name__ == "__main__":
    main()
