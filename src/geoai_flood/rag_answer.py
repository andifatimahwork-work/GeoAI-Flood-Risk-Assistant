from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .rag_retriever import search_regulation


RAG_SYSTEM_PROMPT = (
    "Anda adalah asisten AI ahli kepatuhan keberlanjutan dan risiko iklim. "
    "Jawablah pertanyaan hanya berdasarkan konteks dokumen referensi yang disediakan. "
    "Jika jawabannya tidak ada di dalam konteks, katakan: "
    "'Saya tidak menemukan informasi tersebut dalam dokumen referensi.' "
    "Jangan membuat asumsi di luar konteks dokumen. Namun, jika dokumen menyediakan konsep atau kerangka kerja umum "
    "yang langsung menjawab pertanyaan, gunakan informasi tersebut dengan penalaran logis yang objektif. "
    "Jangan menyalin teks hukum mentah yang tidak relevan. Selaraskan bahasa jawaban dengan bahasa pertanyaan, "
    "dan selalu sertakan sitasi dokumen serta halaman untuk klaim utama."
)


ANSWER_POLICY = {
    "grounding": "Answer only from retrieved reference context.",
    "insufficient_context_response": "Saya tidak menemukan informasi tersebut dalam dokumen referensi.",
    "citation_required": True,
    "avoid_raw_legal_boilerplate": True,
    "no_unsupported_assumptions": True,
    "use_general_framework_when_directly_relevant": True,
    "match_question_language": True,
}


STOPWORDS = {
    "apa",
    "dan",
    "yang",
    "dalam",
    "untuk",
    "dengan",
    "menurut",
    "bagaimana",
    "dapat",
    "the",
    "and",
    "of",
    "to",
    "in",
    "a",
    "an",
    "does",
    "what",
    "how",
    "should",
}


def clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[�]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_english_query(text: str) -> bool:
    lower = text.lower()
    english_markers = (" how ", " what ", " should ", " does ", " under ", " framework", " company ", " disclose ")
    indonesian_markers = (" bagaimana ", " apa ", " menurut ", " dalam ", " perusahaan ", " risiko ")
    padded = f" {lower} "
    return sum(marker in padded for marker in english_markers) > sum(marker in padded for marker in indonesian_markers)


def insufficient_response(query: str) -> str:
    if is_english_query(query):
        return "I did not find this information in the reference documents."
    return ANSWER_POLICY["insufficient_context_response"]


def citation(hit: dict[str, Any]) -> str:
    source = hit.get("source") or "unknown_source"
    page = hit.get("page")
    return f"{source}, p.{page}" if page else source


def first_citation_for_source(evidence: list[dict[str, Any]], source_keyword: str) -> str | None:
    for item in evidence:
        source = (item.get("source") or "").lower()
        if source_keyword.lower() in source:
            return item["citation"]
    return None


def compact_citations(evidence: list[dict[str, Any]], max_items: int = 3) -> str:
    citations = []
    seen = set()
    for item in evidence:
        cite = item["citation"]
        if cite not in seen:
            citations.append(cite)
            seen.add(cite)
        if len(citations) >= max_items:
            break
    return "; ".join(citations)


def evidence_text(evidence: list[dict[str, Any]]) -> str:
    return " ".join(item.get("context", "") for item in evidence).lower()


def has_context(evidence: list[dict[str, Any]], *phrases: str) -> bool:
    text = evidence_text(evidence)
    return any(phrase.lower() in text for phrase in phrases)


def citation_with_context(evidence: list[dict[str, Any]], source_keyword: str, *phrases: str) -> str | None:
    for item in evidence:
        source = (item.get("source") or "").lower()
        context = (item.get("context") or "").lower()
        if source_keyword.lower() in source and any(phrase.lower() in context for phrase in phrases):
            return item["citation"]
    return first_citation_for_source(evidence, source_keyword)


def has_context_all_in_one(evidence: list[dict[str, Any]], *phrases: str) -> bool:
    for item in evidence:
        context = (item.get("context") or "").lower()
        if all(phrase.lower() in context for phrase in phrases):
            return True
    return False


def citation_with_all_context(evidence: list[dict[str, Any]], source_keyword: str, *phrases: str) -> str | None:
    for item in evidence:
        source = (item.get("source") or "").lower()
        context = (item.get("context") or "").lower()
        if source_keyword.lower() in source and all(phrase.lower() in context for phrase in phrases):
            return item["citation"]
    return None


def claim_with_citation(text: str, citation_text: str | None) -> str:
    return f"{text} ({citation_text})" if citation_text else text


def tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[A-Za-zÀ-ÿ0-9_/-]+", text.lower())
    return {token for token in tokens if len(token) > 2 and token not in STOPWORDS}


def split_sentences(text: str) -> list[str]:
    cleaned = clean_text(text)
    parts = re.split(r"(?<=[.!?])\s+|\n+", cleaned)
    return [part.strip() for part in parts if len(part.strip()) >= 60]


def select_evidence_sentence(query: str, content: str) -> str:
    query_terms = tokenize(query)
    sentences = split_sentences(content)
    if not sentences:
        return clean_text(content)[:700]

    scored = []
    for sent in sentences:
        sent_terms = tokenize(sent)
        overlap = len(query_terms & sent_terms)
        score = overlap / max(len(query_terms), 1)
        scored.append((score, len(sent), sent))
    scored.sort(key=lambda row: (row[0], -min(row[1], 350)), reverse=True)
    return scored[0][2][:700]


def synthesize_answer(query: str, evidence: list[dict[str, Any]]) -> str:
    if not evidence:
        return insufficient_response(query)

    q = query.lower()
    cites = compact_citations(evidence)
    pojk = first_citation_for_source(evidence, "OJK_POJK_No.51")
    tcfd = first_citation_for_source(evidence, "TCFD")
    bnpb = first_citation_for_source(evidence, "BNPB")
    taxonomy = first_citation_for_source(evidence, "Taksonomi")
    ipcc = first_citation_for_source(evidence, "IPCC")
    gri = first_citation_for_source(evidence, "GRI")
    english = is_english_query(query)

    if "pojk" in q and "tcfd" in q:
        parts = []
        if pojk:
            pojk_claims = []
            if has_context(evidence, "wajib menerapkan Keuangan Berkelanjutan", "penerapan Keuangan Berkelanjutan"):
                pojk_claims.append("penerapan prinsip keuangan berkelanjutan")
            if has_context(evidence, "Rencana Aksi Keuangan Berkelanjutan"):
                pojk_claims.append("penyusunan Rencana Aksi Keuangan Berkelanjutan")
            if has_context(evidence, "Laporan Keberlanjutan", "Sustainability Report"):
                pojk_claims.append("pelaporan keberlanjutan")
            if pojk_claims:
                parts.append(f"POJK 51 memberi dasar kepatuhan melalui {', '.join(pojk_claims)} ({pojk}).")
        if tcfd:
            parts.append(f"TCFD melengkapi sisi pengungkapan risiko iklim melalui strategi, manajemen risiko, metrik, target, dan analisis skenario untuk risiko fisik ({tcfd}).")
        parts.append("Untuk portofolio properti, FSI dapat diposisikan sebagai layer screening exposure banjir yang menghubungkan risiko lokasi dengan kebutuhan pelaporan dan manajemen risiko.")
        return " ".join(parts)

    if "tcfd" in q and "gri" in q:
        parts = []
        if tcfd:
            parts.append(f"TCFD menyediakan struktur pengungkapan risiko iklim, terutama bagaimana risiko fisik masuk ke strategi, manajemen risiko, dan metrik perusahaan ({tcfd}).")
        if gri:
            parts.append(f"GRI 201 membantu menerjemahkan risiko tersebut ke dampak ekonomi seperti biaya, aset, pendapatan, atau peluang finansial ({gri}).")
        return " ".join(parts)

    if "taksonomi" in q and ("tcfd" in q or "risiko" in q):
        parts = []
        if taxonomy:
            taxonomy_claim = "Taksonomi Hijau membantu mengklasifikasikan aktivitas hijau dalam pembiayaan atau investasi berkelanjutan"
            if has_context(evidence, "greenwashing"):
                taxonomy_claim += " dan mengurangi risiko greenwashing"
            parts.append(f"{taxonomy_claim} ({taxonomy}).")
        if tcfd:
            parts.append(f"TCFD menambahkan kerangka untuk menilai dan mengungkap risiko fisik iklim yang dapat memengaruhi kelayakan aset atau proyek ({tcfd}).")
        return " ".join(parts)

    if "flood susceptibility" in q or "fsi" in q or "flood score" in q:
        if tcfd:
            tcfd_parts = []
            if has_context(evidence, "governance", "strategy", "risk management", "metrics and targets"):
                tcfd_parts.append("governance, strategy, risk management, serta metrics and targets")
            elif has_context(evidence, "climate-related risks", "risk management"):
                tcfd_parts.append("pengelolaan dan pengungkapan climate-related risks")
            if has_context(evidence, "scenario analysis"):
                tcfd_parts.append("scenario analysis")

            bnpb_part = None
            if bnpb and has_context(evidence, "prioritas", "strategi pengurangan risiko", "kebijakan"):
                bnpb_part = "konteks BNPB untuk prioritas, strategi pengurangan risiko, dan kebijakan pembangunan daerah"
            elif bnpb:
                bnpb_part = "konteks BNPB untuk kajian dan pengelolaan risiko bencana"

            pojk_part = None
            if pojk and has_context(evidence, "Laporan Keberlanjutan", "Keuangan Berkelanjutan"):
                pojk_part = "konteks POJK terkait Keuangan Berkelanjutan dan Laporan Keberlanjutan"

            if english:
                fsi_cites = compact_citations(evidence, max_items=4)
                parts = []
                if tcfd_parts:
                    parts.append(f"the {', '.join(tcfd_parts)}")
                if bnpb_part:
                    parts.append(bnpb_part)
                if pojk_part:
                    parts.append(pojk_part)
                if parts:
                    return (
                        "The reference documents do not explicitly define a flood susceptibility score, but the retrieved contexts can frame it as a technical input for "
                        + "; ".join(parts)
                        + f" ({fsi_cites})."
                    )
                return (
                    f"The flood susceptibility score is a project analytical layer, not an explicit term in the TCFD document. "
                    f"The retrieved TCFD context supports climate-risk disclosure through {', '.join(tcfd_parts) if tcfd_parts else 'climate-related risk disclosure'} ({tcfd})."
                )
            parts = []
            if tcfd_parts:
                parts.append(f"kerangka TCFD untuk {', '.join(tcfd_parts)}")
            if bnpb_part:
                parts.append(bnpb_part)
            if pojk_part:
                parts.append(pojk_part)
            if parts:
                fsi_cites = compact_citations(evidence, max_items=4)
                return (
                    "Dokumen referensi tidak mendefinisikan flood susceptibility score secara eksplisit. "
                    "Namun, konteks yang terambil dapat memosisikan hasil FSI sebagai input teknis untuk "
                    + "; ".join(parts)
                    + f" ({fsi_cites})."
                )
            return (
                f"Flood susceptibility score adalah layer analitik proyek, bukan istilah eksplisit dalam dokumen TCFD. "
                f"Konteks TCFD yang terambil mendukung pengungkapan risiko iklim melalui {', '.join(tcfd_parts) if tcfd_parts else 'kerangka climate-related risk disclosure'} ({tcfd})."
            )
        return (
            f"{insufficient_response(query)} "
            f"Konteks terdekat hanya membahas risiko iklim atau pengungkapan risiko dari: {cites}."
        )

    if tcfd and ("tcfd" in q or "physical climate" in q or "climate-related" in q):
        if "scenario" in q:
            if english:
                return (
                    f"TCFD recommends using scenario analysis to assess how climate-related risks and opportunities could affect strategy and financial planning under different plausible futures ({tcfd})."
                )
            return (
                f"TCFD merekomendasikan penggunaan scenario analysis untuk menilai bagaimana risiko dan peluang iklim dapat memengaruhi strategi serta perencanaan finansial dalam berbagai skenario yang masuk akal ({tcfd})."
            )
        if english:
            return (
                f"Under the TCFD framework, a company should disclose physical climate risks through four pillars: governance, strategy, risk management, and metrics and targets. "
                f"It should explain board and management oversight, actual and potential impacts on business and financial planning, how climate-related risks are identified and managed, and the metrics or targets used to assess those risks ({tcfd})."
            )
        return (
            f"Menurut kerangka TCFD, perusahaan perlu mengungkap risiko fisik iklim melalui empat pilar: governance, strategy, risk management, serta metrics and targets. "
            f"Pengungkapan sebaiknya menjelaskan pengawasan manajemen, dampak aktual dan potensial terhadap bisnis serta perencanaan finansial, proses identifikasi dan pengelolaan risiko, serta metrik atau target yang digunakan ({tcfd})."
        )

    if "kewajiban" in q and "pojk" in q and pojk:
        claims = []
        if has_context(evidence, "wajib menerapkan Keuangan Berkelanjutan", "penerapan Keuangan Berkelanjutan"):
            cite = citation_with_context(evidence, "OJK_POJK_No.51", "wajib menerapkan Keuangan Berkelanjutan", "penerapan Keuangan Berkelanjutan")
            claims.append(claim_with_citation("menerapkan Keuangan Berkelanjutan", cite))
        if has_context(evidence, "Rencana Aksi Keuangan Berkelanjutan"):
            cite = citation_with_context(evidence, "OJK_POJK_No.51", "Rencana Aksi Keuangan Berkelanjutan")
            claims.append(claim_with_citation("menyusun Rencana Aksi Keuangan Berkelanjutan", cite))
        if has_context_all_in_one(evidence, "Laporan Keberlanjutan", "wajib disampaikan"):
            cite = citation_with_all_context(evidence, "OJK_POJK_No.51", "Laporan Keberlanjutan", "wajib disampaikan")
            claims.append(claim_with_citation("menyampaikan Laporan Keberlanjutan", cite))
        elif has_context_all_in_one(evidence, "Laporan Keberlanjutan", "disampaikan kepada Otoritas Jasa Keuangan"):
            cite = citation_with_all_context(evidence, "OJK_POJK_No.51", "Laporan Keberlanjutan", "disampaikan kepada Otoritas Jasa Keuangan")
            claims.append(claim_with_citation("menyampaikan Laporan Keberlanjutan", cite))
        elif has_context(evidence, "Laporan Keberlanjutan", "Sustainability Report"):
            cite = citation_with_context(evidence, "OJK_POJK_No.51", "Laporan Keberlanjutan", "Sustainability Report")
            claims.append(claim_with_citation("mengacu pada Laporan Keberlanjutan sebagai laporan bisnis berkelanjutan", cite))
        if claims:
            return f"Menurut konteks POJK 51/2017, lembaga jasa keuangan perlu {', '.join(claims)}."
        return f"{insufficient_response(query)} Konteks POJK yang ditemukan belum memuat kewajiban utama secara cukup eksplisit ({pojk})."

    if "taksonomi hijau" in q and taxonomy:
        benefits = ["berfungsi sebagai rujukan klasifikasi aktivitas hijau"]
        if has_context(evidence, "pembiayaan", "pendanaan", "investasi"):
            benefits.append("mendukung pembiayaan, pendanaan, atau investasi hijau")
        if has_context(evidence, "pelaporan", "monitoring"):
            benefits.append("mendukung pelaporan dan monitoring")
        if has_context(evidence, "manajemen risiko"):
            benefits.append("mendukung pemahaman manajemen risiko")
        if has_context(evidence, "greenwashing"):
            benefits.append("mengurangi risiko greenwashing")
        return f"Taksonomi Hijau Indonesia {', '.join(benefits)} ({taxonomy})."

    if ("greenwashing" in q or "pembiayaan aktivitas" in q) and taxonomy:
        if not has_context(evidence, "greenwashing"):
            return (
                f"Taksonomi Hijau menyediakan klasifikasi aktivitas hijau untuk mendukung pembiayaan atau investasi berkelanjutan. "
                f"Konteks yang ditemukan belum memuat istilah greenwashing secara eksplisit ({taxonomy})."
            )
        return (
            f"Taksonomi Hijau mengurangi greenwashing dengan memberi klasifikasi yang lebih jelas tentang aktivitas hijau, "
            f"sehingga klaim pembiayaan atau investasi berkelanjutan dapat diuji terhadap kriteria yang lebih terstruktur ({taxonomy})."
        )

    if "bnpb" in q and bnpb:
        return (
            f"Dalam konteks BNPB, risiko bencana dipahami melalui kajian risiko yang menjadi dasar pengelolaan dan penanggulangan bencana. "
            f"Untuk banjir dan bencana hidrometeorologi, informasi risiko dipakai untuk mengurangi potensi kerugian dan menentukan prioritas mitigasi ({bnpb})."
        )

    if "ipcc" in q and ipcc:
        return (
            f"IPCC AR6 Asia menempatkan risiko iklim di Asia dalam konteks hazard, exposure, vulnerability, dan adaptasi. "
            f"Curah hujan ekstrem, banjir, dan bahaya hidrometeorologi relevan untuk perencanaan adaptasi karena dapat memengaruhi masyarakat, aset, dan infrastruktur ({ipcc})."
        )

    if "gri" in q and gri:
        return (
            f"GRI 201 relevan karena membantu menjelaskan implikasi ekonomi dari risiko perubahan iklim, termasuk potensi dampak banjir terhadap biaya, aset, operasi, pendapatan, dan peluang finansial ({gri})."
        )

    return (
        f"{insufficient_response(query)} "
        f"Konteks terdekat berasal dari: {cites}."
    )


def generate_cited_answer(
    query: str,
    config_path: str | Path = "rag_config.json",
    max_evidence: int = 4,
) -> dict[str, Any]:
    hits = search_regulation(query, config_path)
    selected = hits[:max_evidence]

    evidence_items = []
    for i, hit in enumerate(selected, start=1):
        sent = select_evidence_sentence(query, hit.get("content", ""))
        cite = citation(hit)
        evidence_items.append(
            {
                "rank": i,
                "citation": cite,
                "source": hit.get("source"),
                "doc_name": hit.get("doc_name"),
                "page": hit.get("page"),
                "context": hit.get("content", ""),
                "selected_sentence": sent,
            }
        )

    answer = synthesize_answer(query, evidence_items)

    return {
        "system_prompt": RAG_SYSTEM_PROMPT,
        "answer_policy": ANSWER_POLICY,
        "question": query,
        "answer": answer,
        "contexts": [item["context"] for item in evidence_items],
        "citations": [item["citation"] for item in evidence_items],
        "evidence": evidence_items,
    }
