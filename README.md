## Struktur

```text
config/
  config.yaml              # Parameter utama pipeline
  remap_config.json        # Mapping ESA WorldCover ke 5 kelas LULC
data/
  samples/ablation_locations.csv
notebooks/
  01b_eda.ipynb            # Notebook EDA sebelum training
  colab_train_unet.py      # Script notebook-style untuk Google Colab
scripts/
  1_mosaic_gee_exports.py
  2_gee_export.py
  3_preprocess_tiles.py
  4_eda.py
  5_train_unet.py
  6_evaluate.py
  7_ablation.py
  8_build_fsi.py
  9_validate_rasters.py
  10_gee_export_worldpop.py
src/geoai_flood/
  ... modul pipeline ...
```

## Setup VS Code

Jalankan dari terminal VS Code:

```powershell
cd "E:\AI ML\GeoAI_Flood_Susceptibility"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Autentikasi Google Earth Engine:

```powershell
earthengine authenticate
```

## Alur Minggu 1

### 1. Export data GEE

Edit `config/config.yaml`, terutama `drive.folder`, `aoi`, dan `gee.esa_worldcover_asset` jika punya asset ESA WorldCover 2023 di GEE. Default memakai asset publik yang umum tersedia.

```powershell
python scripts/2_gee_export.py --config config/config.yaml
```

Output akan masuk ke Google Drive folder yang diset di config:

- `s2_composite_jabodetabek`
- `esa_worldcover_jabodetabek`
- `jrc_gsw_occurrence_jabodetabek`
- `worldpop_population_density_jabodetabek`

Jika Sentinel-2, ESA, dan JRC sudah ada, hanya perlu WorldPop untuk eksperimen 2:

```powershell
python scripts/10_gee_export_worldpop.py --config config/config.yaml
```

### 2. Download atau mount data, lalu preprocessing

Letakkan hasil export GeoTIFF di folder yang sesuai dengan `paths.raw_dir`, lalu jalankan:

Jika Google Earth Engine menghasilkan banyak file seperti `s2_composite_jabodetabek-0000000000-0000000256.tif`, download semua file `.tif` dari Drive ke satu folder lokal sementara, lalu mosaic dulu:

```powershell
python scripts/1_mosaic_gee_exports.py --input-dir "E:\AI ML\GeoAI_Flood_Susceptibility\data\gee_exports" --output-dir data/raw
```

Hasil mosaic yang dibuat:

- `data/raw/s2_composite_jabodetabek.tif`
- `data/raw/esa_worldcover_jabodetabek.tif`
- `data/raw/jrc_gsw_occurrence_jabodetabek.tif`
- `data/raw/worldpop_population_density_jabodetabek.tif`

Setelah itu jalankan preprocessing:

```powershell
python scripts/3_preprocess_tiles.py --config config/config.yaml
```

Output penting:

- `outputs/tiles/train_list.txt`
- `outputs/tiles/val_list.txt`
- `outputs/dataset_stats.json`
- `outputs/remap_config.json`
- `data/samples/ablation_locations.csv` berisi 30 titik Jabodetabek: Jakarta Utara, Jakarta Pusat, Bekasi, Tangerang, Depok, dan Bogor.

### 3. EDA sebelum training

Jalankan EDA setelah preprocessing dan sebelum upload ke Colab:

```powershell
python scripts/4_eda.py --config config/config.yaml
```

Atau pakai runner yang sudah disediakan:

```powershell
.\scripts\run_eda.ps1
```

Jika PowerShell policy bermasalah, pakai:

```powershell
scripts\run_eda.bat
```

Output EDA:

- `outputs/eda/class_distribution.png`
- `outputs/eda/class_weights.json`
- `outputs/eda/bad_tiles.txt`
- `outputs/eda/band_stats_plot.png`
- `outputs/eda/sample_visual_inspection.png`
- `outputs/eda/red_nir_scatter.png`
- `outputs/tiles/train_list_clean.txt`
- `outputs/tiles/val_list_clean.txt`

Training otomatis memakai `class_weights.json` dan clean list jika file tersebut sudah ada.

Perbaikan eksperimen 2 yang diterapkan oleh EDA/training:

- `Informal` didrop dari kelas U-Net karena ESA WorldCover tidak punya label eksplisit untuk informal settlement.
- Train/val clean list dibuat ulang dengan stratified split berdasarkan dominant class per tile.
- Band Sentinel-2 diclip ke range `[0, 1]` sebelum normalisasi.
- Tile yang satu kelasnya mendominasi `>95%` dibuang dari clean list.
- Fine-tuning LR diturunkan ke `1e-5`.
- Loss memakai Dice + weighted Focal CE.
- `class_weights.json` dihitung dari pixel count aktual pada clean train split.
- Weighted sampler menaikkan peluang tile yang mengandung `Lahan_Terbuka`.
- Early stopping memonitor `val_miou` dengan `patience=15` dan `min_delta=0.001`.
- Overlap Sawah-Vegetasi dicatat sebagai limitasi pseudo-label ESA, bukan bug kode.

### 4. Training di Google Colab

Upload/sync project ini ke Google Drive. Di Colab, buka `notebooks/colab_train_unet.py`, lalu jalankan cell per cell. Script itu akan:

- mount Google Drive,
- install dependency,
- menjalankan warm-up 10 epoch,
- fine-tune 50 epoch dengan LR `1e-5`, total maksimum 60 epoch,
- menyimpan `best_model.pth`,
- menyimpan checkpoint tiap 10 epoch,
- menyimpan `training_log_exp2.csv`.

Saat kembali ke VS Code, copy atau arahkan `training.checkpoint_dir` ke folder Drive yang berisi `best_model.pth`.

### 5. Evaluasi dan ablation

```powershell
python scripts/6_evaluate.py --config config/config.yaml --checkpoint "path\to\best_model.pth"
python scripts/7_ablation.py --config config/config.yaml
```

Evaluasi membuat `outputs/evaluation_exp2/lulc_unet_prediction.tif`, `confusion_matrix.png`, dan `confusion_matrix_normalized.png`.

### 6. FSI GIS fusion

Siapkan raster GIS yang dibutuhkan sesuai path di `config/config.yaml`:

- DEM SRTM atau elevation raster
- CHIRPS rainfall mean
- river distance raster
- WorldPop population density raster
- LULC prediksi U-Net yang sudah direproject/resample ke extent referensi
- JRC GSW occurrence sebagai reference extent

Lalu:

```powershell
python scripts/8_build_fsi.py --config config/config.yaml
python scripts/9_validate_rasters.py --config config/config.yaml
```

Output akhir Minggu 1:

- `best_model.pth`
- `confusion_matrix.png`
- `confusion_matrix_normalized.png`
- `dataset_stats.json`
- `remap_config.json`
- `config.yaml`
- `fsi_map.tif`
- `fsi_baseline_esa.tif`
- `fsi_ci_low.tif`
- `fsi_ci_high.tif`
- `jrc_gsw_occurrence.tif`
- `ablation_results.csv`
- `training_log.csv`
- `worldpop_population_density_jabodetabek.tif`

## Catatan penting

- `Informal` tidak tersedia sebagai kelas eksplisit di ESA WorldCover, sehingga pada eksperimen 2 kelas ini tidak dilatih di U-Net. Risiko permukiman padat ditangkap lewat layer WorldPop pada FSI.
- Formula FSI eksperimen 2: `0.28*Elevasi + 0.23*Slope + 0.18*Rainfall + 0.13*DistSungai + 0.10*LULC + 0.08*PopDensity`.
- Elevasi memakai transformasi `log_inverse` agar wilayah sangat rendah seperti Jakarta Utara mendapat skor risiko elevasi jauh lebih tinggi dibanding dataran tinggi seperti Bogor.
- Parameter cloud mask disimpan di `config/config.yaml` agar konsisten dengan pipeline inference Minggu 2/3.
- `dataset_stats.json` wajib dipakai lagi saat inference agar normalisasi input sama dengan training.

## Minggu 2 - RAG

PDF regulasi disimpan di:

```text
docs/regulatory/
```

Build index ChromaDB:

```powershell
python scripts/16_build_rag_index.py --config rag_config.json --reset
```

Smoke test retrieval:

```powershell
python scripts/17_test_rag_retrieval.py --config rag_config.json
```

Prepare context pack untuk jawaban RAG dengan sitasi:

```powershell
python scripts/18_prepare_rag_context.py --config rag_config.json
```

Untuk satu pertanyaan spesifik:

```powershell
python scripts/18_prepare_rag_context.py --config rag_config.json --query "Bagaimana risiko banjir dan bencana hidrometeorologi dijelaskan dalam dokumen BNPB?"
```

Output RAG:

```text
chroma_db/
outputs/rag_index_summary.json
outputs/rag_retrieval_smoke_test.json
outputs/rag_context_pack.json
outputs/rag_context_pack.md
```

## Minggu 2 - On-The-Fly FSI Tool

Agent memakai on-the-fly computation untuk input koordinat user:

```text
koordinat user -> fetch layer dari GEE -> prediksi LULC U-Net -> hitung FSI point-level -> return result
```

FSI raster final Minggu 1 tetap dipakai sebagai artifact validasi, visualisasi Streamlit, dan credibility artifact. Mode lokal masih disediakan sebagai fallback/testing agar hasil GIS yang sudah valid tidak perlu dihitung ulang.

Smoke test tiga koordinat:

```powershell
python scripts/19_test_flood_score.py --config config/config.yaml --mode local
```

Test satu koordinat:

```powershell
python scripts/19_test_flood_score.py --config config/config.yaml --lat -6.1214 --lon 106.7741 --mode local
```

Output:

```text
outputs/flood_score_smoke_test.json
```

Untuk mode GEE live, siapkan environment:

```powershell
copy .env.example .env
```

Isi minimal:

```text
GEE_SERVICE_ACCOUNT_KEY=path\to\gee_service_account.json
EARTHENGINE_PROJECT=your-google-cloud-project-id
```

Lalu jalankan:

```powershell
python scripts/19_test_flood_score.py --config config/config.yaml --mode live
```

Catatan penting: river distance saat ini masih bisa fallback ke raster lokal `data/gis/river_distance_30m.tif`. Kalau mau full server-side GEE, publish river feature/vector ke GEE asset lalu isi `GEE_RIVER_FC_ASSET`.

## Minggu 2 - Agent Integration

Agent menggabungkan dua tool:

- `get_flood_score_tool(lat, lon)` untuk FSI on-the-fly/fallback lokal.
- `search_regulation_tool(query)` untuk RAG regulasi dengan sitasi.

Smoke test tool tanpa memanggil LLM:

```powershell
python scripts/24_test_agent_tools.py --config config/config.yaml --rag-config rag_config.json --mode local
```

Jika ingin menjalankan agent dengan Groq:

```powershell
$env:GROQ_API_KEY="isi_api_key"
python scripts/24_test_agent_tools.py --config config/config.yaml --rag-config rag_config.json --mode local --run-agent
```

Untuk production-like mode dengan GEE:

```powershell
$env:FSI_COMPUTE_MODE="live"
python scripts/24_test_agent_tools.py --config config/config.yaml --rag-config rag_config.json --mode live --run-agent
```

Cache memakai `diskcache` dengan default folder `cache/` dan TTL 24 jam (`CACHE_TTL_SECONDS=86400`).

### GEE Live Integration

Untuk lokal, bisa pakai auth Earth Engine personal:

```powershell
earthengine authenticate
```

Lalu isi `.env`:

```text
EARTHENGINE_PROJECT=your-google-cloud-project-id
```

Untuk deployment/server, pakai service account:

```text
GEE_SERVICE_ACCOUNT_KEY=path\to\service_account.json
EARTHENGINE_PROJECT=your-google-cloud-project-id
```

Cek readiness tanpa menjalankan U-Net:

```powershell
python scripts/26_check_gee_live.py --config config/config.yaml
```

Test full live FSI:

```powershell
python scripts/26_check_gee_live.py --config config/config.yaml --full
```

Run agent untuk query user bebas:

```powershell
python scripts/25_run_agent_query.py --query "Gudang logistik kami di Bekasi koordinat -6.23, 107.01 apakah layak dijadikan collateral kredit?" --mode local
```

Contoh multi-lokasi:

```powershell
python scripts/25_run_agent_query.py --query "Kami punya 3 gudang: Cakung (-6.1831, 106.9370), Cilincing (-6.1077, 106.9238), Marunda (-6.0993, 106.9720). Mana yang wajib masuk laporan TCFD?" --mode local
```

Kalau hanya ingin cek parser, FSI, dan RAG tanpa memanggil LLM:

```powershell
python scripts/25_run_agent_query.py --query "Serpong (-6.3194, 106.6641) vs Bekasi Timur (-6.2641, 107.0101), mana yang lebih aman dikembangkan?" --mode local --dry-run
```

## Minggu 3 - Streamlit UI MVP

### FastAPI backend

File API untuk submission:

```text
apps/fastapi_app.py
```

Jalankan dari root project:

```powershell
cd "E:\AI ML\GeoAI_Flood_Susceptibility"
uvicorn apps.fastapi_app:app --reload --host 127.0.0.1 --port 8000
```

Local API links:

```text
API health: http://127.0.0.1:8000/health
API docs  : http://127.0.0.1:8000/docs
Endpoint  : POST http://127.0.0.1:8000/analyze
```

Contoh request body untuk `/analyze`:

```json
{
  "query": "Gudang logistik kami di Bekasi koordinat -6.23, 107.01 apakah layak dijadikan collateral kredit?",
  "mode": "auto",
  "provider": "groq",
  "dry_run": false
}
```

Jika hanya ingin mengecek FSI dan RAG tanpa LLM, set `dry_run` menjadi `true`.

### Streamlit frontend

File UI untuk submission:

```text
apps/streamlit_app.py
```

Install library UI/API saja jika belum ada:

```powershell
pip install streamlit fastapi uvicorn[standard]
```

Jalankan UI:

```powershell
streamlit run apps/streamlit_app.py
```

Jika ingin agent menyusun rekomendasi dengan Groq, set API key dulu:

```powershell
$env:GROQ_API_KEY="isi_api_key_groq"
streamlit run apps/streamlit_app.py
```

Fitur UI MVP:

- input query bebas dengan koordinat,
- contoh skenario perbankan, ESG/TCFD, dan developer,
- FSI score per lokasi,
- peta lokasi,
- agent recommendation report,
- panel sitasi RAG,
- download JSON dan Markdown report.

## Minggu 2 - RAG Evaluation

Test set evaluasi disimpan di:

```text
test_queries.json
```

Evaluasi retrieval source matching untuk 20 query:

```powershell
python scripts/20_evaluate_rag_retrieval.py --config rag_config.json --queries test_queries.json
```

Generate jawaban RAG lokal berbasis evidence dan sitasi:

```powershell
python scripts/21_generate_rag_answers.py --config rag_config.json --queries test_queries.json
```

Validasi struktur dan grounding output jawaban:

```powershell
python scripts/23_validate_rag_generated_answers.py --input outputs/rag_generated_answers.json
```

Jalankan RAGAS jika evaluator API key sudah tersedia:

```powershell
python scripts/22_run_ragas_eval.py --input outputs/rag_generated_answers.json
```

Jika belum ada API key, buat readiness report dulu:

```powershell
python scripts/22_run_ragas_eval.py --input outputs/rag_generated_answers.json --allow-no-api-key
```

Output:

```text
outputs/rag_retrieval_eval_report.json
outputs/rag_generated_answers.json
outputs/rag_generated_answers_validation.json
outputs/ragas_report.json
```
