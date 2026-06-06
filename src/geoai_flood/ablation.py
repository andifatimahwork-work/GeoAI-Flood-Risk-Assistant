from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import rasterio

from .config import resolve_path


def sample_raster(path: Path, lat: float, lon: float) -> float:
    with rasterio.open(path) as src:
        value = next(src.sample([(lon, lat)]))[0]
    return float(value)


def run_ablation(cfg: dict[str, Any], baseline_fsi: str | Path | None = None, proposed_fsi: str | Path | None = None) -> Path:
    root = Path(cfg["_project_root"])
    locations_path = resolve_path(cfg["paths"]["ablation_locations"], root)
    baseline = resolve_path(baseline_fsi or cfg["fsi"].get("output_baseline_fsi_tif", cfg["fsi"]["output_fsi_tif"]), root)
    proposed = resolve_path(proposed_fsi or cfg["fsi"]["output_fsi_tif"], root)
    df = pd.read_csv(locations_path)
    rows = []
    for row in df.itertuples(index=False):
        base = sample_raster(baseline, float(row.lat), float(row.lon))
        prop = sample_raster(proposed, float(row.lat), float(row.lon))
        rows.append(
            {
                "name": row.name,
                "city": row.city,
                "lat": row.lat,
                "lon": row.lon,
                "fsi_baseline_esa": base,
                "fsi_proposed_unet": prop,
                "delta_fsi": prop - base,
            }
        )
    out_dir = resolve_path(cfg["paths"]["output_dir"], root)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "ablation_results.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out
