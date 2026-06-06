from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geoai_flood.config import load_config, resolve_path
from geoai_flood.gis_fsi import esa_to_custom_lulc, read_matched, read_reference


def read(path: Path) -> tuple[np.ndarray, float | int | None]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan
    return arr, nodata


def pct(count: int, total: int) -> float:
    return float(count / max(total, 1) * 100.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--jrc-water-threshold", type=float, default=10.0)
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["_project_root"])

    out_dir = resolve_path(cfg["validation"]["output_dir"], root)
    out_dir.mkdir(parents=True, exist_ok=True)

    fsi, _ = read(resolve_path(cfg["validation"]["fsi_tif"], root))
    jrc, _ = read(resolve_path(cfg["validation"]["jrc_gsw_tif"], root))
    _, ref_profile = read_reference(resolve_path(cfg["validation"]["jrc_gsw_tif"], root))

    unet_lulc = read_matched(resolve_path(cfg["fsi"]["lulc_unet_tif"], root), ref_profile, Resampling.nearest)
    esa_raw = read_matched(resolve_path(cfg["fsi"]["lulc_esa_tif"], root), ref_profile, Resampling.nearest)
    esa_lulc = esa_to_custom_lulc(np.where(np.isfinite(esa_raw), esa_raw, -9999).astype(np.int16), cfg).astype(np.float32)

    class_to_id = {name: i for i, name in enumerate(cfg["classes"]["names"])}
    air_id = class_to_id.get("Air")

    total = fsi.size
    valid = np.isfinite(fsi) & np.isfinite(jrc)
    invalid = ~valid
    jrc_water = np.isfinite(jrc) & (jrc >= args.jrc_water_threshold)
    jrc_permanentish_water = np.isfinite(jrc) & (jrc >= 90.0)
    unet_gap = ~np.isfinite(unet_lulc)
    unet_air = np.isfinite(unet_lulc) & (unet_lulc.astype(np.int16) == air_id) if air_id is not None else np.zeros_like(valid)
    esa_air = np.isfinite(esa_lulc) & (esa_lulc.astype(np.int16) == air_id) if air_id is not None else np.zeros_like(valid)
    any_lulc_air = unet_air | esa_air

    categories = {
        "total_pixels": np.ones_like(valid, dtype=bool),
        "valid_for_jrc_validation": valid,
        "masked_or_nodata_total": invalid,
        "masked_and_jrc_occurrence_ge_10_possible_laut_or_water": invalid & jrc_water,
        "masked_and_jrc_occurrence_ge_90_permanentish_water": invalid & jrc_permanentish_water,
        "masked_due_to_unet_tile_gap_or_prediction_nodata": invalid & unet_gap,
        "masked_due_to_lulc_air_unet_or_esa": invalid & any_lulc_air,
        "masked_due_to_unet_air": invalid & unet_air,
        "masked_due_to_esa_air": invalid & esa_air,
        "masked_other_unexplained": invalid & ~(jrc_water | unet_gap | any_lulc_air),
    }

    rows = []
    for name, mask in categories.items():
        count = int(mask.sum())
        rows.append(
            {
                "category": name,
                "pixels": count,
                "percent_of_total": pct(count, total),
                "percent_of_masked": pct(count, int(invalid.sum())) if name != "total_pixels" else None,
            }
        )

    # Overlap table for the three requested causes.
    cause_masks = {
        "jrc_water_ge_10": invalid & jrc_water,
        "unet_tile_gap": invalid & unet_gap,
        "lulc_air": invalid & any_lulc_air,
    }
    overlap_rows = []
    names = list(cause_masks)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            m = cause_masks[a] & cause_masks[b]
            overlap_rows.append(
                {
                    "overlap": f"{a} & {b}",
                    "pixels": int(m.sum()),
                    "percent_of_total": pct(int(m.sum()), total),
                    "percent_of_masked": pct(int(m.sum()), int(invalid.sum())),
                }
            )

    summary_path = out_dir / "validation_coverage_audit.csv"
    overlap_path = out_dir / "validation_coverage_overlap.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    pd.DataFrame(overlap_rows).to_csv(overlap_path, index=False)

    payload = {
        "note": "Audit only. Does not modify FSI or validation outputs.",
        "total_pixels": int(total),
        "valid_pixels": int(valid.sum()),
        "valid_percent": pct(int(valid.sum()), total),
        "masked_pixels": int(invalid.sum()),
        "masked_percent": pct(int(invalid.sum()), total),
        "files": {
            "summary_csv": str(summary_path),
            "overlap_csv": str(overlap_path),
        },
    }
    json_path = out_dir / "validation_coverage_audit.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print(f"Coverage audit CSV: {summary_path}")
    print(f"Overlap audit CSV: {overlap_path}")


if __name__ == "__main__":
    main()
