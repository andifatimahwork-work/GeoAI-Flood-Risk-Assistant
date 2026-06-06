from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geoai_flood.config import load_config, resolve_path
from geoai_flood.gis_fsi import esa_to_custom_lulc, lulc_to_risk, read_matched, read_reference


def sample(path: Path, lat: float, lon: float) -> float:
    with rasterio.open(path) as src:
        return float(next(src.sample([(lon, lat)]))[0])


def plot_fsi(cfg: dict) -> Path:
    root = Path(cfg["_project_root"])
    fsi_path = resolve_path(cfg["fsi"]["output_fsi_tif"], root)
    out_dir = fsi_path.parent
    with rasterio.open(fsi_path) as src:
        fsi = src.read(1)
        if src.nodata is not None:
            fsi = np.where(fsi == src.nodata, np.nan, fsi)
    valid = np.isfinite(fsi)
    vmin = float(cfg["fsi"].get("plot_vmin", 0.5))
    vmax = float(cfg["fsi"].get("plot_vmax", 1.0))
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(np.where(valid, fsi, np.nan), cmap="RdYlGn_r", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label="FSI Score")
    ax.set_title("Flood Susceptibility Index - Jabodetabek")
    ax.axis("off")
    out = out_dir / "fsi_map_proper.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(fsi[valid].ravel(), bins=80, color="#3A7CA5")
    ax.set_title("FSI Score Distribution")
    ax.set_xlabel("FSI")
    ax.set_ylabel("Pixel count")
    fig.tight_layout()
    fig.savefig(out_dir / "fsi_histogram.png", dpi=150)
    plt.close(fig)

    summary = pd.DataFrame(
        [
            {
                "valid_pixels": int(valid.sum()),
                "nodata_pixels": int((~valid).sum()),
                "masked_lulc_classes": ",".join(cfg["fsi"].get("mask_lulc_classes", [])),
                "min": float(np.nanmin(fsi)),
                "p01": float(np.nanpercentile(fsi, 1)),
                "p05": float(np.nanpercentile(fsi, 5)),
                "p50": float(np.nanpercentile(fsi, 50)),
                "p95": float(np.nanpercentile(fsi, 95)),
                "p99": float(np.nanpercentile(fsi, 99)),
                "max": float(np.nanmax(fsi)),
                "count_around_0_62_pm_0_005": int(((fsi >= 0.615) & (fsi <= 0.625) & valid).sum()),
            }
        ]
    )
    summary.to_csv(out_dir / "fsi_distribution_masked_summary.csv", index=False)
    return out


def write_audit_tables(cfg: dict) -> tuple[Path, Path]:
    root = Path(cfg["_project_root"])
    locations = pd.read_csv(resolve_path(cfg["paths"]["ablation_locations"], root))
    _, ref_profile = read_reference(resolve_path(cfg["fsi"]["reference_tif"], root))

    esa_raw = read_matched(resolve_path(cfg["fsi"]["lulc_esa_tif"], root), ref_profile, Resampling.nearest).astype(np.int16)
    esa_lulc = esa_to_custom_lulc(esa_raw, cfg)
    unet_lulc = read_matched(resolve_path(cfg["fsi"]["lulc_unet_tif"], root), ref_profile, Resampling.nearest).astype(np.int16)
    esa_risk = lulc_to_risk(esa_lulc, cfg)
    unet_risk = lulc_to_risk(unet_lulc, cfg)

    out_dir = resolve_path(cfg["fsi"]["output_fsi_tif"], root).parent
    tmp_paths = {
        "esa_lulc": out_dir / "_tmp_esa_lulc.tif",
        "unet_lulc": out_dir / "_tmp_unet_lulc.tif",
        "esa_lulc_score": out_dir / "_tmp_esa_lulc_score.tif",
        "unet_lulc_score": out_dir / "_tmp_unet_lulc_score.tif",
    }
    profile = ref_profile.copy()
    profile.update(count=1, dtype="float32", nodata=None, compress="deflate")
    with rasterio.open(tmp_paths["esa_lulc"], "w", **profile) as dst:
        dst.write(esa_lulc.astype(np.float32), 1)
    with rasterio.open(tmp_paths["unet_lulc"], "w", **profile) as dst:
        dst.write(unet_lulc.astype(np.float32), 1)
    with rasterio.open(tmp_paths["esa_lulc_score"], "w", **profile) as dst:
        dst.write(esa_risk.astype(np.float32), 1)
    with rasterio.open(tmp_paths["unet_lulc_score"], "w", **profile) as dst:
        dst.write(unet_risk.astype(np.float32), 1)

    component_dir = resolve_path(cfg["fsi"]["output_components_dir"], root)
    component_paths = {
        "elevation_score": component_dir / "elevation_score.tif",
        "slope_score": component_dir / "slope_score.tif",
        "rainfall_score": component_dir / "rainfall_score.tif",
        "river_distance_score": component_dir / "river_distance_score.tif",
        "population_score": component_dir / "population_score.tif",
    }
    raw_paths = {
        "elevation_raw": resolve_path(cfg["fsi"]["elevation_tif"], root),
        "rainfall_raw": resolve_path(cfg["fsi"]["rainfall_tif"], root),
    }

    class_names = cfg["classes"]["names"]
    rows = []
    for row in locations.itertuples(index=False):
        lat = float(row.lat)
        lon = float(row.lon)
        esa_class = int(sample(tmp_paths["esa_lulc"], lat, lon))
        unet_class = int(sample(tmp_paths["unet_lulc"], lat, lon))
        rows.append(
            {
                "name": row.name,
                "city": row.city,
                "lat": lat,
                "lon": lon,
                "esa_class_id": esa_class,
                "esa_class": class_names[esa_class] if 0 <= esa_class < len(class_names) else "ignore",
                "unet_class_id": unet_class,
                "unet_class": class_names[unet_class] if 0 <= unet_class < len(class_names) else "ignore",
                "same_lulc_class": esa_class == unet_class,
                "esa_lulc_score": sample(tmp_paths["esa_lulc_score"], lat, lon),
                "unet_lulc_score": sample(tmp_paths["unet_lulc_score"], lat, lon),
                "elevation_raw_m": sample(raw_paths["elevation_raw"], lat, lon),
                "rainfall_raw_mm_year": sample(raw_paths["rainfall_raw"], lat, lon),
                **{k: sample(v, lat, lon) for k, v in component_paths.items()},
                "fsi_baseline_esa": sample(resolve_path(cfg["fsi"]["output_baseline_fsi_tif"], root), lat, lon),
                "fsi_proposed_unet": sample(resolve_path(cfg["fsi"]["output_fsi_tif"], root), lat, lon),
            }
        )
    audit = pd.DataFrame(rows)
    audit["delta_lulc_score"] = audit["unet_lulc_score"] - audit["esa_lulc_score"]
    audit["delta_fsi"] = audit["fsi_proposed_unet"] - audit["fsi_baseline_esa"]
    audit_path = out_dir / "ablation_lulc_component_audit.csv"
    audit.to_csv(audit_path, index=False)

    summary = pd.DataFrame(
        [
            {
                "same_lulc_locations": int(audit["same_lulc_class"].sum()),
                "total_locations": int(len(audit)),
                "same_lulc_fraction": float(audit["same_lulc_class"].mean()),
                "delta_fsi_zero_count_1e_12": int((audit["delta_fsi"].abs() < 1e-12).sum()),
                "bogor_mean_elevation_raw_m": float(audit[audit["city"].str.contains("Bogor", case=False)]["elevation_raw_m"].mean()),
                "bogor_mean_elevation_score": float(audit[audit["city"].str.contains("Bogor", case=False)]["elevation_score"].mean()),
                "bogor_mean_rainfall_raw": float(audit[audit["city"].str.contains("Bogor", case=False)]["rainfall_raw_mm_year"].mean()),
                "bogor_mean_rainfall_score": float(audit[audit["city"].str.contains("Bogor", case=False)]["rainfall_score"].mean()),
            }
        ]
    )
    summary_path = out_dir / "fsi_ablation_audit_summary.csv"
    summary.to_csv(summary_path, index=False)
    for path in tmp_paths.values():
        path.unlink(missing_ok=True)
    return audit_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    png = plot_fsi(cfg)
    audit, summary = write_audit_tables(cfg)
    print(f"FSI plot: {png}")
    print(f"Ablation/LULC audit: {audit}")
    print(f"Audit summary: {summary}")


if __name__ == "__main__":
    main()
