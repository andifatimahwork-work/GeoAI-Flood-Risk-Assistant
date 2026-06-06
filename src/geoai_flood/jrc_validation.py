from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

from .config import resolve_path


def read_raster(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        profile = {
            "crs": str(src.crs),
            "bounds": tuple(float(v) for v in src.bounds),
            "shape": (src.height, src.width),
            "transform": tuple(float(v) for v in src.transform),
            "res": tuple(float(v) for v in src.res),
            "nodata": src.nodata,
        }
    return arr, profile


def assert_aligned(fsi_profile: dict[str, Any], jrc_profile: dict[str, Any]) -> None:
    keys = ["crs", "bounds", "shape", "transform"]
    mismatches = {key: (fsi_profile[key], jrc_profile[key]) for key in keys if fsi_profile[key] != jrc_profile[key]}
    if mismatches:
        raise ValueError(f"FSI and JRC rasters are not aligned: {mismatches}")


def confusion_counts(pred: np.ndarray, truth: np.ndarray) -> dict[str, int]:
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    tn = int((~pred & ~truth).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def metrics_from_counts(counts: dict[str, int]) -> dict[str, float]:
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]
    return {
        "hit_rate_recall": tp / max(tp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "specificity": tn / max(tn + fp, 1),
        "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
        "f1": (2 * tp) / max(2 * tp + fp + fn, 1),
    }


def plot_validation_map(
    fsi: np.ndarray,
    jrc_mask: np.ndarray,
    fsi_pred: np.ndarray,
    valid: np.ndarray,
    out_path: Path,
) -> None:
    hit = fsi_pred & jrc_mask & valid
    miss = (~fsi_pred) & jrc_mask & valid
    false_alarm = fsi_pred & (~jrc_mask) & valid

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fsi_plot = np.where(valid, fsi, np.nan)
    im = axes[0].imshow(fsi_plot, cmap="RdYlGn_r", vmin=0.5, vmax=1.0)
    axes[0].set_title("FSI Score")
    axes[0].axis("off")
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].imshow(np.where(valid, jrc_mask, np.nan), cmap="Blues")
    axes[1].set_title("JRC GSW occurrence >= 10%")
    axes[1].axis("off")

    overlay = np.zeros((*fsi.shape, 4), dtype=np.float32)
    overlay[false_alarm] = [1.0, 0.75, 0.10, 0.9]
    overlay[miss] = [0.90, 0.10, 0.10, 0.9]
    overlay[hit] = [0.10, 0.65, 0.20, 0.9]
    axes[2].imshow(np.where(valid, fsi, np.nan), cmap="Greys", vmin=0.5, vmax=1.0)
    axes[2].imshow(overlay)
    axes[2].set_title("Validation overlay: green hit, red miss, yellow false alarm")
    axes[2].axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def ablation_reference(cfg: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    root = Path(cfg["_project_root"])
    candidates = [
        resolve_path(cfg["paths"]["output_dir"], root) / "ablation_results.csv",
        resolve_path(cfg["fsi"]["output_fsi_tif"], root).parent / "ablation_results.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            if {"lat", "lon", "delta_fsi"}.issubset(df.columns):
                top = df.reindex(df["delta_fsi"].abs().sort_values(ascending=False).index).head(10)
                top_path = out_dir / "ablation_top_delta_for_validation.csv"
                top.to_csv(top_path, index=False)
                return {"available": True, "path": str(path), "top_delta_path": str(top_path)}
    return {"available": False, "path": None, "top_delta_path": None}


def run_jrc_validation(cfg: dict[str, Any]) -> Path:
    root = Path(cfg["_project_root"])
    validation_cfg = cfg.get("validation", {})
    out_dir = resolve_path(validation_cfg.get("output_dir", "outputs/validation_exp2"), root)
    out_dir.mkdir(parents=True, exist_ok=True)

    fsi_path = resolve_path(validation_cfg.get("fsi_tif", cfg["fsi"]["output_fsi_tif"]), root)
    baseline_path = resolve_path(cfg["fsi"]["output_baseline_fsi_tif"], root)
    jrc_path = resolve_path(validation_cfg.get("jrc_gsw_tif", cfg["paths"]["jrc_gsw_tif"]), root)
    fsi_threshold = float(validation_cfg.get("fsi_threshold", 0.50))
    jrc_threshold = float(validation_cfg.get("jrc_occurrence_threshold", 10.0))

    fsi, fsi_profile = read_raster(fsi_path)
    jrc, jrc_profile = read_raster(jrc_path)
    assert_aligned(fsi_profile, jrc_profile)

    valid = np.isfinite(fsi) & np.isfinite(jrc)
    jrc_mask = jrc >= jrc_threshold
    fsi_pred = fsi > fsi_threshold
    counts = confusion_counts(fsi_pred[valid], jrc_mask[valid])
    metrics = metrics_from_counts(counts)

    results: dict[str, Any] = {
        "fsi_path": str(fsi_path),
        "jrc_gsw_path": str(jrc_path),
        "thresholds": {
            "fsi_positive": fsi_threshold,
            "jrc_occurrence_percent": jrc_threshold,
        },
        "alignment": {
            "crs": fsi_profile["crs"],
            "bounds": fsi_profile["bounds"],
            "shape": fsi_profile["shape"],
            "res": fsi_profile["res"],
        },
        "valid_pixels": int(valid.sum()),
        "masked_or_nodata_pixels": int((~valid).sum()),
        "jrc_positive_pixels": int((jrc_mask & valid).sum()),
        "fsi_positive_pixels": int((fsi_pred & valid).sum()),
        "confusion_matrix": {
            "labels": ["JRC_negative", "JRC_positive"],
            "matrix_pred_rows_truth_cols": [
                [counts["tn"], counts["fn"]],
                [counts["fp"], counts["tp"]],
            ],
            **counts,
        },
        "metrics": metrics,
        "target": {
            "hit_rate_recall_min": 0.65,
            "pass": bool(metrics["hit_rate_recall"] >= 0.65),
        },
        "note": "Validation excludes FSI NoData pixels, including LULC water pixels masked during FSI generation.",
        "ablation_reference": ablation_reference(cfg, out_dir),
    }

    if baseline_path.exists():
        baseline, baseline_profile = read_raster(baseline_path)
        assert_aligned(baseline_profile, jrc_profile)
        baseline_valid = np.isfinite(baseline) & np.isfinite(jrc)
        baseline_pred = baseline > fsi_threshold
        baseline_counts = confusion_counts(baseline_pred[baseline_valid], jrc_mask[baseline_valid])
        results["baseline_esa"] = {
            "path": str(baseline_path),
            "valid_pixels": int(baseline_valid.sum()),
            "confusion_matrix": baseline_counts,
            "metrics": metrics_from_counts(baseline_counts),
        }

    result_path = out_dir / "validation_results.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    plot_validation_map(fsi, jrc_mask, fsi_pred, valid, out_dir / "validation_map.png")
    pd.DataFrame([results["metrics"]]).to_csv(out_dir / "validation_metrics.csv", index=False)
    return result_path
