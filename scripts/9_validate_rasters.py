from __future__ import annotations

import argparse
import sys
from pathlib import Path

import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geoai_flood.config import load_config, resolve_path


def raster_signature(path: Path) -> dict[str, object]:
    with rasterio.open(path) as src:
        return {
            "path": str(path),
            "crs": str(src.crs),
            "bounds": tuple(round(v, 8) for v in src.bounds),
            "shape": (src.height, src.width),
            "transform": tuple(round(v, 12) for v in src.transform),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["_project_root"])
    paths = [
        resolve_path(cfg["fsi"]["reference_tif"], root),
        resolve_path(cfg["fsi"]["output_fsi_tif"], root),
        resolve_path(cfg["fsi"]["output_baseline_fsi_tif"], root),
        resolve_path(cfg["fsi"]["output_ci_low_tif"], root),
        resolve_path(cfg["fsi"]["output_ci_high_tif"], root),
    ]
    signatures = [raster_signature(path) for path in paths]
    ref = signatures[0]
    ok = True
    for sig in signatures:
        print(sig)
        for key in ["crs", "bounds", "shape", "transform"]:
            if sig[key] != ref[key]:
                ok = False
                print(f"Mismatch for {sig['path']} on {key}: {sig[key]} != {ref[key]}")
    if not ok:
        raise SystemExit(1)
    print("Raster alignment OK: CRS, bounds, shape, and transform match reference.")


if __name__ == "__main__":
    main()
