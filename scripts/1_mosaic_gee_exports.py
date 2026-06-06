from __future__ import annotations

import argparse
from pathlib import Path

import rasterio
from rasterio.merge import merge


EXPECTED_PREFIXES = {
    "s2_composite_jabodetabek": "s2_composite_jabodetabek.tif",
    "esa_worldcover_jabodetabek": "esa_worldcover_jabodetabek.tif",
    "jrc_gsw_occurrence_jabodetabek": "jrc_gsw_occurrence_jabodetabek.tif",
    "worldpop_population_density_jabodetabek": "worldpop_population_density_jabodetabek.tif",
    "elevation_srtm_30m_jabodetabek": "elevation_srtm_30m.tif",
    "slope_srtm_30m_jabodetabek": "slope_srtm_30m.tif",
    "chirps_mean_annual_1990_2023_jabodetabek": "chirps_mean_annual_1990_2023.tif",
}


def find_parts(input_dir: Path, prefix: str) -> list[Path]:
    direct = input_dir / f"{prefix}.tif"
    if direct.exists():
        return [direct]
    parts = sorted(input_dir.glob(f"{prefix}-*.tif"))
    if not parts:
        raise FileNotFoundError(f"No GeoTIFF parts found for prefix '{prefix}' in {input_dir}")
    return parts


def mosaic(parts: list[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    datasets = [rasterio.open(path) for path in parts]
    try:
        arr, transform = merge(datasets)
        profile = datasets[0].profile.copy()
        profile.update(
            height=arr.shape[1],
            width=arr.shape[2],
            transform=transform,
            count=arr.shape[0],
            compress="deflate",
            tiled=True,
        )
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(arr)
    finally:
        for ds in datasets:
            ds.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mosaic sharded Google Earth Engine GeoTIFF exports into the single filenames expected by the pipeline."
    )
    parser.add_argument("--input-dir", required=True, help="Folder containing downloaded GEE .tif files.")
    parser.add_argument("--output-dir", default="data/raw", help="Folder for merged output GeoTIFFs.")
    parser.add_argument("--prefix", default=None, help="Optional single export prefix to mosaic.")
    parser.add_argument("--output-name", default=None, help="Optional output filename when --prefix is used.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if args.prefix:
        prefixes = {args.prefix: args.output_name or EXPECTED_PREFIXES.get(args.prefix, f"{args.prefix}.tif")}
    else:
        prefixes = EXPECTED_PREFIXES
    for prefix, output_name in prefixes.items():
        parts = find_parts(input_dir, prefix)
        output_path = output_dir / output_name
        print(f"{prefix}: {len(parts)} file(s) -> {output_path}")
        mosaic(parts, output_path)
    print("Mosaic complete.")


if __name__ == "__main__":
    main()
