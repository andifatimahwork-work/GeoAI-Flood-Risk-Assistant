from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geoai_flood.geoai_tools import get_flood_score


SAMPLE_POINTS = [
    {"name": "Jakarta Utara - coastal urban", "lat": -6.1214, "lon": 106.7741},
    {"name": "Bogor - inland higher elevation", "lat": -6.5950, "lon": 106.8167},
    {"name": "Depok - urban transition", "lat": -6.4025, "lon": 106.7942},
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--lat", type=float, default=None)
    parser.add_argument("--lon", type=float, default=None)
    parser.add_argument("--mode", choices=["auto", "live", "local"], default="auto")
    parser.add_argument("--out", default="outputs/flood_score_smoke_test.json")
    args = parser.parse_args()

    if args.lat is not None or args.lon is not None:
        if args.lat is None or args.lon is None:
            raise ValueError("Use both --lat and --lon for a custom point.")
        points = [{"name": "custom", "lat": args.lat, "lon": args.lon}]
    else:
        points = SAMPLE_POINTS

    results = []
    for point in points:
        result = get_flood_score(point["lat"], point["lon"], args.config, mode=args.mode)
        result["name"] = point["name"]
        results.append(result)
        print(f"\n{point['name']}")
        print(f"  lat/lon : {point['lat']}, {point['lon']}")
        print(f"  FSI     : {result['fsi_score']} ({result['category']})")
        print(f"  CI      : {result['ci_low']} - {result['ci_high']}")
        print(f"  LULC    : {result['lulc']['class_name']}")
        print(f"  Mode    : {result['computation_mode']} cached={result['cached']}")
        if result.get("warnings"):
            print(f"  Warning : {'; '.join(result['warnings'])}")
        print(f"  Breakdown: {result['breakdown']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved flood score smoke test: {out}")


if __name__ == "__main__":
    main()
