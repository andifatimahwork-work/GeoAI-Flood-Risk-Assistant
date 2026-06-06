from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geoai_flood.config import load_config
from geoai_flood.geoai_tools import gee_live_status, gee_raw_values, get_flood_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--lat", type=float, default=-6.1214)
    parser.add_argument("--lon", type=float, default=106.7741)
    parser.add_argument("--full", action="store_true", help="Also run U-Net LULC and full live FSI. Raw-only is faster.")
    args = parser.parse_args()

    status = gee_live_status(args.config)
    safe_status = dict(status)
    if safe_status.get("earthengine_project"):
        safe_status["earthengine_project"] = str(safe_status["earthengine_project"])
    print("GEE live readiness:")
    print(json.dumps(safe_status, indent=2, ensure_ascii=False))

    if not status["ready_for_local_live"]:
        print("\nBelum siap untuk live mode.")
        print("Opsi lokal termudah:")
        print("1. Jalankan: earthengine authenticate")
        print("2. Isi .env: EARTHENGINE_PROJECT=your-google-cloud-project-id")
        print("\nOpsi deployment:")
        print("1. Buat service account JSON.")
        print("2. Isi .env: GEE_SERVICE_ACCOUNT_KEY=path\\to\\service_account.json")
        print("3. Isi .env: EARTHENGINE_PROJECT=your-google-cloud-project-id")
        raise SystemExit(1)

    cfg = load_config(args.config)
    print(f"\nTesting GEE raw layers at lat={args.lat}, lon={args.lon} ...")
    raw = gee_raw_values(args.lat, args.lon, cfg)
    print(json.dumps(raw, indent=2, ensure_ascii=False))

    if args.full:
        print("\nTesting full live FSI, including Sentinel-2 patch and U-Net CPU inference ...")
        result = get_flood_score(args.lat, args.lon, args.config, mode="live", use_cache=False)
        keep = {
            "fsi_score": result.get("fsi_score"),
            "category": result.get("category"),
            "ci_low": result.get("ci_low"),
            "ci_high": result.get("ci_high"),
            "lulc": result.get("lulc"),
            "breakdown": result.get("breakdown"),
            "weighted_breakdown": result.get("weighted_breakdown"),
            "warnings": result.get("warnings"),
            "computation_mode": result.get("computation_mode"),
        }
        print(json.dumps(keep, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
