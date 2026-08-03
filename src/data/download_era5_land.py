"""Download monthly ERA5-Land variables required by the study.

The request is intentionally parameterized: users must set the geographical
bounding box and choose an output directory appropriate to their local data
access and storage policy. Authentication is handled by ``cdsapi``.
"""

from __future__ import annotations

import argparse
from calendar import monthrange
from pathlib import Path

import cdsapi


VARIABLES = {
    "tp": "total_precipitation",
    "t2m": "2m_temperature",
    "e": "total_evaporation",
    "sd": "snow_depth",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Download a monthly ERA5-Land subset.")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True, choices=range(1, 13))
    parser.add_argument("--output-dir", type=Path, default=Path("data/private/era5_land"))
    parser.add_argument(
        "--area",
        type=float,
        nargs=4,
        metavar=("N", "W", "S", "E"),
        required=True,
        help="Bounding box in CDS order: north west south east.",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        choices=sorted(VARIABLES),
        default=sorted(VARIABLES),
        help="Short names for the study variables.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    days = [f"{day:02d}" for day in range(1, monthrange(args.year, args.month)[1] + 1)]
    hours = [f"{hour:02d}:00" for hour in range(24)]
    client = cdsapi.Client()

    for short_name in args.variables:
        output_dir = args.output_dir / short_name
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / f"era5_land_{short_name}_{args.year}_{args.month:02d}.grib"
        if target.exists():
            print(f"Skipping existing file: {target}")
            continue
        request = {
            "variable": [VARIABLES[short_name]],
            "year": str(args.year),
            "month": f"{args.month:02d}",
            "day": days,
            "time": hours,
            "data_format": "grib",
            "download_format": "unarchived",
            "area": args.area,
        }
        print(f"Downloading {VARIABLES[short_name]} to {target}")
        client.retrieve("reanalysis-era5-land", request).download(str(target))


if __name__ == "__main__":
    main()
