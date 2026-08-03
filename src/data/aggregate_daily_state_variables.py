"""Derive basin-average daily mean temperature or snow depth from ERA5-Land."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rioxarray  # noqa: F401; activates the rio accessor
import xarray as xr


def parse_args():
    parser = argparse.ArgumentParser(description="Create a basin-average daily mean ERA5-Land series.")
    parser.add_argument("--input", type=Path, required=True, help="NetCDF or GRIB file containing one variable.")
    parser.add_argument("--variable", required=True, help="Dataset variable name, e.g. t2m or sd.")
    parser.add_argument("--shapefile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-column", required=True)
    parser.add_argument("--kelvin-to-celsius", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = xr.open_dataset(args.input, engine="cfgrib" if args.input.suffix.lower() == ".grib" else None)
    data = dataset[args.variable]
    if not data.rio.crs:
        data = data.rio.write_crs("EPSG:4326")
    basin = gpd.read_file(args.shapefile).to_crs("EPSG:4326")
    clipped = data.rio.clip(basin.geometry, basin.crs, drop=True)
    spatial_dims = [dim for dim in clipped.dims if dim.lower() in {"latitude", "longitude", "lat", "lon", "x", "y"}]
    daily = clipped.resample(time="1D").mean(skipna=True).mean(dim=spatial_dims, skipna=True)
    result = daily.to_dataframe(name=args.output_column).reset_index()[["time", args.output_column]]
    result = result.rename(columns={"time": "Date"})
    if args.kelvin_to_celsius:
        result[args.output_column] = result[args.output_column] - 273.15
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
