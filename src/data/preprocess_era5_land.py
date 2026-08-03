"""Regenerate ERA5-Land daily precipitation and evaporation predictors.

This script fixes the key aggregation issue in the previous workflow:
ERA5-Land total precipitation (tp) and total evaporation (e) are accumulated
forecast variables. Daily values should therefore be derived from the 24 h
accumulation at forecast step 24, not from an arithmetic mean of hourly
accumulated fields.

The script does not overwrite the existing summary CSV. It writes corrected
Pz/Pi/Ez/Ei series and a merged `summary_corrected.csv` into a separate output
directory.
"""

from __future__ import annotations

import argparse
import calendar
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401 - activates the .rio accessor
import xarray as xr
from rasterio.enums import Resampling


DATA_ROOT = Path(os.environ.get('RUNOFF_DATA_ROOT', 'data/private'))
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OLD_SUMMARY = Path(os.environ.get('RUNOFF_DATA_PATH', str(DATA_ROOT / 'model_input.csv')))
DEFAULT_OUTPUT_DIR = Path(os.environ.get('ERA5_OUTPUT_DIR', 'outputs/era5_daily_corrected'))
P_DIR = Path(os.environ.get('ERA5_PRECIP_DIR', str(DATA_ROOT / 'era5_land' / 'tp')))
E_DIR = Path(os.environ.get('ERA5_EVAP_DIR', str(DATA_ROOT / 'era5_land' / 'e')))
SHP_DIR = Path(os.environ.get('BASIN_SHP_DIR', str(DATA_ROOT / 'shapefiles')))


@dataclass(frozen=True)
class BasinConfig:
    label: str
    shp_path: Path


@dataclass(frozen=True)
class VariableConfig:
    label: str
    var_name: str
    raw_dir: Path
    filename_template: str
    output_columns: Dict[str, str]


BASINS: Dict[str, BasinConfig] = {
    "z": BasinConfig("z", SHP_DIR / "zmd.shp"),
    "i": BasinConfig("i", SHP_DIR / "qujian.shp"),
}

VARIABLES: Dict[str, VariableConfig] = {
    "P": VariableConfig(
        label="P",
        var_name="tp",
        raw_dir=P_DIR,
        filename_template="era5_land_tp_{year}_{month:02d}.grib",
        output_columns={"z": "Pz", "i": "Pi"},
    ),
    "E": VariableConfig(
        label="E",
        var_name="e",
        raw_dir=E_DIR,
        filename_template="era5_land_evap_{year}_{month:02d}.grib",
        output_columns={"z": "Ez", "i": "Ei"},
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate daily ERA5-Land P/E predictors using step-24 "
            "accumulations."
        )
    )
    parser.add_argument("--start-year", type=int, default=2006)
    parser.add_argument("--end-year", type=int, default=2020)
    parser.add_argument(
        "--months",
        type=str,
        default="1-12",
        help="Month selection, e.g. '1-12' or '1,2,7'.",
    )
    parser.add_argument(
        "--variables",
        type=str,
        default="P,E",
        help="Variables to process: P, E, or P,E.",
    )
    parser.add_argument(
        "--basins",
        type=str,
        default="z,i",
        help="Basins to process: z, i, or z,i.",
    )
    parser.add_argument(
        "--old-summary",
        type=Path,
        default=DEFAULT_OLD_SUMMARY,
        help="Existing summary CSV used to keep Q/T/S and merge corrected P/E.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for corrected outputs.",
    )
    parser.add_argument(
        "--resample-resolution",
        type=float,
        default=0.005,
        help=(
            "Optional spatial resampling resolution in degrees. Use 0 to skip "
            "resampling and only clip/mask the native grid."
        ),
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Only write corrected P/E series, without merged summary CSV.",
    )
    return parser.parse_args()


def parse_selection(text: str, valid: Iterable[str]) -> List[str]:
    valid_set = set(valid)
    selected = [item.strip() for item in text.split(",") if item.strip()]
    invalid = [item for item in selected if item not in valid_set]
    if invalid:
        raise ValueError(f"Invalid selection {invalid}; valid values are {sorted(valid_set)}")
    return selected


def parse_months(text: str) -> List[int]:
    text = text.strip()
    if "-" in text:
        start, end = [int(x) for x in text.split("-", 1)]
        months = list(range(start, end + 1))
    else:
        months = [int(x.strip()) for x in text.split(",") if x.strip()]
    invalid = [m for m in months if m < 1 or m > 12]
    if invalid:
        raise ValueError(f"Invalid months: {invalid}")
    return months


def open_grib_var(path: Path, var_name: str) -> xr.DataArray:
    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
    if var_name not in ds.data_vars:
        raise KeyError(f"{var_name!r} not found in {path}; variables={list(ds.data_vars)}")
    da = ds[var_name]
    # Keep the dataset alive through the DataArray object until it is loaded.
    return da


def date_window(year: int, month: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    start_date = pd.Timestamp(year=year, month=month, day=1)
    end_day = calendar.monthrange(year, month)[1]
    end_date = pd.Timestamp(year=year, month=month, day=end_day)
    return start_date, end_date


def next_year_month(year: int, month: int) -> Tuple[int, int]:
    if month == 12:
        return year + 1, 1
    return year, month + 1


def select_step24_accumulation(
    da: xr.DataArray,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> xr.DataArray:
    """Select day-D 24 h accumulations using valid_time - 1 day as date."""
    if "step" not in da.dims and "step" not in da.coords:
        raise ValueError("Accumulated ERA5-Land variable has no 'step' coordinate.")

    step_values = pd.to_timedelta(da["step"].values)
    matches = np.where(step_values == pd.Timedelta(hours=24))[0]
    if len(matches) != 1:
        raise ValueError(f"Expected one 24 h step; found {len(matches)} matches.")

    da24 = da.isel(step=int(matches[0])) if "step" in da.dims else da
    if "valid_time" in da24.coords:
        dates = pd.to_datetime(da24["valid_time"].values).normalize() - pd.Timedelta(days=1)
    else:
        dates = pd.to_datetime(da24["time"].values).normalize()
    da24 = da24.assign_coords(time=dates)
    da24 = da24.sel(time=slice(start_date, end_date))
    return da24


def standardize_spatial_dims(da: xr.DataArray) -> xr.DataArray:
    if "longitude" in da.dims and "latitude" in da.dims:
        x_dim, y_dim = "longitude", "latitude"
    elif "lon" in da.dims and "lat" in da.dims:
        x_dim, y_dim = "lon", "lat"
    elif "x" in da.dims and "y" in da.dims:
        x_dim, y_dim = "x", "y"
    else:
        raise ValueError(f"Could not infer spatial dimensions from {da.dims}")

    da = da.rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim, inplace=False)
    da = da.rio.write_crs("EPSG:4326", inplace=False)
    return da


def clip_resample_average(
    da: xr.DataArray,
    basin_gdf: gpd.GeoDataFrame,
    resample_resolution: Optional[float],
) -> pd.Series:
    """Clip, optionally resample, mask again, and spatially average."""
    da = standardize_spatial_dims(da)
    basin = basin_gdf.to_crs("EPSG:4326")

    clipped = da.rio.clip(
        basin.geometry,
        basin.crs,
        drop=False,
        all_touched=True,
    )

    if resample_resolution and resample_resolution > 0:
        clipped = clipped.rio.reproject(
            dst_crs="EPSG:4326",
            resolution=resample_resolution,
            resampling=Resampling.bilinear,
        )
        clipped = clipped.rio.clip(
            basin.geometry,
            basin.crs,
            drop=True,
            all_touched=True,
        )

    x_dim = clipped.rio.x_dim
    y_dim = clipped.rio.y_dim
    spatial_mean = clipped.mean(dim=[y_dim, x_dim], skipna=True).load()
    values = np.asarray(spatial_mean.values, dtype=float)
    dates = pd.to_datetime(spatial_mean["time"].values).normalize()
    return pd.Series(values, index=dates)


def process_one_month(
    variable: VariableConfig,
    basin: BasinConfig,
    year: int,
    month: int,
    resample_resolution: Optional[float],
) -> pd.DataFrame:
    start_date, end_date = date_window(year, month)
    next_year, next_month = next_year_month(year, month)
    candidate_paths = [
        variable.raw_dir / variable.filename_template.format(year=year, month=month),
        variable.raw_dir / variable.filename_template.format(year=next_year, month=next_month),
    ]
    basin_gdf = gpd.read_file(basin.shp_path)

    col = variable.output_columns[basin.label]
    pieces: List[pd.Series] = []
    for path in candidate_paths:
        if not path.exists():
            print(f"WARNING: missing boundary file {path}")
            continue
        da = open_grib_var(path, variable.var_name)
        da24 = select_step24_accumulation(da, start_date, end_date)
        if da24.sizes.get("time", 0) == 0:
            continue
        pieces.append(clip_resample_average(da24, basin_gdf, resample_resolution))

    if not pieces:
        raise ValueError(f"No data for {variable.label}{basin.label} {year}-{month:02d}")

    series = pd.concat(pieces).sort_index()
    series = series.groupby(level=0).mean()
    expected = pd.date_range(start_date, end_date, freq="D")
    series = series.reindex(expected)

    out = series.rename(col).reset_index()
    out.columns = ["Date", col]
    return out


def process_series(
    variable: VariableConfig,
    basin: BasinConfig,
    years: Iterable[int],
    months: Iterable[int],
    output_dir: Path,
    resample_resolution: Optional[float],
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for year in years:
        for month in months:
            print(f"Processing {variable.label}{basin.label}: {year}-{month:02d}")
            frames.append(
                process_one_month(variable, basin, year, month, resample_resolution)
            )

    if not frames:
        raise ValueError("No monthly frames were generated.")

    result = pd.concat(frames, ignore_index=True)
    result["Date"] = pd.to_datetime(result["Date"])
    result = result.sort_values("Date").drop_duplicates("Date", keep="last")

    col = variable.output_columns[basin.label]
    out_path = output_dir / f"corrected_{col}.csv"
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {out_path}")
    return result


def expected_dates(start_year: int, end_year: int, months: List[int]) -> pd.DatetimeIndex:
    frames: List[pd.DatetimeIndex] = []
    for year in range(start_year, end_year + 1):
        for month in months:
            start = pd.Timestamp(year=year, month=month, day=1)
            end = pd.Timestamp(year=year, month=month, day=calendar.monthrange(year, month)[1])
            frames.append(pd.date_range(start, end, freq="D"))
    return pd.DatetimeIndex([]).append(frames)


def merge_corrected_summary(
    old_summary_path: Path,
    corrected: Dict[str, pd.DataFrame],
    output_dir: Path,
) -> pd.DataFrame:
    old = pd.read_csv(old_summary_path)
    old["Date"] = pd.to_datetime(old["Date"])
    merged = old.copy()

    for col, df in corrected.items():
        tmp = df.copy()
        tmp["Date"] = pd.to_datetime(tmp["Date"])
        tmp = tmp[["Date", col]]
        if col in merged.columns:
            merged = merged.drop(columns=[col])
        merged = merged.merge(tmp, on="Date", how="left")

    original_order = ["Date", "Q_shigu", "Qz", "Pz", "Tz", "Ez", "Sz", "Pi", "Ti", "Ei", "Si"]
    cols = [c for c in original_order if c in merged.columns] + [
        c for c in merged.columns if c not in original_order
    ]
    merged = merged[cols]

    out_path = output_dir / "summary_corrected.csv"
    to_write = merged.copy()
    to_write["Date"] = to_write["Date"].dt.strftime("%Y-%m-%d")
    to_write.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {out_path}")
    return merged


def write_quality_reports(
    old_summary_path: Path,
    merged: pd.DataFrame,
    corrected_cols: List[str],
    output_dir: Path,
) -> None:
    reports: List[pd.DataFrame] = []
    merged = merged.copy()
    merged["Date"] = pd.to_datetime(merged["Date"])

    for col in corrected_cols:
        annual = merged.groupby(merged["Date"].dt.year)[col].sum(min_count=1)
        reports.append(
            pd.DataFrame(
                {
                    "variable": col,
                    "year": annual.index,
                    "annual_sum": annual.values,
                }
            )
        )

    annual_report = pd.concat(reports, ignore_index=True)
    annual_path = output_dir / "annual_sums_corrected.csv"
    annual_report.to_csv(annual_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {annual_path}")

    summary_rows = []
    old = pd.read_csv(old_summary_path)
    old["Date"] = pd.to_datetime(old["Date"])
    for col in corrected_cols:
        old_s = old.set_index("Date")[col] if col in old.columns else None
        new_s = merged.set_index("Date")[col]
        row = {
            "variable": col,
            "new_daily_min": new_s.min(),
            "new_daily_max": new_s.max(),
            "new_daily_mean": new_s.mean(),
            "new_total_sum": new_s.sum(),
            "new_annual_sum_mean": new_s.groupby(new_s.index.year).sum().mean(),
            "new_annual_sum_min": new_s.groupby(new_s.index.year).sum().min(),
            "new_annual_sum_max": new_s.groupby(new_s.index.year).sum().max(),
        }
        if old_s is not None:
            aligned = pd.concat([old_s.rename("old"), new_s.rename("new")], axis=1).dropna()
            row.update(
                {
                    "old_daily_mean": aligned["old"].mean(),
                    "old_total_sum": aligned["old"].sum(),
                    "mean_abs_difference": (aligned["new"] - aligned["old"]).abs().mean(),
                    "max_abs_difference": (aligned["new"] - aligned["old"]).abs().max(),
                    "ratio_new_to_old_sum": aligned["new"].sum() / aligned["old"].sum()
                    if aligned["old"].sum() != 0
                    else np.nan,
                }
            )
        summary_rows.append(row)

    summary_path = output_dir / "old_vs_corrected_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {summary_path}")


def main() -> None:
    args = parse_args()
    months = parse_months(args.months)
    var_keys = parse_selection(args.variables, VARIABLES)
    basin_keys = parse_selection(args.basins, BASINS)
    years = range(args.start_year, args.end_year + 1)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    missing = []
    for basin_key in basin_keys:
        if not BASINS[basin_key].shp_path.exists():
            missing.append(BASINS[basin_key].shp_path)
    for var_key in var_keys:
        var = VARIABLES[var_key]
        for year in years:
            for month in months:
                path = var.raw_dir / var.filename_template.format(year=year, month=month)
                if not path.exists():
                    missing.append(path)
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(str(p) for p in missing[:50]))

    corrected: Dict[str, pd.DataFrame] = {}
    for var_key in var_keys:
        var = VARIABLES[var_key]
        for basin_key in basin_keys:
            basin = BASINS[basin_key]
            df = process_series(
                variable=var,
                basin=basin,
                years=years,
                months=months,
                output_dir=output_dir,
                resample_resolution=args.resample_resolution,
            )
            col = var.output_columns[basin.label]
            corrected[col] = df

    expected = expected_dates(args.start_year, args.end_year, months)
    for col, df in corrected.items():
        got = pd.DatetimeIndex(pd.to_datetime(df["Date"]))
        missing_dates = expected.difference(got)
        if len(missing_dates):
            print(f"WARNING: {col} missing {len(missing_dates)} dates.")

    if args.no_merge:
        return

    merged = merge_corrected_summary(args.old_summary, corrected, output_dir)
    write_quality_reports(args.old_summary, merged, list(corrected), output_dir)


if __name__ == "__main__":
    main()
