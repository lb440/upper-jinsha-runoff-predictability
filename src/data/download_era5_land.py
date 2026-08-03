"""Download ERA5-Land Jan 2021 P/E boundary files.

The corrected daily aggregation uses step-24 accumulations and assigns them to
valid_time - 1 day. Therefore, the daily P/E values for 2020-12-31 require the
2021-01-01 00:00 valid-time boundary record. Downloading the full Jan 2021
monthly files keeps the local file convention identical to the existing
2006-2020 GRIB archive.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import cdsapi


DATA_ROOT = Path.home() / "Desktop" / "\u6570\u636e"
P_DIR = DATA_ROOT / "Total precipitation\uff08\u603b\u964d\u6c34\u91cf\uff09"
E_DIR = P_DIR / "Total evaporation\uff08\u603b\u84b8\u53d1\uff09"

AREA = [36, 90, 25, 105]
YEAR = "2021"
MONTH = "01"
DAYS = [f"{day:02d}" for day in range(1, 32)]
TIMES = [f"{hour:02d}:00" for hour in range(24)]


def unzip_first_grib(zip_path: Path, target_grib: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        grib_members = [name for name in zf.namelist() if name.lower().endswith(".grib")]
        if not grib_members:
            raise RuntimeError(f"No GRIB file found in {zip_path}")
        member = grib_members[0]
        tmp_path = target_grib.with_suffix(".tmp.grib")
        with zf.open(member) as src, open(tmp_path, "wb") as dst:
            dst.write(src.read())
        os.replace(tmp_path, target_grib)


def download_variable(variable: str, zip_path: Path, target_grib: Path) -> None:
    if target_grib.exists():
        print(f"Skip existing {target_grib}")
        return

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    request = {
        "variable": [variable],
        "year": YEAR,
        "month": MONTH,
        "day": DAYS,
        "time": TIMES,
        "data_format": "grib",
        "download_format": "zip",
        "area": AREA,
    }

    print(f"Downloading {variable} -> {zip_path}")
    client = cdsapi.Client()
    client.retrieve("reanalysis-era5-land", request).download(str(zip_path))
    unzip_first_grib(zip_path, target_grib)
    print(f"Wrote {target_grib}")


def main() -> None:
    download_variable(
        variable="total_precipitation",
        zip_path=P_DIR / "era5_land_tp_2021_01.zip",
        target_grib=P_DIR / "era5_land_tp_2021_01.grib",
    )
    download_variable(
        variable="total_evaporation",
        zip_path=E_DIR / "era5_land_evap_2021_01.zip",
        target_grib=E_DIR / "era5_land_evap_2021_01.grib",
    )


if __name__ == "__main__":
    main()
