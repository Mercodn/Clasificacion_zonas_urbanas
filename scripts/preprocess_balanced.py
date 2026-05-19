"""Preprocess additional balanced dataset and merge with existing VIIRS points.

Creates `data_prepared.csv` containing rows with columns:
  system:index,avg_rad,.geo

Strategy
- Extract centroid (lon, lat) from POLYGON/MULTIPOLYGON geometries by
  averaging vertices (approximate centroid).
- Use `ln_mean_NTL` column (if present) as a proxy for nighttime lights;
  reconstruct raw NTL via exp(ln_mean_NTL) and scale it to match the
  magnitude of existing `avg_rad` in `data.csv` using medians. This keeps
  the new points on a comparable scale for the downstream model.

The script is conservative: it writes `data_prepared.csv` and does not
overwrite original data files.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
import math

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXISTING = ROOT / "data.csv"
BALANCED = ROOT / "balanced_dataSet_more_pollutants.csv"
OUT = ROOT / "data_prepared.csv"


COORD_PAIR_RE = re.compile(r"(-?\d+\.\d+)\s+(-?\d+\.\d+)")


def centroid_from_geometry(geom: str) -> tuple[float, float]:
    """Approximate centroid by averaging all coordinate pairs found.
    Works for POLYGON and MULTIPOLYGON WKT-like strings present in the CSV.
    """
    matches = COORD_PAIR_RE.findall(geom)
    if not matches:
        raise ValueError("no coordinates found in geometry")
    lons = [float(m[0]) for m in matches]
    lats = [float(m[1]) for m in matches]
    return float(np.mean(lons)), float(np.mean(lats))


def load_existing(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # ensure expected columns
    if "avg_rad" not in df.columns or ".geo" not in df.columns:
        raise RuntimeError("existing data.csv missing required columns")
    return df


def prepare_balanced(path: Path, ref_median: float) -> pd.DataFrame:
    df = pd.read_csv(path)

    # compute centroids
    lons = []
    lats = []
    for geom in df["geometry"].astype(str):
        try:
            lon, lat = centroid_from_geometry(geom)
        except Exception:
            lon, lat = math.nan, math.nan
        lons.append(lon)
        lats.append(lat)

    df["lon"] = lons
    df["lat"] = lats

    # Choose a column to derive avg_rad proxy. Prefer ln_mean_NTL, then ln_sum_NTL.
    if "ln_mean_NTL" in df.columns:
        raw_ntl = np.exp(df["ln_mean_NTL"].fillna(0).astype(float))
    elif "ln_sum_NTL" in df.columns:
        raw_ntl = np.exp(df["ln_sum_NTL"].fillna(0).astype(float))
    else:
        # fallback: use mean_PM25 as very rough proxy
        raw_ntl = df.get("mean_PM25", pd.Series(np.zeros(len(df)))).astype(float)

    # scale raw_ntl to match median of existing avg_rad so magnitudes are comparable
    med_new = float(np.nanmedian(raw_ntl.replace([np.inf, -np.inf], np.nan).fillna(0)))
    med_ref = float(ref_median)
    if med_new <= 0 or not np.isfinite(med_new):
        scale = 1.0
    else:
        scale = med_ref / med_new if med_new > 0 else 1.0

    avg_rad = raw_ntl * scale

    # Build .geo field in the same format as the project's data.csv
    geo_jsons = []
    for lon, lat in zip(df["lon"], df["lat"]):
        if pd.isna(lon) or pd.isna(lat):
            geo_jsons.append("")
        else:
            geo = {"geodesic": False, "type": "Point", "coordinates": [float(lon), float(lat)]}
            geo_jsons.append(json.dumps(geo))

    out = pd.DataFrame({
        "system:index": df.get("ID", np.arange(len(df))).astype(str),
        "avg_rad": avg_rad,
        ".geo": geo_jsons,
    })
    # drop rows without valid geometry
    out = out[out[".geo"] != ""].reset_index(drop=True)
    return out


def main() -> None:
    if not EXISTING.exists():
        raise FileNotFoundError(f"Existing data file not found: {EXISTING}")
    if not BALANCED.exists():
        raise FileNotFoundError(f"Balanced dataset not found: {BALANCED}")

    existing = load_existing(EXISTING)
    ref_median = existing["avg_rad"].median()

    balanced_prepared = prepare_balanced(BALANCED, ref_median=ref_median)

    # concat and write
    merged = pd.concat([existing, balanced_prepared.rename(columns={"system:index":"system:index"})], ignore_index=True)
    merged = merged[["system:index", "avg_rad", ".geo"]]
    merged.to_csv(OUT, index=False)
    print(f"Wrote prepared data to: {OUT}")


if __name__ == "__main__":
    main()
