"""Fail-fast checksum, schema and invariant validation for load-bearing inputs."""
from __future__ import annotations

import hashlib
import json
import os

import pandas as pd
import pyarrow.parquet as pq


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "data_external", "inputs_manifest.json")


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_columns(frame: pd.DataFrame, columns: set[str], label: str,
                    errors: list[str]) -> None:
    missing = columns - set(frame.columns)
    if missing:
        errors.append(f"{label}: missing columns {sorted(missing)}")


def main() -> None:
    with open(MANIFEST) as handle:
        manifest = json.load(handle)
    errors = []

    print("validating SHA-256 checksums ...")
    for relative, expected in manifest["files"].items():
        path = os.path.join(ROOT, relative)
        if not os.path.exists(path):
            errors.append(f"missing input: {relative}")
            continue
        observed = sha256(path)
        if observed != expected:
            errors.append(f"checksum mismatch: {relative}")
        print(f"  {'OK' if observed == expected else 'FAIL'}  {relative}")

    print("validating schemas and scientific invariants ...")
    temp_path = os.path.join(ROOT, "heat", "meteoblue_temperature.parquet")
    parquet = pq.ParquetFile(temp_path)
    if parquet.metadata.num_rows != 19_801_824:
        errors.append("temperature parquet: unexpected row count")
    temp_columns = set(parquet.schema.names)
    require_columns(pd.DataFrame(columns=list(temp_columns)),
                    {"timestamp", "locationID", "parameter", "value", "qc_flag",
                     "rc_flag"},
                    "temperature parquet", errors)

    qa = pd.read_csv(os.path.join(
        ROOT, "data_external", "meteoblue_qa_metadata.csv"))
    require_columns(qa, {"category", "parameter", "name", "bit", "bitwise",
                         "check-name", "check-desc"},
                    "meteoblue QA metadata", errors)
    qc_bits = set(qa.loc[qa["name"] == "qc_flag", "bit"].astype(int))
    rc_bits = set(qa.loc[qa["name"] == "rc_flag", "bit"].astype(int))
    if qc_bits != {1, 2, 3, 4, 5, 6} or rc_bits != {1}:
        errors.append("meteoblue QA metadata: unexpected flag definitions")

    locations = pd.read_csv(os.path.join(
        ROOT, "heat", "meteoblue_station_locations.csv"))
    require_columns(locations, {"locationID", "EKoord", "NKoord", "masl"},
                    "station locations", errors)
    if len(locations) != 93 or locations.locationID.nunique() != 93:
        errors.append("station locations: expected 93 unique usable stations")

    synoptic = pd.read_csv(os.path.join(
        ROOT, "data", "inputs", "official_meteoswiss_hourly_zurich_synoptic.csv"))
    require_columns(synoptic, {"timestamp_utc", "station_abbr", "tre200h0",
                               "ure200h0", "fkl010h0", "rre150h0", "gre000h0"},
                    "MeteoSwiss synoptic", errors)
    if set(synoptic.station_abbr.dropna().unique()) != {"KLO", "REH", "SMA"}:
        errors.append("MeteoSwiss synoptic: unexpected station set")

    screen = pd.read_csv(os.path.join(
        ROOT, "data", "inputs", "citywide_grid_priority_screen.csv"))
    screen_required = {
        "recordid", "centroid_easting_2056", "centroid_northing_2056", "pers_n",
        "est65_qmargin", "est80_qmargin", "gp_mean_night_min_c",
        "gp_mean_night_min_c_sd", "gp_pct_tropical_nights",
        "gp_pct_tropical_nights_sd", "gp_hottest10_mean_night_min_c",
        "gp_hottest10_mean_night_min_c_sd", "ka_temp_night", "ka_coldair_flow",
        "canopy_cover_250m_direct", "bldg_footprint_frac_250m_direct",
        "routed_refuge_distance_m"
    }
    require_columns(screen, screen_required, "terminal screen layer", errors)
    if len(screen) != 4217:
        errors.append("terminal screen layer: expected 4217 heat-supported cells")
    complete_screen = screen_required - {"ka_temp_night", "ka_coldair_flow"}
    present = complete_screen & set(screen.columns)
    if screen[list(present)].isna().any().any():
        errors.append("terminal screen layer: missing values in required columns")
    if screen.ka_temp_night.notna().sum() != 4201:
        errors.append("terminal screen layer: FITNAH temperature coverage changed")
    if abs(screen.pers_n.sum() - 446_327) > 1e-6:
        errors.append("terminal screen layer: population total changed")

    grids = set()
    era5_dir = os.path.join(ROOT, "data_external", "era5_pinned")
    for name in ["centre", "nw", "ne", "sw", "se"]:
        with open(os.path.join(
                era5_dir, f"openmeteo_era5_zurich_{name}.json")) as handle:
            payload = json.load(handle)
        grids.add((payload["latitude"], payload["longitude"]))
        times = payload["hourly"]["time"]
        if (times[0], times[-1]) != ("2020-06-01T00:00", "2025-08-31T23:00"):
            errors.append(f"ERA5 {name}: unexpected time span")
        if any(x is None for x in payload["hourly"]["cloud_cover"]):
            errors.append(f"ERA5 {name}: missing cloud-cover values")
    if grids != {(47.5, 8.5), (47.25, 8.5)}:
        errors.append(f"ERA5: unexpected returned grid cells {sorted(grids)}")

    if errors:
        print("\nINPUT VALIDATION FAILED")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)
    print("\nINPUT VALIDATION PASSED")
    n_files = len(manifest["files"])
    print(f"  {n_files}/{n_files} files match; schemas, counts, coverage and "
          "grid mapping are valid")


if __name__ == "__main__":
    main()
