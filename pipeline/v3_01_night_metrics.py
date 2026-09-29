"""
V3 STEP 1 -- Night metrics and nocturnal cooling rates from raw 15-min data.

Rebuilds the station-night layer from the raw meteoblue series, adding the
cooling-rate quantities the mechanism analysis needs. Nothing here depends on
any previous v1/v2 output.

Uses DuckDB for out-of-core aggregation (19.8M rows, ~4 GB RAM available).

Definitions (fixed here, varied in the sensitivity step):
  night      : 20:00-08:00 LOCAL time, labelled by the EVENING date
  local time : Europe/Zurich (raw archive is UTC)
  summer     : JJA (June, July, August)
  coverage   : a station-night needs >= 24 of 48 quarter-hour samples
  QC         : qc_flag == 0 only

Cooling-rate quantities per station-night:
  t_even     : mean T over 20:00-21:59
  t_2200     : T at 22:00 exactly
  t_0200     : T at 02:00 exactly
  cool_rate  : -(t_0200 - t_2200) / 4h   -> degC/h, POSITIVE = cooling
  cool_total : t_even - night_min        -> total evening-to-minimum drop

Outputs (outputs_v3/):
  v3_station_night.parquet   station x night panel
  v3_night_city.csv          city aggregate per night
  v3_station_coverage.csv    per-station coverage diagnostics
"""
from __future__ import annotations

import os
import duckdb
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs_v3")
os.makedirs(OUT, exist_ok=True)

RAW = os.path.join(ROOT, "heat", "meteoblue_temperature.parquet")

MIN_SAMPLES = 24
MIN_STATIONS_NIGHT = 70
SUMMER_MONTHS = "(6, 7, 8)"

STATION_NIGHT_SQL = f"""
WITH src AS (
    SELECT
        locationID,
        -- archive is UTC; convert to local wall-clock for night labelling
        timezone('Europe/Zurich', CAST(timestamp AS TIMESTAMPTZ)) AS loc_ts,
        TRY_CAST(value AS DOUBLE) AS t_c,
        TRY_CAST(qc_flag AS INTEGER) AS qc
    FROM read_parquet('{RAW}')
),
clean AS (
    SELECT locationID, loc_ts, t_c
    FROM src
    WHERE qc = 0 AND t_c IS NOT NULL AND t_c BETWEEN -30 AND 50
),
nightly AS (
    SELECT
        locationID,
        t_c,
        EXTRACT(hour FROM loc_ts)   AS hh,
        EXTRACT(minute FROM loc_ts) AS mi,
        -- samples before 08:00 belong to the previous evening's night
        CASE WHEN EXTRACT(hour FROM loc_ts) >= 20
             THEN CAST(loc_ts AS DATE)
             ELSE CAST(loc_ts AS DATE) - INTERVAL 1 DAY END AS night_date
    FROM clean
    WHERE EXTRACT(hour FROM loc_ts) >= 20 OR EXTRACT(hour FROM loc_ts) < 8
)
SELECT
    locationID,
    CAST(night_date AS DATE)                                    AS night_date,
    COUNT(*)                                                    AS n_samples,
    MIN(t_c)                                                    AS night_min_c,
    AVG(t_c)                                                    AS night_mean_c,
    MAX(t_c)                                                    AS night_max_c,
    AVG(CASE WHEN hh IN (20, 21) THEN t_c END)                  AS t_even_c,
    AVG(CASE WHEN hh = 22 AND mi = 0 THEN t_c END)              AS t_2200_c,
    AVG(CASE WHEN hh = 2  AND mi = 0 THEN t_c END)              AS t_0200_c
FROM nightly
WHERE EXTRACT(month FROM night_date) IN {SUMMER_MONTHS}
GROUP BY locationID, night_date
HAVING COUNT(*) >= {MIN_SAMPLES}
ORDER BY night_date, locationID
"""


def main() -> None:
    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='2500MB'")
    con.execute("PRAGMA threads=4")

    print("aggregating station-nights in DuckDB (out-of-core) ...")
    sn = con.execute(STATION_NIGHT_SQL).df()
    print(f"  station-nights: {len(sn):,}   stations: {sn.locationID.nunique()}")

    sn["night_date"] = pd.to_datetime(sn["night_date"])
    sn["cool_rate_c_per_h"] = -(sn.t_0200_c - sn.t_2200_c) / 4.0
    sn["cool_total_c"] = sn.t_even_c - sn.night_min_c
    sn["year"] = sn.night_date.dt.year
    sn["month"] = sn.night_date.dt.month

    print(f"  cooling-rate coverage: {sn.cool_rate_c_per_h.notna().mean():.1%} of station-nights")

    # per-station coverage diagnostics
    cov = (sn.groupby("locationID")
             .agg(n_nights=("night_date", "size"),
                  first_night=("night_date", "min"),
                  last_night=("night_date", "max"),
                  n_cool_rate=("cool_rate_c_per_h", lambda s: s.notna().sum()),
                  mean_tmin=("night_min_c", "mean"))
             .reset_index().sort_values("n_nights", ascending=False))
    cov.to_csv(os.path.join(OUT, "v3_station_coverage.csv"), index=False)

    # ---- city-night dispersion layer ------------------------------------
    print("aggregating to city-nights ...")
    g = sn.groupby("night_date")

    def qd(col, hi, lo):
        return g[col].quantile(hi) - g[col].quantile(lo)

    city = pd.DataFrame({
        "n_stations": g["night_min_c"].size(),
        "city_mean_tmin_c": g["night_min_c"].mean(),
        "city_median_tmin_c": g["night_min_c"].median(),
        "sd_tmin_c": g["night_min_c"].std(ddof=1),
        "p90p10_tmin_c": qd("night_min_c", 0.9, 0.1),
        "iqr_tmin_c": qd("night_min_c", 0.75, 0.25),
        "range_tmin_c": g["night_min_c"].max() - g["night_min_c"].min(),
        "frac_ge18_c": g["night_min_c"].apply(lambda s: (s >= 18).mean()),
        "frac_ge20_c": g["night_min_c"].apply(lambda s: (s >= 20).mean()),
        "city_mean_cool_rate": g["cool_rate_c_per_h"].mean(),
        "sd_cool_rate": g["cool_rate_c_per_h"].std(ddof=1),
        "p90p10_cool_rate": qd("cool_rate_c_per_h", 0.9, 0.1),
        "city_mean_cool_total": g["cool_total_c"].mean(),
        "sd_cool_total": g["cool_total_c"].std(ddof=1),
        "n_cool_rate": g["cool_rate_c_per_h"].apply(lambda s: s.notna().sum()),
    }).reset_index()

    city["year"] = city.night_date.dt.year
    city["month"] = city.night_date.dt.month

    city_ok = city[city.n_stations >= MIN_STATIONS_NIGHT].copy()
    print(f"  city-nights: {len(city)} total, {len(city_ok)} with >={MIN_STATIONS_NIGHT} stations")
    print(f"  span: {city_ok.night_date.min().date()} -> {city_ok.night_date.max().date()}")
    print("\n  nights per year:")
    print(city_ok.year.value_counts().sort_index().to_string())

    sn.to_parquet(os.path.join(OUT, "v3_station_night.parquet"), index=False)
    city_ok.to_csv(os.path.join(OUT, "v3_night_city.csv"), index=False)
    print(f"\nwrote -> {OUT}")


if __name__ == "__main__":
    main()
