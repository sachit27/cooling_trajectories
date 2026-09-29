"""Balanced within-night coverage sensitivity for the night-minimum index.

The primary station-night rule requires at least 24 of 48 observations between
20:00 and 08:00. This diagnostic additionally requires at least 8 of 16
scheduled observations in each four-hour third of the night. It tests whether
late-night missingness changes the city-warmth index or its association with
midpoint-aligned dusk dispersion.
"""
from __future__ import annotations

import os
import duckdb
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs_robust")
RAW = os.path.join(ROOT, "heat", "meteoblue_temperature.parquet")


def main() -> None:
    query = f"""
    WITH src AS (
      SELECT locationID,
             timezone('Europe/Zurich', CAST(timestamp AS TIMESTAMPTZ)) loc_ts,
             TRY_CAST(value AS DOUBLE) t_c,
             TRY_CAST(qc_flag AS INTEGER) qc
      FROM read_parquet('{RAW}')
    ), clean AS (
      SELECT * FROM src
      WHERE qc=0 AND t_c IS NOT NULL AND t_c BETWEEN -30 AND 50
    ), observations AS (
      SELECT locationID, t_c, EXTRACT(hour FROM loc_ts) hh,
             CASE WHEN EXTRACT(hour FROM loc_ts)>=20 THEN CAST(loc_ts AS DATE)
                  ELSE CAST(loc_ts AS DATE)-INTERVAL 1 DAY END night_date
      FROM clean
      WHERE EXTRACT(hour FROM loc_ts)>=20 OR EXTRACT(hour FROM loc_ts)<8
    ), station_nights AS (
      SELECT locationID, CAST(night_date AS DATE) night_date,
             COUNT(*) n_samples,
             SUM(CASE WHEN hh>=20 THEN 1 ELSE 0 END) n_20_24,
             SUM(CASE WHEN hh<4 THEN 1 ELSE 0 END) n_00_04,
             SUM(CASE WHEN hh>=4 AND hh<8 THEN 1 ELSE 0 END) n_04_08,
             MIN(t_c) night_min_c
      FROM observations
      WHERE EXTRACT(month FROM night_date) IN (6,7,8)
      GROUP BY locationID, night_date
      HAVING COUNT(*)>=24 AND n_20_24>=8 AND n_00_04>=8 AND n_04_08>=8
    )
    SELECT night_date, COUNT(*) n_stations,
           AVG(night_min_c) city_mean_tmin_balanced_c,
           STDDEV_SAMP(night_min_c) sd_tmin_balanced_c
    FROM station_nights
    GROUP BY night_date
    HAVING COUNT(*)>=70
    ORDER BY night_date
    """
    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='2500MB'")
    con.execute("PRAGMA threads=4")
    balanced = con.execute(query).df()
    balanced["night_date"] = pd.to_datetime(balanced.night_date)
    balanced = balanced[balanced.night_date.dt.year.between(2020, 2025)].copy()

    default = pd.read_csv(
        os.path.join(ROOT, "outputs_v3", "v3_night_city.csv"),
        parse_dates=["night_date"],
        usecols=["night_date", "city_mean_tmin_c"],
    )
    synoptic = pd.read_csv(
        os.path.join(ROOT, "outputs_v3", "v3_night_synoptic.csv"),
        parse_dates=["night_date"],
        usecols=["night_date", "regime"],
    )
    metric = pd.read_csv(
        os.path.join(OUT, "v3_17_dusk_metric_by_night.csv"),
        parse_dates=["night_date"],
    )
    dusk = "dusk_sd__dynamic_all__published_corrected_values__interval_midpoint"
    joined = (balanced.merge(default, on="night_date", validate="one_to_one")
              .merge(synoptic, on="night_date", validate="one_to_one")
              .merge(metric[["night_date", dusk]], on="night_date",
                     validate="one_to_one"))
    joined["year"] = joined.night_date.dt.year

    rows = []
    periods = {
        "development": (2020, 2021, 2022, 2023),
        "internal_temporal_evaluation": (2024, 2025),
        "all_summers": (2020, 2021, 2022, 2023, 2024, 2025),
    }
    for period, years in periods.items():
        d = joined[joined.year.isin(years) & (joined.regime == "calm_clear")]
        d = d.dropna(subset=[dusk])
        rows.append({
            "period": period,
            "n_nights": len(d),
            "rho_dusk_with_default_warmth": stats.spearmanr(
                d[dusk], d.city_mean_tmin_c).statistic,
            "rho_dusk_with_balanced_coverage_warmth": stats.spearmanr(
                d[dusk], d.city_mean_tmin_balanced_c).statistic,
            "rho_default_vs_balanced_warmth": stats.spearmanr(
                d.city_mean_tmin_c, d.city_mean_tmin_balanced_c).statistic,
            "balanced_rule": (
                ">=24 of 48 observations and >=8 of 16 in each of "
                "20:00-24:00, 00:00-04:00, and 04:00-08:00"
            ),
        })

    balanced.to_csv(
        os.path.join(OUT, "v3_tmin_balanced_coverage_city_nights.csv"), index=False
    )
    pd.DataFrame(rows).to_csv(
        os.path.join(OUT, "v3_tmin_balanced_coverage_summary.csv"), index=False
    )
    print(f"balanced-coverage city nights: {len(balanced)}")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
