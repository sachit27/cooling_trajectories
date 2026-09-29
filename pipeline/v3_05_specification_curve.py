"""
V3 STEP 5 -- Specification curve over the dispersion-warmth estimand.

The audit showed this estimand moves from ~0 to +0.42 depending on the adjustment
set. Rather than defend one specification, we estimate ALL of them and report the
distribution. This is the honest representation of what night-level observational
data can say.

Crossed factors (pre-declared in PRE_ANALYSIS_PLAN.md section 6):
  night window   : 20-08 / 21-07 / 22-06        (recomputed from raw 15-min data)
  spread metric  : SD / p90-p10 / IQR
  regime cut     : 33-67 / 25-75 / 20-80 tertile-equivalents
  sample         : all nights / calm-clear only
  adjustment set : 8 sets from unadjusted to full+precip
  years          : all / dev / holdout

Outputs:
  v3_specification_curve.csv    every specification, one row each
  v3_specification_summary.csv  distribution summaries
"""
from __future__ import annotations

import os
import itertools
import numpy as np
import pandas as pd
import duckdb

import v3_core as C

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = C.OUT
RAW = os.path.join(ROOT, "heat", "meteoblue_temperature.parquet")

ADJ_SETS = {
    "unadjusted": [],
    "clearness": ["clearness_wm2"],
    "day_radiation": ["day_rad_wm2"],
    "wind": ["night_wind_ms"],
    "clearness+wind": ["clearness_wm2", "night_wind_ms"],
    "dayrad+wind": ["day_rad_wm2", "night_wind_ms"],
    "full(dayrad+clear+wind)": ["day_rad_wm2", "clearness_wm2", "night_wind_ms"],
    "full+precip": ["day_rad_wm2", "clearness_wm2", "night_wind_ms", "night_precip_mm"],
}

WINDOWS = {"20-08": (20, 8), "21-07": (21, 7), "22-06": (22, 6)}
SPREADS = ["sd_tmin_c", "p90p10_tmin_c", "iqr_tmin_c"]
CUTS = {"33/67": (1 / 3, 2 / 3), "25/75": (0.25, 0.75), "20/80": (0.20, 0.80)}

# Coverage threshold: 50% of the quarter-hours in each window, matching the
# proportional rule used in v3_01_night_metrics (MIN_SAMPLES=24 for 48 slots).
COVERAGE_FRACTION = 0.5


def city_nights_for_window(start_h: int, end_h: int) -> pd.DataFrame:
    """Recompute the city-night dispersion layer for an alternative night window."""
    window_hours = (24 - start_h) + end_h
    min_samples = max(1, int(window_hours * 4 * COVERAGE_FRACTION))
    sql = f"""
    WITH src AS (
        SELECT locationID,
               timezone('Europe/Zurich', CAST(timestamp AS TIMESTAMPTZ)) AS loc_ts,
               TRY_CAST(value AS DOUBLE) AS t_c,
               TRY_CAST(qc_flag AS INTEGER) AS qc
        FROM read_parquet('{RAW}')
    ),
    clean AS (
        SELECT locationID, loc_ts, t_c FROM src
        WHERE qc = 0 AND t_c IS NOT NULL AND t_c BETWEEN -30 AND 50
    ),
    nightly AS (
        SELECT locationID, t_c,
               CASE WHEN EXTRACT(hour FROM loc_ts) >= {start_h}
                    THEN CAST(loc_ts AS DATE)
                    ELSE CAST(loc_ts AS DATE) - INTERVAL 1 DAY END AS night_date
        FROM clean
        WHERE EXTRACT(hour FROM loc_ts) >= {start_h} OR EXTRACT(hour FROM loc_ts) < {end_h}
    ),
    sn AS (
        SELECT locationID, CAST(night_date AS DATE) AS night_date,
               COUNT(*) AS n, MIN(t_c) AS night_min_c
        FROM nightly
        WHERE EXTRACT(month FROM night_date) IN (6,7,8)
        GROUP BY 1,2
        HAVING COUNT(*) >= {min_samples}
    )
    SELECT night_date,
           COUNT(*) AS n_stations,
           AVG(night_min_c) AS city_mean_tmin_c,
           STDDEV_SAMP(night_min_c) AS sd_tmin_c,
           QUANTILE_CONT(night_min_c, 0.9) - QUANTILE_CONT(night_min_c, 0.1) AS p90p10_tmin_c,
           QUANTILE_CONT(night_min_c, 0.75) - QUANTILE_CONT(night_min_c, 0.25) AS iqr_tmin_c
    FROM sn GROUP BY 1 HAVING COUNT(*) >= 70 ORDER BY 1
    """
    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='2000MB'")
    # Parallel floating-point aggregation changed the last 4--5 decimal places
    # across otherwise identical runs. A single DuckDB thread makes the
    # specification layer byte-reproducible.
    con.execute("PRAGMA threads=1")
    df = con.execute(sql).df()
    con.close()
    df["night_date"] = pd.to_datetime(df["night_date"])
    return df


def partial_r(df, x, y, covs):
    """Point estimate only -- permutation p-values are computed for headline specs."""
    return C.partial_spearman_point(df, x, y, covs, min_n=30)


def main() -> None:
    syn = pd.read_csv(os.path.join(OUT, "v3_night_synoptic.csv"), parse_dates=["night_date"])

    print("recomputing city-night layers for each night window ...")
    layers = {}
    for wname, (sh, eh) in WINDOWS.items():
        layers[wname] = city_nights_for_window(sh, eh)
        print(f"  {wname}: {len(layers[wname])} city-nights")

    rows = []
    for wname, cut_name, spread, adj_name, yrs in itertools.product(
            WINDOWS, CUTS, SPREADS, ADJ_SETS, ["all", "dev", "holdout"]):

        city = layers[wname]
        d = city.merge(syn, on="night_date", how="inner")
        d = d[d.night_date.dt.year.between(2020, 2025)].copy()
        d["year"] = d.night_date.dt.year

        # Estimate every alternative regime threshold on the development years
        # and apply it unchanged to all, development, and withheld rows.  This
        # keeps the descriptive holdout rows free of transductive thresholding.
        lo_q, hi_q = CUTS[cut_name]
        threshold_sample = d[d.year.isin(C.DEV_YEARS)]
        lo, hi = threshold_sample.calm_clear_index.quantile([lo_q, hi_q])
        d["regime"] = np.select(
            [d.calm_clear_index >= hi, d.calm_clear_index <= lo],
            ["calm_clear", "cloudy_windy"], default="mixed")

        if yrs == "dev":
            d = d[d.year.isin(C.DEV_YEARS)]
        elif yrs == "holdout":
            d = d[d.year.isin(C.HOLDOUT_YEARS)]

        covs = ADJ_SETS[adj_name]

        for samp in ["all_nights", "calm_clear_only"]:
            sub = d if samp == "all_nights" else d[d.regime == "calm_clear"]
            r, n = partial_r(sub, "city_mean_tmin_c", spread, covs)
            rows.append({
                "night_window": wname, "regime_cut": cut_name, "spread_metric": spread,
                "adjustment_set": adj_name, "years": yrs, "sample": samp,
                "rho": r, "n": n,
            })

    sc = pd.DataFrame(rows).dropna(subset=["rho"])
    sc.to_csv(os.path.join(OUT, "v3_specification_curve.csv"), index=False)
    print(f"\n{len(sc)} specifications estimated")

    print("\n=== DISTRIBUTION OF THE ESTIMATE ACROSS ALL SPECIFICATIONS ===")
    print(f"  median rho = {sc.rho.median():+.3f}")
    print(f"  IQR        = [{sc.rho.quantile(.25):+.3f}, {sc.rho.quantile(.75):+.3f}]")
    print(f"  full range = [{sc.rho.min():+.3f}, {sc.rho.max():+.3f}]")
    print(f"  share positive = {(sc.rho > 0).mean():.1%}")

    print("\n=== what drives the spread: median rho by factor level ===")
    summ = []
    for factor in ["adjustment_set", "spread_metric", "night_window", "regime_cut",
                   "sample", "years"]:
        g = sc.groupby(factor).rho.agg(["median", "min", "max", "size"]).round(3)
        g.index.name = "level"
        g = g.reset_index()
        g.insert(0, "factor", factor)
        summ.append(g)
        print(f"\n-- {factor}")
        print(g[["level", "median", "min", "max", "size"]].to_string(index=False))
    pd.concat(summ, ignore_index=True).to_csv(
        os.path.join(OUT, "v3_specification_summary.csv"), index=False)

    print(f"\nwrote -> {OUT}/v3_specification_curve.csv")


if __name__ == "__main__":
    main()
