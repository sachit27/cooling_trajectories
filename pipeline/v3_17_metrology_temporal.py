"""
V3 STEP 17 -- Measurement-system audit and calendar-time sensitivity analysis.

This module addresses two limitations of the original analysis without changing any
of its frozen outputs:

1. It audits the *published* quality-control and radiation-correction fields in the
   Zurich 15-minute archive, coverage of the changing station network, and the
   near-collocated Kollerwiese pair.
2. It re-evaluates the dusk-dispersion association with calendar-day rather than
   selected-event time as the resampling unit.  Calm-clear nights are selected only
   after complete JJA calendars have been assembled, so gaps between selected events
   are retained.  A continuous-weather, seasonality- and year-adjusted all-night HAC
   model provides a complementary analysis that does not condition on a thresholded
   regime.

The 2024--2025 split is called an ``internal temporal evaluation`` here.  It is not
described as confirmatory or as an external replication: the dusk estimand was
developed after the literal pre-analysis plan and the plan has no external timestamp.

Important measurement semantics
---------------------------------
The official metadata state that ``timestamp`` is the end of the preceding 15-minute
averaging interval.  The primary sunset alignment therefore uses timestamp minus
7.5 minutes (the interval midpoint); the published endpoint convention is retained as
a sensitivity analysis.  ``rc_flag == 1`` means that the distributed value has already
been radiation-corrected.  Excluding those records is a missing-data sensitivity, not
a recovery of the unavailable uncorrected readings.

Outputs (outputs_robust/):
  v3_17_qa_flag_summary.csv
  v3_17_radiation_correction_by_hour.csv
  v3_17_dusk_rc_by_night.csv
  v3_17_dusk_rc_summary.csv
  v3_17_station_coverage_audit.csv
  v3_17_station_year_offsets.csv
  v3_17_kollerwiese_paired_nights.csv
  v3_17_kollerwiese_summary.csv
  v3_17_dusk_metric_by_night.csv
  v3_17_dusk_associations.csv
  v3_17_trajectory_associations.csv
  v3_17_calendar_block_regression.csv
  v3_17_calendar_hac_models.csv
  v3_17_audit_manifest.json
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
from dataclasses import dataclass
import duckdb
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import statsmodels.api as sm
import statsmodels


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "heat", "meteoblue_temperature.parquet")
STATION_PANEL = os.path.join(ROOT, "outputs_v3", "v3_station_night.parquet")
CITY_NIGHTS = os.path.join(ROOT, "outputs_v3", "v3_night_city.csv")
SYNOPTIC = os.path.join(ROOT, "outputs_v3", "v3_night_synoptic.csv")
STATIONS = os.path.join(ROOT, "heat", "meteoblue_station_locations.csv")
QA_METADATA = os.path.join(ROOT, "data_external", "meteoblue_qa_metadata.csv")
OUT = os.path.join(ROOT, "outputs_robust")

YEARS = tuple(range(2020, 2026))
DEV_YEARS = (2020, 2021, 2022, 2023)
EVAL_YEARS = (2024, 2025)
LAT, LON = 47.3769, 8.5417
SEED = 20260721
BLOCK_DAYS = 14
N_BOOT = int(os.environ.get("V3_17_N_BOOT", "3000"))
MIN_DYNAMIC_STATIONS = 60
MIN_DUSK_BINS = 3
CORE_THRESHOLDS = (0.80, 0.90, 0.95)
CORE_BIN_COMPLETENESS = 0.75

DATASET_URL = (
    "https://data.stadt-zuerich.ch/dataset/"
    "ugz_stadtklima_zuerich_temperaturmessungen_messnetz_meteoblue"
)
QA_METADATA_URL = (
    DATASET_URL + "/download/"
    "ugz_stadtklima_zuerich_temperaturmessungen_messnetz_meteoblue_metadaten_qa.csv"
)

KOLLER_1 = "Zürich-Wiedikon-Kollerwiese 1"
KOLLER_2 = "Zürich-Wiedikon-Kollerwiese 2"


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_and_validate_qa_metadata() -> tuple[pd.DataFrame, dict[int, str], str]:
    """Read the retained official flag dictionary and fail on an altered schema."""
    meta = pd.read_csv(QA_METADATA, encoding="utf-8-sig")
    required = {"category", "parameter", "name", "bit", "bitwise",
                "check-name", "check-desc"}
    if not required.issubset(meta.columns):
        raise ValueError(f"QA metadata lacks columns: {sorted(required - set(meta.columns))}")
    meta = meta[meta.parameter.eq("T")].copy()
    qc = meta[meta.name.eq("qc_flag")].sort_values("bit")
    rc = meta[meta.name.eq("rc_flag")]
    if qc.bit.astype(int).tolist() != [1, 2, 3, 4, 5, 6]:
        raise ValueError("official QA metadata no longer defines qc_flag checks 1--6")
    if len(rc) != 1 or int(rc.iloc[0].bit) != 1:
        raise ValueError("official QA metadata no longer has the expected rc_flag=1 row")
    definitions = {0: "valid after the publisher's quality-control workflow"}
    definitions.update({
        int(r.bit): f"{r['check-name']}: {r['check-desc']}"
        for _, r in qc.iterrows()
    })
    rc_definition = f"{rc.iloc[0]['check-name']}: {rc.iloc[0]['check-desc']}"
    return meta, definitions, rc_definition


def jja_calendar(years=YEARS) -> pd.DataFrame:
    parts = []
    for year in years:
        dates = pd.date_range(f"{year}-06-01", f"{year}-08-31", freq="D")
        parts.append(pd.DataFrame({"night_date": dates, "year": year}))
    return pd.concat(parts, ignore_index=True)


def sunset_local(dates: pd.Series) -> pd.DataFrame:
    """NOAA apparent sunset approximation, returned as naive Europe/Zurich time."""
    idx = pd.DatetimeIndex(pd.to_datetime(dates).drop_duplicates().sort_values())
    doy = idx.dayofyear.to_numpy()
    gamma = 2 * np.pi / 365.0 * (doy - 1)
    eqtime = 229.18 * (
        0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma)
    )
    decl = (
        0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma)
    )
    lat = np.deg2rad(LAT)
    hour_angle = np.rad2deg(np.arccos(np.clip(
        np.cos(np.deg2rad(90.833)) / (np.cos(lat) * np.cos(decl))
        - np.tan(lat) * np.tan(decl), -1, 1
    )))
    sunset_min_utc = 720 - 4 * (LON - hour_angle) - eqtime
    utc = pd.DatetimeIndex([
        date + pd.Timedelta(minutes=float(minutes))
        for date, minutes in zip(idx, sunset_min_utc)
    ]).tz_localize("UTC")
    local = utc.tz_convert("Europe/Zurich").tz_localize(None)
    return pd.DataFrame({"night_date": idx, "sunset_local": local})


def load_nights() -> pd.DataFrame:
    city = pd.read_csv(CITY_NIGHTS, parse_dates=["night_date"])
    syn = pd.read_csv(SYNOPTIC, parse_dates=["night_date"])
    d = city.merge(syn, on="night_date", how="inner", suffixes=("", "_syn"))
    d = d[d.night_date.dt.year.isin(YEARS)].copy()
    d["year"] = d.night_date.dt.year
    d["doy"] = d.night_date.dt.dayofyear
    return d.sort_values("night_date").reset_index(drop=True)


def qa_and_hourly_audits(
    con: duckdb.DuckDBPyConnection,
    qc_definitions: dict[int, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One archive scan for QA counts, then one study-period scan for RC by hour."""
    local_expr = "timezone('Europe/Zurich', CAST(timestamp AS TIMESTAMPTZ))"
    qa_sql = f"""
    WITH src AS (
      SELECT TRY_CAST(qc_flag AS INTEGER) AS qc,
             TRY_CAST(rc_flag AS INTEGER) AS rc,
             TRY_CAST(value AS DOUBLE) AS value,
             {local_expr} AS local_ts
      FROM read_parquet('{RAW}')
      WHERE parameter = 'T'
    )
    SELECT qc,
           COUNT(*) AS n_full,
           COUNT(value) AS n_value_full,
           COUNT(*) FILTER (
             WHERE EXTRACT(year FROM local_ts) BETWEEN 2020 AND 2025
               AND EXTRACT(month FROM local_ts) IN (6,7,8)
           ) AS n_study,
           COUNT(value) FILTER (
             WHERE EXTRACT(year FROM local_ts) BETWEEN 2020 AND 2025
               AND EXTRACT(month FROM local_ts) IN (6,7,8)
           ) AS n_value_study,
           COUNT(*) FILTER (WHERE rc = 1) AS n_rc1_full,
           COUNT(*) FILTER (
             WHERE rc = 1
               AND EXTRACT(year FROM local_ts) BETWEEN 2020 AND 2025
               AND EXTRACT(month FROM local_ts) IN (6,7,8)
           ) AS n_rc1_study
    FROM src GROUP BY qc ORDER BY qc
    """
    wide = con.execute(qa_sql).df()
    rows: list[dict] = []
    for scope, n_col, v_col, rc_col in [
        ("full_published_archive", "n_full", "n_value_full", "n_rc1_full"),
        ("JJA_2020_2025_local_time", "n_study", "n_value_study", "n_rc1_study"),
    ]:
        total = int(wide[n_col].sum())
        for r in wide.itertuples(index=False):
            qc = int(r.qc) if pd.notna(r.qc) else -1
            n = int(getattr(r, n_col))
            n_value = int(getattr(r, v_col))
            rows.append({
                "scope": scope,
                "qc_flag": qc,
                "qc_definition": qc_definitions.get(qc, "missing or undocumented flag"),
                "n_records": n,
                "fraction_of_scope": n / total if total else np.nan,
                "n_nonmissing_values": n_value,
                "nonmissing_value_fraction": n_value / n if n else np.nan,
                "n_missing_values": n - n_value,
                "n_rc_flag_1": int(getattr(r, rc_col)),
            })
    qa = pd.DataFrame(rows)

    rc_hour_sql = f"""
    WITH src AS (
      SELECT locationID,
             {local_expr} AS local_ts,
             TRY_CAST(value AS DOUBLE) AS value,
             TRY_CAST(qc_flag AS INTEGER) AS qc,
             TRY_CAST(rc_flag AS INTEGER) AS rc
      FROM read_parquet('{RAW}')
      WHERE parameter = 'T'
    )
    SELECT CAST(EXTRACT(hour FROM local_ts) AS INTEGER) AS published_interval_end_hour,
           COUNT(*) AS n_records,
           COUNT(*) FILTER (WHERE rc = 1) AS n_rc_flag_1,
           COUNT(DISTINCT locationID) AS n_stations
    FROM src
    WHERE qc = 0 AND value IS NOT NULL
      AND EXTRACT(year FROM local_ts) BETWEEN 2020 AND 2025
      AND EXTRACT(month FROM local_ts) IN (6,7,8)
    GROUP BY published_interval_end_hour
    ORDER BY published_interval_end_hour
    """
    rc_hour = con.execute(rc_hour_sql).df()
    rc_hour["rc_flag_1_fraction"] = rc_hour.n_rc_flag_1 / rc_hour.n_records
    rc_hour["scope"] = "valid_nonmissing_JJA_2020_2025"
    rc_hour["time_semantics"] = "local hour of published interval-end timestamp"
    return qa, rc_hour


def load_evening_slots(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load only local 19:00--23:59 values; this encloses all JJA dusk bands."""
    sql = f"""
    WITH src AS (
      SELECT locationID,
             timezone('Europe/Zurich', CAST(timestamp AS TIMESTAMPTZ)) AS local_ts,
             TRY_CAST(value AS DOUBLE) AS t_c,
             TRY_CAST(qc_flag AS INTEGER) AS qc,
             TRY_CAST(rc_flag AS INTEGER) AS rc
      FROM read_parquet('{RAW}')
      WHERE parameter = 'T'
    )
    SELECT locationID, CAST(local_ts AS TIMESTAMP) AS local_ts, t_c, rc
    FROM src
    WHERE qc = 0 AND t_c IS NOT NULL AND t_c BETWEEN -30 AND 50
      AND EXTRACT(year FROM local_ts) BETWEEN 2020 AND 2025
      AND EXTRACT(month FROM local_ts) IN (6,7,8)
      AND EXTRACT(hour FROM local_ts) BETWEEN 19 AND 23
    ORDER BY local_ts, locationID
    """
    d = con.execute(sql).df()
    d["local_ts"] = pd.to_datetime(d.local_ts)
    d["night_date"] = d.local_ts.dt.normalize()
    return d


def station_coverage_and_offsets(nights: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sn = pd.read_parquet(STATION_PANEL)
    sn["night_date"] = pd.to_datetime(sn.night_date)
    sn = sn[sn.night_date.dt.year.isin(YEARS)].copy()
    eligible_dates = set(nights.night_date)
    sn = sn[sn.night_date.isin(eligible_dates)].copy()
    city_mean = sn.groupby("night_date").night_min_c.mean().rename("panel_city_mean")
    sn = sn.merge(city_mean, on="night_date", how="left")
    sn["within_night_anomaly_c"] = sn.night_min_c - sn.panel_city_mean
    sn["year"] = sn.night_date.dt.year

    n_observed_nights = nights.night_date.nunique()
    n_calendar_nights = len(jja_calendar())
    cov = (sn.groupby("locationID")
           .agg(n_station_nights=("night_date", "nunique"),
                first_night=("night_date", "min"),
                last_night=("night_date", "max"),
                mean_samples_per_night=("n_samples", "mean"),
                mean_tmin_c=("night_min_c", "mean"),
                mean_within_night_anomaly_c=("within_night_anomaly_c", "mean"))
           .reset_index())
    cov["fraction_of_observed_city_nights"] = cov.n_station_nights / n_observed_nights
    cov["fraction_of_all_JJA_calendar_days"] = cov.n_station_nights / n_calendar_nights
    counts = (sn.pivot_table(index="locationID", columns="year", values="night_date",
                             aggfunc="nunique", fill_value=0)
              .reindex(columns=YEARS, fill_value=0))
    counts.columns = [f"n_nights_{int(y)}" for y in counts.columns]
    cov = cov.merge(counts.reset_index(), on="locationID", how="left")
    for threshold in CORE_THRESHOLDS:
        cov[f"fixed_core_{int(threshold * 100)}"] = (
            cov.fraction_of_observed_city_nights >= threshold
        )
    cov = cov.sort_values(["n_station_nights", "locationID"], ascending=[False, True])

    sy = (sn.groupby(["locationID", "year"])
          .agg(n_nights=("night_date", "nunique"),
               mean_within_night_anomaly_c=("within_night_anomaly_c", "mean"),
               sd_within_night_anomaly_c=("within_night_anomaly_c", "std"))
          .reset_index().sort_values(["locationID", "year"]))
    sy["previous_year"] = sy.groupby("locationID").year.shift()
    sy["previous_n_nights"] = sy.groupby("locationID").n_nights.shift()
    sy["change_from_previous_available_year_c"] = (
        sy.mean_within_night_anomaly_c
        - sy.groupby("locationID").mean_within_night_anomaly_c.shift()
    )
    sy["years_are_consecutive"] = sy.year - sy.previous_year == 1
    sy["well_observed_pair"] = (
        sy.years_are_consecutive & (sy.n_nights >= 30) & (sy.previous_n_nights >= 30)
    )
    sy["absolute_change_ge_0_5_c"] = (
        sy.well_observed_pair &
        (sy.change_from_previous_available_year_c.abs() >= 0.5)
    )
    return sn, cov, sy


def kollerwiese_audit(sn: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = ["night_date", "night_min_c", "n_samples"]
    a = sn.loc[sn.locationID == KOLLER_1, cols].rename(columns={
        "night_min_c": "tmin_pair_1_c", "n_samples": "n_samples_pair_1"
    })
    b = sn.loc[sn.locationID == KOLLER_2, cols].rename(columns={
        "night_min_c": "tmin_pair_2_c", "n_samples": "n_samples_pair_2"
    })
    pair = a.merge(b, on="night_date", how="inner").sort_values("night_date")
    pair["difference_pair_1_minus_2_c"] = pair.tmin_pair_1_c - pair.tmin_pair_2_c
    pair["absolute_difference_c"] = pair.difference_pair_1_minus_2_c.abs()
    diff = pair.difference_pair_1_minus_2_c
    station_meta = pd.read_csv(STATIONS)
    m = station_meta.set_index("locationID")
    dx = float(m.loc[KOLLER_1, "EKoord"] - m.loc[KOLLER_2, "EKoord"])
    dy = float(m.loc[KOLLER_1, "NKoord"] - m.loc[KOLLER_2, "NKoord"])
    summary = pd.DataFrame([{
        "pair_1": KOLLER_1,
        "pair_2": KOLLER_2,
        "coordinate_separation_m": float(np.hypot(dx, dy)),
        "n_paired_nights": len(pair),
        "first_paired_night": pair.night_date.min(),
        "last_paired_night": pair.night_date.max(),
        "mean_difference_pair_1_minus_2_c": diff.mean(),
        "median_difference_c": diff.median(),
        "sd_difference_c": diff.std(ddof=1),
        "mae_between_pair_c": diff.abs().mean(),
        "rmse_between_pair_c": float(np.sqrt(np.mean(diff ** 2))),
        "q025_difference_c": diff.quantile(0.025),
        "q975_difference_c": diff.quantile(0.975),
        "min_difference_c": diff.min(),
        "max_difference_c": diff.max(),
        "fraction_abs_difference_le_0_3_c": (diff.abs() <= 0.3).mean(),
        "pearson_r": stats.pearsonr(pair.tmin_pair_1_c, pair.tmin_pair_2_c).statistic,
        "interpretation": (
            "near-collocated field repeatability diagnostic; not a traceable calibration "
            "or a network-wide estimate of measurement uncertainty"
        ),
    }])
    return pair, summary


def prepare_dusk_slots(slots: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    ss = sunset_local(calendar.night_date)
    d = slots.merge(ss, on="night_date", how="inner")
    d["interval_midpoint_local"] = d.local_ts - pd.Timedelta(minutes=7.5)
    d["hss_midpoint"] = (
        (d.interval_midpoint_local - d.sunset_local).dt.total_seconds() / 3600.0
    )
    d["hss_endpoint"] = ((d.local_ts - d.sunset_local).dt.total_seconds() / 3600.0)
    return d


def aggregate_dusk(
    prepared: pd.DataFrame,
    hss_column: str,
    station_ids: set[str] | None = None,
    rc_policy: str = "published_corrected_values",
    minimum_stations: int = MIN_DYNAMIC_STATIONS,
) -> pd.DataFrame:
    d = prepared[(prepared[hss_column] >= 0.0) & (prepared[hss_column] <= 1.0)].copy()
    if station_ids is not None:
        d = d[d.locationID.isin(station_ids)]
    if rc_policy == "exclude_rc_flag_1":
        d = d[d.rc == 0]
    elif rc_policy != "published_corrected_values":
        raise ValueError(rc_policy)
    # Assign the mean interval support to the nearest 15-minute sunset-relative bin.
    d["hbin"] = (d[hss_column] * 4.0).round() / 4.0
    # Defensive collapse in case a source revision contains duplicate station/timestamps.
    cell = (d.groupby(["night_date", "hbin", "locationID"], as_index=False)
            .agg(t_c=("t_c", "mean"), rc=("rc", "max")))
    per_bin = (cell.groupby(["night_date", "hbin"], as_index=False)
               .agg(sd_c=("t_c", "std"), n_stations=("locationID", "nunique"),
                    n_rc_flag_1=("rc", lambda x: int((x == 1).sum()))))
    per_bin = per_bin[per_bin.n_stations >= minimum_stations]
    out = (per_bin.groupby("night_date", as_index=False)
           .agg(dusk_sd_c=("sd_c", "mean"),
                n_dusk_bins=("hbin", "nunique"),
                mean_stations_per_bin=("n_stations", "mean"),
                min_stations_in_bin=("n_stations", "min"),
                n_station_bin_records=("n_stations", "sum"),
                n_rc_flag_1=("n_rc_flag_1", "sum")))
    out = out[out.n_dusk_bins >= MIN_DUSK_BINS].copy()
    out["rc_flag_1_fraction"] = out.n_rc_flag_1 / out.n_station_bin_records
    return out


def rank_partial_rho(d: pd.DataFrame, adjusted: bool) -> float:
    cols = ["warmth", "dusk_sd_c", "doy", "year"]
    r = d[cols].dropna().copy()
    if len(r) < 8:
        return np.nan
    rx = stats.rankdata(r.warmth)
    ry = stats.rankdata(r.dusk_sd_c)
    if not adjusted:
        return float(np.corrcoef(rx, ry)[0, 1])
    doy_c = (r.doy.to_numpy(float) - 196.5) / 30.0
    year_dummies = pd.get_dummies(r.year.astype(str), drop_first=True, dtype=float).to_numpy()
    z = np.column_stack([np.ones(len(r)), doy_c, doy_c ** 2, year_dummies])
    # Explicit contractions avoid a reproducible Accelerate/NumPy small-matrix
    # matmul fault on macOS when block resampling creates Fortran-contiguous arrays.
    bx = np.linalg.lstsq(z, rx, rcond=None)[0]
    by = np.linalg.lstsq(z, ry, rcond=None)[0]
    ex = rx - np.einsum("ij,j->i", z, bx)
    ey = ry - np.einsum("ij,j->i", z, by)
    if np.std(ex) == 0 or np.std(ey) == 0:
        return np.nan
    return float(np.corrcoef(ex, ey)[0, 1])


def calendar_block_bootstrap_ci(
    calendar_frame: pd.DataFrame,
    years: tuple[int, ...],
    adjusted: bool,
    n_boot: int = N_BOOT,
    block_days: int = BLOCK_DAYS,
    seed: int = SEED,
) -> tuple[float, float, int]:
    """Moving-block bootstrap of complete daily summer rows within each year.

    Blocks are sampled from the full 92-day JJA series, including missing data and
    non-calm-clear days.  Calm-clear selection is applied only inside the statistic.
    Blocks never cross a summer boundary and are not allowed to wrap August to June.
    """
    frame = calendar_frame[calendar_frame.year.isin(years)].copy()
    groups = [g.sort_values("night_date").reset_index(drop=True)
              for _, g in frame.groupby("year", sort=True)]
    rng = np.random.default_rng(seed)
    boot = np.full(n_boot, np.nan)
    for b in range(n_boot):
        pieces = []
        for g in groups:
            n = len(g)
            length = min(block_days, n)
            idx: list[int] = []
            while len(idx) < n:
                start = int(rng.integers(0, n - length + 1))
                idx.extend(range(start, start + length))
            pieces.append(g.iloc[idx[:n]])
        bs = pd.concat(pieces, ignore_index=True)
        selected = bs[bs.calm_clear.fillna(False)]
        boot[b] = rank_partial_rho(selected, adjusted=adjusted)
    good = boot[np.isfinite(boot)]
    if len(good) < max(20, n_boot // 2):
        return np.nan, np.nan, len(good)
    lo, hi = np.percentile(good, [2.5, 97.5])
    return float(lo), float(hi), len(good)


def trajectory_calendar_associations(nights: pd.DataFrame) -> pd.DataFrame:
    """Dependence-preserving intervals for complete fixed-window profiles.

    The trajectory stage writes scalar metrics only when all quarter-hour bins
    through the relevant endpoint are present. This function places those
    outcomes on complete JJA calendars before calm-clear selection and uses the
    same within-summer 14-day moving-block procedure as the primary dusk result.
    """
    path = os.path.join(ROOT, "outputs_v3", "v3_traj_pernight.csv")
    traj = pd.read_csv(path, parse_dates=["night_date"])
    required = {"night_date", "integ_sd", "decay", "integ7", "decay7"}
    if not required.issubset(traj.columns):
        raise ValueError(
            "v3_traj_pernight.csv must be regenerated by the corrected "
            "v3_14_trajectory_framework.py"
        )
    base = jja_calendar().merge(nights, on=["night_date", "year"], how="left")
    base = base.merge(traj[sorted(required)], on="night_date", how="left")
    base["calm_clear"] = base.regime.eq("calm_clear")
    rows = []
    metrics = {
        "integrated_0_9h": "integ_sd",
        "post_peak_slope_0_9h": "decay",
        "integrated_0_7h": "integ7",
        "post_peak_slope_0_7h": "decay7",
    }
    scopes = [("development", DEV_YEARS),
              ("internal_temporal_evaluation", EVAL_YEARS)]
    for metric_i, (metric, column) in enumerate(metrics.items()):
        frame = base.copy()
        frame["dusk_sd_c"] = frame[column]
        for scope_i, (scope, years) in enumerate(scopes):
            selected = frame[
                frame.year.isin(years) & frame.calm_clear &
                frame.warmth.notna() & frame.dusk_sd_c.notna()
            ]
            rho = rank_partial_rho(selected, adjusted=False)
            lo, hi, n_valid = calendar_block_bootstrap_ci(
                frame, years, adjusted=False, n_boot=N_BOOT,
                block_days=BLOCK_DAYS,
                seed=SEED + 1000 + metric_i * 17 + scope_i,
            )
            rows.append({
                "analysis_scope": scope,
                "years": ",".join(map(str, years)),
                "metric": metric,
                "rho": rho,
                "ci_low": lo,
                "ci_high": hi,
                "n_nights": len(selected),
                "n_boot_valid": n_valid,
                "block_length_calendar_days": BLOCK_DAYS,
                "profile_completeness_rule": (
                    "all quarter-hour bins from sunset through endpoint"
                ),
            })
    return pd.DataFrame(rows)


def all_night_calendar_block_regression(
    metric_table: pd.DataFrame, nights: pd.DataFrame
) -> pd.DataFrame:
    """All-night level model with complete-calendar block uncertainty."""
    primary = (
        "dusk_sd__dynamic_all__published_corrected_values__interval_midpoint"
    )
    frame = jja_calendar().merge(nights, on=["night_date", "year"], how="left")
    frame = frame.merge(metric_table[["night_date", primary]],
                        on="night_date", how="left")
    frame = frame.rename(columns={primary: "dusk_sd_c"})
    frame["season"] = (frame.doy - 196.5) / 30.0
    weather = ["day_rad_wm2", "clearness_wm2", "night_wind_ms", "night_rh_pct"]
    design = pd.DataFrame({
        "intercept": np.ones(len(frame)),
        "warmth_c": frame.warmth,
        "season": frame.season,
        "season_sq": frame.season ** 2,
    })
    for col in weather:
        design[f"z_{col}"] = (
            (frame[col] - frame[col].mean()) / frame[col].std(ddof=0)
        )
    design = pd.concat([
        design,
        pd.get_dummies(frame.year.astype(str), prefix="year",
                       drop_first=True, dtype=float),
    ], axis=1)
    X = design.to_numpy(float)
    y = frame.dusk_sd_c.to_numpy(float)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    beta = np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]

    year_indices = [g.index.to_numpy() for _, g in frame.groupby("year", sort=True)]
    rng = np.random.default_rng(SEED + 2000)
    boots = np.full((N_BOOT, X.shape[1]), np.nan)
    for b in range(N_BOOT):
        sampled = []
        for original in year_indices:
            n = len(original)
            idx = []
            while len(idx) < n:
                start = int(rng.integers(0, n - BLOCK_DAYS + 1))
                idx.extend(original[start:start + BLOCK_DAYS])
            sampled.extend(idx[:n])
        sampled = np.asarray(sampled, int)
        sampled = sampled[ok[sampled]]
        if len(sampled) >= X.shape[1] + 5:
            boots[b] = np.linalg.lstsq(X[sampled], y[sampled], rcond=None)[0]
    lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
    return pd.DataFrame({
        "term": design.columns.astype(str),
        "estimate": beta,
        "ci_low": lo,
        "ci_high": hi,
        "n_nights": int(ok.sum()),
        "n_boot": int(np.isfinite(boots).all(axis=1).sum()),
        "block_length_calendar_days": BLOCK_DAYS,
    })


@dataclass(frozen=True)
class MetricSpec:
    station_set: str
    rc_policy: str
    timestamp_convention: str
    metric_column: str


def association_rows(metric_table: pd.DataFrame, nights: pd.DataFrame) -> list[dict]:
    result: list[dict] = []
    calendar = jja_calendar().merge(nights, on=["night_date", "year"], how="left")
    calendar["calm_clear"] = calendar.regime.eq("calm_clear")
    calendar = calendar.merge(metric_table, on="night_date", how="left")

    metric_specs = []
    for col in metric_table.columns:
        if not col.startswith("dusk_sd__"):
            continue
        _, station_set, rc_policy, timestamp = col.split("__", 3)
        metric_specs.append(MetricSpec(station_set, rc_policy, timestamp, col))

    scopes = [
        ("development", DEV_YEARS, "developmental_description"),
        ("internal_temporal_evaluation", EVAL_YEARS, "internal_temporal_evaluation"),
        ("all_summers", YEARS, "full_record_sensitivity"),
    ]
    for spec_i, spec in enumerate(metric_specs):
        for scope, years, status in scopes:
            # To keep the audit tractable, run every scope for the primary dynamic
            # metric, but only the internal evaluation for secondary sensitivities.
            primary = (
                spec.station_set == "dynamic_all" and
                spec.rc_policy == "published_corrected_values" and
                spec.timestamp_convention == "interval_midpoint"
            )
            if not primary and scope != "internal_temporal_evaluation":
                continue
            f = calendar[calendar.year.isin(years)].copy()
            f["dusk_sd_c"] = f[spec.metric_column]
            selected = f[f.calm_clear & f.warmth.notna() & f.dusk_sd_c.notna()]
            for adjusted in (False, True):
                rho = rank_partial_rho(selected, adjusted=adjusted)
                lo, hi, n_valid = calendar_block_bootstrap_ci(
                    f, years, adjusted=adjusted,
                    seed=SEED + spec_i * 101 + (1 if adjusted else 0),
                )
                result.append({
                    "analysis_scope": scope,
                    "years": ",".join(map(str, years)),
                    "evidence_status": status,
                    "regime": "calm_clear",
                    "station_set": spec.station_set,
                    "radiation_correction_policy": spec.rc_policy,
                    "timestamp_convention": spec.timestamp_convention,
                    "estimand": ("day_of_year_and_year_adjusted_partial_spearman"
                                  if adjusted else "spearman"),
                    "rho": rho,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n_nights": len(selected),
                    "n_bootstrap_valid": n_valid,
                    "inference_method": (
                        "summer-stratified full-calendar moving-block bootstrap; "
                        f"{BLOCK_DAYS}-day non-wrapping blocks; calm-clear selection inside replicate"
                    ),
                    "p_value": np.nan,
                    "p_value_note": "bootstrap interval reported; no selected-event circular-shift p-value",
                })

    # Descriptive within-year estimates establish that the sign is not supplied by
    # composition across summers.  Each gets a full-calendar block interval.
    primary_col = "dusk_sd__dynamic_all__published_corrected_values__interval_midpoint"
    if primary_col in calendar:
        for i, year in enumerate(YEARS):
            f = calendar[calendar.year == year].copy()
            f["dusk_sd_c"] = f[primary_col]
            selected = f[f.calm_clear & f.warmth.notna() & f.dusk_sd_c.notna()]
            rho = rank_partial_rho(selected, adjusted=False)
            lo, hi, n_valid = calendar_block_bootstrap_ci(
                f, (year,), adjusted=False, seed=SEED + 500 + i
            )
            result.append({
                "analysis_scope": f"single_summer_{year}",
                "years": str(year),
                "evidence_status": "within_summer_descriptive_consistency",
                "regime": "calm_clear",
                "station_set": "dynamic_all",
                "radiation_correction_policy": "published_corrected_values",
                "timestamp_convention": "interval_midpoint",
                "estimand": "spearman",
                "rho": rho,
                "ci_low": lo,
                "ci_high": hi,
                "n_nights": len(selected),
                "n_bootstrap_valid": n_valid,
                "inference_method": (
                    "single-summer full-calendar moving-block bootstrap; "
                    f"{BLOCK_DAYS}-day non-wrapping blocks"
                ),
                "p_value": np.nan,
                "p_value_note": "descriptive consistency check; no multiplicity-adjusted test",
            })
    return result


def all_night_hac_models(metric_table: pd.DataFrame, nights: pd.DataFrame) -> pd.DataFrame:
    """All-night models on near-complete daily time, with explicit season/year terms."""
    primary_col = "dusk_sd__dynamic_all__published_corrected_values__interval_midpoint"
    d = nights.merge(metric_table[["night_date", primary_col]], on="night_date", how="left")
    d = d.rename(columns={primary_col: "dusk_sd_c", "city_mean_tmin_c": "warmth"})
    d = d.sort_values("night_date").copy()
    weather = ["day_rad_wm2", "clearness_wm2", "night_wind_ms", "night_rh_pct"]
    needed = ["dusk_sd_c", "warmth", "doy", "year"] + weather
    d = d.dropna(subset=needed).copy()
    d["doy_scaled"] = (d.doy - 196.5) / 30.0
    for col in weather:
        d[f"z_{col}"] = (d[col] - d[col].mean()) / d[col].std(ddof=0)
    year_dummies = pd.get_dummies(d.year.astype(str), prefix="year", drop_first=True,
                                  dtype=float)
    X = pd.DataFrame({
        "intercept": 1.0,
        "warmth_c": d.warmth,
        "doy_scaled": d.doy_scaled,
        "doy_scaled_sq": d.doy_scaled ** 2,
        **{f"z_{c}": d[f"z_{c}"] for c in weather},
    }, index=d.index)
    X = pd.concat([X, year_dummies.set_axis(d.index)], axis=1).astype(float)
    m = sm.OLS(d.dusk_sd_c.to_numpy(float), X.to_numpy(float)).fit(
        cov_type="HAC", cov_kwds={"maxlags": BLOCK_DAYS}
    )
    names = X.columns.tolist()
    ci = np.asarray(m.conf_int())
    rows = []
    for i, name in enumerate(names):
        rows.append({
            "model": "level_all_nights_weather_season_year_adjusted",
            "term": name,
            "estimate": m.params[i],
            "std_error_HAC": m.bse[i],
            "ci_low_HAC": ci[i, 0],
            "ci_high_HAC": ci[i, 1],
            "p_value_HAC": m.pvalues[i],
            "maxlags_calendar_days_approx": BLOCK_DAYS,
            "n_nights": int(m.nobs),
            "r_squared": m.rsquared,
            "evidence_status": "full_record_sensitivity",
        })

    # Rank analogue: coefficient is the standardized rank slope conditional on the
    # same weather, seasonal and year terms.  This supplies robust rank-based inference
    # without rotating only the irregularly selected calm-clear events.
    rank_cols = ["dusk_sd_c", "warmth"] + weather
    rd = d[rank_cols].rank(method="average")
    rd = (rd - rd.mean()) / rd.std(ddof=0)
    RX = X.copy()
    RX["warmth_c"] = rd.warmth
    for col in weather:
        RX[f"z_{col}"] = rd[col]
    rm = sm.OLS(rd.dusk_sd_c.to_numpy(float), RX.to_numpy(float)).fit(
        cov_type="HAC", cov_kwds={"maxlags": BLOCK_DAYS}
    )
    rci = np.asarray(rm.conf_int())
    for i, name in enumerate(names):
        rows.append({
            "model": "rank_all_nights_weather_season_year_adjusted",
            "term": ("rank_warmth" if name == "warmth_c" else name),
            "estimate": rm.params[i],
            "std_error_HAC": rm.bse[i],
            "ci_low_HAC": rci[i, 0],
            "ci_high_HAC": rci[i, 1],
            "p_value_HAC": rm.pvalues[i],
            "maxlags_calendar_days_approx": BLOCK_DAYS,
            "n_nights": int(rm.nobs),
            "r_squared": rm.rsquared,
            "evidence_status": "full_record_sensitivity",
        })
    return pd.DataFrame(rows)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    required = [RAW, STATION_PANEL, CITY_NIGHTS, SYNOPTIC, STATIONS, QA_METADATA]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"missing required inputs: {missing}")

    qa_metadata, qc_definitions, rc_definition = load_and_validate_qa_metadata()
    print("validated retained official QA metadata ...")
    print("loading analysis nights and station panel ...")
    nights = load_nights().rename(columns={"city_mean_tmin_c": "warmth"})
    calendar = jja_calendar()
    print(f"  {len(nights)} observed analysis nights / {len(calendar)} JJA calendar days")

    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='2500MB'")
    con.execute("PRAGMA threads=4")
    print("auditing published QC and radiation-correction flags ...")
    qa, rc_hour = qa_and_hourly_audits(con, qc_definitions)
    print("loading the evening subset for sunset-aligned checks ...")
    evening = load_evening_slots(con)
    con.close()

    sn, coverage, offsets = station_coverage_and_offsets(nights)
    pair, pair_summary = kollerwiese_audit(sn)
    prepared = prepare_dusk_slots(evening, calendar)

    # Dusk correction-use audit is based on the scientifically primary interval midpoint.
    dusk_records = prepared[(prepared.hss_midpoint >= 0) & (prepared.hss_midpoint <= 1)].copy()
    dusk_records = dusk_records.merge(
        nights[["night_date", "regime"]], on="night_date", how="left"
    )
    rc_by_night = (dusk_records.groupby("night_date", as_index=False)
                   .agg(year=("night_date", lambda x: int(x.iloc[0].year)),
                        regime=("regime", "first"),
                        n_valid_records=("t_c", "size"),
                        n_stations=("locationID", "nunique"),
                        n_rc_flag_1=("rc", lambda x: int((x == 1).sum()))))
    rc_by_night["rc_flag_1_fraction"] = rc_by_night.n_rc_flag_1 / rc_by_night.n_valid_records
    rc_by_night["dusk_definition"] = "apparent sunset to +1h; 15-minute interval midpoint"
    rc_summary_rows = []
    grouped_specs = [
        ("all_study_nights", [], [((), rc_by_night)]),
        ("by_year", ["year"], rc_by_night.groupby("year", sort=True)),
        ("by_regime", ["regime"], rc_by_night.groupby("regime", sort=True)),
    ]
    for scope, group_cols, grouped in grouped_specs:
        for key, g in grouped:
            key = key if isinstance(key, tuple) else (key,)
            row = {
                "scope": scope,
                "year": np.nan,
                "regime": "all",
                "n_nights": int(g.night_date.nunique()),
                "n_valid_records": int(g.n_valid_records.sum()),
                "n_rc_flag_1": int(g.n_rc_flag_1.sum()),
                "n_nights_with_any_rc_flag_1": int((g.n_rc_flag_1 > 0).sum()),
            }
            for col, value in zip(group_cols, key):
                row[col] = value
            row["rc_flag_1_fraction"] = (
                row["n_rc_flag_1"] / row["n_valid_records"]
                if row["n_valid_records"] else np.nan
            )
            rc_summary_rows.append(row)
    rc_summary = pd.DataFrame(rc_summary_rows)
    rc_summary["dusk_definition"] = "apparent sunset to +1h; 15-minute interval midpoint"

    # Core membership is defined upstream of the dusk outcome from station-night
    # availability over the 546 observed analysis nights.
    station_sets: dict[str, set[str] | None] = {"dynamic_all": None}
    for threshold in CORE_THRESHOLDS:
        name = f"fixed_core_{int(threshold * 100)}"
        station_sets[name] = set(coverage.loc[coverage[name], "locationID"])

    metric = calendar[["night_date"]].copy()
    metric_meta: list[dict] = []
    for set_name, members in station_sets.items():
        if members is None:
            minimum = MIN_DYNAMIC_STATIONS
        else:
            minimum = max(20, int(np.ceil(CORE_BIN_COMPLETENESS * len(members))))
        for rc_policy in ("published_corrected_values", "exclude_rc_flag_1"):
            # The RC exclusion is needed for the dynamic network only; repeating it
            # for every fixed core would conflate two missingness perturbations.
            if rc_policy == "exclude_rc_flag_1" and set_name != "dynamic_all":
                continue
            for timestamp, hss in (
                ("interval_midpoint", "hss_midpoint"),
                ("published_endpoint", "hss_endpoint"),
            ):
                # Endpoint sensitivity is needed only for the otherwise-primary model.
                if timestamp == "published_endpoint" and not (
                    set_name == "dynamic_all" and rc_policy == "published_corrected_values"
                ):
                    continue
                agg = aggregate_dusk(
                    prepared, hss, station_ids=members, rc_policy=rc_policy,
                    minimum_stations=minimum,
                )
                col = f"dusk_sd__{set_name}__{rc_policy}__{timestamp}"
                agg = agg.rename(columns={"dusk_sd_c": col})
                metric = metric.merge(agg[["night_date", col]], on="night_date", how="left")
                metric_meta.append({
                    "column": col,
                    "station_set": set_name,
                    "n_stations_in_fixed_set": None if members is None else len(members),
                    "minimum_stations_per_15min_bin": minimum,
                    "radiation_correction_policy": rc_policy,
                    "timestamp_convention": timestamp,
                    "n_nights_with_metric": int(agg[col].notna().sum()),
                })

    # Add key diagnostic columns so each night-level result is directly inspectable.
    metric = metric.merge(
        rc_by_night[["night_date", "n_stations", "n_rc_flag_1", "rc_flag_1_fraction"]]
        .rename(columns={"n_stations": "dusk_n_stations_raw"}),
        on="night_date", how="left",
    )

    associations = pd.DataFrame(association_rows(metric, nights))
    trajectory_associations = trajectory_calendar_associations(nights)
    calendar_regression = all_night_calendar_block_regression(metric, nights)
    hac = all_night_hac_models(metric, nights)

    paths = {
        "qa_flag_summary": os.path.join(OUT, "v3_17_qa_flag_summary.csv"),
        "radiation_correction_by_hour": os.path.join(OUT, "v3_17_radiation_correction_by_hour.csv"),
        "dusk_rc_by_night": os.path.join(OUT, "v3_17_dusk_rc_by_night.csv"),
        "dusk_rc_summary": os.path.join(OUT, "v3_17_dusk_rc_summary.csv"),
        "station_coverage": os.path.join(OUT, "v3_17_station_coverage_audit.csv"),
        "station_year_offsets": os.path.join(OUT, "v3_17_station_year_offsets.csv"),
        "kollerwiese_paired": os.path.join(OUT, "v3_17_kollerwiese_paired_nights.csv"),
        "kollerwiese_summary": os.path.join(OUT, "v3_17_kollerwiese_summary.csv"),
        "dusk_metric": os.path.join(OUT, "v3_17_dusk_metric_by_night.csv"),
        "dusk_associations": os.path.join(OUT, "v3_17_dusk_associations.csv"),
        "trajectory_associations": os.path.join(
            OUT, "v3_17_trajectory_associations.csv"
        ),
        "calendar_block_regression": os.path.join(
            OUT, "v3_17_calendar_block_regression.csv"
        ),
        "calendar_hac": os.path.join(OUT, "v3_17_calendar_hac_models.csv"),
    }
    for frame, key in [
        (qa, "qa_flag_summary"), (rc_hour, "radiation_correction_by_hour"),
        (rc_by_night, "dusk_rc_by_night"), (rc_summary, "dusk_rc_summary"),
        (coverage, "station_coverage"),
        (offsets, "station_year_offsets"), (pair, "kollerwiese_paired"),
        (pair_summary, "kollerwiese_summary"), (metric, "dusk_metric"),
        (associations, "dusk_associations"),
        (trajectory_associations, "trajectory_associations"),
        (calendar_regression, "calendar_block_regression"),
        (hac, "calendar_hac"),
    ]:
        frame.to_csv(paths[key], index=False)

    qa_full = qa[qa.scope == "full_published_archive"]
    qa_study = qa[qa.scope == "JJA_2020_2025_local_time"]
    rc_valid = rc_hour.n_records.sum()
    rc_corrected = rc_hour.n_rc_flag_1.sum()
    primary_eval = associations[
        (associations.analysis_scope == "internal_temporal_evaluation") &
        (associations.station_set == "dynamic_all") &
        (associations.radiation_correction_policy == "published_corrected_values") &
        (associations.timestamp_convention == "interval_midpoint")
    ]
    manifest = {
        "script": "pipeline/v3_17_metrology_temporal.py",
        "evidence_label": "internal temporal evaluation (not confirmatory replication)",
        "random_seed": SEED,
        "calendar_block_bootstrap": {
            "draws": N_BOOT,
            "block_length_days": BLOCK_DAYS,
            "unit": "full JJA calendar days within summers",
            "selection": "calm-clear indicator resampled jointly; selection occurs inside each replicate",
            "blocks_cross_summer_boundaries": False,
            "blocks_wrap_august_to_june": False,
        },
        "measurement_semantics": {
            "primary_timestamp": "midpoint of preceding 15-minute averaging interval",
            "radiation_correction_exclusion": (
                "missing-data sensitivity only; uncorrected source readings are unavailable"
            ),
            "qc_flag_6": qc_definitions[6],
            "rc_flag_1": rc_definition,
            "flag_dictionary_rows_validated": int(len(qa_metadata)),
        },
        "official_sources": {
            "dataset": DATASET_URL,
            "qa_metadata": QA_METADATA_URL,
        },
        "input_sha256": {os.path.relpath(p, ROOT): sha256(p) for p in required},
        "software": {
            "python": platform.python_version(), "duckdb": duckdb.__version__,
            "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "statsmodels": statsmodels.__version__,
        },
        "headline_diagnostics": {
            "n_observed_analysis_nights": int(len(nights)),
            "n_JJA_calendar_days": int(len(calendar)),
            "n_stations": int(coverage.locationID.nunique()),
            "qa_full_archive_records": int(qa_full.n_records.sum()),
            "qa_study_records": int(qa_study.n_records.sum()),
            "valid_JJA_records_in_hourly_rc_audit": int(rc_valid),
            "radiation_corrected_fraction_valid_JJA": float(rc_corrected / rc_valid),
            "dusk_radiation_corrected_fraction": float(
                rc_by_night.n_rc_flag_1.sum() / rc_by_night.n_valid_records.sum()
            ),
            "kollerwiese_paired_nights": int(pair_summary.n_paired_nights.iloc[0]),
            "primary_internal_evaluation": json.loads(
                primary_eval.to_json(orient="records")
            ),
            "well_observed_consecutive_station_year_changes_ge_0_5_c": int(
                offsets.absolute_change_ge_0_5_c.sum()
            ),
        },
        "metric_definitions": metric_meta,
        "outputs": {
            key: {"path": os.path.relpath(path, ROOT),
                  "columns": pd.read_csv(path, nrows=0).columns.tolist()}
            for key, path in paths.items()
        },
        "limitations": [
            "The distributed qc_flag>0 records are normally blank, so failed values cannot be restored and alternative pre-QC filters cannot be evaluated.",
            "rc_flag exclusion removes already-corrected observations; it does not compare corrected with uncorrected readings.",
            "The Kollerwiese pair is about 22 m apart and is not a traceable reference calibration.",
            "Sensor model, device-level calibration coefficients, and operator maintenance/replacement logs are not present in the retained inputs.",
            "The temporal evaluation is internal to the same city network and processing system.",
            "HAC lags index the nearly complete observed daily series; the full-calendar bootstrap explicitly retains the six missing calendar days.",
        ],
    }
    manifest_path = os.path.join(OUT, "v3_17_audit_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)

    print("\nkey results")
    print(primary_eval[["estimand", "rho", "ci_low", "ci_high", "n_nights"]]
          .round(4).to_string(index=False))
    warmth_hac = hac[hac.term.isin(["warmth_c", "rank_warmth"])]
    print(warmth_hac[["model", "term", "estimate", "ci_low_HAC", "ci_high_HAC",
                      "p_value_HAC", "n_nights"]].round(4).to_string(index=False))
    print(f"\nwrote {len(paths) + 1} audit outputs -> {OUT}")


if __name__ == "__main__":
    main()
