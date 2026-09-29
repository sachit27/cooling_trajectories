"""
V3 STEP 14 -- The cooling trajectory as the scientific object.

Promotes the sunset-aligned dispersion CURVE from supporting evidence to the
central object, per senior-review guidance. Three deliverables:

(1) A COMPACT TRAJECTORY FRAMEWORK -- three per-night scalars that summarise the
    shape of the intra-urban dispersion curve on the sunset-aligned axis:
      dusk_sd    : cross-station SD of temperature at sunset..sunset+1h
      integ_sd   : integrated cross-station SD, sunset -> sunset+9h  (degC-hours)
      decay      : linear slope of cross-station SD from its peak to +9h
                   (degC per hour; negative = the contrast erodes overnight)
    Each is regressed on continuous city warmth (calm-clear nights), development
    then evaluated once on the withheld 2024-2025 window.

(2) A FUNCTIONAL MODEL of the dispersion curve on the sunset axis:
      SD(t) ~ f(hours_since_sunset) x continuous city warmth.
    We fit a full-rank natural-cubic-spline-in-time interaction model to the
    night-level cross-station SD curves and report E[SD(t)] as a function of
    warmth. Complete-calendar, within-summer moving-block resampling provides
    descriptive curve intervals while preserving multi-day dependence.

(3) CONTINUOUS SYNOPTIC PHYSICS. Regimes are retained for communication only;
    the models here use continuous daytime radiation (storage input), nocturnal
    longwave-residual clearness (radiative cooling potential) and wind
    (ventilation). This pre-empts the arbitrary-threshold criticism.

Evidential status: dusk_sd and integ_sd were developed after the literal frozen
plan.  Their 2024--2025 results are therefore post-development internal temporal
evaluations, not preregistered confirmation or independent replication.  The
decay x warmth interaction is exploratory.  The circular-shift p-values written
here are retained as a sensitivity analysis; the calendar-preserving analyses
in v3_17 are used for primary temporal uncertainty statements.

Outputs (outputs_v3/):
  v3_traj_pernight.csv              per-night trajectory scalars + drivers
  v3_traj_confirmatory.csv          legacy filename: internal temporal evaluation
  v3_traj_continuous_drivers.csv    scalars ~ continuous synoptic drivers (HAC/partial)
  v3_traj_functional_fit.csv        fitted dispersion curve E[SD(t)] by warmth level
  v3_traj_functional_model.csv      model coefficients + diagnostics
"""
from __future__ import annotations

import os
import duckdb
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm

import v3_core as C
import v3_17_metrology_temporal as temporal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = C.OUT
RAW = os.path.join(ROOT, "heat", "meteoblue_temperature.parquet")
LAT, LON = 47.3769, 8.5417
SEED = 20260721
N_BOOT = 3000

# sunset-aligned analysis window (hours relative to local sunset)
SS_LO, SS_HI = 0.0, 9.0          # sunset to sunset+9h
DUSK_LO, DUSK_HI = 0.0, 1.0      # dusk band for the scalar


# --------------------------------------------------------------------------
def sunset_local(nights: pd.Series) -> pd.DataFrame:
    """NOAA sunset per evening date, returned as naive local wall-clock."""
    idx = pd.DatetimeIndex(pd.to_datetime(nights.unique()))
    doy = idx.dayofyear.values
    gamma = 2 * np.pi / 365 * (doy - 1)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
                       - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma))
    decl = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
            - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
            - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma))
    lat = np.deg2rad(LAT)
    ha = np.rad2deg(np.arccos(np.clip(
        np.cos(np.deg2rad(90.833)) / (np.cos(lat) * np.cos(decl))
        - np.tan(lat) * np.tan(decl), -1, 1)))
    sunset_min_utc = 720 - 4 * (LON - ha) - eqtime
    ss_utc = [d + pd.Timedelta(minutes=m) for d, m in zip(idx, sunset_min_utc)]
    ss = pd.DataFrame({"night_date": idx})
    ss["sunset_local"] = (pd.Series(ss_utc).dt.tz_localize("UTC")
                          .dt.tz_convert("Europe/Zurich").dt.tz_localize(None))
    return ss


SLOT_SQL = f"""
WITH src AS (
    SELECT locationID,
           timezone('Europe/Zurich', CAST(timestamp AS TIMESTAMPTZ)) AS loc_ts_tz,
           TRY_CAST(value AS DOUBLE) AS t_c, TRY_CAST(qc_flag AS INTEGER) AS qc
    FROM read_parquet('{RAW}')
),
clean AS (
    SELECT locationID, CAST(loc_ts_tz AS TIMESTAMP) AS loc_ts, t_c
    FROM src WHERE qc = 0 AND t_c IS NOT NULL AND t_c BETWEEN -30 AND 50
),
nightly AS (
    SELECT locationID, loc_ts, t_c,
           CASE WHEN EXTRACT(hour FROM loc_ts) >= 18 THEN CAST(loc_ts AS DATE)
                ELSE CAST(loc_ts AS DATE) - INTERVAL 1 DAY END AS night_date
    FROM clean
    WHERE EXTRACT(hour FROM loc_ts) >= 18 OR EXTRACT(hour FROM loc_ts) < 8
)
SELECT night_date, loc_ts, locationID, t_c
FROM nightly
WHERE EXTRACT(month FROM night_date) IN (6,7,8)
  AND EXTRACT(year FROM night_date) BETWEEN 2020 AND 2025
"""


def load_slots():
    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='2300MB'"); con.execute("PRAGMA threads=4")
    df = con.execute(SLOT_SQL).df(); con.close()
    df["night_date"] = pd.to_datetime(df["night_date"])
    df["loc_ts"] = pd.to_datetime(df["loc_ts"])
    # Published timestamps mark the end of the preceding 15-minute averaging
    # interval. Associate each value with the interval midpoint.
    df["loc_ts"] = df["loc_ts"] - pd.Timedelta(minutes=7.5)
    return df


def year_block_boot(df, statfun, n_boot=N_BOOT, seed=SEED):
    """Generic year-block bootstrap CI for a statistic of a (x,y,year) frame."""
    rng = np.random.default_rng(seed)
    yrs = df.year.unique()
    groups = {y: df[df.year == y] for y in yrs}
    obs = statfun(df)
    boots = np.full(n_boot, np.nan)
    for i in range(n_boot):
        bs = pd.concat([groups[y] for y in rng.choice(yrs, len(yrs))])
        try:
            boots[i] = statfun(bs)
        except Exception:
            pass
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return obs, lo, hi


def spearman_stat(col):
    return lambda d: stats.spearmanr(d.warmth, d[col]).statistic


def main():
    nights = C.load_nights()
    ss = sunset_local(nights.night_date)

    print("loading 15-min slots ...")
    slots = load_slots()
    slots = slots.merge(ss, on="night_date")
    slots["hss"] = (slots.loc_ts - slots.sunset_local).dt.total_seconds() / 3600.0
    win = slots[(slots.hss >= -1.0) & (slots.hss <= SS_HI + 0.5)].copy()

    # cross-station SD per (night, quarter-hour-since-sunset)
    win["hbin"] = (win.hss * 4).round() / 4
    persd = (win.groupby(["night_date", "hbin"])
             .t_c.agg(sd="std", n="size").reset_index())
    persd = persd[persd.n >= 60]

    exact = slots[slots.hss.between(DUSK_LO, DUSK_HI)].copy()
    exact['hbin'] = (exact.hss * 4).round() / 4
    exact_sd = exact.groupby(['night_date', 'hbin']).t_c.agg(sd='std', n='size')
    exact_dusk = exact_sd[exact_sd.n >= 60].groupby('night_date').sd.agg(['mean', 'size'])
    exact_dusk.loc[exact_dusk['size'] < 3, 'mean'] = np.nan

    # ---- (1) per-night trajectory scalars ------------------------------
    print("computing trajectory scalars ...")
    rows = []
    for nd, g in persd.groupby("night_date"):
        g = g.sort_values("hbin")
        dusk = exact_dusk['mean'].get(nd, np.nan)
        seg = g[(g.hbin >= SS_LO) & (g.hbin <= SS_HI)]
        if len(seg) < 20:
            continue
        early = seg[seg.hbin <= 7.0]
        expected9 = np.arange(0.0, 9.0 + 1e-9, 0.25)
        expected7 = np.arange(0.0, 7.0 + 1e-9, 0.25)
        complete9 = np.array_equal(seg.hbin.to_numpy(float), expected9)
        complete7 = np.array_equal(early.hbin.to_numpy(float), expected7)
        integ = np.trapezoid(seg.sd.values, seg.hbin.values) if complete9 else np.nan
        integ7 = np.trapezoid(early.sd.values, early.hbin.values) if complete7 else np.nan
        # Decay: slope from the profile maximum to the fixed endpoint. Scalar
        # comparisons require every quarter-hour bin, so unequal endpoints are
        # never treated as equivalent integrals or slopes.
        pk = seg.loc[seg.sd.idxmax()]
        tail = seg[seg.hbin >= pk.hbin]
        decay = (np.polyfit(tail.hbin, tail.sd, 1)[0]
                 if complete9 and len(tail) >= 4 else np.nan)
        pk7 = early.loc[early.sd.idxmax()]
        tail7 = early[early.hbin >= pk7.hbin]
        decay7 = (np.polyfit(tail7.hbin, tail7.sd, 1)[0]
                  if complete7 and len(tail7) >= 4 else np.nan)
        rows.append({"night_date": nd, "dusk_sd": dusk, "integ_sd": integ,
                     "decay": decay, "integ7": integ7, "decay7": decay7,
                     "complete9": complete9, "complete7": complete7,
                     "n_profile_bins_0_9": len(seg),
                     "peak_hss": pk.hbin, "peak_sd": pk.sd})
    traj = pd.DataFrame(rows)

    # attach warmth, regime, continuous drivers
    keep = ["night_date", "year", "regime", "city_mean_tmin_c", "day_rad_wm2",
            "clearness_wm2", "night_wind_ms", "night_rh_pct", "night_precip_mm"]
    traj = traj.merge(nights[keep], on="night_date")
    traj = traj.rename(columns={"city_mean_tmin_c": "warmth"})
    traj.to_csv(os.path.join(OUT, "v3_traj_pernight.csv"), index=False)
    print(f"  {len(traj)} nights with trajectory scalars "
          f"({(traj.regime=='calm_clear').sum()} calm-clear)")
    print(f"  median peak at sunset+{traj[traj.regime=='calm_clear'].peak_hss.median():.2f} h")

    # ---- (1b) post-development temporal evaluation --------------------
    # NOTE ON INFERENCE. The withheld window spans only two summers, so a
    # year-block bootstrap is degenerate (~3 distinct resamples) and produces
    # artificially tight intervals. We therefore report, per scalar and sample:
    #   * an exact circular-shift p-value, independently stratified by summer,
    #     retained as a sensitivity test subject to cyclic-invariance; and
    #   * complete-calendar 14-day moving-block intervals. Selected-event
    #     circular shifts are retained as legacy sensitivity outputs only.
    print("\ninternal temporal evaluation (calm-clear), development vs withheld:")
    cc = traj[traj.regime == "calm_clear"]
    dev = cc[cc.year.isin(C.DEV_YEARS)]
    ho = cc[cc.year.isin(C.HOLDOUT_YEARS)]

    conf = []
    plan = {"dusk_sd": "post-development temporal evaluation",
            "integ_sd": "post-development temporal evaluation",
            "decay": "exploratory temporal evaluation"}
    for col in ["dusk_sd", "integ_sd", "decay"]:
        for lab, d in [("development", dev), ("withheld", ho)]:
            dd = d[["warmth", col, "year"]].dropna()
            test = C.block_permutation_spearman(
                dd.warmth, dd[col], groups=dd.year)
            o, pp, n = test["rho"], test["p_block_perm"], test["n"]
            cal = temporal.jja_calendar().merge(
                traj[['night_date', 'warmth', 'regime', col]],
                on='night_date', how='left')
            cal['doy'] = cal.night_date.dt.dayofyear
            cal['calm_clear'] = cal.regime.eq('calm_clear')
            cal['dusk_sd_c'] = cal[col]
            years = C.DEV_YEARS if lab == 'development' else C.HOLDOUT_YEARS
            lo, hi, _ = temporal.calendar_block_bootstrap_ci(
                cal, years, False, n_boot=N_BOOT, seed=SEED)
            conf.append({"scalar": col, "status": plan[col], "sample": lab,
                         "rho": o, "block_perm_p": pp,
                         "ci_lo": lo, "ci_hi": hi, "n": n,
                         "n_randomizations": test["n_randomizations"]})
            print(f"  {col:9s} [{lab:11s}] rho={o:+.3f} block-perm p={pp:.4f} "
                  f"CI[{lo:+.3f},{hi:+.3f}] n={n}  ({plan[col]})")
    conf = pd.DataFrame(conf)
    conf["p_bh_within_sample"] = np.nan
    for sample_name, idx in conf.groupby("sample").groups.items():
        conf.loc[idx, "p_bh_within_sample"] = C.bh_correct(
            conf.loc[idx, "block_perm_p"].tolist())
    conf.to_csv(os.path.join(OUT, "v3_traj_confirmatory.csv"), index=False)

    # ---- (3) continuous synoptic drivers of each scalar ----------------
    print("\ncontinuous-driver models (all nights, HAC; partial for warmth):")
    drv = []
    z = lambda s: (s - s.mean()) / s.std()
    T = traj.copy()
    for c in ["warmth", "day_rad_wm2", "clearness_wm2", "night_wind_ms",
              "night_rh_pct"]:
        T["z_" + c] = z(T[c])
    for col in ["dusk_sd", "integ_sd", "decay"]:
        sub = T.dropna(subset=[col]).sort_values("night_date")
        m = smf.ols(f"{col} ~ z_warmth + z_day_rad_wm2 + z_clearness_wm2 + "
                    f"z_night_wind_ms + z_night_rh_pct", data=sub).fit(
            cov_type="HAC", cov_kwds={"maxlags": 7})
        for term in ["z_warmth", "z_day_rad_wm2", "z_clearness_wm2",
                     "z_night_wind_ms", "z_night_rh_pct"]:
            drv.append({"scalar": col, "driver": term,
                        "beta": m.params[term], "se": m.bse[term],
                        "p": m.pvalues[term], "n": int(m.nobs), "r2": m.rsquared})
        print(f"  {col:9s}: warmth beta={m.params['z_warmth']:+.3f} "
              f"(p={m.pvalues['z_warmth']:.2g}); clearness "
              f"{m.params['z_clearness_wm2']:+.3f}; dayrad "
              f"{m.params['z_day_rad_wm2']:+.3f}; R2={m.rsquared:.2f}")
    pd.DataFrame(drv).to_csv(os.path.join(OUT, "v3_traj_continuous_drivers.csv"), index=False)

    # ---- (2) functional model of the dispersion curve ------------------
    # Model E[SD(t)] with a natural cubic spline in hours-since-sunset whose shape
    # is allowed to scale with continuous warmth. Fit on calm-clear night-level SD
    # curves. Then predict SD(t) at low/mean/high warmth with calendar-block bands.
    print("\nfunctional model of the dispersion curve (calm-clear) ...")
    curve = persd.merge(cc[["night_date", "warmth", "year"]], on="night_date")
    curve = curve[(curve.hbin >= SS_LO) & (curve.hbin <= SS_HI)].copy()
    curve["wz"] = z(curve.warmth)
    # natural cubic spline basis in time (df=5)
    from patsy import dmatrix, build_design_matrices
    # ``0 +`` is essential: cr() spans the constant, so adding Patsy's default
    # intercept makes the design rank deficient and destabilises predictions.
    basis = dmatrix("0 + cr(hbin, df=5)", {"hbin": curve.hbin},
                    return_type="dataframe")
    B = np.asarray(basis, float)
    y_curve = curve.sd.to_numpy(float)
    wz_curve = curve.wz.to_numpy(float)
    X_full = np.column_stack([B, B * wz_curve[:, None]])
    X_no_warmth = B
    # Same curve shape plus a time-constant warmth offset. Comparing this to the
    # full interaction isolates time-varying shape change from a level change.
    X_level_only = np.column_stack([B, wz_curve])
    rank_full = np.linalg.matrix_rank(X_full)
    rank_no_warmth = np.linalg.matrix_rank(X_no_warmth)
    rank_level_only = np.linalg.matrix_rank(X_level_only)
    if (rank_full, rank_no_warmth, rank_level_only) != (10, 5, 6):
        raise RuntimeError("functional spline design is not full rank")
    coef = np.linalg.lstsq(X_full, y_curve, rcond=None)[0]
    # Explicit contractions avoid spurious Accelerate/NumPy 2.2 matmul warnings
    # for these small Fortran-contiguous spline matrices on macOS.
    fitted = np.einsum("ij,j->i", X_full, coef)
    rss_full = np.sum((y_curve - fitted) ** 2)
    r2_full = 1 - rss_full / np.sum((y_curve - y_curve.mean()) ** 2)

    tg = np.arange(SS_LO, SS_HI + 1e-9, 0.25)
    wlevels = [("cool (10th pct)", np.percentile(curve.wz, 10)),
               ("mean warmth", 0.0),
               ("hot (90th pct)", np.percentile(curve.wz, 90))]

    def predict_curve(model, wz_val):
        bg = np.asarray(build_design_matrices(
            [basis.design_info], {"hbin": tg})[0], float)
        xp = np.column_stack([bg, bg * wz_val])
        return np.einsum("ij,j->i", xp, model)

    # point estimate
    point = {lab: predict_curve(coef, wz) for lab, wz in wlevels}

    # Complete-calendar moving-block prediction bands. Calendar days are sampled
    # within each summer before selecting calm-clear profiles, so irregular gaps
    # between selected events and multi-day weather dependence are retained.
    rng0 = np.random.default_rng(SEED)
    ndl0 = curve.night_date.unique()
    grp0 = {pd.Timestamp(n): np.flatnonzero(curve.night_date.to_numpy() == n) for n in ndl0}

    def calendar_curve_indices(rng, block_days=14):
        sampled_groups = []
        for year in (*C.DEV_YEARS, *C.HOLDOUT_YEARS):
            dates = pd.date_range(f"{year}-06-01", f"{year}-08-31", freq="D")
            n_days = len(dates)
            chosen = []
            while len(chosen) < n_days:
                start = int(rng.integers(0, n_days - block_days + 1))
                chosen.extend(dates[start:start + block_days])
            for date in chosen[:n_days]:
                idx = grp0.get(date)
                if idx is not None:
                    sampled_groups.append(idx)
        return np.concatenate(sampled_groups)

    boot_curves = {lab: [] for lab, _ in wlevels}
    for _ in range(1000):
        idx = calendar_curve_indices(rng0)
        try:
            coef_b = np.linalg.lstsq(X_full[idx], y_curve[idx], rcond=None)[0]
            for lab, wz in wlevels:
                boot_curves[lab].append(predict_curve(coef_b, wz))
        except Exception:
            pass
    fit_rows = []
    if any(len(values) != 1000 for values in boot_curves.values()):
        raise RuntimeError('A functional calendar-bootstrap fit failed; do not publish incomplete bands')
    for lab, _ in wlevels:
        arr = np.array(boot_curves[lab])
        lo = np.percentile(arr, 2.5, axis=0)
        hi = np.percentile(arr, 97.5, axis=0)
        for t, p, l, h in zip(tg, point[lab], lo, hi):
            fit_rows.append({"warmth_level": lab, "hss": t, "sd_fit": p,
                             "ci_lo": l, "ci_hi": h})
    fit = pd.DataFrame(fit_rows)
    fit.to_csv(os.path.join(OUT, "v3_traj_functional_fit.csv"), index=False)

    # summarise: peak time and peak-vs-dawn drop at each warmth level
    fsummary = []
    for wl, g in fit.groupby("warmth_level"):
        g = g.sort_values("hss")
        pk = g.loc[g.sd_fit.idxmax()]
        dawn = g.sd_fit.iloc[-1]
        fsummary.append({"warmth_level": wl, "peak_hss": pk.hss,
                         "peak_sd": pk.sd_fit, "dawn_sd": dawn,
                         "erosion": pk.sd_fit - dawn})
    fs = pd.DataFrame(fsummary)

    diag = pd.DataFrame([{
        "n_obs": len(y_curve), "n_nights": curve.night_date.nunique(),
        "rank_full": rank_full, "rank_no_warmth": rank_no_warmth,
        "rank_level_only": rank_level_only, "r2_descriptive": r2_full,
        "curve_interval_method": "14-day complete-calendar moving blocks within summer",
        "n_curve_boot": 1000, "seed": SEED}])
    fs.to_csv(os.path.join(OUT, "v3_traj_functional_model.csv"), index=False)
    print(fs.round(3).to_string(index=False))
    print(f"  descriptive functional R2={r2_full:.2f}; "
          f"nights={curve.night_date.nunique()}; calendar-block bands")
    diag.to_csv(os.path.join(OUT, "v3_traj_functional_diag.csv"), index=False)

    print(f"\nwrote -> {OUT}/v3_traj_*.csv")


if __name__ == "__main__":
    main()
