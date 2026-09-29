"""Revision analyses (September 2026): temporal part.

Adds, with one inference method throughout (summer-stratified 14-day moving
calendar blocks, calm-clear selection inside each replicate):

* dispersion in four sunset-relative windows (-1 to 0 h, 0 to 1 h, 1 to 2 h,
  3 to 5 h) and their association with city warmth;
* partial association of dusk dispersion with warmth after adjusting for
  daytime global radiation (and additionally for season and year);
* tercile profiles (sunset- and clock-aligned) with calendar-block bands;
* a natural-spline profile model fitted from -2 h to +9 h;
* the evening/minimum/total-cooling decomposition (Table 1) and the anchor
  sensitivity (Fig. S2) with calendar-block intervals;
* evening ventilation timing on hot evenings.

Outputs are written to ``outputs_revision``.
"""
from __future__ import annotations

import os
import sys

import duckdb
import numpy as np
import pandas as pd
from patsy import build_design_matrices, dmatrix
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v3_17_metrology_temporal as T  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "heat", "meteoblue_temperature.parquet")
OUT = os.path.join(ROOT, "outputs_revision")
os.makedirs(OUT, exist_ok=True)
SLOT_CACHE = os.path.join(OUT, "_slots_17_08.parquet")

SEED = 20260721
N_BOOT = int(os.environ.get("V3_25_N_BOOT", "3000"))
BLOCK = 14
DEV, EVAL, ALL = (2020, 2021, 2022, 2023), (2024, 2025), tuple(range(2020, 2026))
WINDOWS = {"pre_m1_0": (-1.0, -0.25), "dusk_0_1": (0.0, 1.0),
           "w_1_2": (1.0, 2.0), "w_3_5": (3.0, 5.0)}
MIN_ST = 60


# --------------------------------------------------------------------------
def load_slots() -> pd.DataFrame:
    if os.path.exists(SLOT_CACHE):
        return pd.read_parquet(SLOT_CACHE)
    sql = f"""
    WITH src AS (
      SELECT locationID,
             CAST(timezone('Europe/Zurich', CAST(timestamp AS TIMESTAMPTZ)) AS TIMESTAMP) AS loc_ts,
             TRY_CAST(value AS DOUBLE) AS t_c, TRY_CAST(qc_flag AS INTEGER) AS qc,
             TRY_CAST(rc_flag AS INTEGER) AS rc
      FROM read_parquet('{RAW}') WHERE parameter = 'T')
    SELECT locationID, loc_ts, t_c, rc,
           CASE WHEN EXTRACT(hour FROM loc_ts) >= 12 THEN CAST(loc_ts AS DATE)
                ELSE CAST(loc_ts AS DATE) - INTERVAL 1 DAY END AS night_date
    FROM src
    WHERE qc = 0 AND t_c IS NOT NULL AND t_c BETWEEN -30 AND 50
      AND (EXTRACT(hour FROM loc_ts) >= 17 OR EXTRACT(hour FROM loc_ts) < 8
           OR (EXTRACT(hour FROM loc_ts) = 8 AND EXTRACT(minute FROM loc_ts) = 0))
    """
    con = duckdb.connect()
    d = con.execute(sql).df()
    con.close()
    d["night_date"] = pd.to_datetime(d.night_date)
    d = d[d.night_date.dt.month.isin([6, 7, 8]) & d.night_date.dt.year.between(2020, 2025)]
    d["loc_ts"] = pd.to_datetime(d.loc_ts) - pd.Timedelta(minutes=7.5)  # midpoint
    d = d[["night_date", "locationID", "loc_ts", "t_c", "rc"]].reset_index(drop=True)
    d.to_parquet(SLOT_CACHE, index=False)
    return d


def nights_table() -> pd.DataFrame:
    n = T.load_nights().rename(columns={"city_mean_tmin_c": "warmth"})
    n["calm_clear"] = n.regime.eq("calm_clear")
    return n


def calendar(nights: pd.DataFrame, extra: pd.DataFrame | None = None) -> pd.DataFrame:
    cal = T.jja_calendar().merge(nights, on=["night_date", "year"], how="left")
    if extra is not None:
        cal = cal.merge(extra, on="night_date", how="left")
    cal["calm_clear"] = cal.calm_clear.fillna(False).astype(bool)
    cal["doy"] = cal.night_date.dt.dayofyear
    return cal.sort_values("night_date").reset_index(drop=True)


def block_indices(cal: pd.DataFrame, years, rng) -> np.ndarray:
    idx = []
    for _, g in cal[cal.year.isin(years)].groupby("year", sort=True):
        orig = g.index.to_numpy()
        n = len(orig)
        take = []
        while len(take) < n:
            s = int(rng.integers(0, n - BLOCK + 1))
            take.extend(orig[s:s + BLOCK])
        idx.extend(take[:n])
    return np.asarray(idx, int)


def rank_resid(v: np.ndarray, Z: np.ndarray | None) -> np.ndarray:
    r = stats.rankdata(v)
    if Z is None:
        return r
    b = np.linalg.lstsq(Z, r, rcond=None)[0]
    return r - np.einsum("ij,j->i", Z, b)


def covariates(d: pd.DataFrame, adjust: tuple[str, ...]) -> np.ndarray | None:
    if not adjust:
        return None
    cols = [np.ones(len(d))]
    for a in adjust:
        if a == "season":
            s = (d.doy.to_numpy(float) - 196.5) / 30.0
            cols += [s, s ** 2]
        elif a == "year":
            dm = pd.get_dummies(d.year.astype(str), drop_first=True, dtype=float)
            cols += [dm[c].to_numpy() for c in dm.columns]
        else:
            cols.append(stats.rankdata(d[a].to_numpy(float)))
    return np.column_stack(cols)


def prho(d: pd.DataFrame, x: str, y: str, adjust=()) -> float:
    d = d.dropna(subset=[x, y] + [a for a in adjust if a not in ("season", "year")])
    if len(d) < 8:
        return np.nan
    Z = covariates(d, adjust)
    ex, ey = rank_resid(d[x].to_numpy(float), Z), rank_resid(d[y].to_numpy(float), Z)
    if np.std(ex) == 0 or np.std(ey) == 0:
        return np.nan
    return float(np.corrcoef(ex, ey)[0, 1])


def cb_ci(cal, years, stat, seed, n_boot=N_BOOT, select_cc=True):
    rng = np.random.default_rng(seed)
    out = np.full(n_boot, np.nan)
    for b in range(n_boot):
        bs = cal.loc[block_indices(cal, years, rng)]
        if select_cc:
            bs = bs[bs.calm_clear]
        out[b] = stat(bs)
    good = out[np.isfinite(out)]
    lo, hi = np.percentile(good, [2.5, 97.5])
    return float(lo), float(hi)


# --------------------------------------------------------------------------
def main() -> None:
    nights = nights_table()
    ss = T.sunset_local(nights.night_date)
    print("loading slots", flush=True)
    sl = load_slots().merge(ss, on="night_date")
    sl = sl[sl.night_date.isin(nights.night_date)]
    sl["u"] = (sl.loc_ts - sl.sunset_local).dt.total_seconds() / 3600.0
    sl["hbin"] = (sl.u * 4).round() / 4
    print(f"  {len(sl):,} quarter-hour values", flush=True)

    # per night x sunset-relative bin cross-station SD
    per = (sl[(sl.hbin >= -2.0) & (sl.hbin <= 9.0)]
           .groupby(["night_date", "hbin"]).t_c.agg(sd="std", n="size").reset_index())
    per = per[per.n >= MIN_ST]
    per.to_csv(os.path.join(OUT, "rev_sd_by_night_bin.csv"), index=False)

    # window metrics (dusk window uses exact 0..1 membership like the primary metric)
    wrows = {}
    for name, (a, b) in WINDOWS.items():
        if name == "dusk_0_1":
            d = sl[sl.u.between(0.0, 1.0)]
            g = d.groupby(["night_date", "hbin"]).t_c.agg(sd="std", n="size").reset_index()
            g = g[g.n >= MIN_ST]
        else:
            g = per[(per.hbin >= a) & (per.hbin <= b)]
        m = g.groupby("night_date").sd.agg(["mean", "size"])
        m.loc[m["size"] < 3, "mean"] = np.nan
        wrows[name] = m["mean"]
    win = pd.DataFrame(wrows).reset_index()
    ref = pd.read_csv(os.path.join(ROOT, "outputs_robust", "v3_17_dusk_metric_by_night.csv"),
                      parse_dates=["night_date"])
    chk = win.merge(ref, on="night_date")
    prim = "dusk_sd__dynamic_all__published_corrected_values__interval_midpoint"
    print("  max |dusk - primary| =", float((chk.dusk_0_1 - chk[prim]).abs().max()))
    win.to_csv(os.path.join(OUT, "rev_window_metrics_by_night.csv"), index=False)

    cal = calendar(nights, win)
    rows = []
    k = 0
    for scope, years in [("development", DEV), ("evaluation", EVAL), ("all", ALL)]:
        sel = cal[cal.year.isin(years) & cal.calm_clear]
        for name in WINDOWS:
            st = (lambda d, name=name: prho(d, "warmth", name))
            lo, hi = cb_ci(cal, years, st, SEED + 100 + k); k += 1
            rows.append(dict(scope=scope, analysis="window", window=name, adjust="none",
                             rho=st(sel), ci_low=lo, ci_high=hi,
                             n=int(sel[["warmth", name]].dropna().shape[0])))
        for adj_name, adj in [("day_radiation", ("day_rad_wm2",)),
                              ("day_radiation+season+year", ("day_rad_wm2", "season", "year")),
                              ("season+year", ("season", "year"))]:
            st = (lambda d, adj=adj: prho(d, "warmth", "dusk_0_1", adj))
            lo, hi = cb_ci(cal, years, st, SEED + 100 + k); k += 1
            rows.append(dict(scope=scope, analysis="dusk_partial", window="dusk_0_1",
                             adjust=adj_name, rho=st(sel), ci_low=lo, ci_high=hi,
                             n=int(sel[["warmth", "dusk_0_1"]].dropna().shape[0])))
        st = lambda d: prho(d, "day_rad_wm2", "dusk_0_1")
        lo, hi = cb_ci(cal, years, st, SEED + 100 + k); k += 1
        rows.append(dict(scope=scope, analysis="dusk_vs_day_radiation", window="dusk_0_1",
                         adjust="none", rho=st(sel), ci_low=lo, ci_high=hi, n=len(sel)))
        st = lambda d: prho(d, "day_rad_wm2", "dusk_0_1", ("warmth",))
        lo, hi = cb_ci(cal, years, st, SEED + 100 + k); k += 1
        rows.append(dict(scope=scope, analysis="dusk_vs_day_radiation", window="dusk_0_1",
                         adjust="warmth", rho=st(sel), ci_low=lo, ci_high=hi, n=len(sel)))
        print(f"  {scope} done", flush=True)
    assoc = pd.DataFrame(rows)
    assoc.to_csv(os.path.join(OUT, "rev_dusk_window_associations.csv"), index=False)
    print(assoc.round(3).to_string(index=False), flush=True)

    # ---------------- tercile profiles with calendar-block bands ----------
    cc = nights[nights.calm_clear]
    q1, q2 = cc.warmth.quantile([1 / 3, 2 / 3])
    terc = cc.assign(tercile=np.select([cc.warmth >= q2, cc.warmth <= q1],
                                       ["warm", "cool"], "middle"))[["night_date", "tercile"]]
    base_cal = calendar(nights, terc)

    def tercile_profiles(perframe, offset_col, label, seed):
        pv = perframe.pivot(index="night_date", columns=offset_col, values="sd")
        pv = base_cal[["night_date"]].merge(pv, left_on="night_date", right_index=True, how="left")
        M = pv.drop(columns="night_date").to_numpy(float)
        offs = pv.columns[1:].to_numpy(float)
        tt = base_cal.tercile.to_numpy()
        res = []
        rng = np.random.default_rng(seed)
        boots = {t: [] for t in ["cool", "middle", "warm"]}
        for b in range(N_BOOT):
            idx = block_indices(base_cal, ALL, rng)
            for t in boots:
                sel = idx[tt[idx] == t]
                boots[t].append(np.nanmean(M[sel], axis=0))
        for t in boots:
            sel = np.where(tt == t)[0]
            est = np.nanmean(M[sel], axis=0)
            nn = np.sum(np.isfinite(M[sel]), axis=0)
            B = np.asarray(boots[t])
            lo, hi = np.nanpercentile(B, [2.5, 97.5], axis=0)
            res.append(pd.DataFrame(dict(align=label, tercile=t, offset_h=offs, sd_mean=est,
                                         ci_lo=lo, ci_hi=hi, n_nights=nn)))
        return pd.concat(res)

    prof_ss = tercile_profiles(per[per.hbin.between(-2, 9)], "hbin", "sunset", SEED + 7)
    prof_ss.to_csv(os.path.join(OUT, "rev_profile_sunset.csv"), index=False)
    # clock alignment 18:00-08:00
    base = sl.night_date + pd.Timedelta(hours=18)
    sl["off_clock"] = ((sl.loc_ts - base).dt.total_seconds() / 3600.0 * 4).round() / 4
    perc = (sl[sl.off_clock.between(0, 13.75)].groupby(["night_date", "off_clock"])
            .t_c.agg(sd="std", n="size").reset_index())
    perc = perc[perc.n >= MIN_ST]
    prof_clock = tercile_profiles(perc, "off_clock", "clock", SEED + 8)
    prof_clock.to_csv(os.path.join(OUT, "rev_profile_clock.csv"), index=False)
    print("  profiles done", flush=True)
    for t in ["warm", "middle", "cool"]:
        p = prof_ss[prof_ss.tercile == t].set_index("offset_h").sd_mean
        print(f"   {t}: -2h {p.get(-2.0):.2f} 0h {p.get(0.0):.2f} max {p.max():.2f} at {p.idxmax()} +7h {p.get(7.0):.2f} +9h {p.get(9.0):.2f}")

    # ---------------- natural spline model from -2 h to +9 h --------------
    curve = per[per.hbin.between(-2, 9)].merge(cc[["night_date", "warmth"]], on="night_date")
    mu, sd = curve.drop_duplicates("night_date").warmth.agg(["mean", "std"])
    basis = dmatrix("0 + cr(hbin, df=6)", {"hbin": curve.hbin}, return_type="dataframe")
    info = basis.design_info
    tg = np.arange(-2, 9 + 1e-9, 0.25)
    bg = np.asarray(build_design_matrices([info], {"hbin": tg})[0], float)
    wz_levels = {"cool (10th pct)": None, "mean warmth": 0.0, "warm (90th pct)": None}
    night_w = cc.set_index("night_date").warmth
    wz_levels["cool (10th pct)"] = (np.percentile(night_w, 10) - mu) / sd
    wz_levels["warm (90th pct)"] = (np.percentile(night_w, 90) - mu) / sd

    def fit_predict(cv: pd.DataFrame):
        B = np.asarray(build_design_matrices([info], {"hbin": cv.hbin})[0], float)
        wz = ((cv.warmth - mu) / sd).to_numpy(float)
        X = np.column_stack([B, B * wz[:, None]])
        coef = np.linalg.lstsq(X, cv.sd.to_numpy(float), rcond=None)[0]
        return {lab: np.einsum("ij,j->i", np.column_stack([bg, bg * w]), coef)
                for lab, w in wz_levels.items()}

    point = fit_predict(curve)
    groups = {nd: g for nd, g in curve.groupby("night_date")}
    rng = np.random.default_rng(SEED + 9)
    bs_store = {lab: [] for lab in point}
    for b in range(N_BOOT):
        idx = block_indices(base_cal, ALL, rng)
        nds = base_cal.night_date.to_numpy()[idx]
        parts = [groups[n] for n in nds if n in groups]
        pr = fit_predict(pd.concat(parts, ignore_index=True))
        for lab in pr:
            bs_store[lab].append(pr[lab])
    fit_rows = []
    for lab in point:
        lo, hi = np.percentile(np.asarray(bs_store[lab]), [2.5, 97.5], axis=0)
        fit_rows.append(pd.DataFrame(dict(warmth_level=lab, hss=tg, sd_fit=point[lab],
                                          ci_lo=lo, ci_hi=hi)))
    fit = pd.concat(fit_rows)
    fit.to_csv(os.path.join(OUT, "rev_functional_fit.csv"), index=False)
    for lab in point:
        f = fit[fit.warmth_level == lab].set_index("hss").sd_fit
        print(f"   spline {lab}: max {f.max():.2f} at {f.idxmax():+.2f} h; 0h {f.loc[0.0]:.2f}; +9h {f.loc[9.0]:.2f}; drop max->9 {f.max()-f.loc[9.0]:.2f}")
    print(f"   warmth percentiles (C): p10 {np.percentile(night_w,10):.2f} p90 {np.percentile(night_w,90):.2f}")

    # ---------------- decomposition (Table 1) ------------------------------
    sn = pd.read_parquet(os.path.join(ROOT, "outputs_v3", "v3_station_night.parquet"))

    def moments(x):
        p = x[["t_even_c", "night_min_c"]].dropna()
        return pd.Series({"sd_t_even": p.t_even_c.std(ddof=1),
                          "sd_tmin": p.night_min_c.std(ddof=1),
                          "sd_cool_total": (p.t_even_c - p.night_min_c).std(ddof=1)})
    mom = sn.groupby("night_date").apply(moments, include_groups=False).reset_index()
    mom["night_date"] = pd.to_datetime(mom.night_date)
    cal2 = calendar(nights.drop(columns=["sd_cool_total"], errors="ignore"),
                    mom.merge(terc, on="night_date", how="left"))
    dec = []
    for i, (col, lab) in enumerate([("sd_t_even", "Evening temperature (20:00-22:00)"),
                                    ("sd_tmin", "Night minimum"),
                                    ("sd_cool_total", "Total cooling")]):
        sel = cal2[cal2.calm_clear]
        st = (lambda d, col=col: prho(d, "warmth", col))
        lo, hi = cb_ci(cal2, ALL, st, SEED + 300 + i)
        dec.append(dict(quantity=lab, column=col, rho=st(sel), ci_low=lo, ci_high=hi,
                        n=int(sel[col].notna().sum()),
                        sd_cool_tercile=sel[sel.tercile == "cool"][col].mean(),
                        sd_warm_tercile=sel[sel.tercile == "warm"][col].mean()))
    sel = cal2[cal2.calm_clear]
    pc = prho(sel, "warmth", "sd_cool_total", ("sd_t_even",))
    lo, hi = cb_ci(cal2, ALL, lambda d: prho(d, "warmth", "sd_cool_total", ("sd_t_even",)), SEED + 310)
    dec.append(dict(quantity="Total cooling | evening SD (partial)", column="partial",
                    rho=pc, ci_low=lo, ci_high=hi, n=len(sel)))
    dec = pd.DataFrame(dec)
    dec.to_csv(os.path.join(OUT, "rev_decomposition.csv"), index=False)
    print(dec.round(3).to_string(index=False), flush=True)

    # ---------------- anchor sensitivity (Fig. S2) ---------------------------
    A = sl.merge(nights[["night_date", "calm_clear"]], on="night_date")
    A = A[A.calm_clear]
    A["hour"] = A.loc_ts.dt.hour + A.loc_ts.dt.minute / 60.0

    def cool_total(ev):
        evm = ev.groupby(["night_date", "locationID"]).t_c.mean()
        # night minimum over the published 20:00-08:00 window (interval end times)
        # as in v3_12: midpoint hour within 20:00-08:00
        nightpart = A[(A.hour >= 20) | (A.hour < 8)]
        mn = nightpart.groupby(["night_date", "locationID"]).t_c.min()
        return evm - mn

    def rate(a, b):
        ta = a.groupby(["night_date", "locationID"]).t_c.mean()
        tb = b.groupby(["night_date", "locationID"]).t_c.mean()
        return -(tb - ta) / 4.0

    near = lambda s, v: np.isclose(s, v, atol=0.13)
    variants = {
        "Total cooling, evening 20-22 (primary)": cool_total(A[(A.hour >= 20) & (A.hour < 22)]),
        "Total cooling, evening 19-21": cool_total(A[(A.hour >= 19) & (A.hour < 21)]),
        "Total cooling, evening 21-23": cool_total(A[(A.hour >= 21) & (A.hour < 23)]),
        "Total cooling, sunset to +2 h": cool_total(A[(A.u >= 0) & (A.u < 2)]),
        "Cooling rate, 22-02 (primary)": rate(A[near(A.hour, 22.0)], A[near(A.hour, 2.0)]),
        "Cooling rate, 21-01": rate(A[near(A.hour, 21.0)], A[near(A.hour, 1.0)]),
        "Cooling rate, 23-03": rate(A[near(A.hour, 23.0)], A[near(A.hour, 3.0)]),
        "Cooling rate, sunset +1 to +5 h": rate(A[near(A.u, 1.0)], A[near(A.u, 5.0)]),
    }
    arows = []
    for i, (name, q) in enumerate(variants.items()):
        q = q.rename("q").reset_index()
        pern = q.groupby("night_date").q.agg(sd="std", n="size").reset_index()
        pern = pern[pern.n >= MIN_ST][["night_date", "sd"]].rename(columns={"sd": "anchor_sd"})
        c3 = calendar(nights, pern)
        sel = c3[c3.calm_clear]
        st = lambda d: prho(d, "warmth", "anchor_sd")
        lo, hi = cb_ci(c3, ALL, st, SEED + 400 + i)
        arows.append(dict(variant=name, rho=st(sel), ci_low=lo, ci_high=hi,
                          n=int(sel.anchor_sd.notna().sum())))
    anch = pd.DataFrame(arows)
    anch.to_csv(os.path.join(OUT, "rev_anchor_sensitivity.csv"), index=False)
    print(anch.round(3).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
