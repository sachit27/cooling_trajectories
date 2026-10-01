"""Revision analyses (September 2026): measurement, spatial and screen part.

* Fluntern co-location: hour-of-day differences and the time shift that best
  aligns the network record with MeteoSwiss SwissMetNet.
* City-mean cooling rate by sunset-relative window (context for lag effects).
* Station-trait models with modelled cold-air flow as a fourth predictor and
  counts of distinct spatial partitions.
* A Gaussian-process field of station dusk anomalies, its spatial CV and a
  screen sensitivity in which it replaces the night-minimum heat layer.
* A 65+ share population option, indicator correlations, FITNAH comparison.
* Evening ventilation timing on hot evenings.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v3_06_priority_screen as screen  # noqa: E402
import v3_15_station_traits as traits_mod  # noqa: E402
import v3_17_metrology_temporal as T  # noqa: E402
import v3_18_spatial_rebuild as sp  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs_revision")
RAW = os.path.join(ROOT, "heat", "meteoblue_temperature.parquet")
SLOT_CACHE = os.path.join(OUT, "_slots_17_08.parquet")
SEED = 20260721
summary: dict[str, object] = {}


def sig(labels, ids):
    return sp.partition_signature(np.asarray(labels), np.asarray(ids))


# --------------------------------------------------------------------------
def fluntern() -> None:
    d = pd.read_parquet(RAW, columns=["timestamp", "locationID", "value", "qc_flag"])
    d = d[(d.locationID == "Zürich-Fluntern") & (d.qc_flag == 0) & d.value.notna()]
    d["ts"] = pd.to_datetime(d.timestamp, utc=True)
    d = d[d.ts.dt.year.between(2020, 2025)]
    q = d.set_index("ts").value.sort_index()
    smn = pd.read_csv(os.path.join(ROOT, "outputs_robust",
                                   "official_meteoswiss_hourly_zurich_synoptic.csv"),
                      usecols=["station_abbr", "timestamp_utc", "tre200h0", "gre000h0"])
    smn = smn[smn.station_abbr == "SMA"].copy()
    smn["he"] = pd.to_datetime(smn.timestamp_utc, utc=True)
    # network values interpolated to a 5-minute grid, so that shifts finer than
    # the 15-minute reporting step can be evaluated
    grid = pd.date_range(q.index.min().floor("h"), q.index.max().ceil("h"), freq="5min")
    # published value = mean of the interval ending at its timestamp -> midpoint
    qm = pd.Series(q.values, index=q.index - pd.Timedelta(minutes=7.5))
    qi = qm.reindex(qm.index.union(grid)).interpolate(method="time", limit=3).reindex(grid)
    rows, hourly = [], []
    for shift in range(-60, 61, 5):
        s = qi.copy()
        s.index = s.index - pd.Timedelta(minutes=shift)  # positive = network lags
        # hourly mean over the interval (he-60 min, he]
        lab = s.index.ceil("h")
        hm = s.groupby(lab).agg(["mean", "count"])
        hm = hm[hm["count"] >= 11]["mean"]
        m = smn.merge(hm.rename("net"), left_on="he", right_index=True)
        m["loc"] = (m.he - pd.Timedelta(minutes=30)).dt.tz_convert("Europe/Zurich")
        m = m[m["loc"].dt.month.isin([6, 7, 8])]
        e = m.net - m.tre200h0
        h = m["loc"].dt.hour
        eve = h.between(17, 22)
        rows.append(dict(shift_min=shift, rmse_all=float(np.sqrt((e ** 2).mean())),
                         rmse_17_22=float(np.sqrt((e[eve] ** 2).mean())),
                         bias_17_22=float(e[eve].mean()),
                         rmse_night_22_05=float(np.sqrt((e[(h >= 22) | (h <= 5)] ** 2).mean())),
                         n=int(len(e))))
        if shift == 0:
            hourly = pd.DataFrame(dict(local_hour=h, diff=e, rad=m.gre000h0,
                                       date=m["loc"].dt.date))
    sh = pd.DataFrame(rows)
    sh.to_csv(os.path.join(OUT, "rev_fluntern_shift.csv"), index=False)
    hod = hourly.groupby("local_hour")["diff"].agg(["mean", "std", "count"]).reset_index()
    hod.to_csv(os.path.join(OUT, "rev_fluntern_hour_of_day.csv"), index=False)
    best_all = sh.loc[sh.rmse_all.idxmin()]
    best_eve = sh.loc[sh.rmse_17_22.idxmin()]
    # evening difference against daily global radiation
    dayrad = hourly[hourly.local_hour.between(8, 18)].groupby("date").rad.mean()
    eve = hourly[hourly.local_hour.between(19, 21)].groupby("date")["diff"].mean()
    j = pd.concat([dayrad.rename("rad"), eve.rename("eve")], axis=1).dropna()
    r = stats.spearmanr(j.rad, j.eve)
    summary["fluntern"] = dict(
        best_shift_all_min=int(best_all.shift_min), rmse_all_at_best=best_all.rmse_all,
        rmse_all_at_0=float(sh.loc[sh.shift_min == 0, "rmse_all"].iloc[0]),
        best_shift_evening_min=int(best_eve.shift_min),
        rmse_evening_at_best=best_eve.rmse_17_22,
        rmse_evening_at_0=float(sh.loc[sh.shift_min == 0, "rmse_17_22"].iloc[0]),
        bias_evening_at_0=float(sh.loc[sh.shift_min == 0, "bias_17_22"].iloc[0]),
        bias_evening_at_best=float(best_eve.bias_17_22),
        hour_mean_diff=hod.set_index("local_hour")["mean"].round(3).to_dict(),
        evening_diff_vs_dayrad_rho=float(r.statistic), evening_diff_vs_dayrad_p=float(r.pvalue),
        n_days=int(len(j)))
    print(json.dumps(summary["fluntern"], indent=1, default=float), flush=True)


# --------------------------------------------------------------------------
def city_cooling_rate() -> None:
    per = pd.read_parquet(SLOT_CACHE)
    nights = T.load_nights()
    ss = T.sunset_local(nights.night_date)
    per = per.merge(ss, on="night_date").merge(nights[["night_date", "regime"]], on="night_date")
    per = per[per.regime == "calm_clear"]
    per["hbin"] = (((per.loc_ts - per.sunset_local).dt.total_seconds() / 3600) * 4).round() / 4
    cm = per.groupby(["night_date", "hbin"]).t_c.mean().unstack()
    rate = -cm.diff(axis=1) * 4  # K per hour, positive = cooling
    out = {}
    for name, (a, b) in {"pre_m1_0": (-1, -0.25), "dusk_0_1": (0.25, 1.0),
                         "w_1_2": (1.25, 2.0), "w_3_5": (3.25, 5.0)}.items():
        cols = [c for c in rate.columns if a <= c <= b]
        out[name] = float(rate[cols].mean(axis=1).median())
    summary["city_mean_cooling_rate_K_per_h_median_calm_clear"] = out
    print("cooling rate by window", out, flush=True)


# --------------------------------------------------------------------------
def station_traits() -> pd.DataFrame:
    tr = pd.read_csv(os.path.join(ROOT, "outputs_v3", "v3_station_traits.csv"))
    context = traits_mod.nearest_cell_context(tr)
    tr["coldair_flow"] = tr.locationID.map(context.set_index("locationID").coldair_flow)
    rows, rep = [], []
    ids = tr.locationID.to_numpy()
    coords = tr[["EKoord", "NKoord"]].to_numpy(float)
    sigs = [sig(traits_mod.spatial_blocks(coords, SEED + r), ids) for r in range(50)]
    summary["trait_partitions_distinct_of_50"] = int(len(set(sigs)))
    for preds, tag in [(["elevation", "bldg_frac", "canopy"], "three"),
                       (["elevation", "bldg_frac", "canopy", "coldair_flow"], "four"),
                       (["elevation", "bldg_frac", "canopy"], "three_matched_cold")]:
        for target in ["dusk_anom", "integ_anom", "decay_tend"]:
            source = tr if tag == "three" else tr.dropna(subset=["coldair_flow"])
            sub = source.dropna(subset=[target] + preds)
            X, y = sub[preds].to_numpy(float), sub[target].to_numpy(float)
            cc = sub[["EKoord", "NKoord"]].to_numpy(float)
            from sklearn.linear_model import RidgeCV
            from sklearn.preprocessing import StandardScaler
            Xs = StandardScaler().fit_transform(X)
            ys = (y - y.mean()) / y.std()
            coef = RidgeCV(alphas=np.logspace(-2, 3, 30)).fit(Xs, ys).coef_
            r2s, drops = [], {p: [] for p in preds}
            for r in range(50):
                b = traits_mod.spatial_blocks(cc, SEED + r)
                full = traits_mod.prediction_metrics(y, traits_mod.blocked_predictions(X, y, b))["r2"]
                r2s.append(full)
                for j, p in enumerate(preds):
                    red = traits_mod.prediction_metrics(
                        y, traits_mod.blocked_predictions(np.delete(X, j, 1), y, b))["r2"]
                    drops[p].append(full - red)
            for p, c in zip(preds, coef):
                rows.append(dict(model=tag, target=target, predictor=p, n_stations=len(sub), std_coef=float(c),
                                 cv_r2_median=float(np.median(r2s)),
                                 cv_r2_q05=float(np.quantile(r2s, .05)),
                                 cv_r2_q95=float(np.quantile(r2s, .95)),
                                 delta_r2_median=float(np.median(drops[p])),
                                 delta_r2_q05=float(np.quantile(drops[p], .05)),
                                 delta_r2_q95=float(np.quantile(drops[p], .95))))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "rev_station_trait_models.csv"), index=False)
    print(res.round(3).to_string(index=False), flush=True)
    # predictor correlations
    summary["trait_predictor_spearman"] = tr[["elevation", "bldg_frac", "canopy", "coldair_flow"]] \
        .corr("spearman").round(2).to_dict()
    return tr


# --------------------------------------------------------------------------
def dusk_anomaly_targets() -> pd.DataFrame:
    """Per-station dusk anomaly (0 to +1 h after sunset) on calm-clear nights,
    with a between-night standard error for the GP noise term."""
    sl = pd.read_parquet(SLOT_CACHE)
    nights = T.load_nights()
    cc = nights[nights.regime == "calm_clear"][["night_date"]]
    ss = T.sunset_local(nights.night_date)
    sl = sl.merge(cc, on="night_date").merge(ss, on="night_date")
    sl["u"] = (sl.loc_ts - sl.sunset_local).dt.total_seconds() / 3600
    sl = sl[sl.u.between(0, 1)].copy()
    sl["hbin"] = (sl.u * 4).round() / 4
    sl["anom"] = sl.t_c - sl.groupby(["night_date", "hbin"]).t_c.transform("mean")
    pn = sl.groupby(["locationID", "night_date"]).anom.mean().reset_index()
    tg = pn.groupby("locationID").anom.agg(dusk_anom="mean", sd="std", n="size").reset_index()
    tg["se"] = tg.sd / np.sqrt(tg.n)
    loc = pd.read_csv(os.path.join(ROOT, "outputs_robust", "v3_spatial_rebuild_station_targets.csv"))
    tg = tg.merge(loc[["locationID", "EKoord", "NKoord", "adjusted_mean_night_min_c",
                       "between_summer_se_c"]], on="locationID")
    tg.to_csv(os.path.join(OUT, "rev_station_dusk_targets.csv"), index=False)
    old = pd.read_csv(os.path.join(ROOT, "outputs_v3", "v3_station_traits.csv"))
    chk = tg.merge(old[["locationID", "dusk_anom"]], on="locationID", suffixes=("", "_old"))
    summary["dusk_target_vs_trait_rho"] = float(stats.spearmanr(chk.dusk_anom, chk.dusk_anom_old).statistic)
    summary["dusk_vs_tmin_station_rho"] = float(stats.spearmanr(tg.dusk_anom, tg.adjusted_mean_night_min_c).statistic)
    return tg


def dusk_field(tg: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    origin = tg[["EKoord", "NKoord"]].to_numpy(float).mean(axis=0)
    xk = (tg[["EKoord", "NKoord"]].to_numpy(float) - origin) / 1000
    y = tg.dusk_anom.to_numpy(float)
    se = tg.se.to_numpy(float)
    m = sp.fit_gp(xk, y, se, SEED, restarts=8)
    ks = sp.summarize_kernel(m, y)
    gx = (grid[["centroid_easting_2056", "centroid_northing_2056"]].to_numpy(float) - origin) / 1000
    mu, sdv = m.predict(gx, return_std=True)
    summary["dusk_gp_kernel"] = ks.__dict__ if hasattr(ks, "__dict__") else dict(ks._asdict())
    # repeated spatial-block CV (same 6-block K-means design as the night-minimum field)
    ids = tg.locationID.to_numpy()
    r2, rmse, rho, sigs = [], [], [], []
    for r in range(20):
        b = traits_mod.spatial_blocks(tg[["EKoord", "NKoord"]].to_numpy(float), SEED + r)
        sigs.append(sig(b, ids))
        pred = np.full(len(y), np.nan)
        for k in np.unique(b):
            tr_, te = b != k, b == k
            mm = sp.fit_gp(xk[tr_], y[tr_], se[tr_], SEED, restarts=0)
            pred[te] = mm.predict(xk[te])
        r2.append(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))
        rmse.append(float(np.sqrt(np.mean((y - pred) ** 2))))
        rho.append(float(stats.spearmanr(y, pred).statistic))
    summary["dusk_gp_cv"] = dict(r2_median=float(np.median(r2)), r2_q025=float(np.quantile(r2, .025)),
                                 r2_q975=float(np.quantile(r2, .975)), rmse_median=float(np.median(rmse)),
                                 rho_median=float(np.median(rho)), distinct_partitions=len(set(sigs)))
    print("dusk GP", summary["dusk_gp_kernel"], summary["dusk_gp_cv"], flush=True)
    return pd.DataFrame({"recordid": grid.recordid, "gp_dusk_anom_c": mu, "gp_dusk_anom_sd_c": sdv})


# --------------------------------------------------------------------------
def screen_sensitivity(grid: pd.DataFrame) -> None:
    base_c = screen.components(grid, "rebuilt_gp", screen.POPULATION_LENSES["qmargin"])
    base_score = screen.score(base_c, screen.W_MAIN).to_numpy()
    base = screen.top_k_mask(base_score, 0.1)
    dist = pd.read_csv(os.path.join(ROOT, "outputs_robust", "v3_20_distinct_decision_frequency.csv"))
    g2 = grid.merge(dist[["recordid", "baseline_top10"]], on="recordid", how="left")
    assert (g2.baseline_top10.to_numpy(bool) == base).all()
    # nominal 54-spec frequency (reproduce the stable-81 set)
    masks = []
    for field in ["rebuilt_gp", "rebuilt_idw"]:
        for w in screen.WEIGHT_SCENARIOS.values():
            for lens in screen.POPULATION_LENSES.values():
                masks.append(screen.top_k_mask(screen.score(screen.components(grid, field, lens), w), .1))
    freq54 = np.mean(masks, axis=0)
    stable = base & (freq54 >= 0.9)
    summary["screen_n_weight_scenarios"] = len(screen.WEIGHT_SCENARIOS)
    summary["screen_stable81"] = int(stable.sum())
    # dusk field replacing the heat layer
    heat_dusk = screen.pct_rank(grid.gp_dusk_anom_c)
    rows = []
    dmasks = []
    for wname, w in screen.WEIGHT_SCENARIOS.items():
        for lname, lens in screen.POPULATION_LENSES.items():
            c = screen.components(grid, "rebuilt_gp", lens, heat_override=heat_dusk)
            m = screen.top_k_mask(screen.score(c, w), .1)
            dmasks.append(m)
            if wname == "baseline" and lname == "qmargin":
                dbase = m
    dfreq = np.mean(dmasks, axis=0)
    summary["dusk_field_screen"] = dict(
        baseline_overlap_with_tmin_baseline=int((dbase & base).sum()),
        stable81_in_dusk_baseline=int((dbase & stable).sum()),
        stable81_freq_ge_0_9_across_27_dusk_specs=int((stable & (dfreq >= 0.9)).sum()),
        stable81_freq_ge_0_5_across_27_dusk_specs=int((stable & (dfreq >= 0.5)).sum()),
        baseline422_freq_ge_0_9_across_27_dusk_specs=int((base & (dfreq >= 0.9)).sum()),
        rho_heat_layers=float(stats.spearmanr(grid.gp_dusk_anom_c, grid.gp_adjusted_mean_night_min_c).statistic),
        rho_scores=float(stats.spearmanr(
            screen.score(screen.components(grid, "rebuilt_gp", screen.POPULATION_LENSES["qmargin"], heat_override=heat_dusk), screen.W_MAIN),
            base_score).statistic))
    # 65+ share as the population layer
    share = (grid.est65_qmargin / grid.pers_n).clip(0, 1)
    c = base_c.copy()
    c["elderly"] = screen.pct_rank(share)
    sm = screen.top_k_mask(screen.score(c, screen.W_MAIN), .1)
    summary["share65_screen"] = dict(baseline_overlap=int((sm & base).sum()),
                                     stable81_retained=int((sm & stable).sum()))
    # indicator correlations
    X = pd.DataFrame({"heat_tmin": grid.gp_adjusted_mean_night_min_c, "heat_dusk": grid.gp_dusk_anom_c,
                      "older65_count": grid.est65_qmargin, "residents": grid.pers_n,
                      "older65_share": share, "canopy": grid.canopy_cover_250m_direct,
                      "building": grid.bldg_footprint_frac_250m_direct,
                      "gp_sd": grid.gp_adjusted_mean_night_min_c_sd})
    corr = X.corr("spearman")
    corr.to_csv(os.path.join(OUT, "rev_screen_indicator_correlations.csv"))
    summary["indicator_rho"] = {k: float(corr.loc[a, b]) for k, (a, b) in {
        "older65_count_vs_residents": ("older65_count", "residents"),
        "heat_vs_building": ("heat_tmin", "building"),
        "heat_vs_canopy": ("heat_tmin", "canopy"),
        "canopy_vs_building": ("canopy", "building"),
        "heat_vs_gp_sd": ("heat_tmin", "gp_sd"),
        "dusk_vs_tmin": ("heat_dusk", "heat_tmin"),
        "share_vs_count": ("older65_share", "older65_count")}.items()}
    out = grid[["recordid", "centroid_easting_2056", "centroid_northing_2056"]].copy()
    out["baseline_top10"] = base
    out["freq54"] = freq54
    out["stable90"] = stable
    out["dusk_baseline_top10"] = dbase
    out["dusk_freq27"] = dfreq
    out.to_csv(os.path.join(OUT, "rev_screen_cells.csv"), index=False)
    print(json.dumps({k: summary[k] for k in ["screen_stable81", "dusk_field_screen", "share65_screen", "indicator_rho"]}, indent=1), flush=True)


def fitnah(grid: pd.DataFrame, tg: pd.DataFrame) -> None:
    g = grid.dropna(subset=["ka_temp_night"])
    ce, cn = g.centroid_easting_2056.to_numpy(), g.centroid_northing_2056.to_numpy()
    vals, dists = [], []
    for _, s in tg.iterrows():
        d2 = (ce - s.EKoord) ** 2 + (cn - s.NKoord) ** 2
        j = int(np.argmin(d2))
        vals.append(g.ka_temp_night.iloc[j]); dists.append(np.sqrt(d2[j]))
    tg = tg.assign(fitnah=vals, fitnah_dist=dists)
    ok = tg.fitnah_dist <= 250
    summary["fitnah"] = dict(
        grid_rho=float(stats.spearmanr(g.gp_adjusted_mean_night_min_c, g.ka_temp_night).statistic),
        grid_n=int(len(g)),
        station_rho_tmin=float(stats.spearmanr(tg.adjusted_mean_night_min_c[ok], tg.fitnah[ok]).statistic),
        station_rho_dusk=float(stats.spearmanr(tg.dusk_anom[ok], tg.fitnah[ok]).statistic),
        station_n=int(ok.sum()))
    print("FITNAH", summary["fitnah"], flush=True)


# --------------------------------------------------------------------------
def km_median(g: pd.DataFrame) -> float:
    """Product-limit median; events precede censor removals at tied times."""
    risk, survival = len(g), 1.0
    for t, at in g.groupby("delay", sort=True):
        events = int((~at.censored).sum())
        survival *= 1.0 - events / risk
        if survival <= 0.5 + 1e-12:
            return float(t)
        risk -= len(at)
    return float("nan")


def ventilation(tr: pd.DataFrame) -> None:
    """Hours after sunset until outdoor air first falls below a threshold on hot evenings."""
    sl = pd.read_parquet(SLOT_CACHE)
    nights = T.load_nights()
    ss = T.sunset_local(nights.night_date)
    sl = sl.merge(ss, on="night_date")
    sl = sl[sl.night_date.isin(nights.night_date)]
    sl["u"] = (sl.loc_ts - sl.sunset_local).dt.total_seconds() / 3600
    sl["hbin"] = (sl.u * 4).round() / 4
    # keep the whole night: the slot cache ends at 08:00 local (interval end)
    sl = sl[sl.hbin >= -0.25]
    at_ss = sl[sl.hbin.between(-0.25, 0.25)].groupby("night_date").t_c.mean()
    hot = at_ss[at_ss >= 25.0].index
    summary["vent_hot_evenings_n"] = int(len(hot))
    summary["vent_hot_evenings_by_year"] = pd.Series(pd.DatetimeIndex(hot).year).value_counts().sort_index().to_dict()
    h = sl[sl.night_date.isin(hot) & (sl.hbin >= 0)]
    res = []
    for thr in [20.0, 22.0, 24.0]:
        below = h[h.t_c < thr].groupby(["night_date", "locationID"]).hbin.min()
        seen = h.groupby(["night_date", "locationID"]).hbin.agg(["max", "size"])
        seen = seen[seen["size"] >= 30]
        dd = seen.join(below.rename("first_below"), how="left")
        dd["censored"] = dd.first_below.isna()
        # Right censor at the last observed quarter-hour bin; no extra unobserved interval.
        dd["delay"] = dd.first_below.fillna(dd["max"])
        st = dd.groupby("locationID").agg(median_delay=("delay", "median"),
                                          share_censored=("censored", "mean"),
                                          n=("delay", "size")).reset_index()
        st = st[st.n >= 20].merge(tr[["locationID", "bldg_frac", "canopy", "elevation"]], on="locationID")
        # Use the same qualifying-station denominator for all summaries.
        dd = dd[dd.index.get_level_values("locationID").isin(st.locationID)].copy()
        km = dd.groupby(level="locationID").apply(km_median)
        st["km_median_delay"] = st.locationID.map(km)
        qb = pd.qcut(st.bldg_frac, 4, labels=False)
        dd["quartile"] = dd.index.get_level_values("locationID").map(dict(zip(st.locationID, qb)))
        dd.reset_index().to_csv(os.path.join(OUT, f"rev_cooling_records_{int(thr)}c.csv"), index=False)
        res.append(dict(threshold_c=thr, n_station_nights=int(len(dd)),
                        n_stations=int(len(st)),
                        km_undefined_stations=int(st.km_median_delay.isna().sum()),
                        km_low_bldg_quartile=float(st.km_median_delay[qb == 0].median()) if st.km_median_delay[qb == 0].notna().all() else np.nan,
                        km_high_bldg_quartile=float(st.km_median_delay[qb == 3].median()) if st.km_median_delay[qb == 3].notna().all() else np.nan,
                        **{f"censored_quartile_{q+1}": float(dd.loc[dd.quartile == q, "censored"].mean()) for q in range(4)},
                        share_censored=float(dd.censored.mean()),
                        city_median_delay_h=float(dd.delay.median()),
                        station_median_delay_p10=float(st.median_delay.quantile(.1)),
                        station_median_delay_p90=float(st.median_delay.quantile(.9)),
                        rho_bldg=float(stats.spearmanr(st.bldg_frac, st.median_delay).statistic),
                        p_bldg=float(stats.spearmanr(st.bldg_frac, st.median_delay).pvalue),
                        rho_canopy=float(stats.spearmanr(st.canopy, st.median_delay).statistic),
                        rho_elev=float(stats.spearmanr(st.elevation, st.median_delay).statistic),
                        delay_low_bldg_quartile=float(st.median_delay[qb == 0].median()),
                        delay_high_bldg_quartile=float(st.median_delay[qb == 3].median())))
        if thr == 22.0:
            st.to_csv(os.path.join(OUT, "rev_ventilation_station_22c.csv"), index=False)
    v = pd.DataFrame(res)
    v.to_csv(os.path.join(OUT, "rev_ventilation_summary.csv"), index=False)
    print(v.round(3).to_string(index=False), flush=True)


def main() -> None:
    fluntern()
    city_cooling_rate()
    tr = station_traits()
    tg = dusk_anomaly_targets()
    grid = pd.read_csv(os.path.join(ROOT, "outputs_robust", "citywide_grid_priority_screen.csv")).merge(
        pd.read_csv(os.path.join(ROOT, "outputs_robust", "v3_spatial_rebuild_grid.csv"))[
            ["recordid", "gp_adjusted_mean_night_min_c", "gp_adjusted_mean_night_min_c_sd",
             "idw2_adjusted_mean_night_min_c"]], on="recordid", validate="one_to_one")
    df = dusk_field(tg, grid)
    grid = grid.merge(df, on="recordid")
    grid[["recordid", "gp_dusk_anom_c", "gp_dusk_anom_sd_c"]].to_csv(
        os.path.join(OUT, "rev_dusk_field_grid.csv"), index=False)
    screen_sensitivity(grid)
    fitnah(grid, tg)
    ventilation(tr)
    cv = pd.read_csv(os.path.join(ROOT, "outputs_robust", "v3_spatial_rebuild_cv_metrics.csv"))
    summary["tmin_gp_cv_distinct_partitions"] = int(cv[cv.row_type == "repeat"].partition_signature.nunique())
    with open(os.path.join(OUT, "rev_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)


if __name__ == "__main__":
    main()
