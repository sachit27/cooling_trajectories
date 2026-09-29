"""
V3 STEP 8 -- Collect every number that will appear in the manuscript.

Single source of truth. Any figure quoted in RESULTS_V3.md or the manuscript must
appear in the JSON this writes. Run this before writing prose, and again after,
to check the prose against it.

Output: outputs_v3/v3_verified_numbers.json  (+ printed report)
"""
from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd
from scipy import stats

import v3_core as C
from v3_02_synoptic import SIGMA

OUT = C.OUT
V = {}


def rec(key, value, note=""):
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    V[key] = {"value": value, "note": note}
    return value


def main() -> None:
    rec("stefan_boltzmann_constant_mantissa", SIGMA / 1e-8,
        "mantissa in sigma = 5.670374419 x 10^-8 W m^-2 K^-4")

    # ---------------- data layer -------------------------------------
    sn = pd.read_parquet(os.path.join(OUT, "v3_station_night.parquet"))
    city = pd.read_csv(os.path.join(OUT, "v3_night_city.csv"), parse_dates=["night_date"])
    d = C.load_nights()
    dev, ho = C.sample(d, "dev"), C.sample(d, "holdout")

    # Distinguish station-night QC from the subsequent city-night eligibility
    # rule. Six JJA dates in 2020--2025 had valid station records but fewer than
    # 70 reporting stations and do not contribute to any city-night analysis.
    sn_window = sn[sn.night_date.dt.year.between(2020, 2025)].copy()
    analysis_dates = set(d.night_date)
    sn_analysis = sn_window[sn_window.night_date.isin(analysis_dates)].copy()
    rec("n_station_nights", int(len(sn_analysis)),
        "station-nights on the 546 city-night-eligible analysis dates")
    rec("n_station_nights_before_city_eligibility", int(len(sn_window)),
        "QC-clean JJA station-nights before the >=70-station city-night rule")
    rec("n_station_nights_excluded_city_eligibility",
        int(len(sn_window) - len(sn_analysis)))
    rec("n_dates_excluded_city_eligibility",
        int(sn_window.loc[~sn_window.night_date.isin(analysis_dates),
                          "night_date"].nunique()))
    rec("n_station_nights_all_years", int(len(sn)),
        "full panel incl. 2019 (not used in analysis)")
    rec("n_quarterhour_readings",
        int(sn_analysis.n_samples.sum()),
        "retained 15-min readings on city-night-eligible analysis dates")
    rec("n_quarterhour_readings_before_city_eligibility",
        int(sn_window.n_samples.sum()),
        "retained 15-min readings before the city-night eligibility rule")
    rec("n_stations", int(sn_analysis.locationID.nunique()))
    rec("cooling_rate_coverage_pct",
        round(100 * sn_analysis.cool_rate_c_per_h.notna().mean(), 1))
    rec("n_city_nights_all", int(len(city)))
    rec("n_nights_2020_2025", int(len(d)), "after merge with synoptic")
    rec("n_nights_dev", int(len(dev)))
    rec("n_nights_holdout", int(len(ho)))
    rec("n_calm_clear_dev", int((dev.regime == "calm_clear").sum()))
    rec("n_calm_clear_holdout", int((ho.regime == "calm_clear").sum()))

    # Retained-input reference comparison: 15-minute network station named
    # Zürich-Fluntern versus hourly SwissMetNet SMA night minima.
    ref = sn_analysis[sn_analysis.locationID == "Zürich-Fluntern"][[
        "night_date", "night_min_c"]]
    syn_hourly = pd.read_csv(os.path.join(
        C.ROOT, "data", "inputs", "official_meteoswiss_hourly_zurich_synoptic.csv"),
        parse_dates=["night_date"])
    sma = syn_hourly[(syn_hourly.station_abbr == "SMA") &
                     ((syn_hourly.local_hour >= 20) |
                      (syn_hourly.local_hour < 8))]
    sma = (sma.groupby("night_date").tre200h0.min()
           .rename("sma_night_min_c").reset_index())
    ref = ref.merge(sma, on="night_date", how="inner").dropna()
    rec("fluntern_reference_n", int(len(ref)))
    rec("fluntern_reference_mean_bias_c", round(float(
        (ref.night_min_c - ref.sma_night_min_c).mean()), 3),
        "meteoblue minus SwissMetNet SMA hourly night minimum")
    rec("fluntern_reference_pearson_r", round(float(stats.pearsonr(
        ref.night_min_c, ref.sma_night_min_c).statistic), 3))

    # ---------------- clearness model --------------------------------
    cm = pd.read_csv(os.path.join(OUT, "v3_clearness_model.csv"))
    rec("clearness_model_r2", round(float(cm.loc[cm.term == "r_squared", "coef"].iloc[0]), 3))
    rec("clearness_model_resid_sd_wm2",
        round(float(cm.loc[cm.term == "resid_sd_wm2", "coef"].iloc[0]), 1))

    reg = d.groupby("regime").agg(
        n=("night_date", "size"), lw=("lw_in_wm2_mean", "mean"),
        wind=("night_wind_ms", "mean"), tmin=("city_mean_tmin_c", "mean"),
        sd=("sd_tmin_c", "mean")).round(2)
    for r in reg.index:
        rec(f"regime_{r}_n", int(reg.loc[r, "n"]))
        rec(f"regime_{r}_lwin_wm2", float(reg.loc[r, "lw"]))
        rec(f"regime_{r}_mean_sd_tmin", float(reg.loc[r, "sd"]))
        rec(f"regime_{r}_mean_tmin", float(reg.loc[r, "tmin"]))

    # ---------------- the two confounders ----------------------------
    for v, name in [("day_rad_wm2", "dayrad"), ("clearness_wm2", "clearness")]:
        rec(f"conf_{name}_rho_warmth",
            round(float(stats.spearmanr(dev.city_mean_tmin_c, dev[v],
                                        nan_policy="omit").statistic), 3))
        rec(f"conf_{name}_rho_dispersion",
            round(float(stats.spearmanr(dev.sd_tmin_c, dev[v],
                                        nan_policy="omit").statistic), 3))
    rec("conf_dayrad_clearness_corr",
        round(float(stats.spearmanr(dev.day_rad_wm2, dev.clearness_wm2,
                                    nan_policy="omit").statistic), 3))
    # ---------------- specification curve ----------------------------
    sc = pd.read_csv(os.path.join(OUT, "v3_specification_curve.csv"))
    rec("spec_n", int(len(sc)))
    rec("spec_median_rho", round(float(sc.rho.median()), 3))
    rec("spec_q25", round(float(sc.rho.quantile(.25)), 3))
    rec("spec_q75", round(float(sc.rho.quantile(.75)), 3))
    rec("spec_min", round(float(sc.rho.min()), 3))
    rec("spec_max", round(float(sc.rho.max()), 3))
    rec("spec_share_positive_pct", round(100 * float((sc.rho > 0).mean()), 1))
    rec("spec_share_negative_pct", round(100 * float((sc.rho < 0).mean()), 1))
    bya = sc.groupby("adjustment_set").rho.median().round(3)
    for k, vv in bya.items():
        rec(f"spec_median_by_adj__{k}", float(vv))
    rec("spec_adjustment_range",
        round(float(bya.max() - bya.min()), 3), "median rho spread across adjustment sets")
    for f in ["night_window", "regime_cut", "spread_metric"]:
        g = sc.groupby(f).rho.median()
        rec(f"spec_range_by_{f}", round(float(g.max() - g.min()), 3))

    # ---------------- hold-out confirmatory --------------------------
    hoc = pd.read_csv(os.path.join(OUT, "holdout_confirmatory.csv"))
    for _, r in hoc.iterrows():
        rec(f"{r.id}_dev", round(float(r.dev_estimate), 4))
        rec(f"{r.id}_holdout", round(float(r.estimate), 4))
        rec(f"{r.id}_p_bh", round(float(r.p_bh), 4))
        rec(f"{r.id}_verdict", str(r.verdict))
    rec("n_replicated", int(hoc.verdict.str.startswith("replicated").sum()))

    # ---------------- applied layer ----------------------------------
    scr = pd.read_csv(os.path.join(OUT, "v3_priority_screen.csv"))
    cmp_ = pd.read_csv(os.path.join(OUT, "v3_screen_gp_vs_idw_comparison.csv")).iloc[0]
    rec("screen_n_cells", int(len(scr)))
    rec("screen_gp_idw_spearman", round(float(cmp_.score_spearman), 3))
    rec("screen_gp_idw_jaccard", round(float(cmp_.highest_tier_jaccard), 3))
    rec("screen_cells_reclassified",
        int(cmp_.n_highest_gp - cmp_.n_overlap),
        "GP highest-tier cells that IDW did not rank highest")

    hi = scr[scr.tier_gp == "highest_priority"]
    rec("screen_n_highest", int(len(hi)))
    rec("screen_highest_p90", int((hi.p_highest_tier >= 0.90).sum()))
    rec("screen_highest_p80", int((hi.p_highest_tier >= 0.80).sum()))
    rec("screen_highest_median_p", round(float(hi.p_highest_tier.median()), 3))
    rec("screen_frequency_interpretation", "sensitivity frequency, not probability")

    rebuilt_unc = pd.read_csv(os.path.join(
        OUT, "v3_priority_rebuilt_field_uncertainty_summary.csv"))
    ru10 = rebuilt_unc[rebuilt_unc.top_percent == 10].iloc[0]
    rec("screen_rebuilt_p90", int(ru10.n_nominal_frequency_ge_0_90))
    rec("screen_rebuilt_p80", int(ru10.n_nominal_frequency_ge_0_80))
    rec("screen_rebuilt_median_frequency",
        round(float(ru10.median_frequency_nominal), 3))
    dm = pd.read_csv(os.path.join(
        OUT, "v3_priority_decision_multiverse_summary.csv"))
    dm10 = dm[(dm.top_percent == 10) & dm.scope.str.startswith(
        "all_heat_fields")].iloc[0]
    rec("screen_decision_n_specs", int(dm10.n_specifications))
    rec("screen_decision_p90", int(dm10.n_baseline_frequency_ge_0_90))
    rec("screen_decision_p80", int(dm10.n_baseline_frequency_ge_0_80))
    rec("screen_decision_median_frequency",
        round(float(dm10.median_frequency_baseline), 3))
    for scope, tag in [
        ("weights_at_qmargin", "weights"),
        ("population_lenses_at_baseline_weights", "population_lenses"),
        ("heat_fields_at_baseline_decision", "heat_fields"),
        ("leave_one_out_at_qmargin", "leave_one_out"),
    ]:
        row = dm[(dm.top_percent == 10) & (dm.scope == scope)].iloc[0]
        rec(f"screen_decision_{tag}_n_specs", int(row.n_specifications))
        rec(f"screen_decision_{tag}_p80",
            int(row.n_baseline_frequency_ge_0_80))
        rec(f"screen_decision_{tag}_p90",
            int(row.n_baseline_frequency_ge_0_90))
    us = pd.read_csv(os.path.join(
        OUT, "v3_priority_uncertainty_sensitivity_summary.csv"))
    us10 = us[us.top_percent == 10]
    rec("screen_legacy_errorcorr_p90_min",
        int(us10.n_nominal_frequency_ge_0_90.min()))
    rec("screen_legacy_errorcorr_p90_max",
        int(us10.n_nominal_frequency_ge_0_90.max()))
    hc = pd.read_csv(os.path.join(
        OUT, "v3_priority_heat_outcome_correlations.csv"))
    rec("screen_heat_outcome_rho_min",
        round(float(hc.spearman_rho_across_cells.min()), 3))
    rec("screen_heat_outcome_rho_max",
        round(float(hc.spearman_rho_across_cells.max()), 3))

    ts = pd.read_csv(os.path.join(OUT, "v3_priority_tier_summary.csv"))
    for _, r in ts.iterrows():
        t = r.tier_gp
        rec(f"tier_{t}_cells", int(r.n_cells))
        rec(f"tier_{t}_population", int(r.population))
        rec(f"tier_{t}_age65", round(float(r.age65_official)))
        rec(f"tier_{t}_age80", round(float(r.age80_official)))
        rec(f"tier_{t}_mean_tmin", round(float(r.mean_gp_tmin_c), 2))
        rec(f"tier_{t}_median_refuge_m", round(float(r.median_refuge_m)))

    rf = pd.read_csv(os.path.join(OUT, "v3_refuge_decircularised.csv"))
    for spec, g in rf.groupby("specification"):
        g = g.set_index("tier").reindex(
            ["baseline_monitoring", "elevated_monitoring", "high_priority", "highest_priority"])
        tag = "circular" if "with cooling" in spec else "decircularised"
        rec(f"refuge_{tag}_low_m", round(float(g.median_routed_m.iloc[0])))
        rec(f"refuge_{tag}_high_m", round(float(g.median_routed_m.iloc[-1])))
        rec(f"refuge_{tag}_gradient_m",
            round(float(g.median_routed_m.iloc[-1] - g.median_routed_m.iloc[0])))

    # city totals
    rec("city_population_supported", int(scr.pers_n.sum()))
    rec("city_age65_supported", round(float(scr.est65_qmargin.sum())))
    rec("city_age80_supported", round(float(scr.est80_qmargin.sum())))


    # ---------------- ERA5 validation (step 10) ----------------------
    era5_val = os.path.join(OUT, "v3_era5_clearness_validation.csv")
    if os.path.exists(era5_val):
        ev = pd.read_csv(era5_val)
        tot = ev[ev.era5_variable == "era5_cc_mean"].iloc[0]
        rec("era5_rho_clearness_vs_cloud", round(float(tot.rho_clearness_index), 3))
        rec("era5_rho_raw_lwin_vs_cloud", round(float(tot.rho_raw_lwin), 3))
        rec("era5_n_nights", int(tot.n))
        low = ev[ev.era5_variable == "era5_cclow_mean"].iloc[0]
        rec("era5_rho_clearness_vs_lowcloud", round(float(low.rho_clearness_index), 3))
        for _, r in ev[ev.era5_variable == "regime_mean"].iterrows():
            reg = r.label.split("=")[1]
            rec(f"era5_cloud_pct_regime_{reg}", round(float(r.rho_raw_lwin), 1))

        sv = pd.read_csv(os.path.join(OUT, "v3_era5_spatial_variability.csv"))
        rec("era5_spatial_sd_median_pp", round(float(sv.era5_cc_sd_spatial.median()), 1))
        rec("era5_spatial_sd_p90_pp", round(float(sv.era5_cc_sd_spatial.quantile(.9)), 1))
        rec("era5_spatial_range_median_pp", round(float(sv.era5_cc_range_spatial.median()), 1))
        rec("era5_between_night_sd_pp", round(float(sv.era5_cc_mean.std()), 1))
        rec("era5_spatial_to_temporal_ratio",
            round(float(sv.era5_cc_sd_spatial.mean() / sv.era5_cc_mean.std()), 2))

        rs = pd.read_csv(os.path.join(OUT, "v3_era5_regime_sensitivity.csv"))
        for _, r in rs.iterrows():
            rec(f"era5_{r.id}_estimate", round(float(r.estimate), 4))
            rec(f"era5_{r.id}_p_bh", round(float(r.p_bh), 4))
            rec(f"era5_{r.id}_longwave_estimate", round(float(r.estimate_longwave), 3))
            rec(f"era5_{r.id}_longwave_p_bh", round(float(r.p_bh_longwave), 4))
        m = pd.read_csv(os.path.join(OUT, "v3_night_synoptic_era5.csv"))
        rec("era5_regime_agreement_pct",
            round(100 * float((m.regime == m.regime_era5).mean()), 1))
        gm = pd.read_csv(os.path.join(OUT, "v3_era5_grid_mapping.csv"))
        rec("era5_unique_grid_cells", int(gm[
            ["era5_grid_lat", "era5_grid_lon"]].drop_duplicates().shape[0]))
        rec("era5_cloud_available", True,
            "model-pinned ERA5 downloaded 2026-07-28; five requests, two unique cells")


    # ---------------- observable stability (step 11) ------------------
    obs = os.path.join(OUT, "v3_observable_stability_summary.csv")
    if os.path.exists(obs):
        o = pd.read_csv(obs).set_index("outcome")
        for k, tag in [("sd_tmin_c", "endpoint"), ("sd_cool_total", "trajectory_total"),
                       ("sd_cool_rate", "trajectory_rate"),
                       ("city_mean_cool_rate", "level_rate")]:
            if k in o.index:
                r = o.loc[k]
                rec(f"obs_{tag}_median_rho", round(float(r.median_rho), 3))
                rec(f"obs_{tag}_min_rho", round(float(r.min_rho), 3))
                rec(f"obs_{tag}_max_rho", round(float(r.max_rho), 3))
                rec(f"obs_{tag}_adj_range", round(float(r.adj_range), 3))
                rec(f"obs_{tag}_sign_consistency_pct",
                    round(100 * float(r.sign_consistency), 1))
                rec(f"obs_{tag}_n_specs", int(r.n_specs))
        cur = pd.read_csv(os.path.join(OUT, "v3_observable_stability_curve.csv"))
        rec("obs_n_specs_total", int(len(cur)))


    # ---------------- mechanism deep-dive (step 12) --------------------
    prof_c = os.path.join(OUT, "v3_profile_clock.csv")
    if os.path.exists(prof_c):
        for tag, f in [("clock", "v3_profile_clock.csv"), ("sunset", "v3_profile_sunset.csv")]:
            pr = pd.read_csv(os.path.join(OUT, f))
            for tc in ["hot", "cool"]:
                g = pr[pr.tercile == tc].sort_values("offset_h")
                rec(f"prof_{tag}_{tc}_start_sd", round(float(g.sd_mean.iloc[0]), 2))
                rec(f"prof_{tag}_{tc}_end_sd", round(float(g.sd_mean.iloc[-1]), 2))
        # Register the four profile-point SD values quoted in the manuscript
        # abstract and results (sunset and +9h, by warm/cool tercile).
        ps = pd.read_csv(os.path.join(OUT, "v3_profile_sunset.csv"))
        for _tc in ["hot", "cool"]:
            _sub = ps[ps.tercile == _tc].sort_values("offset_h")
            _sunset = _sub.iloc[(_sub.offset_h - 0).abs().argsort().iloc[0]]
            _dawn = _sub.iloc[(_sub.offset_h - 9).abs().argsort().iloc[0]]
            rec(f"prof_sunset_{_tc}_sunset_sd",
                round(float(_sunset.sd_mean), 3),
                "cross-station SD at sunset, sunset-aligned profile")
            rec(f"prof_sunset_{_tc}_plus9h_sd",
                round(float(_dawn.sd_mean), 3),
                "cross-station SD at +9h after sunset, sunset-aligned profile")
        an = pd.read_csv(os.path.join(OUT, "v3_anchor_sensitivity.csv"))
        for _, r in an.iterrows():
            key = (r.variant.replace(" ", "_").replace("(", "").replace(")", "")
                   .replace("->", "_").replace("..", "_").replace("+", "p"))
            rec(f"anchor_{key}_rho", round(float(r.rho), 3))
            rec(f"anchor_{key}_lo", round(float(r.ci_lo), 3))
            rec(f"anchor_{key}_hi", round(float(r.ci_hi), 3))
        dec = pd.read_csv(os.path.join(OUT, "v3_variance_decomposition.csv"))
        for _, r in dec.iterrows():
            if "SD evening" in r.quantity:
                rec("decomp_evening_rho", round(float(r.rho_warmth), 3))
                rec("decomp_evening_lo", round(float(r.ci_lo), 3))
                rec("decomp_evening_hi", round(float(r.ci_hi), 3))
            if "SD night minimum" in r.quantity:
                rec("decomp_min_rho", round(float(r.rho_warmth), 3))
                rec("decomp_min_lo", round(float(r.ci_lo), 3))
                rec("decomp_min_hi", round(float(r.ci_hi), 3))
            if "SD total cooling" in r.quantity:
                rec("decomp_ct_rho", round(float(r.rho_warmth), 3))
                rec("decomp_ct_lo", round(float(r.ci_lo), 3))
                rec("decomp_ct_hi", round(float(r.ci_hi), 3))
            if "partial" in r.quantity:
                rec("decomp_partial_rho", round(float(r.rho_warmth), 3))
                rec("decomp_partial_p", round(float(r.p_value), 4))
        eff = pd.read_csv(os.path.join(OUT, "v3_effect_sizes.csv"))
        for _, r in eff.iterrows():
            q = r.quantity.replace(" ", "_").replace("-", "_")
            rec(f"eff_{q}_{r.tercile}_c", round(float(r.mean_c), 2))
            rec(f"eff_{q}_{r.tercile}_lo", round(float(r.ci_lo), 2))
            rec(f"eff_{q}_{r.tercile}_hi", round(float(r.ci_hi), 2))


    # ---------------- decay non-replication robustness ---------------
    import statsmodels.formula.api as _smf
    _tr = pd.read_csv(os.path.join(OUT, "v3_traj_pernight.csv"), parse_dates=["night_date"])
    _tr["year"] = _tr.night_date.dt.year
    _z = lambda s: (s - s.mean()) / s.std()
    def _decay_p(dd):
        dd = dd.dropna(subset=["decay", "warmth", "clearness_wm2",
                               "night_wind_ms", "day_rad_wm2"]).sort_values("night_date")
        for c in ["warmth", "clearness_wm2", "night_wind_ms", "day_rad_wm2"]:
            dd["z_" + c] = _z(dd[c])
        m = _smf.ols("decay ~ z_warmth + z_clearness_wm2 + z_night_wind_ms + z_day_rad_wm2",
                     data=dd).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
        return float(m.pvalues["z_warmth"]), int(m.nobs)
    _ho = _tr[_tr.year.isin(C.HOLDOUT_YEARS)]
    _pw, _nw = _decay_p(_ho)
    # Distinguish incomplete fixed windows from complete profiles whose peak
    # leaves too few observations for a slope. Missing dusk does not discard an
    # otherwise eligible trajectory row.
    _holdout_dates = set(pd.to_datetime(ho.night_date))
    _trajectory_dates = set(pd.to_datetime(_ho.night_date))
    _missing_trajectory_dates = sorted(_holdout_dates - _trajectory_dates)
    _late_peak_dates = sorted(pd.to_datetime(
        _ho.loc[_ho.complete9 & _ho.decay.isna(), "night_date"]))
    _incomplete_dates = sorted(pd.to_datetime(_ho.loc[~_ho.complete9, 'night_date']))
    _driver_cols = ["warmth", "clearness_wm2", "night_wind_ms", "day_rad_wm2"]
    _missing_driver_n = int(_ho[_driver_cols].isna().any(axis=1).sum())
    assert len(ho) == 178
    _excluded = _ho[['decay'] + _driver_cols].isna().any(axis=1)
    assert _nw == len(_ho) - int(_excluded.sum())
    rec("decay_holdout_missing_trajectory_n", len(_missing_trajectory_dates),
        "eligible holdout night lacking the sunset-to-dusk trajectory segment")
    rec("decay_holdout_late_peak_n", len(_late_peak_dates),
        "complete trajectories with a peak at or after +8.5 h and fewer than four peak-to-+9 h points")
    rec('decay_holdout_incomplete_window_n', len(_incomplete_dates),
        'nine-hour profiles missing at least one required quarter-hour bin')
    rec("decay_holdout_late_peak_threshold_h", 8.5,
        "late-peak threshold implied by requiring four 15-minute points through +9 h")
    rec("decay_holdout_missing_driver_n", _missing_driver_n,
        "missing continuous meteorological covariates among trajectory rows")
    rec("decay_withheld_allnights_p", round(_pw, 3),
        "legacy all-night HAC sensitivity; exclusions follow complete-window and peak-support rules")
    rec("decay_withheld_allnights_n", _nw)
    _cch = _ho[_ho.regime == "calm_clear"]
    _cch = _cch.assign(rel=_cch.decay / _cch.peak_sd)
    from scipy import stats as _st
    _ret = C.block_permutation_spearman(
        _cch.warmth, _cch.rel, groups=_cch.year)
    rec("decay_retention_withheld_p",
        round(float(_ret["p_block_perm"]), 3),
        "retention-ratio metric, calm-clear withheld; summer-stratified shift test")

    # ---------------- trajectory framework (step 14) ------------------
    tc = os.path.join(OUT, "v3_traj_confirmatory.csv")
    if os.path.exists(tc):
        cf = pd.read_csv(tc)
        for _, r in cf.iterrows():
            k = f"traj_{r.scalar}_{r['sample']}"
            framework_note = (
                "v3_14 trajectory framework; midpoint-aligned profile with "
                "complete-calendar summer-stratified moving-block interval"
            )
            rec(f"{k}_rho", round(float(r.rho), 3), framework_note)
            rec(f"{k}_lo", round(float(r.ci_lo), 3), framework_note)
            rec(f"{k}_hi", round(float(r.ci_hi), 3), framework_note)
            rec(f"{k}_shift_p", round(float(r.block_perm_p), 4))
            rec(f"{k}_bh_p", round(float(r.p_bh_within_sample), 4))
            rec(f"{k}_n_randomizations", int(r.n_randomizations))
        dr = pd.read_csv(os.path.join(OUT, "v3_traj_continuous_drivers.csv"))
        for sc in ["dusk_sd", "integ_sd", "decay"]:
            w = dr[(dr.scalar == sc) & (dr.driver == "z_warmth")].iloc[0]
            rec(f"traj_{sc}_warmth_beta", round(float(w.beta), 3))
            rec(f"traj_{sc}_warmth_p", float(f"{w.p:.2g}"))
            rec(f"traj_{sc}_r2", round(float(w.r2), 2))
            rec(f"traj_{sc}_n", int(w.n))
        fm = pd.read_csv(os.path.join(OUT, "v3_traj_functional_model.csv"))
        for _, r in fm.iterrows():
            wl = r.warmth_level.split()[0]
            rec(f"traj_curve_{wl}_peak_hss", round(float(r.peak_hss), 2))
            rec(f"traj_curve_{wl}_peak_sd", round(float(r.peak_sd), 2))
            rec(f"traj_curve_{wl}_erosion", round(float(r.erosion), 2))
        dg = pd.read_csv(os.path.join(OUT, "v3_traj_functional_diag.csv")).iloc[0]
        rec("traj_curve_calendar_replicates", int(dg.n_curve_boot),
            "pointwise descriptive intervals; no whole-curve significance test")
        rec("traj_functional_rank_full", int(dg.rank_full))
        rec("traj_functional_nights", int(dg.n_nights))


    # ---------------- station-trait model (step 15) -------------------
    stm = os.path.join(OUT, "v3_station_trait_models.csv")
    if os.path.exists(stm):
        sm = pd.read_csv(stm)
        for tgt in ["dusk_anom", "integ_anom", "decay_tend"]:
            g = sm[sm.target == tgt]
            rec(f"strait_{tgt}_cv_r2", round(float(g.cv_r2.iloc[0]), 3))
            rec(f"strait_{tgt}_cv_r2_q05", round(float(g.cv_r2_q05.iloc[0]), 3))
            rec(f"strait_{tgt}_cv_r2_q95", round(float(g.cv_r2_q95.iloc[0]), 3))
            top = g.sort_values("delta_cv_r2_drop", ascending=False).iloc[0]
            rec(f"strait_{tgt}_top_pred", str(top.predictor))
            rec(f"strait_{tgt}_top_coef", round(float(top.std_coef), 2))
            rec(f"strait_{tgt}_top_delta_cv_r2",
                round(float(top.delta_cv_r2_drop), 3),
                "median leave-one-predictor-out change across repeated spatial blocks")
            rec(f"strait_{tgt}_top_delta_positive_share",
                round(float(top.delta_positive_share), 3))
            elev = g[g.predictor == "elevation"].iloc[0]
            rec(f"strait_{tgt}_elevation_delta_cv_r2",
                round(float(elev.delta_cv_r2_drop), 3))
        st = pd.read_csv(os.path.join(OUT, "v3_station_traits.csv"))
        rec("strait_n_stations", int(len(st)))


    # block-permutation p (non-degenerate) for the confirmatory scalars
    _cf = pd.read_csv(os.path.join(OUT, "v3_traj_confirmatory.csv"))
    if "block_perm_p" in _cf.columns:
        for _, r in _cf.iterrows():
            rec(f"traj_{r.scalar}_{r['sample']}_blockperm_p", round(float(r.block_perm_p), 4))
    # FITNAH external validation (step 16)
    _fv = os.path.join(OUT, "v3_fitnah_validation.csv")
    if os.path.exists(_fv):
        fv = pd.read_csv(_fv)
        gp = fv[fv.quantity.str.startswith("GP field")]
        idw = fv[fv.quantity.str.startswith("IDW field")]
        raw = gp[(gp.quantity.str.contains("Spearman", regex=False)) &
                 (~gp.quantity.str.contains("detrended", case=False)) &
                 (~gp.quantity.str.contains("cold-air", case=False))].iloc[0]
        pear = gp[gp.quantity.str.contains("Pearson", regex=False)].iloc[0]
        det = gp[gp.quantity.str.contains("detrended", case=False)].iloc[0]
        idw_raw = idw[(idw.quantity.str.contains("Spearman", regex=False)) &
                      (~idw.quantity.str.contains("detrended", case=False))].iloc[0]
        idw_pear = idw[idw.quantity.str.contains("Pearson", regex=False)].iloc[0]
        idw_det = idw[idw.quantity.str.contains("detrended", case=False)].iloc[0]
        cold = fv[fv.quantity.str.contains("cold-air", case=False)].iloc[0]
        rec("fitnah_gp_spearman", round(float(raw["value"]), 3),
            "GP night-heat field vs independent FITNAH model")
        rec("fitnah_gp_ci_lo", round(float(raw["ci_lo"]), 3))
        rec("fitnah_gp_ci_hi", round(float(raw["ci_hi"]), 3))
        rec("fitnah_gp_pearson", round(float(pear["value"]), 3))
        rec("fitnah_gp_detrended_spearman", round(float(det["value"]), 3))
        rec("fitnah_gp_detrended_ci_lo", round(float(det["ci_lo"]), 3))
        rec("fitnah_gp_detrended_ci_hi", round(float(det["ci_hi"]), 3))
        rec("fitnah_idw_spearman", round(float(idw_raw["value"]), 3),
            "IDW night-heat field vs independent FITNAH model on identical cells")
        rec("fitnah_idw_ci_lo", round(float(idw_raw["ci_lo"]), 3))
        rec("fitnah_idw_ci_hi", round(float(idw_raw["ci_hi"]), 3))
        rec("fitnah_idw_pearson", round(float(idw_pear["value"]), 3))
        rec("fitnah_idw_detrended_spearman", round(float(idw_det["value"]), 3))
        rec("fitnah_idw_detrended_ci_lo", round(float(idw_det["ci_lo"]), 3))
        rec("fitnah_idw_detrended_ci_hi", round(float(idw_det["ci_hi"]), 3))
        rec("fitnah_coldair_spearman", round(float(cold["value"]), 3),
            "cooler where FITNAH cold-air drainage is stronger")
        rec("fitnah_n_cells", int(raw["n"]))

    # ---------------- metrology + calendar time (step 17) ------------
    qa_path = os.path.join(C.ROOT, "outputs_robust", "v3_17_qa_flag_summary.csv")
    if os.path.exists(qa_path):
        qa = pd.read_csv(qa_path)
        qfull = qa[qa.scope == "full_published_archive"].set_index("qc_flag")
        for flag in range(7):
            rec(f"qa_archive_flag_{flag}_n", int(qfull.loc[flag, "n_records"]))
            rec(f"qa_archive_flag_{flag}_pct",
                round(100 * float(qfull.loc[flag, "fraction_of_scope"]), 3))
        qj = qa[qa.scope == "JJA_2020_2025_local_time"].set_index("qc_flag")
        for flag in range(7):
            rec(f"qa_jja_flag_{flag}_n", int(qj.loc[flag, "n_records"]))
            rec(f"qa_jja_flag_{flag}_pct",
                round(100 * float(qj.loc[flag, "fraction_of_scope"]), 3))
        rc = pd.read_csv(os.path.join(
            C.ROOT, "outputs_robust", "v3_17_dusk_rc_summary.csv"))
        all_rc = rc[rc.scope == "all_study_nights"].iloc[0]
        rec("dusk_rc_n_records", int(all_rc.n_valid_records))
        rec("dusk_rc_n_corrected", int(all_rc.n_rc_flag_1))
        hourly_rc = pd.read_csv(os.path.join(
            C.ROOT, "outputs_robust", "v3_17_radiation_correction_by_hour.csv"))
        for hour in (19, 20, 21):
            row = hourly_rc[hourly_rc.published_interval_end_hour == hour].iloc[0]
            rec(f"rc_hour_{hour}_pct", round(100 * float(row.rc_flag_1_fraction), 2))
        kw = pd.read_csv(os.path.join(
            C.ROOT, "outputs_robust", "v3_17_kollerwiese_summary.csv")).iloc[0]
        for key, col, nd in [
            ("koller_distance_m", "coordinate_separation_m", 1),
            ("koller_n_nights", "n_paired_nights", 0),
            ("koller_mean_diff_c", "mean_difference_pair_1_minus_2_c", 3),
            ("koller_sd_diff_c", "sd_difference_c", 3),
            ("koller_rmse_diff_c", "rmse_between_pair_c", 3),
            ("koller_max_abs_diff_c", "min_difference_c", 1),
        ]:
            val = abs(float(kw[col])) if key == "koller_max_abs_diff_c" else float(kw[col])
            rec(key, int(val) if nd == 0 else round(val, nd))

        da = pd.read_csv(os.path.join(
            C.ROOT, "outputs_robust", "v3_17_dusk_associations.csv"))
        for name, mask in {
            "dusk_dev_midpoint": (
                (da.analysis_scope == "development") &
                (da.timestamp_convention == "interval_midpoint") &
                (da.station_set == "dynamic_all") & (da.estimand == "spearman")),
            "dusk_eval_midpoint": (
                (da.analysis_scope == "internal_temporal_evaluation") &
                (da.timestamp_convention == "interval_midpoint") &
                (da.station_set == "dynamic_all") & (da.estimand == "spearman") &
                (da.radiation_correction_policy == "published_corrected_values")),
            "dusk_eval_doy_year_adjusted": (
                (da.analysis_scope == "internal_temporal_evaluation") &
                (da.timestamp_convention == "interval_midpoint") &
                (da.station_set == "dynamic_all") &
                (da.estimand == "day_of_year_and_year_adjusted_partial_spearman") &
                (da.radiation_correction_policy == "published_corrected_values")),
            "dusk_eval_endpoint": (
                (da.analysis_scope == "internal_temporal_evaluation") &
                (da.timestamp_convention == "published_endpoint") &
                (da.estimand == "spearman")),
        }.items():
            row = da[mask].iloc[0]
            primary_note = (
                "v3_17 metrology-temporal analysis; summer-stratified "
                "full-calendar moving-block bootstrap interval"
            )
            rec(f"{name}_rho", round(float(row.rho), 3), primary_note)
            rec(f"{name}_lo", round(float(row.ci_low), 3), primary_note)
            rec(f"{name}_hi", round(float(row.ci_high), 3), primary_note)
        for core in (80, 90, 95):
            row = da[(da.analysis_scope == "internal_temporal_evaluation") &
                     (da.station_set == f"fixed_core_{core}") &
                     (da.estimand == "spearman")].iloc[0]
            rec(f"dusk_core_{core}_rho", round(float(row.rho), 3))
            rec(f"dusk_core_{core}_lo", round(float(row.ci_low), 3))
            rec(f"dusk_core_{core}_hi", round(float(row.ci_high), 3))
        cov = pd.read_csv(os.path.join(
            C.ROOT, "outputs_robust", "v3_17_station_coverage_audit.csv"))
        for core in (80, 90, 95):
            rec(f"dusk_core_{core}_n_stations", int(cov[f"fixed_core_{core}"].sum()))
        hac = pd.read_csv(os.path.join(
            C.ROOT, "outputs_robust", "v3_17_calendar_hac_models.csv"))
        warmth = hac[(hac.model == "level_all_nights_weather_season_year_adjusted") &
                     (hac.term == "warmth_c")].iloc[0]
        rec("dusk_hac_warmth_beta_per_c", round(float(warmth.estimate), 3))
        rec("dusk_hac_warmth_lo", round(float(warmth.ci_low_HAC), 3))
        rec("dusk_hac_warmth_hi", round(float(warmth.ci_high_HAC), 3))
        rec("dusk_hac_n", int(warmth.n_nights))
        rec("dusk_hac_r2", round(float(warmth.r_squared), 2))

    # ---------------- spatial rebuild (step 18) ----------------------
    diag_path = os.path.join(C.ROOT, "outputs_robust",
                             "v3_spatial_rebuild_diagnostics.csv")
    if os.path.exists(diag_path):
        dg = pd.read_csv(diag_path)
        def dnum(metric):
            return float(dg.loc[dg.metric == metric, "numeric_value"].iloc[0])
        rec("spatial_rebuild_station_nights", int(dnum("n_station_nights")))
        rec("spatial_matern_length_km", round(dnum("matern_length_scale"), 2))
        rec("spatial_effective_range_km",
            round(dnum("matern_correlation_0p05_range"), 2))
        rec("spatial_white_noise_sd_c", round(dnum("white_noise_sd_c"), 2))
        rec("spatial_grid_nearest_station_median_m",
            round(dnum("grid_nearest_station_median")))
        rec("spatial_grid_nearest_station_p90_m",
            round(dnum("grid_nearest_station_p90")))
        cv = pd.read_csv(os.path.join(
            C.ROOT, "outputs_robust", "v3_spatial_rebuild_cv_metrics.csv"))
        for row_type, tag in [("q025", "lo"), ("median", "median"),
                              ("q975", "hi")]:
            row = cv[cv.row_type == row_type].iloc[0]
            rec(f"spatial_cv_r2_{tag}", round(float(row.r2), 3))
            rec(f"spatial_cv_rmse_{tag}", round(float(row.rmse_c), 3))
            rec(f"spatial_cv_mae_{tag}", round(float(row.mae_c), 3))
            rec(f"spatial_cv_rho_{tag}", round(float(row.spearman_rho), 3))
            rec(f"spatial_cv_coverage_{tag}",
                round(float(row.gp_coverage95), 3))
            rec(f"spatial_cv_gp_interval_width_{tag}",
                round(float(row.gp_mean_interval_width_c), 3))
            rec(f"spatial_cv_target_augmented_coverage_{tag}",
                round(float(row.target_augmented_coverage95), 3))
            rec(f"spatial_cv_target_augmented_interval_width_{tag}",
                round(float(row.target_augmented_mean_interval_width_c), 3))
        comp = pd.read_csv(os.path.join(
            C.ROOT, "outputs_robust", "v3_spatial_rebuild_comparisons.csv"))
        for name, label in [
            ("rebuilt adjusted GP vs terminal GP", "spatial_vs_terminal_gp"),
            ("rebuilt adjusted GP vs retained FITNAH", "spatial_vs_fitnah"),
            ("rebuilt adjusted GP vs retained FITNAH (quadratic-coordinate detrended)",
             "spatial_vs_fitnah_detrended"),
        ]:
            row = comp[comp.comparison == name].iloc[0]
            rec(f"{label}_rho", round(float(row.spearman_rho), 3))
        targets = pd.read_csv(os.path.join(
            C.ROOT, "outputs_robust", "v3_spatial_rebuild_station_targets.csv"))
        k10 = targets[targets.locationID.str.startswith("K10")].iloc[0]
        rec("spatial_k10_coverage_adjustment_c",
            round(float(k10.adjusted_mean_night_min_c -
                        k10.raw_mean_night_min_c), 3))

    # ---------------- targeted submission corrections ----------------
    corrected = os.path.join(C.ROOT, 'outputs_robust', 'v3_17_trajectory_associations.csv')
    if os.path.exists(corrected):
        for _, row in pd.read_csv(corrected).iterrows():
            scope = ('development' if row.analysis_scope == 'development' else 'withheld')
            for col in ['rho', 'ci_low', 'ci_high', 'n_nights']:
                rec(f"corrected_{row.metric}_{scope}_{col}", float(row[col]), corrected)
            if row.metric in ['integrated_0_9h', 'post_peak_slope_0_9h']:
                scalar = 'integ_sd' if row.metric == 'integrated_0_9h' else 'decay'
                for old, new in [('rho', 'rho'), ('lo', 'ci_low'), ('hi', 'ci_high')]:
                    rec(f'traj_{scalar}_{scope}_{old}', round(float(row[new]), 3), corrected)
        regression = os.path.join(C.ROOT, 'outputs_robust', 'v3_17_calendar_block_regression.csv')
        row = pd.read_csv(regression).query("term == 'warmth_c'").iloc[0]
        for col in ['estimate', 'ci_low', 'ci_high', 'n_nights']:
            rec(f'dusk_calendar_regression_{col}', float(row[col]), regression)
        summary_path = os.path.join(C.ROOT, 'outputs_robust', 'v3_20_correction_summary.json')
        with open(summary_path) as f:
            for name, value in json.load(f).items():
                rec('correction_'+name, value, summary_path)
        matched_path = os.path.join(C.ROOT, 'outputs_robust', 'v3_20_fluntern_matched_hourly.csv')
        for _, row in pd.read_csv(matched_path).iterrows():
            for col in ['n', 'bias', 'rmse', 'mae', 'q025', 'q975']:
                rec(f'matched_reference_{row.support}_{col}', float(row[col]), matched_path)
        strict_path = os.path.join(C.ROOT, 'outputs_robust', 'fold_contained_spatial_cv.csv')
        for col in ['r2', 'rmse', 'rho']:
            for q in [.025, .5, .975]:
                rec(f'fold_contained_{col}_{q}',
                    float(pd.read_csv(strict_path)[col].quantile(q)), strict_path)
        st = pd.read_csv(os.path.join(OUT, 'v3_station_traits.csv'))
        rec('station_context_median_offset_m', float(st.dist_nearest_cell_m.median()))
        rec('station_context_n_over_100m', int((st.dist_nearest_cell_m > 100).sum()))
        rec('station_context_n_over_250m', int((st.dist_nearest_cell_m > 250).sum()))
        rec('station_trait_dusk_cv_percent', 100*V['strait_dusk_anom_cv_r2']['value'])
        rec('spatial_predictive_coverage_percent', 100*V['spatial_cv_coverage_median']['value'])
        balanced_path = os.path.join(C.ROOT, 'outputs_robust', 'v3_tmin_balanced_coverage_summary.csv')
        b = pd.read_csv(balanced_path).query("period == 'internal_temporal_evaluation'").iloc[0]
        rec('dusk_balanced_evaluation_rho', float(b.rho_dusk_with_balanced_coverage_warmth), balanced_path)
        clear_path = os.path.join(OUT, 'v3_clearness_model.csv')
        c = pd.read_csv(clear_path).query("term == 'n_hours_fit_2020_2023'").iloc[0]
        rec('clearness_calibration_all_seasons_n_hours', int(c.coef), clear_path)
        rec('external_anet2022_training_rmse_K', .92,
            'Externally attributed training result, Anet 2022; not a study validation metric')

    # ---------------- historical availability snapshot ---------------
    rec("june2026_available_at_20260728", False,
        "CKAN datastore max timestamp 2026-05-31T23:45Z, checked 2026-07-28")
    rec("june2026_checked_date", "2026-07-28",
        "CKAN datastore re-checked directly; still ends 2026-05-31")

    with open(os.path.join(OUT, "v3_verified_numbers.json"), "w") as f:
        json.dump(V, f, indent=2)

    print(f"{len(V)} verified quantities\n")
    for k, o in V.items():
        note = f"   # {o['note']}" if o["note"] else ""
        print(f"  {k:44s} = {o['value']}{note}")
    print(f"\nwrote -> {OUT}/v3_verified_numbers.json")


if __name__ == "__main__":
    main()
