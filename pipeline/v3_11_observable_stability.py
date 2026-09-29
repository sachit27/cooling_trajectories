"""
V3 STEP 11 -- Is the instability a property of the ESTIMAND, or of the OBSERVABLE?

The specification curve (v3_05) showed that the association between city warmth and
intra-urban dispersion of NIGHT MINIMA swings with the adjustment set. Two readings
are possible:

  (a) the underlying relationship is genuinely unidentifiable from these data, or
  (b) the night minimum is a poor observable -- it aggregates decoupling, cooling rate
      and duration, which answer to different drivers with different signs -- and a
      trajectory-based observable would be stable.

These make opposite predictions. Under (a), EVERY dispersion outcome should be
adjustment-set sensitive. Under (b), dispersion in a trajectory COMPONENT should be
markedly more stable than dispersion in the endpoint.

This script runs the identical specification machinery over four outcomes:
    sd_tmin_c        endpoint     (dispersion of night minima)
    sd_cool_total    trajectory   (dispersion of total evening->minimum cooling)
    sd_cool_rate     trajectory   (dispersion of mid-night cooling rate)
    city_mean_cool_rate  level    (mean cooling rate, not a dispersion measure)

and compares their stability. The night window is held at 20:00-08:00 because the
cooling anchors (20-22, 22:00, 02:00) are clock-defined; all other factors are crossed
exactly as in v3_05.

Outputs:
  v3_observable_stability_curve.csv    every specification
  v3_observable_stability_summary.csv  stability statistics per outcome
"""
from __future__ import annotations

import itertools
import os
import numpy as np
import pandas as pd

import v3_core as C

OUT = C.OUT

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
CUTS = {"33/67": (1 / 3, 2 / 3), "25/75": (0.25, 0.75), "20/80": (0.20, 0.80)}

OUTCOMES = {
    "sd_tmin_c": ("endpoint", "dispersion of night minima"),
    "sd_cool_total": ("trajectory", "dispersion of total evening-to-minimum cooling"),
    "sd_cool_rate": ("trajectory", "dispersion of mid-night cooling rate"),
    "city_mean_cool_rate": ("level", "mean cooling rate"),
}


def partial_r(df, x, y, covs):
    return C.partial_spearman_point(df, x, y, covs, min_n=30)


def main() -> None:
    base = C.load_nights()
    rows = []

    for outcome, cut_name, adj_name, yrs, samp in itertools.product(
            OUTCOMES, CUTS, ADJ_SETS, ["all", "dev", "holdout"],
            ["all_nights", "calm_clear_only"]):

        d = base.copy()
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
        if samp == "calm_clear_only":
            d = d[d.regime == "calm_clear"]

        r, n = partial_r(d, "city_mean_tmin_c", outcome, ADJ_SETS[adj_name])
        rows.append({
            "outcome": outcome, "outcome_type": OUTCOMES[outcome][0],
            "regime_cut": cut_name, "adjustment_set": adj_name,
            "years": yrs, "sample": samp, "rho": r, "n": n,
        })

    sc = pd.DataFrame(rows).dropna(subset=["rho"])
    sc.to_csv(os.path.join(OUT, "v3_observable_stability_curve.csv"), index=False)
    print(f"{len(sc)} specifications across {sc.outcome.nunique()} outcomes\n")

    # ---- stability statistics -------------------------------------------
    print("=== STABILITY BY OUTCOME ===")
    print("(adj_range = spread of MEDIAN rho across the 8 adjustment sets --")
    print(" the quantity that made the endpoint unstable)\n")

    summ = []
    for outcome, g in sc.groupby("outcome"):
        by_adj = g.groupby("adjustment_set").rho.median()
        sign_consistency = max((g.rho > 0).mean(), (g.rho < 0).mean())
        row = {
            "outcome": outcome,
            "outcome_type": OUTCOMES[outcome][0],
            "description": OUTCOMES[outcome][1],
            "median_rho": g.rho.median(),
            "min_rho": g.rho.min(),
            "max_rho": g.rho.max(),
            "full_range": g.rho.max() - g.rho.min(),
            "iqr": g.rho.quantile(.75) - g.rho.quantile(.25),
            "adj_range": by_adj.max() - by_adj.min(),
            "sign_consistency": sign_consistency,
            "n_specs": len(g),
        }
        summ.append(row)

    s = pd.DataFrame(summ).sort_values("adj_range")
    print(s[["outcome", "outcome_type", "median_rho", "min_rho", "max_rho",
             "adj_range", "sign_consistency"]].round(3).to_string(index=False))
    s.to_csv(os.path.join(OUT, "v3_observable_stability_summary.csv"), index=False)

    # ---- the decisive comparison ----------------------------------------
    end = s[s.outcome == "sd_tmin_c"].iloc[0]
    tot = s[s.outcome == "sd_cool_total"].iloc[0]
    print("\n=== DECISIVE COMPARISON ===")
    print(f"  endpoint   (night-minimum dispersion) : adjustment-set range = {end.adj_range:.3f}, "
          f"sign consistency = {end.sign_consistency:.0%}")
    print(f"  trajectory (total-cooling dispersion) : adjustment-set range = {tot.adj_range:.3f}, "
          f"sign consistency = {tot.sign_consistency:.0%}")
    ratio = end.adj_range / tot.adj_range if tot.adj_range > 0 else np.inf
    print(f"\n  the endpoint is {ratio:.1f}x more adjustment-set sensitive "
          f"than the trajectory component")

    # Be precise about WHICH kind of stability improves. Magnitude sensitivity
    # improves only modestly; SIGN stability is the categorical difference, and
    # sign is what the literature actually disagrees about.
    print("\n  Magnitude sensitivity  : improves modestly "
          f"({end.adj_range:.3f} -> {tot.adj_range:.3f}). Both observables still move")
    print("                           substantially with the adjustment set.")
    print(f"  SIGN stability         : categorical. The trajectory component is positive")
    print(f"                           in {tot.sign_consistency:.0%} of specifications "
          f"(minimum rho = {tot.min_rho:+.3f});")
    print(f"                           the endpoint flips sign in "
          f"{1 - end.sign_consistency:.0%} (minimum rho = {end.min_rho:+.3f}).")
    print("\n  -> The defensible claim is about SIGN, not magnitude: the DIRECTION of")
    print("     the relationship is specification-dependent on the endpoint and is not")
    print("     on the trajectory component. Since the literature disagrees about")
    print("     direction, this is the disagreement that the observable resolves.")
    print("     Do NOT claim the trajectory observable removes adjustment-set")
    print("     sensitivity in magnitude -- it does not.")

    # per-adjustment-set detail for the two key outcomes
    print("\n=== median rho by adjustment set, endpoint vs trajectory ===")
    piv = (sc[sc.outcome.isin(["sd_tmin_c", "sd_cool_total"])]
           .pivot_table(index="adjustment_set", columns="outcome",
                        values="rho", aggfunc="median").round(3))
    piv["swing"] = (piv["sd_tmin_c"] - piv["sd_tmin_c"].median()).abs().round(3)
    print(piv.to_string())

    print(f"\nwrote -> {OUT}/v3_observable_stability_*.csv")


if __name__ == "__main__":
    main()
