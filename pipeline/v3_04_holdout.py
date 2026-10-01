"""
V3 STEP 4 -- INTERNAL TEMPORAL EVALUATION (summers 2024-2025).

Evaluates five development-derived hypotheses, with the
estimators defined in v3_core.py. No new estimators, no re-metricisation.
These hypotheses follow development-sample exploration. No prospective plan
is retained in this package, so these results are not labelled preregistered
confirmation or independent replication.  The output filename is retained for
backward compatibility.

Outputs (outputs_v3/):
  holdout_confirmatory.csv
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

import v3_core as C

OUT = C.OUT

# Retained development-sample values; no prospective plan is available in this package.
DEV = {
    "H1": {"stat": "spearman_rho", "dev_estimate": 0.2450, "predicted": "positive"},
    "H2": {"stat": "partial_rho", "dev_estimate": 0.1950, "predicted": "positive"},
    "H3": {"stat": "spearman_rho", "dev_estimate": 0.3539, "predicted": "positive"},
    "H4": {"stat": "hac_slope", "dev_estimate": 0.0193, "predicted": "positive"},
    "H5": {"stat": "hac_slope", "dev_estimate": 0.0025, "predicted": "null"},
}

FULL_SET = ["day_rad_wm2", "clearness_wm2", "night_wind_ms"]


def main() -> None:
    d = C.load_nights()
    ho = C.sample(d, "holdout")
    cc = ho[ho.regime == "calm_clear"].copy()
    print(f"HOLD-OUT: {len(ho)} nights ({ho.year.min()}-{ho.year.max()}), "
          f"{len(cc)} calm-clear\n")

    rows = []

    # H1 -- dispersion vs warmth within calm-clear
    h1 = C.block_permutation_spearman(
        cc.city_mean_tmin_c, cc.sd_tmin_c, groups=cc.year)
    rows.append({"id": "H1", "description": "SD(Tmin) vs city warmth, calm-clear",
                 "estimate": h1["rho"], "p_raw": h1["p_block_perm"], "n": h1["n"]})

    # H2 -- same, full physical adjustment set, all regimes
    h2 = C.partial_spearman(ho, "city_mean_tmin_c", "sd_tmin_c", FULL_SET)
    rows.append({"id": "H2", "description": "SD(Tmin) vs warmth | dayrad+clearness+wind",
                 "estimate": h2["r_partial"], "p_raw": h2["p_block_perm"], "n": h2["n"]})

    # H3 -- dispersion of total cooling vs warmth, calm-clear
    h3 = C.block_permutation_spearman(
        cc.city_mean_tmin_c, cc.sd_cool_total, groups=cc.year)
    rows.append({"id": "H3", "description": "SD(total cooling) vs warmth, calm-clear",
                 "estimate": h3["rho"], "p_raw": h3["p_block_perm"], "n": h3["n"]})

    # H4 -- mean cooling rate vs warmth, calm-clear
    m4 = C.hac_ols(cc.dropna(subset=["city_mean_cool_rate", "city_mean_tmin_c"]),
                   "city_mean_cool_rate ~ city_mean_tmin_c")
    rows.append({"id": "H4", "description": "mean cooling rate vs warmth, calm-clear",
                 "estimate": m4.params["city_mean_tmin_c"],
                 "p_raw": m4.pvalues["city_mean_tmin_c"], "n": int(m4.nobs)})

    # H5 -- DECLARED NULL: cooling-rate dispersion vs warmth
    m5 = C.hac_ols(cc.dropna(subset=["sd_cool_rate", "city_mean_tmin_c"]),
                   "sd_cool_rate ~ city_mean_tmin_c")
    rows.append({"id": "H5", "description": "SD(cooling rate) vs warmth, calm-clear [NULL]",
                 "estimate": m5.params["city_mean_tmin_c"],
                 "p_raw": m5.pvalues["city_mean_tmin_c"], "n": int(m5.nobs)})

    res = pd.DataFrame(rows)
    res["p_bh"] = C.bh_correct(res.p_raw.tolist())
    res["dev_estimate"] = res.id.map(lambda k: DEV[k]["dev_estimate"])
    res["predicted"] = res.id.map(lambda k: DEV[k]["predicted"])

    def verdict(r):
        if r.predicted == "null":
            return ("not rejected (not equivalence)" if r.p_bh >= 0.05
                    else "null rejected")
        sign_ok = np.sign(r.estimate) == np.sign(r.dev_estimate)
        if sign_ok and r.p_bh < 0.05:
            return "direction held; BH-detectable"
        if sign_ok:
            return "direction held; BH p>=0.05"
        return "direction reversed"

    res["verdict"] = res.apply(verdict, axis=1)

    print("=== INTERNAL TEMPORAL EVALUATION (hold-out 2024-2025) ===")
    print(res[["id", "description", "dev_estimate", "estimate", "p_raw", "p_bh", "verdict"]]
          .round(4).to_string(index=False))

    n_detect = res.verdict.str.contains("BH-detectable").sum()
    print(f"\n  {n_detect} directional hypothesis had the planned sign and BH p<0.05; "
          "H5 is a failure-to-reject, not an equivalence result")

    res.to_csv(os.path.join(OUT, "holdout_confirmatory.csv"), index=False)
    print(f"\nwrote -> {OUT}/holdout_confirmatory.csv")


if __name__ == "__main__":
    main()
