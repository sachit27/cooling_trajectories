"""
V3 STEP 3 -- DEVELOPMENT SAMPLE ONLY (summers 2020-2023).

This script may look at the development sample freely. The hold-out (2024-2025)
is not touched here. Hypotheses are formed from these results, frozen in
PRE_ANALYSIS_PLAN.md, and only then tested on the hold-out.

Outputs (outputs_v3/):
  dev_E1_simpson_decomposition.csv
  dev_E2_saturation.csv
  dev_E3_mechanism.csv
  dev_descriptives.csv
"""
from __future__ import annotations

import os
import pandas as pd

import v3_core as C

OUT = C.OUT


def main() -> None:
    d = C.load_nights()
    dev = C.sample(d, "dev")
    print(f"DEVELOPMENT SAMPLE: {len(dev)} nights, {dev.year.min()}-{dev.year.max()}")
    print(f"  hold-out reserved: {len(C.sample(d, 'holdout'))} nights (NOT examined)\n")

    # ---- descriptives ----------------------------------------------------
    desc = (dev.groupby("regime")
            .agg(n_nights=("night_date", "size"),
                 mean_tmin=("city_mean_tmin_c", "mean"),
                 sd_tmin=("sd_tmin_c", "mean"),
                 p90p10=("p90p10_tmin_c", "mean"),
                 clearness=("clearness_wm2", "mean"),
                 wind=("night_wind_ms", "mean"),
                 cool_rate=("city_mean_cool_rate", "mean"),
                 sd_cool_rate=("sd_cool_rate", "mean"))
            .round(3).reset_index())
    desc.to_csv(os.path.join(OUT, "dev_descriptives.csv"), index=False)
    print("=== descriptives by regime ===")
    print(desc.to_string(index=False))

    # ---- E1: Simpson decomposition --------------------------------------
    print("\n=== E1  marginal vs conditional (Simpson decomposition) ===")
    e1_all = []
    for metric in ["sd_tmin_c", "p90p10_tmin_c"]:
        e1 = C.marginal_vs_conditional(dev, metric)
        e1_all.append(e1)
        print(f"\n-- spread metric: {metric}")
        print(e1[["estimand", "conditioning", "r", "p_block_perm", "n"]]
              .round(4).to_string(index=False))
    e1_df = pd.concat(e1_all, ignore_index=True)
    e1_df.to_csv(os.path.join(OUT, "dev_E1_simpson_decomposition.csv"), index=False)

    # ---- E2: saturation within calm-clear -------------------------------
    print("\n=== E2  within calm-clear: does dispersion grow or shrink with warmth? ===")
    e2 = C.saturation_within_calm_clear(dev)
    e2.to_csv(os.path.join(OUT, "dev_E2_saturation.csv"), index=False)
    print(e2.round(4).to_string(index=False))

    # ---- E3: cooling-rate mechanism -------------------------------------
    print("\n=== E3  cooling-rate mechanism (calm-clear nights) ===")
    e3 = C.cooling_rate_mechanism(dev)
    e3.to_csv(os.path.join(OUT, "dev_E3_mechanism.csv"), index=False)
    print(e3.round(4).to_string(index=False))

    print(f"\nwrote development results -> {OUT}")


if __name__ == "__main__":
    main()
