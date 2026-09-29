"""Retain the block-length and single-summer dusk sensitivity results.

This lightweight step uses the retained night-level dusk metric and the same
complete-calendar moving-block implementation as v3_17. It does not read or
modify the raw sensor archive.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from v3_17_metrology_temporal import (
    EVAL_YEARS,
    N_BOOT,
    SEED,
    YEARS,
    calendar_block_bootstrap_ci,
    jja_calendar,
    load_nights,
    rank_partial_rho,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs_robust"
METRIC = OUT / "v3_17_dusk_metric_by_night.csv"
DEST = OUT / "v3_23_temporal_sensitivity.csv"
PRIMARY_COLUMN = "dusk_sd__dynamic_all__published_corrected_values__interval_midpoint"
BLOCK_SENSITIVITY_SEED = 20240914


def analysis_frame() -> pd.DataFrame:
    nights = load_nights().rename(columns={"city_mean_tmin_c": "warmth"})
    metric = pd.read_csv(METRIC, parse_dates=["night_date"])
    frame = jja_calendar().merge(nights, on=["night_date", "year"], how="left")
    frame = frame.merge(metric[["night_date", PRIMARY_COLUMN]], on="night_date", how="left")
    frame["calm_clear"] = frame.regime.eq("calm_clear")
    frame["dusk_sd_c"] = frame[PRIMARY_COLUMN]
    return frame


def main() -> None:
    frame = analysis_frame()
    selected = frame[
        frame.year.isin(EVAL_YEARS)
        & frame.calm_clear
        & frame.warmth.notna()
        & frame.dusk_sd_c.notna()
    ]
    rho = rank_partial_rho(selected, adjusted=False)
    rows: list[dict] = []

    for block_days in (7, 14, 21, 28):
        seed = SEED if block_days == 14 else BLOCK_SENSITIVITY_SEED
        lo, hi, valid = calendar_block_bootstrap_ci(
            frame,
            EVAL_YEARS,
            adjusted=False,
            n_boot=N_BOOT,
            block_days=block_days,
            seed=seed,
        )
        rows.append(
            {
                "analysis": "block_length_sensitivity",
                "years": "2024,2025",
                "block_length_calendar_days": block_days,
                "rho": rho,
                "ci_low": lo,
                "ci_high": hi,
                "n_nights": len(selected),
                "n_bootstrap_valid": valid,
                "seed": seed,
            }
        )

    for year in EVAL_YEARS:
        year_selected = frame[
            frame.year.eq(year)
            & frame.calm_clear
            & frame.warmth.notna()
            & frame.dusk_sd_c.notna()
        ]
        year_seed = SEED + 500 + YEARS.index(year)
        lo, hi, valid = calendar_block_bootstrap_ci(
            frame,
            (year,),
            adjusted=False,
            n_boot=N_BOOT,
            block_days=14,
            seed=year_seed,
        )
        rows.append(
            {
                "analysis": "single_summer",
                "years": str(year),
                "block_length_calendar_days": 14,
                "rho": rank_partial_rho(year_selected, adjusted=False),
                "ci_low": lo,
                "ci_high": hi,
                "n_nights": len(year_selected),
                "n_bootstrap_valid": valid,
                "seed": year_seed,
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(DEST, index=False)
    print(result.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"wrote {DEST}")


if __name__ == "__main__":
    main()
