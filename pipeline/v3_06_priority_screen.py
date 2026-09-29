"""
V3 STEP 6 -- Exploratory monitoring-priority screen.

This is a decision-sensitivity analysis, not a health-risk model. It combines
ranked heat, older-resident, cooling-deficit, and urban-intensity indicators.
The weights and top-share cut-offs are
explicit planning assumptions; they are not estimated from health outcomes.

The screening analysis uses the following design:

* A coverage-adjusted single GP mean rebuilt by v3_18 is the primary heat
  field, with IDW from the same coverage-adjusted station targets as the
  field-definition sensitivity.
* The four score components use one indicator each: adjusted mean summer
  night minimum, residents aged 65 years or older, inverse canopy fraction,
  and building-footprint fraction. This prevents nested age groups or closely
  related urban-form variables from receiving hidden additional weight.
* Marginal uncertainty for the primary single field is propagated directly.
  The resulting quantity is an uncertainty selection frequency under
  independent cell-level perturbations, not a posterior probability. Spatial
  cross-cell error covariance is unavailable and is not simulated.

The decision multiverse varies:

* two heat-field definitions, five stated weight scenarios plus four
  leave-one-component-out scenarios;
* three population lenses (official quartier allocation, observed lower bound,
  and all residents); and
* fixed monitoring shares of 5%, 10%, 15%, and 20%.

Outputs (outputs_v3/):
  v3_priority_screen.csv
      Backwards-compatible main cell table. ``p_*`` columns are retained only
      as legacy aliases of the rho=0 uncertainty selection frequencies.
  v3_priority_tier_summary.csv
      Descriptive tier table using the baseline specification.
  v3_screen_gp_vs_idw_comparison.csv
      Field-choice sensitivity.
  v3_refuge_decircularised.csv
      Descriptive refuge-access sensitivity.
  v3_priority_heat_outcome_correlations.csv
      Empirical cross-cell Spearman correlations among the heat point fields.
  v3_priority_uncertainty_sensitivity.csv
      Cell-level uncertainty selection frequencies by assumed error correlation.
  v3_priority_uncertainty_sensitivity_summary.csv
      Counts and retention summaries by rho and monitoring share.
  v3_priority_rebuilt_field_uncertainty.csv and *_summary.csv
      Primary single-field marginal sensitivity (cross-cell covariance absent).
  v3_priority_decision_multiverse_cell.csv
      Cell-level stability frequencies over decision specifications.
  v3_priority_decision_multiverse_summary.csv
      Stability counts over decision specifications.
  v3_priority_decision_specifications.csv
      Results and explicit weights for every decision specification.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd
from scipy import stats


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs_v3")
os.makedirs(OUT, exist_ok=True)
SRC = os.path.join(ROOT, "data", "inputs", "citywide_grid_priority_screen.csv")
SPATIAL_REBUILD = os.path.join(ROOT, "outputs_robust", "v3_spatial_rebuild_grid.csv")

SEED = 20260721
N_MC = 2000
RHO_SCENARIOS = (0.0, 0.5, 0.9, 0.95, 1.0)
TOP_SHARES = (0.05, 0.10, 0.15, 0.20)
COMPONENT_NAMES = ("heat", "elderly", "cooling", "intensity")


def normalise_weights(weights: dict[str, float]) -> dict[str, float]:
    """Return non-negative weights scaled to one, retaining explicit zeroes."""
    if set(weights) != set(COMPONENT_NAMES):
        raise ValueError(f"weights must contain exactly {COMPONENT_NAMES}")
    if any(v < 0 for v in weights.values()):
        raise ValueError("weights must be non-negative")
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("at least one weight must be positive")
    return {k: float(weights[k] / total) for k in COMPONENT_NAMES}


# Baseline is the four-component screen. Each domain is represented by one
# observable quantity, which prevents correlated indicators from receiving
# hidden additional weight within a component.
W_MAIN = normalise_weights({
    "heat": 0.40,
    "elderly": 0.30,
    "cooling": 0.20,
    "intensity": 0.10,
})

# Decircularised diagnostic used only for the refuge-distance description.
W_DECIRC = normalise_weights({
    "heat": 0.35,
    "elderly": 0.25,
    "cooling": 0.0,
    "intensity": 0.10,
})

WEIGHT_SCENARIOS: dict[str, dict[str, float]] = {
    "baseline": W_MAIN,
    "equal_components": normalise_weights({k: 1.0 for k in COMPONENT_NAMES}),
    "heat_focused": normalise_weights({
        "heat": 0.50, "elderly": 0.20, "cooling": 0.15,
        "intensity": 0.15,
    }),
    "older_resident_focused": normalise_weights({
        "heat": 0.20, "elderly": 0.50, "cooling": 0.15,
        "intensity": 0.15,
    }),
    "cooling_focused": normalise_weights({
        "heat": 0.20, "elderly": 0.20, "cooling": 0.45,
        "intensity": 0.15,
    }),
}

# Leave-one-out scenarios start from equal weights, making each omission a
# symmetric diagnostic instead of inheriting the normative baseline weights.
for omitted in COMPONENT_NAMES:
    WEIGHT_SCENARIOS[f"leave_out_{omitted}"] = normalise_weights({
        k: float(k != omitted) for k in COMPONENT_NAMES
    })

POPULATION_LENSES: dict[str, tuple[str, str]] = {
    "qmargin": ("est65_qmargin", "est80_qmargin"),
    "lower": ("est65_lower", "est80_lower"),
    # This is deliberately a population-exposure lens rather than a claim that
    # every resident has the age-related susceptibility represented above.
    "all_residents": ("pers_n", "pers_n"),
}


def pct_rank(s: pd.Series, higher_is_worse: bool = True) -> pd.Series:
    """Average-tie percentile rank, preserving missing values."""
    r = s.rank(pct=True, na_option="keep")
    return r if higher_is_worse else 1.0 - r


def components(
    df: pd.DataFrame,
    heat_prefix: str,
    population_columns: tuple[str, str],
    heat_override: np.ndarray | pd.Series | None = None,
) -> pd.DataFrame:
    """Build dimensionless rank components for one population lens."""
    c = pd.DataFrame(index=df.index)
    if heat_override is not None:
        c["heat"] = np.asarray(heat_override)
    elif heat_prefix == "gp":
        c["heat"] = (
            pct_rank(df["gp_mean_night_min_c"])
            + pct_rank(df["gp_pct_tropical_nights"])
            + pct_rank(df["gp_hottest10_mean_night_min_c"])
        ) / 3
    elif heat_prefix == "idw":
        c["heat"] = (
            pct_rank(df["idw_mean_night_min_c"])
            + pct_rank(df["idw_pct_tropical_nights"])
            + pct_rank(df["idw_hottest10_mean_night_min_c"])
        ) / 3
    elif heat_prefix == "rebuilt_gp":
        # Primary, non-redundant heat component.  This field is rebuilt from
        # coverage-adjusted station targets by v3_18 and is independent of the
        # three highly correlated terminal heat summaries below.
        c["heat"] = pct_rank(df["gp_adjusted_mean_night_min_c"])
    elif heat_prefix == "rebuilt_idw":
        c["heat"] = pct_rank(df["idw2_adjusted_mean_night_min_c"])
    else:
        raise ValueError(
            "heat_prefix must be 'gp', 'idw', 'rebuilt_gp', or 'rebuilt_idw'"
        )

    age65_col, _age80_col = population_columns
    # The 65+ estimate already contains the 80+ population. Using both ranks
    # would implicitly give additional weight to the oldest residents.
    c["elderly"] = pct_rank(df[age65_col])
    c["cooling"] = pct_rank(
        df["canopy_cover_250m_direct"], higher_is_worse=False
    )
    c["intensity"] = pct_rank(df["bldg_footprint_frac_250m_direct"])
    return c


def score(c: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Weighted monitoring-priority score on a nominal 0--100 scale."""
    values = sum(c[k] * weights[k] for k in COMPONENT_NAMES)
    return 100.0 * values


def tiers(s: pd.Series) -> pd.Series:
    """Legacy descriptive tiers used by figures and tables."""
    q50, q75, q90 = s.quantile([0.50, 0.75, 0.90])
    return pd.Series(
        np.select(
            [s >= q90, s >= q75, s >= q50],
            ["highest_priority", "high_priority", "elevated_monitoring"],
            default="baseline_monitoring",
        ),
        index=s.index,
    )


def top_k_mask(values: np.ndarray | pd.Series, share: float) -> np.ndarray:
    """Select exactly ceil(share*n) cells, resolving ties by stable row order."""
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("top-k scores contain missing or non-finite values")
    n_select = int(math.ceil(share * len(values)))
    order = np.argsort(-values, kind="mergesort")
    selected = np.zeros(len(values), dtype=bool)
    selected[order[:n_select]] = True
    return selected


def rho_label(rho: float) -> str:
    return f"{int(round(100 * rho)):03d}"


def rank_heat(draw: np.ndarray) -> np.ndarray:
    """Average the percentile ranks of the three simulated heat outcomes."""
    n = draw.shape[0]
    ranked = np.column_stack([
        stats.rankdata(draw[:, j], method="average") / n for j in range(draw.shape[1])
    ])
    return ranked.mean(axis=1)


def empirical_heat_correlations(df: pd.DataFrame) -> pd.DataFrame:
    labels = {
        "gp_mean_night_min_c": "mean_night_min",
        "gp_pct_tropical_nights": "pct_tropical_nights",
        "gp_hottest10_mean_night_min_c": "hottest10_mean_night_min",
    }
    matrix = df[list(labels)].rename(columns=labels).corr(method="spearman")
    rows = []
    for i, outcome_a in enumerate(matrix.columns):
        for outcome_b in matrix.columns[i + 1:]:
            rows.append({
                "outcome_a": outcome_a,
                "outcome_b": outcome_b,
                "spearman_rho_across_cells": matrix.loc[outcome_a, outcome_b],
                "interpretation": (
                    "correlation of point fields; motivates but does not identify "
                    "predictive-error correlation"
                ),
            })
    return pd.DataFrame(rows)


def simulate_uncertainty(
    df: pd.DataFrame,
    c_baseline: pd.DataFrame,
) -> tuple[dict[tuple[float, float], np.ndarray], pd.DataFrame, pd.DataFrame]:
    """Propagate marginal SDs under stated cross-outcome rho scenarios."""
    mu = df[[
        "gp_mean_night_min_c",
        "gp_pct_tropical_nights",
        "gp_hottest10_mean_night_min_c",
    ]].to_numpy(dtype=float)
    sd = df[[
        "gp_mean_night_min_c_sd",
        "gp_pct_tropical_nights_sd",
        "gp_hottest10_mean_night_min_c_sd",
    ]].to_numpy(dtype=float)
    sd = np.nan_to_num(sd, nan=0.0)
    if not np.isfinite(mu).all() or not np.isfinite(sd).all() or (sd < 0).any():
        raise ValueError("invalid GP means or predictive SDs")

    # These non-heat components are fixed within this sensitivity analysis.
    other = {
        k: c_baseline[k].to_numpy(dtype=float)
        for k in ("elderly", "cooling", "intensity")
    }
    frequencies: dict[tuple[float, float], np.ndarray] = {}
    long_frames = []
    summary_rows = []
    nominal_score = score(c_baseline, W_MAIN).to_numpy()

    for rho in RHO_SCENARIOS:
        # Resetting the seed applies common random numbers across rho scenarios,
        # reducing Monte-Carlo noise in comparisons between scenarios.
        rng = np.random.default_rng(SEED)
        counts = {share: np.zeros(len(df), dtype=np.int32) for share in TOP_SHARES}
        sqrt_shared = math.sqrt(rho)
        sqrt_independent = math.sqrt(1.0 - rho)

        for _ in range(N_MC):
            shared = rng.standard_normal((len(df), 1))
            independent = rng.standard_normal(mu.shape)
            z = sqrt_shared * shared + sqrt_independent * independent
            draw = mu + sd * z
            draw[:, 1] = np.clip(draw[:, 1], 0.0, 100.0)
            heat = rank_heat(draw)
            simulated_score = 100.0 * (
                W_MAIN["heat"] * heat
                + W_MAIN["elderly"] * other["elderly"]
                + W_MAIN["cooling"] * other["cooling"]
                + W_MAIN["intensity"] * other["intensity"]
            )
            for share in TOP_SHARES:
                counts[share] += top_k_mask(simulated_score, share)

        cell = pd.DataFrame({"recordid": df["recordid"], "assumed_error_rho": rho})
        for share in TOP_SHARES:
            freq = counts[share] / N_MC
            frequencies[(rho, share)] = freq
            pct = int(round(100 * share))
            cell[f"uncertainty_selection_frequency_top{pct:02d}"] = freq

            nominal = top_k_mask(nominal_score, share)
            nominal_freq = freq[nominal]
            summary_rows.append({
                "assumed_error_rho": rho,
                "top_percent": pct,
                "n_mc": N_MC,
                "n_cells": len(df),
                "n_nominal_selected": int(nominal.sum()),
                "expected_selected_per_draw": float(freq.sum()),
                "n_nominal_frequency_ge_0_80": int((nominal_freq >= 0.80).sum()),
                "n_nominal_frequency_ge_0_90": int((nominal_freq >= 0.90).sum()),
                "median_frequency_nominal": float(np.median(nominal_freq)),
                "n_all_cells_frequency_ge_0_80": int((freq >= 0.80).sum()),
                "n_all_cells_frequency_ge_0_90": int((freq >= 0.90).sum()),
                "quantity": "uncertainty selection frequency under stated sensitivity model",
                "within_cell_error_model": "equicorrelated Gaussian standardized errors",
                "spatial_cross_cell_covariance": "not represented (unavailable)",
            })
        long_frames.append(cell)

    return frequencies, pd.concat(long_frames, ignore_index=True), pd.DataFrame(summary_rows)


def simulate_rebuilt_field_uncertainty(
    df: pd.DataFrame, c_baseline: pd.DataFrame
) -> tuple[dict[float, np.ndarray], pd.DataFrame, pd.DataFrame]:
    """Marginal uncertainty for the primary single, rebuilt heat field.

    This removes the unidentified correlation among three redundant heat
    outcomes.  It still cannot represent posterior covariance between grid
    cells because that covariance is not retained in the portable grid output;
    frequencies are therefore sensitivity diagnostics, not probabilities.
    """
    mu = df["gp_adjusted_mean_night_min_c"].to_numpy(float)
    sd = df["gp_adjusted_mean_night_min_c_sd"].to_numpy(float)
    if not np.isfinite(mu).all() or not np.isfinite(sd).all() or (sd < 0).any():
        raise ValueError("invalid rebuilt GP mean or predictive SD")
    rng = np.random.default_rng(SEED + 1800)
    counts = {share: np.zeros(len(df), dtype=np.int32) for share in TOP_SHARES}
    other = {k: c_baseline[k].to_numpy(float)
             for k in ("elderly", "cooling", "intensity")}
    nominal_score = score(c_baseline, W_MAIN).to_numpy()
    for _ in range(N_MC):
        draw = mu + sd * rng.standard_normal(len(df))
        heat = stats.rankdata(draw, method="average") / len(df)
        simulated_score = 100.0 * (
            W_MAIN["heat"] * heat
            + W_MAIN["elderly"] * other["elderly"]
            + W_MAIN["cooling"] * other["cooling"]
            + W_MAIN["intensity"] * other["intensity"]
        )
        for share in TOP_SHARES:
            counts[share] += top_k_mask(simulated_score, share)

    frequencies = {share: counts[share] / N_MC for share in TOP_SHARES}
    cell = pd.DataFrame({"recordid": df["recordid"]})
    summary_rows = []
    for share in TOP_SHARES:
        pct = int(round(100 * share))
        freq = frequencies[share]
        nominal = top_k_mask(nominal_score, share)
        nominal_freq = freq[nominal]
        cell[f"uncertainty_selection_frequency_top{pct:02d}"] = freq
        summary_rows.append({
            "heat_field": "coverage-adjusted single GP mean",
            "top_percent": pct,
            "n_mc": N_MC,
            "n_cells": len(df),
            "n_nominal_selected": int(nominal.sum()),
            "n_nominal_frequency_ge_0_80": int((nominal_freq >= .80).sum()),
            "n_nominal_frequency_ge_0_90": int((nominal_freq >= .90).sum()),
            "median_frequency_nominal": float(np.median(nominal_freq)),
            "quantity": "uncertainty selection frequency under marginal field sensitivity",
            "spatial_cross_cell_covariance": "not represented (unavailable)",
        })
    return frequencies, cell, pd.DataFrame(summary_rows)


def decision_multiverse(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate heat fields, weights, population lenses, and top shares."""
    heat_fields = {
        "rebuilt_adjusted_gp": "rebuilt_gp",
        "rebuilt_adjusted_idw": "rebuilt_idw",
    }
    primary_field = "rebuilt_adjusted_gp"
    components_by_design = {
        (field_name, lens): components(df, prefix, columns)
        for field_name, prefix in heat_fields.items()
        for lens, columns in POPULATION_LENSES.items()
    }
    baseline_masks = {
        share: top_k_mask(
            score(components_by_design[(primary_field, "qmargin")], W_MAIN), share
        )
        for share in TOP_SHARES
    }

    all_counts = {share: np.zeros(len(df), dtype=np.int32) for share in TOP_SHARES}
    weight_counts = {share: np.zeros(len(df), dtype=np.int32) for share in TOP_SHARES}
    lens_counts = {share: np.zeros(len(df), dtype=np.int32) for share in TOP_SHARES}
    field_counts = {share: np.zeros(len(df), dtype=np.int32) for share in TOP_SHARES}
    loo_counts = {share: np.zeros(len(df), dtype=np.int32) for share in TOP_SHARES}
    config_rows = []

    for field_name in heat_fields:
        for scenario, weights in WEIGHT_SCENARIOS.items():
            family = ("leave_one_component_out" if scenario.startswith("leave_out_")
                      else "stated_weights")
            for lens in POPULATION_LENSES:
                c = components_by_design[(field_name, lens)]
                scenario_score = score(c, weights).to_numpy()
                for share in TOP_SHARES:
                    mask = top_k_mask(scenario_score, share)
                    baseline = baseline_masks[share]
                    union = int(np.logical_or(mask, baseline).sum())
                    overlap = int(np.logical_and(mask, baseline).sum())
                    all_counts[share] += mask
                    if field_name == primary_field and lens == "qmargin":
                        weight_counts[share] += mask
                        if family == "leave_one_component_out":
                            loo_counts[share] += mask
                    if field_name == primary_field and scenario == "baseline":
                        lens_counts[share] += mask
                    if scenario == "baseline" and lens == "qmargin":
                        field_counts[share] += mask

                    row = {
                        "scenario_family": family,
                        "heat_field": field_name,
                        "weight_scenario": scenario,
                        "population_lens": lens,
                        "top_percent": int(round(100 * share)),
                        "n_selected": int(mask.sum()),
                        "overlap_with_baseline_qmargin": overlap,
                        "jaccard_with_baseline_qmargin": overlap / union,
                        "selected_population": float(df.loc[mask, "pers_n"].sum()),
                        "selected_age65_qmargin": float(
                            df.loc[mask, "est65_qmargin"].sum()),
                    }
                    row.update({f"weight_{k}": weights[k]
                                for k in COMPONENT_NAMES})
                    config_rows.append(row)

    n_all = len(heat_fields) * len(WEIGHT_SCENARIOS) * len(POPULATION_LENSES)
    n_weights = len(WEIGHT_SCENARIOS)
    n_lenses = len(POPULATION_LENSES)
    n_fields = len(heat_fields)
    n_loo = sum(name.startswith("leave_out_") for name in WEIGHT_SCENARIOS)

    cell = df[[
        "recordid", "centroid_easting_2056", "centroid_northing_2056", "pers_n"
    ]].copy()
    summary_rows = []
    for share in TOP_SHARES:
        pct = int(round(100 * share))
        freq_all = all_counts[share] / n_all
        freq_weights = weight_counts[share] / n_weights
        freq_lenses = lens_counts[share] / n_lenses
        freq_fields = field_counts[share] / n_fields
        freq_loo = loo_counts[share] / n_loo
        nominal = baseline_masks[share]

        cell[f"baseline_qmargin_top{pct:02d}"] = nominal.astype(int)
        cell[f"decision_frequency_top{pct:02d}_all"] = freq_all
        cell[f"weight_frequency_top{pct:02d}_qmargin"] = freq_weights
        cell[f"population_lens_frequency_top{pct:02d}_baseline"] = freq_lenses
        cell[f"heat_field_frequency_top{pct:02d}_baseline"] = freq_fields
        cell[f"leave_one_out_frequency_top{pct:02d}_qmargin"] = freq_loo

        for scope, frequency, n_specs in (
            ("all_heat_fields_x_weights_x_population_lenses", freq_all, n_all),
            ("weights_at_qmargin", freq_weights, n_weights),
            ("population_lenses_at_baseline_weights", freq_lenses, n_lenses),
            ("heat_fields_at_baseline_decision", freq_fields, n_fields),
            ("leave_one_out_at_qmargin", freq_loo, n_loo),
        ):
            nominal_frequency = frequency[nominal]
            summary_rows.append({
                "scope": scope,
                "top_percent": pct,
                "n_specifications": n_specs,
                "n_cells": len(df),
                "n_baseline_selected": int(nominal.sum()),
                "n_all_cells_frequency_ge_0_80": int((frequency >= 0.80).sum()),
                "n_all_cells_frequency_ge_0_90": int((frequency >= 0.90).sum()),
                "n_all_cells_in_every_specification": int(np.isclose(frequency, 1.0).sum()),
                "n_baseline_frequency_ge_0_80": int((nominal_frequency >= 0.80).sum()),
                "n_baseline_frequency_ge_0_90": int((nominal_frequency >= 0.90).sum()),
                "median_frequency_baseline": float(np.median(nominal_frequency)),
                "quantity": "selection frequency across explicit decision specifications",
            })

    return cell, pd.DataFrame(summary_rows), pd.DataFrame(config_rows)


def main() -> None:
    df = pd.read_csv(SRC)
    rebuilt = pd.read_csv(
        SPATIAL_REBUILD,
        usecols=[
            "recordid", "gp_adjusted_mean_night_min_c",
            "gp_adjusted_mean_night_min_c_sd",
            "idw2_adjusted_mean_night_min_c",
        ],
    )
    df = df.merge(rebuilt, on="recordid", how="left", validate="one_to_one")
    if df[["gp_adjusted_mean_night_min_c", "gp_adjusted_mean_night_min_c_sd",
           "idw2_adjusted_mean_night_min_c"]].isna().any().any():
        raise ValueError("rebuilt spatial heat fields do not cover every screen cell")
    print(f"cells: {len(df)}   (heat-supported evaluation grid)")

    # ---- primary screen: coverage-adjusted, reproducibly rebuilt GP -----
    c_gp = components(df, "rebuilt_gp", POPULATION_LENSES["qmargin"])
    df["score_gp"] = score(c_gp, W_MAIN)
    df["tier_gp"] = tiers(df["score_gp"])
    for component_name in COMPONENT_NAMES:
        df[f"{component_name}_score"] = c_gp[component_name]

    # ---- same adjusted targets, transparent IDW field sensitivity -------
    c_idw = components(df, "rebuilt_idw", POPULATION_LENSES["qmargin"])
    df["score_idw"] = score(c_idw, W_MAIN)
    df["tier_idw"] = tiers(df["score_idw"])

    # The legacy three-outcome terminal composite is retained as an explicit
    # field-definition sensitivity rather than being averaged into the primary
    # heat component.
    c_terminal = components(df, "gp", POPULATION_LENSES["qmargin"])
    df["score_terminal_composite"] = score(c_terminal, W_MAIN)
    df["tier_terminal_composite"] = tiers(df["score_terminal_composite"])

    rho_field = stats.spearmanr(df["score_gp"], df["score_idw"]).statistic
    hi_gp = set(df.index[df["tier_gp"] == "highest_priority"])
    hi_idw = set(df.index[df["tier_idw"] == "highest_priority"])
    jaccard = len(hi_gp & hi_idw) / len(hi_gp | hi_idw)
    print(
        f"\nGP vs IDW screen: score Spearman {rho_field:.3f}; "
        f"highest-tier Jaccard {jaccard:.3f}; "
        f"{len(hi_gp & hi_idw)}/{len(hi_gp)} GP highest-tier cells also IDW highest-tier"
    )
    pd.DataFrame([{
        "score_spearman": rho_field,
        "highest_tier_jaccard": jaccard,
        "n_highest_gp": len(hi_gp),
        "n_highest_idw": len(hi_idw),
        "n_overlap": len(hi_gp & hi_idw),
        "gp_field": "coverage-adjusted rebuilt GP mean",
        "idw_field": "IDW2 from the same adjusted station targets",
    }]).to_csv(os.path.join(OUT, "v3_screen_gp_vs_idw_comparison.csv"), index=False)

    empirical_corr = empirical_heat_correlations(df)
    empirical_corr.to_csv(
        os.path.join(OUT, "v3_priority_heat_outcome_correlations.csv"), index=False
    )
    print("\nempirical Spearman correlations among GP heat point fields:")
    print(empirical_corr[["outcome_a", "outcome_b", "spearman_rho_across_cells"]].to_string(index=False))

    # ---- GP uncertainty with explicit cross-outcome rho sensitivity -----
    print(
        f"\npropagating marginal GP uncertainty ({N_MC} draws per rho; "
        f"rho={RHO_SCENARIOS}) ..."
    )
    frequencies, uncertainty_cell, uncertainty_summary = simulate_uncertainty(
        df, c_terminal)
    uncertainty_cell.to_csv(
        os.path.join(OUT, "v3_priority_uncertainty_sensitivity.csv"), index=False
    )
    uncertainty_summary.to_csv(
        os.path.join(OUT, "v3_priority_uncertainty_sensitivity_summary.csv"), index=False
    )

    # Primary marginal sensitivity for the single rebuilt field. This removes
    # the redundant-outcome correlation assumption but still lacks spatial
    # cross-cell covariance.
    rebuilt_freq, rebuilt_uncertainty_cell, rebuilt_uncertainty_summary = (
        simulate_rebuilt_field_uncertainty(df, c_gp)
    )
    rebuilt_uncertainty_cell.to_csv(
        os.path.join(OUT, "v3_priority_rebuilt_field_uncertainty.csv"), index=False)
    rebuilt_uncertainty_summary.to_csv(
        os.path.join(OUT, "v3_priority_rebuilt_field_uncertainty_summary.csv"),
        index=False,
    )

    # Explicit names. Legacy p_* names are aliases solely for old downstream
    # consumers and must not be interpreted as posterior probabilities.
    for rho_assumed in RHO_SCENARIOS:
        label = rho_label(rho_assumed)
        df[f"uncertainty_selection_frequency_top10_rho_{label}"] = (
            frequencies[(rho_assumed, 0.10)]
        )
    df["uncertainty_selection_frequency_top10_rebuilt_field"] = rebuilt_freq[0.10]
    df["p_highest_tier"] = df["uncertainty_selection_frequency_top10_rebuilt_field"]
    df["p_top_decile_score"] = df["uncertainty_selection_frequency_top10_rebuilt_field"]

    print("\n10% uncertainty-selection sensitivity among nominal baseline cells:")
    print(
        uncertainty_summary.loc[
            uncertainty_summary["top_percent"] == 10,
            [
                "assumed_error_rho",
                "n_nominal_frequency_ge_0_80",
                "n_nominal_frequency_ge_0_90",
                "median_frequency_nominal",
            ],
        ].to_string(index=False)
    )
    print("\nprimary rebuilt-field marginal sensitivity:")
    print(rebuilt_uncertainty_summary.to_string(index=False))

    # ---- decision multiverse -------------------------------------------
    multiverse_cell, multiverse_summary, multiverse_specs = decision_multiverse(df)
    multiverse_cell.to_csv(
        os.path.join(OUT, "v3_priority_decision_multiverse_cell.csv"), index=False
    )
    multiverse_summary.to_csv(
        os.path.join(OUT, "v3_priority_decision_multiverse_summary.csv"), index=False
    )
    multiverse_specs.to_csv(
        os.path.join(OUT, "v3_priority_decision_specifications.csv"), index=False
    )
    print("\ndecision multiverse, 10% monitoring share:")
    print(
        multiverse_summary.loc[
            multiverse_summary["top_percent"] == 10,
            [
                "scope",
                "n_specifications",
                "n_baseline_frequency_ge_0_80",
                "n_baseline_frequency_ge_0_90",
                "median_frequency_baseline",
            ],
        ].to_string(index=False)
    )

    # ---- baseline descriptive tier summary -----------------------------
    summary = (
        df.groupby("tier_gp")
        .agg(
            n_cells=("recordid", "size"),
            population=("pers_n", "sum"),
            age65_official=("est65_qmargin", "sum"),
            age80_official=("est80_qmargin", "sum"),
            age65_observed_lower=("est65_lower", "sum"),
            mean_gp_tmin_c=("gp_adjusted_mean_night_min_c", "mean"),
            mean_canopy=("canopy_cover_250m_direct", "mean"),
            median_refuge_m=("routed_refuge_distance_m", "median"),
            mean_uncertainty_selection_frequency_rebuilt=(
                "uncertainty_selection_frequency_top10_rebuilt_field", "mean"
            ),
        )
        .reindex([
            "highest_priority",
            "high_priority",
            "elevated_monitoring",
            "baseline_monitoring",
        ])
        .round(2)
        .reset_index()
    )
    # Backwards-compatible alias for old readers; see explicit field above.
    summary["mean_p_highest"] = summary[
        "mean_uncertainty_selection_frequency_rebuilt"
    ]
    summary.to_csv(os.path.join(OUT, "v3_priority_tier_summary.csv"), index=False)
    print("\n=== tier summary (baseline GP screen; official quartier margins) ===")
    print(summary.to_string(index=False))

    # ---- refuge distance after removing its overlapping cooling signal --
    print("\n=== refuge access: circular vs cooling-removed score ===")
    df["score_decirc"] = score(c_gp, W_DECIRC)
    df["tier_decirc"] = tiers(df["score_decirc"])
    refuge_rows = []
    for label, tier_col in (
        ("with cooling component (circular)", "tier_gp"),
        ("cooling component REMOVED", "tier_decirc"),
    ):
        grouped = (
            df.groupby(tier_col)
            .agg(
                n_cells=("recordid", "size"),
                population=("pers_n", "sum"),
                median_routed_m=("routed_refuge_distance_m", "median"),
            )
            .reindex([
                "baseline_monitoring",
                "elevated_monitoring",
                "high_priority",
                "highest_priority",
            ])
            .reset_index()
        )
        grouped.columns = ["tier", "n_cells", "population", "median_routed_m"]
        grouped.insert(0, "specification", label)
        refuge_rows.append(grouped)
        low = grouped["median_routed_m"].iloc[0]
        high = grouped["median_routed_m"].iloc[-1]
        print(f"  {label:36s} {low:6.0f} -> {high:6.0f} m   (gradient {high - low:+.0f} m)")
    pd.concat(refuge_rows, ignore_index=True).to_csv(
        os.path.join(OUT, "v3_refuge_decircularised.csv"), index=False
    )

    keep = [
        "recordid",
        "centroid_easting_2056",
        "centroid_northing_2056",
        "pers_n",
        "est65_qmargin",
        "est80_qmargin",
        "est65_lower",
        "gp_mean_night_min_c",
        "gp_mean_night_min_c_sd",
        "gp_pct_tropical_nights",
        "idw_mean_night_min_c",
        "gp_adjusted_mean_night_min_c",
        "gp_adjusted_mean_night_min_c_sd",
        "idw2_adjusted_mean_night_min_c",
        "canopy_cover_250m_direct",
        "bldg_footprint_frac_250m_direct",
        "noise_night_mean_direct",
        "routed_refuge_distance_m",
        "euclid_refuge_distance_m",
        "heat_score",
        "elderly_score",
        "cooling_score",
        "intensity_score",
        "score_gp",
        "tier_gp",
        "score_idw",
        "tier_idw",
        "score_terminal_composite",
        "tier_terminal_composite",
        "score_decirc",
        "tier_decirc",
        "uncertainty_selection_frequency_top10_rebuilt_field",
        *[
            f"uncertainty_selection_frequency_top10_rho_{rho_label(rho)}"
            for rho in RHO_SCENARIOS
        ],
        "p_highest_tier",
        "p_top_decile_score",
    ]
    df[keep].to_csv(os.path.join(OUT, "v3_priority_screen.csv"), index=False)
    print(f"\nwrote planning-screen outputs -> {OUT}")
    print(
        "Interpretation: exploratory monitoring-priority selection under explicit "
        "assumptions; not posterior probability, health risk, or causal effect."
    )


if __name__ == "__main__":
    main()
