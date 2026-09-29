"""
V3 STEP 18 -- Rebuild and validate the observed night-heat spatial field.

This script closes the reproducibility boundary on the observed spatial layer.
It starts from the station-night panel produced by ``v3_01_night_metrics.py``;
no legacy interpolation result is used to fit the new surface.

Scientific design
-----------------
1. Restrict the station panel to the 546 analysis nights retained by the
   city-night coverage rule (JJA 2020--2025).
2. Estimate an additive station + night fixed-effect model.  The station
   effects therefore compare stations on the same nights and are not biased by
   the fact that station coverage ranges from roughly two to six summers.
   Add the equally weighted mean of the fitted night effects to each station
   effect to obtain a coverage-adjusted climatological night-minimum target.
3. Estimate target uncertainty from between-summer variation in the two-way
   fixed-effect model.  The between-summer standard error (SE of the
   annual-mean station effect) is used as a heteroskedastic ``alpha`` term in
   the Gaussian process; it represents the estimation uncertainty in each
   station target.  A separately fitted WhiteKernel represents unresolved
   station-scale variation.  Total station-night residual RMSE is not a
   like-for-like replacement for target SE because it describes variation
   around individual night effects rather than uncertainty in the estimated
   station climatology.  Spatial cross-validation evaluates calibration of the
   fitted model as a whole; it does not uniquely validate either noise term.
4. Fit a Matern-5/2 Gaussian process in LV95 kilometres and evaluate it on the
   retained 100 m centroid lattice.  The lattice spacing is an evaluation
   grid, not a claim of 100 m information content.  Kernel correlation ranges
   and nearest-station distances are reported as the defensible spatial-support
   diagnostics.
5. Validate by repeated leave-one-spatial-block-out prediction.  For each of
   20 KMeans partitions (six compact spatial blocks), the GP is refitted inside
   every training fold.  Report out-of-block R2, RMSE, MAE, Spearman rho, and
   nominal 95% predictive-interval coverage (with and without held-out target
   estimation uncertainty).

The terminal grid remains the carrier for population, urban-form, refuge, and
FITNAH columns because their upstream raw layers are not present in the
recovered pipeline.  Those columns are copied unchanged and explicitly marked
as a provenance boundary; they are never used to fit the rebuilt heat field.

Outputs (outputs_robust/)
-------------------------
v3_spatial_rebuild_station_targets.csv
v3_spatial_rebuild_grid.csv
v3_spatial_rebuild_cv_predictions.csv
v3_spatial_rebuild_cv_metrics.csv
v3_spatial_rebuild_comparisons.csv
v3_spatial_rebuild_diagnostics.csv
"""
from __future__ import annotations

import hashlib
import os
import warnings
from dataclasses import dataclass

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy import optimize, stats
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Matern,
    WhiteKernel,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs_robust")
STATION_NIGHT = os.path.join(ROOT, "outputs_v3", "v3_station_night.parquet")
CITY_NIGHT = os.path.join(ROOT, "outputs_v3", "v3_night_city.csv")
LOCATIONS = os.path.join(ROOT, "heat", "meteoblue_station_locations.csv")
TERMINAL_GRID = os.path.join(ROOT, "data", "inputs", "citywide_grid_priority_screen.csv")

YEARS = tuple(range(2020, 2026))
SEED = 20260721
N_BLOCKS = 6
N_CV_REPEATS = 20
FULL_MODEL_RESTARTS = 8
CV_MODEL_RESTARTS = 0

GRID_LABEL = "100 m evaluation grid (not 100 m model resolution)"
BOUNDARY_LABEL = (
    "retained terminal non-heat/FITNAH layers; upstream raw layers absent "
    "from recovered pipeline"
)


@dataclass(frozen=True)
class KernelSummary:
    signal_sd_c: float
    length_scale_km: float
    white_noise_sd_c: float
    corr_010_range_km: float
    corr_005_range_km: float


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def load_analysis_panel() -> pd.DataFrame:
    """Load station-nights on exactly the city-night analysis date set."""
    station = pd.read_parquet(STATION_NIGHT)
    city = pd.read_csv(CITY_NIGHT, parse_dates=["night_date"])
    _require_columns(
        station,
        {"locationID", "night_date", "night_min_c", "year", "month"},
        "station-night panel",
    )
    _require_columns(city, {"night_date", "year", "n_stations"}, "city-night table")

    station["night_date"] = pd.to_datetime(station["night_date"])
    analysis_dates = city.loc[city.year.isin(YEARS), "night_date"].drop_duplicates()
    panel = station[
        station.night_date.isin(analysis_dates)
        & station.year.isin(YEARS)
        & station.month.isin((6, 7, 8))
        & station.night_min_c.notna()
    ].copy()

    if panel.duplicated(["locationID", "night_date"]).any():
        raise ValueError("station-night panel contains duplicate station/date rows")
    if panel.night_date.nunique() != len(analysis_dates):
        raise ValueError("station panel does not contain every retained city-night date")
    panel = panel.sort_values(["night_date", "locationID"]).reset_index(drop=True)
    return panel


def repair_locations(loc: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    """Validate LV95 coordinates and repair malformed entries from WGS84.

    One archived row stores LV95 values at approximately 100 times their proper
    magnitude.  Rather than silently dropping that temperature record, this
    function detects coordinates far outside the grid region and transforms the
    row's supplied longitude/latitude to EPSG:2056.  The source of every final
    coordinate is retained in the station output.
    """
    _require_columns(
        loc,
        {"locationID", "EKoord", "NKoord", "lonDecimal", "latDecimal", "masl"},
        "station locations",
    )
    _require_columns(
        grid,
        {"centroid_easting_2056", "centroid_northing_2056"},
        "terminal grid",
    )
    out = loc.copy()
    out["EKoord_original"] = out["EKoord"]
    out["NKoord_original"] = out["NKoord"]
    out["EKoord"] = out["EKoord"].astype(float)
    out["NKoord"] = out["NKoord"].astype(float)
    out["coordinate_source"] = "provided_lv95"

    margin_m = 30_000.0
    e_ok = out.EKoord.between(
        grid.centroid_easting_2056.min() - margin_m,
        grid.centroid_easting_2056.max() + margin_m,
    )
    n_ok = out.NKoord.between(
        grid.centroid_northing_2056.min() - margin_m,
        grid.centroid_northing_2056.max() + margin_m,
    )
    invalid = ~(e_ok & n_ok)
    if invalid.any():
        if out.loc[invalid, ["lonDecimal", "latDecimal"]].isna().any(axis=None):
            bad = out.loc[invalid, "locationID"].tolist()
            raise ValueError(f"invalid LV95 coordinates and missing WGS84 for {bad}")
        transformer = Transformer.from_crs(4326, 2056, always_xy=True)
        e_new, n_new = transformer.transform(
            out.loc[invalid, "lonDecimal"].to_numpy(float),
            out.loc[invalid, "latDecimal"].to_numpy(float),
        )
        out.loc[invalid, "EKoord"] = e_new
        out.loc[invalid, "NKoord"] = n_new
        out.loc[invalid, "coordinate_source"] = "wgs84_to_lv95_repair"

    # A second, much tighter validity check after repair.
    e_ok = out.EKoord.between(
        grid.centroid_easting_2056.min() - margin_m,
        grid.centroid_easting_2056.max() + margin_m,
    )
    n_ok = out.NKoord.between(
        grid.centroid_northing_2056.min() - margin_m,
        grid.centroid_northing_2056.max() + margin_m,
    )
    if not (e_ok & n_ok).all():
        bad = out.loc[~(e_ok & n_ok), "locationID"].tolist()
        raise ValueError(f"unresolved invalid LV95 coordinates for {bad}")
    return out


def fit_two_way_fixed_effects(
    panel: pd.DataFrame, tolerance: float = 1e-12, max_iter: int = 10_000
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Estimate y(station, night) = station_effect + night_effect + error.

    Alternating projections solve the unbalanced additive least-squares model.
    Station effects are constrained to have an unweighted mean of zero.  Night
    effects then retain the temperature scale; their equally weighted mean is
    the climatological baseline added to every station effect.
    """
    station_levels, station_code = np.unique(panel.locationID, return_inverse=True)
    night_levels, night_code = np.unique(panel.night_date.values, return_inverse=True)
    y = panel.night_min_c.to_numpy(float)
    n_station, n_night = len(station_levels), len(night_levels)
    station_count = np.bincount(station_code, minlength=n_station)
    night_count = np.bincount(night_code, minlength=n_night)

    station_effect = np.zeros(n_station, dtype=float)
    night_effect = np.zeros(n_night, dtype=float)
    converged = False
    for iteration in range(1, max_iter + 1):
        new_night = np.bincount(
            night_code, weights=y - station_effect[station_code], minlength=n_night
        ) / night_count
        new_station = np.bincount(
            station_code, weights=y - new_night[night_code], minlength=n_station
        ) / station_count
        shift = new_station.mean()
        new_station -= shift
        new_night += shift
        delta = max(
            float(np.max(np.abs(new_station - station_effect))),
            float(np.max(np.abs(new_night - night_effect))),
        )
        station_effect, night_effect = new_station, new_night
        if delta < tolerance:
            converged = True
            break
    if not converged:
        raise RuntimeError(f"two-way fixed effects did not converge in {max_iter} iterations")

    fitted = station_effect[station_code] + night_effect[night_code]
    residual = y - fitted
    climatological_baseline = float(night_effect.mean())

    work = panel[["locationID", "night_date", "year", "night_min_c"]].copy()
    work["night_effect_c"] = night_effect[night_code]
    work["station_effect_observation_c"] = y - night_effect[night_code]
    work["residual_c"] = residual

    annual = (
        work.groupby(["locationID", "year"], as_index=False)
        .station_effect_observation_c.mean()
    )
    summer_se = annual.groupby("locationID").station_effect_observation_c.agg(
        lambda values: values.std(ddof=1) / np.sqrt(values.notna().sum())
        if values.notna().sum() >= 2
        else np.nan
    )

    grouped = panel.groupby("locationID")
    residual_rmse = work.groupby("locationID").residual_c.apply(
        lambda values: float(np.sqrt(np.mean(np.square(values))))
    )
    targets = pd.DataFrame(
        {
            "locationID": station_levels,
            "n_nights": grouped.size().reindex(station_levels).to_numpy(),
            "n_summers": grouped.year.nunique().reindex(station_levels).to_numpy(),
            "first_night": grouped.night_date.min().reindex(station_levels).to_numpy(),
            "last_night": grouped.night_date.max().reindex(station_levels).to_numpy(),
            "raw_mean_night_min_c": grouped.night_min_c.mean()
            .reindex(station_levels)
            .to_numpy(),
            "station_effect_c": station_effect,
            "adjusted_mean_night_min_c": climatological_baseline + station_effect,
            "between_summer_se_c": summer_se.reindex(station_levels).to_numpy(),
            "station_residual_rmse_c": residual_rmse.reindex(station_levels).to_numpy(),
        }
    )
    if targets.between_summer_se_c.isna().any():
        # This should not occur in the six-summer archive, but retaining an
        # explicit conservative fallback makes future partial updates safe.
        fallback = float(targets.between_summer_se_c.median())
        targets["between_summer_se_c_imputed"] = targets.between_summer_se_c.isna()
        targets["between_summer_se_c"] = targets.between_summer_se_c.fillna(fallback)
    else:
        targets["between_summer_se_c_imputed"] = False

    diagnostics = {
        "n_station_nights": float(len(panel)),
        "n_stations": float(n_station),
        "n_nights": float(n_night),
        "fixed_effect_iterations": float(iteration),
        "fixed_effect_final_delta_c": float(delta),
        "climatological_baseline_c": climatological_baseline,
        "panel_residual_rmse_c": float(np.sqrt(np.mean(np.square(residual)))),
    }
    return targets, diagnostics


def base_kernel(initial_length_scale_km: float = 2.0):
    """Matern-5/2 covariance with city-scale, physically expressed bounds."""
    return ConstantKernel(1.0, (1e-2, 1e2)) * Matern(
        length_scale=initial_length_scale_km,
        length_scale_bounds=(0.10, 20.0),
        nu=2.5,
    ) + WhiteKernel(noise_level=0.05, noise_level_bounds=(1e-4, 2.0))


def fit_gp(
    x_km: np.ndarray,
    y_c: np.ndarray,
    target_se_c: np.ndarray,
    seed: int,
    restarts: int,
) -> GaussianProcessRegressor:
    """Fit GP with between-summer estimation uncertainty on normalized y scale.

    ``target_se_c`` carries each station's between-summer standard error from
    the two-way fixed-effect model, representing the estimation uncertainty in
    the coverage-adjusted station target.  This matches the uncertainty scale
    of the response supplied to the GP, which is an estimated station
    climatology.  The separately fitted WhiteKernel captures remaining
    station-scale variation.  Total station-night residual RMSE is not a
    like-for-like replacement for target SE because it describes variation
    around individual night effects rather than uncertainty in the estimated
    station climatology.  Spatial CV evaluates the fitted model as a whole; it
    does not uniquely validate either noise component.
    """
    y_sd = float(np.std(y_c, ddof=0))
    if not np.isfinite(y_sd) or y_sd <= 0:
        raise ValueError("station targets have zero or invalid variance")
    alpha = np.maximum(np.square(target_se_c / y_sd), 1e-10)
    model = GaussianProcessRegressor(
        kernel=base_kernel(),
        alpha=alpha,
        normalize_y=True,
        n_restarts_optimizer=restarts,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x_km, y_c)
    return model


def matern52_correlation(distance_over_length: float) -> float:
    x = np.sqrt(5.0) * distance_over_length
    return float((1.0 + x + x * x / 3.0) * np.exp(-x))


def correlation_range(length_scale_km: float, threshold: float) -> float:
    ratio = optimize.brentq(
        lambda value: matern52_correlation(value) - threshold, 0.0, 100.0
    )
    return float(ratio * length_scale_km)


def summarize_kernel(model: GaussianProcessRegressor, y_c: np.ndarray) -> KernelSummary:
    fitted = model.kernel_
    # Expected structure: ConstantKernel * Matern + WhiteKernel.
    constant = float(fitted.k1.k1.constant_value)
    length_scale = float(np.asarray(fitted.k1.k2.length_scale).squeeze())
    white_noise = float(fitted.k2.noise_level)
    y_sd = float(np.std(y_c, ddof=0))
    return KernelSummary(
        signal_sd_c=float(np.sqrt(constant) * y_sd),
        length_scale_km=length_scale,
        white_noise_sd_c=float(np.sqrt(white_noise) * y_sd),
        corr_010_range_km=correlation_range(length_scale, 0.10),
        corr_005_range_km=correlation_range(length_scale, 0.05),
    )


def idw_predict(
    train_xy_m: np.ndarray, train_y: np.ndarray, query_xy_m: np.ndarray, power: float = 2.0
) -> np.ndarray:
    """All-station inverse-distance weighting, retained only as a sensitivity."""
    delta = query_xy_m[:, None, :] - train_xy_m[None, :, :]
    distance = np.sqrt(np.sum(np.square(delta), axis=2))
    exact = distance <= 1e-9
    safe_distance = np.maximum(distance, 1e-9)
    weights = np.power(safe_distance, -power)
    # Some Accelerate/OpenBLAS builds raise floating-point status warnings for
    # otherwise finite matrix products.  Suppress the status flag locally, then
    # enforce finiteness explicitly so a real numerical failure still aborts.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        prediction = (weights @ train_y) / weights.sum(axis=1)
    if exact.any():
        rows = np.flatnonzero(exact.any(axis=1))
        prediction[rows] = train_y[np.argmax(exact[rows], axis=1)]
    if not np.isfinite(prediction).all():
        raise RuntimeError("non-finite IDW prediction")
    return prediction


def partition_signature(labels: np.ndarray, location_ids: np.ndarray) -> str:
    """Hash a label-invariant representation of a spatial partition."""
    groups = []
    for label in np.unique(labels):
        groups.append(",".join(sorted(location_ids[labels == label].astype(str))))
    canonical = "|".join(sorted(groups))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def repeated_spatial_block_cv(targets: pd.DataFrame, origin_m: np.ndarray):
    coords_m = targets[["EKoord", "NKoord"]].to_numpy(float)
    coords_km = (coords_m - origin_m) / 1000.0
    y = targets.adjusted_mean_night_min_c.to_numpy(float)
    se = targets.between_summer_se_c.to_numpy(float)
    location_ids = targets.locationID.to_numpy(str)
    all_predictions: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []

    for repeat in range(N_CV_REPEATS):
        repeat_seed = SEED + repeat
        labels = KMeans(
            n_clusters=N_BLOCKS,
            n_init=1,
            random_state=repeat_seed,
        ).fit_predict(coords_km)
        signature = partition_signature(labels, location_ids)
        prediction = np.full(len(y), np.nan)
        predictive_sd = np.full(len(y), np.nan)

        for fold in range(N_BLOCKS):
            train = labels != fold
            test = labels == fold
            if train.sum() < 20 or test.sum() == 0:
                raise RuntimeError(
                    f"invalid spatial block split: repeat={repeat}, fold={fold}, "
                    f"train={train.sum()}, test={test.sum()}"
                )
            model = fit_gp(
                coords_km[train],
                y[train],
                se[train],
                seed=repeat_seed * 100 + fold,
                restarts=CV_MODEL_RESTARTS,
            )
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                fold_prediction, fold_sd = model.predict(
                    coords_km[test], return_std=True
                )
            prediction[test] = fold_prediction
            predictive_sd[test] = fold_sd

        if not (np.isfinite(prediction) & np.isfinite(predictive_sd)).all():
            raise RuntimeError(f"incomplete predictions in CV repeat {repeat}")

        # For coverage of an estimated held-out station target, add that
        # target's between-summer estimation SE to the GP predictive variance.
        gp_lower = prediction - 1.96 * predictive_sd
        gp_upper = prediction + 1.96 * predictive_sd
        gp_covered = (y >= gp_lower) & (y <= gp_upper)
        validation_sd = np.sqrt(np.square(predictive_sd) + np.square(se))
        validation_lower = prediction - 1.96 * validation_sd
        validation_upper = prediction + 1.96 * validation_sd
        validation_covered = (y >= validation_lower) & (y <= validation_upper)
        rho = float(stats.spearmanr(y, prediction).statistic)
        metric_rows.append(
            {
                "row_type": "repeat",
                "repeat": repeat,
                "seed": repeat_seed,
                "partition_signature": signature,
                "n_stations": len(y),
                "n_blocks": N_BLOCKS,
                "r2": float(r2_score(y, prediction)),
                "rmse_c": float(np.sqrt(mean_squared_error(y, prediction))),
                "mae_c": float(mean_absolute_error(y, prediction)),
                "spearman_rho": rho,
                "gp_coverage95": float(gp_covered.mean()),
                "target_augmented_coverage95": float(validation_covered.mean()),
                "gp_mean_interval_width_c": float(np.mean(gp_upper - gp_lower)),
                "target_augmented_mean_interval_width_c": float(
                    np.mean(validation_upper - validation_lower)
                ),
            }
        )
        all_predictions.append(
            pd.DataFrame(
                {
                    "repeat": repeat,
                    "seed": repeat_seed,
                    "partition_signature": signature,
                    "fold": labels,
                    "locationID": location_ids,
                    "observed_adjusted_target_c": y,
                    "observed_between_summer_se_c": se,
                    "predicted_target_c": prediction,
                    "gp_predictive_sd_c": predictive_sd,
                    "validation_sd_c": validation_sd,
                    "gp_lower95_c": gp_lower,
                    "gp_upper95_c": gp_upper,
                    "gp_covered95": gp_covered,
                    "validation_lower95_c": validation_lower,
                    "validation_upper95_c": validation_upper,
                    "target_augmented_covered95": validation_covered,
                }
            )
        )

    repeat_metrics = pd.DataFrame(metric_rows)
    numeric_metrics = [
        "r2",
        "rmse_c",
        "mae_c",
        "spearman_rho",
        "gp_coverage95",
        "target_augmented_coverage95",
        "gp_mean_interval_width_c",
        "target_augmented_mean_interval_width_c",
    ]
    summary_rows = []
    for label, quantile in (("q025", 0.025), ("median", 0.5), ("q975", 0.975)):
        row: dict[str, object] = {
            "row_type": label,
            "repeat": np.nan,
            "seed": np.nan,
            "partition_signature": "",
            "n_stations": len(y),
            "n_blocks": N_BLOCKS,
        }
        for metric in numeric_metrics:
            row[metric] = float(repeat_metrics[metric].quantile(quantile))
        summary_rows.append(row)
    metrics = pd.concat([repeat_metrics, pd.DataFrame(summary_rows)], ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    return predictions, metrics


def residualize_coordinates(values: np.ndarray, grid: pd.DataFrame) -> np.ndarray:
    """Remove a low-order broad spatial trend before pattern comparison."""
    e = grid.centroid_easting_2056.to_numpy(float)
    n = grid.centroid_northing_2056.to_numpy(float)
    e = (e - e.mean()) / e.std(ddof=0)
    n = (n - n.mean()) / n.std(ddof=0)
    design = np.column_stack([np.ones(len(values)), e, n, e * e, e * n, n * n])
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        residual = values - design @ np.linalg.lstsq(
            design, values, rcond=None
        )[0]
    if not np.isfinite(residual).all():
        raise RuntimeError("non-finite coordinate-detrended residual")
    return residual


def comparison_row(
    grid: pd.DataFrame,
    left: str,
    right: str,
    label: str,
    scale: str = "raw",
    note: str = "",
) -> dict[str, object]:
    sub = grid[[left, right, "centroid_easting_2056", "centroid_northing_2056"]].dropna()
    x = sub[left].to_numpy(float)
    y = sub[right].to_numpy(float)
    if scale == "coordinate_detrended":
        x = residualize_coordinates(x, sub)
        y = residualize_coordinates(y, sub)
    diff = x - y
    return {
        "comparison": label,
        "left_field": left,
        "right_field": right,
        "scale": scale,
        "n_cells": len(sub),
        "pearson_r": float(stats.pearsonr(x, y).statistic),
        "spearman_rho": float(stats.spearmanr(x, y).statistic),
        "rmse_c": float(np.sqrt(np.mean(np.square(diff)))) if scale == "raw" else np.nan,
        "mae_c": float(np.mean(np.abs(diff))) if scale == "raw" else np.nan,
        "mean_left_minus_right_c": float(np.mean(diff)) if scale == "raw" else np.nan,
        "note": note,
    }


def build_comparisons(grid: pd.DataFrame) -> pd.DataFrame:
    specs = [
        (
            "gp_adjusted_mean_night_min_c",
            "gp_mean_night_min_c",
            "rebuilt adjusted GP vs terminal GP",
            "same nominal variable; terminal upstream interpolation code was missing",
        ),
        (
            "gp_adjusted_mean_night_min_c",
            "idw_mean_night_min_c",
            "rebuilt adjusted GP vs terminal IDW",
            "field-method sensitivity",
        ),
        (
            "idw2_adjusted_mean_night_min_c",
            "idw_mean_night_min_c",
            "rebuilt adjusted IDW2 vs terminal IDW",
            "IDW2 is a transparent field-method sensitivity, not the primary field",
        ),
        (
            "gp_adjusted_mean_night_min_c",
            "idw2_adjusted_mean_night_min_c",
            "rebuilt adjusted GP vs rebuilt adjusted IDW2",
            "same adjusted station targets",
        ),
    ]
    if "ka_temp_night" in grid.columns:
        specs.extend(
            [
                (
                    "gp_adjusted_mean_night_min_c",
                    "ka_temp_night",
                    "rebuilt adjusted GP vs retained FITNAH",
                    "FITNAH is an external-model carrier column; absolute-error metrics are descriptive because climatological baselines may differ",
                ),
                (
                    "idw2_adjusted_mean_night_min_c",
                    "ka_temp_night",
                    "rebuilt adjusted IDW2 vs retained FITNAH",
                    "FITNAH is an external-model carrier column; absolute-error metrics are descriptive because climatological baselines may differ",
                ),
                (
                    "gp_mean_night_min_c",
                    "ka_temp_night",
                    "terminal GP vs retained FITNAH",
                    "contextual benchmark only",
                ),
            ]
        )

    rows = []
    for left, right, label, note in specs:
        if left not in grid.columns or right not in grid.columns:
            continue
        rows.append(comparison_row(grid, left, right, label, "raw", note))
        if right == "ka_temp_night":
            rows.append(
                comparison_row(
                    grid,
                    left,
                    right,
                    f"{label} (quadratic-coordinate detrended)",
                    "coordinate_detrended",
                    "both fields residualized on easting, northing, squared terms, and their interaction",
                )
            )
    return pd.DataFrame(rows)


def diagnostic_row(
    category: str,
    metric: str,
    numeric_value: float | None = None,
    text_value: str = "",
    units: str = "",
    notes: str = "",
) -> dict[str, object]:
    return {
        "category": category,
        "metric": metric,
        "numeric_value": numeric_value,
        "text_value": text_value,
        "units": units,
        "notes": notes,
    }


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    print("loading retained analysis nights and terminal evaluation grid ...")
    panel = load_analysis_panel()
    grid = pd.read_csv(TERMINAL_GRID)
    locations = repair_locations(pd.read_csv(LOCATIONS), grid)
    print(
        f"  {len(panel):,} station-nights; {panel.locationID.nunique()} stations; "
        f"{panel.night_date.nunique()} retained nights"
    )

    print("estimating coverage-adjusted station climatologies ...")
    targets, fe_diag = fit_two_way_fixed_effects(panel)
    targets = targets.merge(
        locations[
            [
                "locationID",
                "EKoord",
                "NKoord",
                "EKoord_original",
                "NKoord_original",
                "coordinate_source",
                "latDecimal",
                "lonDecimal",
                "masl",
            ]
        ],
        on="locationID",
        how="left",
        validate="one_to_one",
    )
    if targets[["EKoord", "NKoord"]].isna().any(axis=None):
        bad = targets.loc[
            targets[["EKoord", "NKoord"]].isna().any(axis=1), "locationID"
        ].tolist()
        raise ValueError(f"station target(s) without coordinates: {bad}")

    raw_adjusted_rho = float(
        stats.spearmanr(
            targets.raw_mean_night_min_c, targets.adjusted_mean_night_min_c
        ).statistic
    )
    print(
        f"  coverage {targets.n_nights.min()}--{targets.n_nights.max()} nights; "
        f"raw vs adjusted station rho={raw_adjusted_rho:.3f}"
    )

    station_xy_m = targets[["EKoord", "NKoord"]].to_numpy(float)
    grid_xy_m = grid[
        ["centroid_easting_2056", "centroid_northing_2056"]
    ].to_numpy(float)
    origin_m = station_xy_m.mean(axis=0)
    station_xy_km = (station_xy_m - origin_m) / 1000.0
    grid_xy_km = (grid_xy_m - origin_m) / 1000.0
    target_y = targets.adjusted_mean_night_min_c.to_numpy(float)
    target_se = targets.between_summer_se_c.to_numpy(float)

    print("fitting full Matern-5/2 GP ...")
    full_model = fit_gp(
        station_xy_km,
        target_y,
        target_se,
        seed=SEED,
        restarts=FULL_MODEL_RESTARTS,
    )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        grid_gp, grid_gp_sd = full_model.predict(grid_xy_km, return_std=True)
    if not (np.isfinite(grid_gp) & np.isfinite(grid_gp_sd)).all():
        raise RuntimeError("non-finite full-field GP prediction")
    kernel = summarize_kernel(full_model, target_y)
    print(f"  fitted kernel: {full_model.kernel_}")
    print(
        f"  length scale {kernel.length_scale_km:.2f} km; "
        f"Matern correlation=0.05 at {kernel.corr_005_range_km:.2f} km"
    )

    grid_idw2 = idw_predict(station_xy_m, target_y, grid_xy_m, power=2.0)
    nearest_distance_m = cKDTree(station_xy_m).query(grid_xy_m, k=1)[0]

    # Preserve every legacy carrier column unchanged, then append distinctly
    # named rebuilt fields and explicit provenance/scale labels.
    rebuilt_grid = grid.copy()
    rebuilt_grid["gp_adjusted_mean_night_min_c"] = grid_gp
    rebuilt_grid["gp_adjusted_mean_night_min_c_sd"] = grid_gp_sd
    rebuilt_grid["idw2_adjusted_mean_night_min_c"] = grid_idw2
    rebuilt_grid["rebuild_nearest_station_m"] = nearest_distance_m
    rebuilt_grid["rebuild_matern_length_scale_m"] = kernel.length_scale_km * 1000.0
    rebuilt_grid["spatial_grid_semantics"] = GRID_LABEL
    rebuilt_grid["terminal_context_upstream_status"] = BOUNDARY_LABEL

    print(
        "running repeated six-block spatial CV "
        f"({N_CV_REPEATS} KMeans partitions; {N_CV_REPEATS * N_BLOCKS} refits) ..."
    )
    cv_predictions, cv_metrics = repeated_spatial_block_cv(targets, origin_m)
    cv_median = cv_metrics[cv_metrics.row_type == "median"].iloc[0]
    print(
        f"  median CV R2={cv_median.r2:+.3f}, RMSE={cv_median.rmse_c:.3f} C, "
        f"MAE={cv_median.mae_c:.3f} C, rho={cv_median.spearman_rho:.3f}, "
        f"GP-only 95% coverage={cv_median.gp_coverage95:.3f}; "
        f"target-augmented coverage={cv_median.target_augmented_coverage95:.3f}"
    )

    comparisons = build_comparisons(rebuilt_grid)
    for row in comparisons.loc[comparisons.scale == "raw"].itertuples():
        print(
            f"  {row.comparison}: Spearman {row.spearman_rho:.3f}; "
            f"Pearson {row.pearson_r:.3f}"
        )

    station_nn = cKDTree(station_xy_m).query(station_xy_m, k=2)[0][:, 1]
    diagnostics = [
        diagnostic_row("sample", key, value, units="count")
        for key, value in fe_diag.items()
        if key.startswith("n_") or key == "fixed_effect_iterations"
    ]
    diagnostics.extend(
        [
            diagnostic_row(
                "fixed_effects",
                "fixed_effect_final_delta_c",
                fe_diag["fixed_effect_final_delta_c"],
                units="degC",
            ),
            diagnostic_row(
                "fixed_effects",
                "climatological_baseline_c",
                fe_diag["climatological_baseline_c"],
                units="degC",
                notes="equally weighted mean fitted night effect",
            ),
            diagnostic_row(
                "fixed_effects",
                "panel_residual_rmse_c",
                fe_diag["panel_residual_rmse_c"],
                units="degC",
            ),
            diagnostic_row(
                "fixed_effects",
                "raw_vs_adjusted_station_spearman",
                raw_adjusted_rho,
            ),
            diagnostic_row(
                "coverage",
                "station_nights_min",
                float(targets.n_nights.min()),
                units="nights",
            ),
            diagnostic_row(
                "coverage",
                "station_nights_median",
                float(targets.n_nights.median()),
                units="nights",
            ),
            diagnostic_row(
                "coverage",
                "station_nights_max",
                float(targets.n_nights.max()),
                units="nights",
            ),
            diagnostic_row(
                "coordinates",
                "coordinates_repaired_from_wgs84",
                float((targets.coordinate_source == "wgs84_to_lv95_repair").sum()),
                units="stations",
            ),
            diagnostic_row(
                "gp_kernel",
                "signal_sd_c",
                kernel.signal_sd_c,
                units="degC",
            ),
            diagnostic_row(
                "gp_kernel",
                "matern_length_scale",
                kernel.length_scale_km,
                units="km",
                notes="Matern-5/2 scale parameter, not map resolution",
            ),
            diagnostic_row(
                "gp_kernel",
                "white_noise_sd_c",
                kernel.white_noise_sd_c,
                units="degC",
                notes="unresolved station-scale variation on the target scale",
            ),
            diagnostic_row(
                "effective_support",
                "matern_correlation_0p10_range",
                kernel.corr_010_range_km,
                units="km",
            ),
            diagnostic_row(
                "effective_support",
                "matern_correlation_0p05_range",
                kernel.corr_005_range_km,
                units="km",
            ),
            diagnostic_row(
                "effective_support",
                "station_nearest_neighbor_median",
                float(np.median(station_nn)),
                units="m",
            ),
            diagnostic_row(
                "effective_support",
                "grid_nearest_station_median",
                float(np.median(nearest_distance_m)),
                units="m",
            ),
            diagnostic_row(
                "effective_support",
                "grid_nearest_station_p90",
                float(np.quantile(nearest_distance_m, 0.90)),
                units="m",
            ),
            diagnostic_row(
                "effective_support",
                "grid_nearest_station_max",
                float(np.max(nearest_distance_m)),
                units="m",
            ),
            diagnostic_row(
                "semantics",
                "evaluation_grid",
                text_value=GRID_LABEL,
            ),
            diagnostic_row(
                "provenance_boundary",
                "terminal_context_layers",
                text_value=BOUNDARY_LABEL,
                notes="copied unchanged; never used to fit the rebuilt heat field",
            ),
            diagnostic_row(
                "provenance",
                "fitted_kernel",
                text_value=str(full_model.kernel_),
            ),
        ]
    )
    diagnostics = pd.DataFrame(diagnostics)

    targets.to_csv(os.path.join(OUT, "v3_spatial_rebuild_station_targets.csv"), index=False)
    rebuilt_grid.to_csv(os.path.join(OUT, "v3_spatial_rebuild_grid.csv"), index=False)
    cv_predictions.to_csv(
        os.path.join(OUT, "v3_spatial_rebuild_cv_predictions.csv"), index=False
    )
    cv_metrics.to_csv(os.path.join(OUT, "v3_spatial_rebuild_cv_metrics.csv"), index=False)
    comparisons.to_csv(
        os.path.join(OUT, "v3_spatial_rebuild_comparisons.csv"), index=False
    )
    diagnostics.to_csv(
        os.path.join(OUT, "v3_spatial_rebuild_diagnostics.csv"), index=False
    )
    print(f"wrote six spatial-rebuild outputs -> {OUT}")


if __name__ == "__main__":
    main()
