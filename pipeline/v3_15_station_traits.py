"""
V3 STEP 15 -- Explain the trajectory with urban form (station-trait model).

Per senior-review guidance (point 4): use urban form NARROWLY, to explain WHERE
the dusk inequality is strongest and WHERE the overnight decay is weakest --
not as a generic driver dump.

Design:
  (1) Per-station trajectory traits on calm-clear nights, on the sunset axis:
        dusk_anom  : mean (station - city) temperature anomaly, sunset..sunset+1h
        integ_anom : mean anomaly integrated sunset..sunset+9h  (degC-hours)
        decay_tend : slope of the station's anomaly from sunset+1h to +9h
                     (>0 : the station's relative warmth GROWS overnight;
                      <0 : it fades)
  (2) A small, theory-led predictor set (no black box, n = 93 stations):
        elevation        -- cold-air drainage / valley position (masl)
        bldg_frac        -- built fraction, heat-storage mass
        canopy           -- canopy cover, radiative shielding / cooling
        coldair_flow     -- cantonal cold-air drainage flow (Klimaanalyse)
        noise_night      -- road/impervious intensity proxy
  (3) Ridge regression with REPEATED spatial-block cross-validation.  Each
      repeat partitions the stations into six geographically compact K-means
      blocks and predicts every block from the other five.  Ridge (not OLS) is
      used because the predictors are collinear and n is small.  Predictor
      importance is the change in honest out-of-block R2 after removing that
      predictor, summarised across repeated partitions.  Coefficients are
      descriptive; no coefficient-wise p-values are reported because the
      predictors can substitute for one another and the available station
      sample cannot support unique-effect inference.

Outputs (outputs_v3/):
  v3_station_traits.csv           per-station traits + covariates
  v3_station_trait_models.csv     coefficients + repeated spatial-CV summaries
  v3_station_trait_cv_repeats.csv one row per target/partition/predictor drop
"""
from __future__ import annotations

import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
import duckdb
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import v3_core as C
from v3_14_trajectory_framework import sunset_local, SLOT_SQL, load_slots, SS_LO, SS_HI

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = C.OUT
LOC = os.path.join(ROOT, "heat", "meteoblue_station_locations.csv")
SCREEN = os.path.join(ROOT, "data", "inputs", "citywide_grid_priority_screen.csv")
SEED = 20260721
DUSK_LO, DUSK_HI = 0.0, 1.0
N_BLOCKS = 6
N_BLOCK_REPEATS = 50


def nearest_cell_context(stations: pd.DataFrame) -> pd.DataFrame:
    """Sample each station's nearest 100 m cell for direct urban-form context."""
    scr = pd.read_csv(SCREEN)
    cx = scr[["centroid_easting_2056", "centroid_northing_2056",
              "canopy_cover_250m_direct", "bldg_footprint_frac_250m_direct",
              "tree_density_250m_direct", "noise_night_mean_direct",
              "ka_coldair_flow"]].dropna(subset=["centroid_easting_2056"]).values
    ce, cn = cx[:, 0], cx[:, 1]
    out = []
    for _, s in stations.iterrows():
        d2 = (ce - s.EKoord) ** 2 + (cn - s.NKoord) ** 2
        j = int(np.argmin(d2))
        out.append({"locationID": s.locationID,
                    "canopy": cx[j, 2], "bldg_frac": cx[j, 3],
                    "tree_density": cx[j, 4], "noise_night": cx[j, 5],
                    "coldair_flow": cx[j, 6],
                    "dist_nearest_cell_m": float(np.sqrt(d2[j]))})
    return pd.DataFrame(out)


def spatial_blocks(coords: np.ndarray, seed: int,
                   n_blocks: int = N_BLOCKS) -> np.ndarray:
    """Create geographically compact holdout regions.

    Coordinates are standardised before clustering so easting and northing
    have comparable leverage.  ``n_init=1`` deliberately retains variation
    among deterministic seeds; the distribution over partitions, rather than
    one favourable split, is the estimand reported below.
    """
    z = StandardScaler().fit_transform(coords)
    return KMeans(n_clusters=n_blocks, n_init=1,
                  random_state=seed).fit_predict(z)


def blocked_predictions(X: np.ndarray, y: np.ndarray,
                        blocks: np.ndarray) -> np.ndarray:
    """Predict each compact spatial block from all remaining stations."""
    pred = np.full(len(y), np.nan)
    for block in np.unique(blocks):
        train, test = blocks != block, blocks == block
        if test.sum() == 0 or train.sum() < 10:
            continue
        scaler = StandardScaler().fit(X[train])
        model = RidgeCV(alphas=np.logspace(-2, 3, 30)).fit(
            scaler.transform(X[train]), y[train])
        pred[test] = model.predict(scaler.transform(X[test]))
    return pred


def prediction_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(pred)
    if valid.sum() < 3:
        return {"r2": np.nan, "rmse": np.nan, "mae": np.nan,
                "spearman": np.nan}
    return {
        "r2": float(r2_score(y[valid], pred[valid])),
        "rmse": float(np.sqrt(mean_squared_error(y[valid], pred[valid]))),
        "mae": float(mean_absolute_error(y[valid], pred[valid])),
        "spearman": float(stats.spearmanr(y[valid], pred[valid]).statistic),
    }


def main():
    nights = C.load_nights()
    cc = nights[nights.regime == "calm_clear"][["night_date", "year"]]
    ss = sunset_local(nights.night_date)

    print("loading slots + computing station anomalies ...")
    slots = load_slots().merge(ss, on="night_date")
    slots = slots.merge(cc, on="night_date")           # calm-clear only
    slots["hss"] = (slots.loc_ts - slots.sunset_local).dt.total_seconds() / 3600.0
    slots = slots[(slots.hss >= SS_LO) & (slots.hss <= SS_HI)].copy()
    slots["hbin"] = (slots.hss * 4).round() / 4

    # city mean per (night, hbin); station anomaly = station - city
    citymean = slots.groupby(["night_date", "hbin"]).t_c.transform("mean")
    slots["anom"] = slots.t_c - citymean

    # per-station x hbin mean anomaly (averaged over calm-clear nights)
    prof = (slots.groupby(["locationID", "hbin"]).anom.mean().reset_index())

    print("computing per-station trajectory traits ...")
    rows = []
    for sid, g in prof.groupby("locationID"):
        g = g.sort_values("hbin")
        dusk = g[(g.hbin >= DUSK_LO) & (g.hbin <= DUSK_HI)].anom.mean()
        integ = np.trapezoid(g.anom.values, g.hbin.values)
        tail = g[g.hbin >= 1.0]
        decay = np.polyfit(tail.hbin, tail.anom, 1)[0] if len(tail) >= 4 else np.nan
        rows.append({"locationID": sid, "dusk_anom": dusk,
                     "integ_anom": integ, "decay_tend": decay})
    traits = pd.DataFrame(rows)

    loc = pd.read_csv(LOC)
    # Repair malformed LV95 entries from the supplied WGS84 coordinates. This
    # applies the same coordinate treatment as the rebuilt heat field.
    loc["EKoord"] = loc["EKoord"].astype(float)
    loc["NKoord"] = loc["NKoord"].astype(float)
    valid = (loc.EKoord.between(2_480_000, 2_840_000)
             & loc.NKoord.between(1_070_000, 1_300_000))
    if (~valid).any():
        if loc.loc[~valid, ["lonDecimal", "latDecimal"]].isna().any(axis=None):
            raise ValueError("invalid LV95 coordinates without WGS84 repair values")
        transformer = Transformer.from_crs(4326, 2056, always_xy=True)
        east, north = transformer.transform(
            loc.loc[~valid, "lonDecimal"].to_numpy(float),
            loc.loc[~valid, "latDecimal"].to_numpy(float),
        )
        loc.loc[~valid, "EKoord"] = east
        loc.loc[~valid, "NKoord"] = north
        print(f"  repaired {(~valid).sum()} station coordinate(s) from WGS84")
    traits = (traits.merge(loc[["locationID", "EKoord", "NKoord", "masl"]], on="locationID")
                    .merge(nearest_cell_context(loc), on="locationID"))
    traits = traits.rename(columns={"masl": "elevation"})
    traits.to_csv(os.path.join(OUT, "v3_station_traits.csv"), index=False)
    print(f"  {len(traits)} stations; median dist to nearest cell "
          f"{traits.dist_nearest_cell_m.median():.0f} m")

    # ---- ridge with spatial-block CV -----------------------------------
    # Restrict the explanatory model to three directly documented physical
    # variables. Supplied cold-air and noise layers remain in the exported
    # station table for reproducibility but do not enter the model.
    preds = ["elevation", "bldg_frac", "canopy"]
    results = []
    repeat_rows = []
    for target in ["dusk_anom", "integ_anom", "decay_tend"]:
        sub = traits.dropna(subset=[target] + preds)
        X = sub[preds].values
        y = sub[target].values
        coords = sub[["EKoord", "NKoord"]].values

        # Full-sample standardised coefficients are descriptive only.  All
        # predictive claims below come from held-out geographic blocks.
        Xs = StandardScaler().fit_transform(X)
        ys = (y - y.mean()) / y.std()
        mfull = RidgeCV(alphas=np.logspace(-2, 3, 30)).fit(Xs, ys)
        for repeat in range(N_BLOCK_REPEATS):
            seed = SEED + repeat
            blocks = spatial_blocks(coords, seed)
            full = prediction_metrics(y, blocked_predictions(X, y, blocks))
            for j, predictor in enumerate(preds):
                reduced = prediction_metrics(
                    y, blocked_predictions(np.delete(X, j, axis=1), y, blocks))
                repeat_rows.append({
                    "target": target, "repeat": repeat, "seed": seed,
                    "dropped_predictor": predictor,
                    "n_blocks": int(len(np.unique(blocks))),
                    "min_block_n": int(np.bincount(blocks).min()),
                    "max_block_n": int(np.bincount(blocks).max()),
                    "cv_r2": full["r2"], "cv_rmse": full["rmse"],
                    "cv_mae": full["mae"], "cv_spearman": full["spearman"],
                    "reduced_cv_r2": reduced["r2"],
                    "delta_cv_r2_drop": full["r2"] - reduced["r2"],
                })

        rr = pd.DataFrame(repeat_rows)
        tr = rr[rr.target == target]
        # Full-model metrics repeat once for each dropped predictor; select one
        # copy per partition before summarising.
        full_repeats = tr.drop_duplicates("repeat")
        cv = full_repeats.cv_r2
        for predictor, coef in zip(preds, mfull.coef_):
            delta = tr.loc[tr.dropped_predictor == predictor,
                           "delta_cv_r2_drop"]
            results.append({
                "target": target, "predictor": predictor,
                "std_coef": float(coef), "n": len(y),
                "n_block_repeats": N_BLOCK_REPEATS,
                "cv_r2": float(cv.median()),
                "cv_r2_q05": float(cv.quantile(.05)),
                "cv_r2_q25": float(cv.quantile(.25)),
                "cv_r2_q75": float(cv.quantile(.75)),
                "cv_r2_q95": float(cv.quantile(.95)),
                "cv_rmse": float(full_repeats.cv_rmse.median()),
                "cv_mae": float(full_repeats.cv_mae.median()),
                "cv_spearman": float(full_repeats.cv_spearman.median()),
                "delta_cv_r2_drop": float(delta.median()),
                "delta_cv_r2_q05": float(delta.quantile(.05)),
                "delta_cv_r2_q95": float(delta.quantile(.95)),
                "delta_positive_share": float((delta > 0).mean()),
            })
        print(f"  {target:11s} repeated spatial CV R2 median "
              f"{cv.median():+.2f} (5--95% {cv.quantile(.05):+.2f} to "
              f"{cv.quantile(.95):+.2f})")

    pd.DataFrame(results).to_csv(
        os.path.join(OUT, "v3_station_trait_models.csv"), index=False)
    pd.DataFrame(repeat_rows).to_csv(
        os.path.join(OUT, "v3_station_trait_cv_repeats.csv"), index=False)
    print(f"\nwrote -> {OUT}/v3_station_traits.csv, "
          "v3_station_trait_models.csv, v3_station_trait_cv_repeats.csv")


if __name__ == "__main__":
    main()
