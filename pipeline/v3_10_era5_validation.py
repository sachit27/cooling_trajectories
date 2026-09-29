"""
V3 STEP 10 -- model-pinned ERA5 validation of measured-longwave clearness.

Closes the gap flagged in MANUSCRIPT_V3.md section 7: the synoptic regime variable
rests on measured incoming longwave at a SINGLE station (Fluntern). ERA5 cloud
cover requested through Open-Meteo with ``models=era5`` provides an independent
temporal check. The five requested Zurich locations collapse to only two distinct
ERA5 grid cells; this product is therefore too coarse to validate intra-basin cloud
homogeneity.

Three questions:

  Q1  Does the longwave-derived clearness index track an independent cloud field?
      (If yes, the single-station construction is defensible.)

  Q2  What variation is visible between the ERA5 cells bracketing Zurich?
      This is descriptive only: a 0.25-degree product cannot resolve the basin.

  Q3  Does the confirmatory family (H1-H5) survive if the regime is defined from
      ERA5 cloud instead of measured longwave?

ERA5 is reanalysis, not observation, so it is a cross-check rather than ground truth.
Disagreement is informative in both directions.

Inputs : data_external/era5_pinned/openmeteo_era5_zurich_{centre,nw,ne,sw,se}.json
Outputs: outputs_v3/v3_era5_clearness_validation.csv
         outputs_v3/v3_era5_spatial_variability.csv
         outputs_v3/v3_era5_grid_mapping.csv
         outputs_v3/v3_era5_regime_sensitivity.csv
         outputs_v3/v3_night_synoptic_era5.csv
"""
from __future__ import annotations

import glob
import json
import os
import numpy as np
import pandas as pd
from scipy import stats

import v3_core as C

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = C.OUT
ERA5 = os.path.join(ROOT, "data_external", "era5_pinned")

POINTS = ["centre", "nw", "ne", "sw", "se"]
POINT_COORDS = {
    "centre": (47.3769, 8.5417), "nw": (47.43, 8.47),
    "ne": (47.43, 8.61), "sw": (47.32, 8.47), "se": (47.32, 8.61),
}


def load_point(name: str) -> pd.DataFrame:
    with open(os.path.join(ERA5, f"openmeteo_era5_zurich_{name}.json")) as f:
        d = json.load(f)
    h = d["hourly"]
    df = pd.DataFrame({
        "ts_utc": pd.to_datetime(h["time"], utc=True),
        f"cc_{name}": pd.to_numeric(h["cloud_cover"], errors="coerce"),
        f"cclow_{name}": pd.to_numeric(h["cloud_cover_low"], errors="coerce"),
        f"wind_{name}": pd.to_numeric(h["wind_speed_10m"], errors="coerce") / 3.6,  # km/h -> m/s
    })
    df.attrs["lat"], df.attrs["lon"] = d["latitude"], d["longitude"]
    return df


def build_era5_nights() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Night means at unique returned ERA5 cells, plus request-to-grid mapping."""
    requested = [(p, load_point(p)) for p in POINTS]
    unique = {}
    map_rows = []
    for name, df in requested:
        key = (float(df.attrs["lat"]), float(df.attrs["lon"]))
        unique.setdefault(key, (name, df))
        req_lat, req_lon = POINT_COORDS[name]
        map_rows.append({"requested_point": name, "requested_lat": req_lat,
                         "requested_lon": req_lon, "era5_grid_lat": key[0],
                         "era5_grid_lon": key[1]})
    selected = list(unique.values())
    used_names = [name for name, _ in selected]
    dfs = [df for _, df in selected]
    e = dfs[0]
    for x in dfs[1:]:
        e = e.merge(x, on="ts_utc", how="outer")

    # convert to local WALL CLOCK and drop the tz before taking dates, so that
    # normalising cannot be re-interpreted as UTC and shift the night label
    loc = e.ts_utc.dt.tz_convert("Europe/Zurich").dt.tz_localize(None)
    e["local_hour"] = loc.dt.hour
    day = loc.dt.normalize()
    e["night_date"] = day.where(loc.dt.hour >= 20, day - pd.Timedelta(days=1))

    night = e[(e.local_hour >= 20) | (e.local_hour < 8)].copy()
    cc_cols = [f"cc_{p}" for p in used_names]

    agg = night.groupby("night_date").agg(
        **{f"{c}": (c, "mean") for c in cc_cols},
        **{f"cclow_{p}": (f"cclow_{p}", "mean") for p in used_names},
        era5_wind_ms=("wind_centre", "mean"),
        n_hours=("ts_utc", "size"),
    ).reset_index()

    agg["era5_cc_mean"] = agg[cc_cols].mean(axis=1)
    agg["era5_cc_sd_spatial"] = agg[cc_cols].std(axis=1, ddof=1)
    agg["era5_cc_range_spatial"] = agg[cc_cols].max(axis=1) - agg[cc_cols].min(axis=1)
    agg["era5_cclow_mean"] = agg[
        [f"cclow_{p}" for p in used_names]].mean(axis=1)
    agg["n_unique_grid_cells"] = len(used_names)
    return agg, pd.DataFrame(map_rows)


def main() -> None:
    print("loading model-pinned ERA5 (5 requests) ...")
    era, grid_map = build_era5_nights()
    n_cells = grid_map[["era5_grid_lat", "era5_grid_lon"]].drop_duplicates().shape[0]
    print(f"  five requested points resolve to {n_cells} distinct ERA5 cells")
    grid_map.to_csv(os.path.join(OUT, "v3_era5_grid_mapping.csv"), index=False)
    era = era[era.night_date.dt.month.isin([6, 7, 8])]
    print(f"  ERA5 summer nights: {len(era)}  "
          f"({era.night_date.min().date()} -> {era.night_date.max().date()})")

    syn = pd.read_csv(os.path.join(OUT, "v3_night_synoptic.csv"), parse_dates=["night_date"])
    m = syn.merge(era, on="night_date", how="inner")
    m = m[m.night_date.dt.year.between(2020, 2025)]
    print(f"  merged with measured-longwave nights: {len(m)}")

    # ---------------- Q1: does measured clearness track ERA5 cloud? -------
    print("\n=== Q1  measured longwave clearness vs independent ERA5 cloud ===")
    rows = []
    for lab, col in [("ERA5 total cloud cover", "era5_cc_mean"),
                     ("ERA5 low cloud cover", "era5_cclow_mean")]:
        s = m[["clearness_wm2", "lw_in_wm2_mean", col]].dropna()
        r_clear = stats.spearmanr(s.clearness_wm2, s[col])
        r_lw = stats.spearmanr(s.lw_in_wm2_mean, s[col])
        rows.append({"era5_variable": col, "label": lab, "n": len(s),
                     "rho_clearness_index": r_clear.statistic, "p_clearness": r_clear.pvalue,
                     "rho_raw_lwin": r_lw.statistic, "p_lwin": r_lw.pvalue})
        print(f"  {lab:26s} vs clearness index : rho = {r_clear.statistic:+.3f} "
              f"(p={r_clear.pvalue:.2g}, n={len(s)})")
        print(f"  {'':26s} vs raw L_in        : rho = {r_lw.statistic:+.3f}")

    # regime means of ERA5 cloud -- does the regime label separate cloud?
    print("\n  ERA5 cloud cover by longwave-derived regime:")
    byreg = m.groupby("regime")[["era5_cc_mean", "era5_cclow_mean", "clearness_wm2"]].mean().round(1)
    print(byreg.to_string().replace("\n", "\n    "))
    for r in byreg.index:
        rows.append({"era5_variable": "regime_mean", "label": f"regime={r}",
                     "n": int((m.regime == r).sum()),
                     "rho_clearness_index": np.nan, "p_clearness": np.nan,
                     "rho_raw_lwin": float(byreg.loc[r, "era5_cc_mean"]), "p_lwin": np.nan})
    # separation test
    cc_clear = m.loc[m.regime == "calm_clear", "era5_cc_mean"].dropna()
    cc_cloud = m.loc[m.regime == "cloudy_windy", "era5_cc_mean"].dropna()
    u = stats.mannwhitneyu(cc_clear, cc_cloud)
    print(f"\n  calm-clear vs cloudy-windy ERA5 cloud: "
          f"{cc_clear.mean():.1f}% vs {cc_cloud.mean():.1f}%  (MWU p={u.pvalue:.2g})")
    pd.DataFrame(rows).to_csv(
        os.path.join(OUT, "v3_era5_clearness_validation.csv"), index=False)

    # ---------------- Q2: spatial variability across the basin -----------
    print("\n=== Q2  variation between the two coarse ERA5 cells ===")
    sv = m[["night_date", "era5_cc_mean", "era5_cc_sd_spatial",
            "era5_cc_range_spatial", "regime"]].dropna()
    print(f"  across-point SD of night cloud cover: "
          f"median {sv.era5_cc_sd_spatial.median():.1f} pp, "
          f"p90 {sv.era5_cc_sd_spatial.quantile(.9):.1f} pp")
    print(f"  across-point range:                   "
          f"median {sv.era5_cc_range_spatial.median():.1f} pp, "
          f"p90 {sv.era5_cc_range_spatial.quantile(.9):.1f} pp")
    # compare within-night spatial spread to between-night spread
    between = sv.era5_cc_mean.std()
    within = sv.era5_cc_sd_spatial.mean()
    ratio = within / between
    print(f"  between-night SD {between:.1f} pp  vs  mean within-night spatial SD "
          f"{within:.1f} pp   ratio {ratio:.2f}")
    print("  -> descriptive only: two 0.25-degree cells cannot establish "
          "within-basin cloud homogeneity")
    sv.to_csv(os.path.join(OUT, "v3_era5_spatial_variability.csv"), index=False)

    # ---------------- Q3: regime redefined from ERA5 ---------------------
    print("\n=== Q3  confirmatory family with an ERA5-defined regime ===")
    train = m.night_date.dt.year.between(2020, 2023)
    z_train = lambda s: (s - s[train].mean()) / s[train].std()
    m["era5_clearness"] = -z_train(m.era5_cc_mean)    # low cloud = clear
    m["era5_calm"] = -z_train(m.era5_wind_ms)
    m["era5_index"] = (m.era5_clearness + m.era5_calm) / 2
    lo, hi = m.loc[train, "era5_index"].quantile([1 / 3, 2 / 3])
    m["regime_era5"] = np.select(
        [m.era5_index >= hi, m.era5_index <= lo],
        ["calm_clear", "cloudy_windy"], default="mixed")

    agree = (m.regime == m.regime_era5).mean()
    print(f"  regime label agreement (longwave vs ERA5): {agree:.1%}")
    print(pd.crosstab(m.regime, m.regime_era5).to_string().replace("\n", "\n    "))

    m.to_csv(os.path.join(OUT, "v3_night_synoptic_era5.csv"), index=False)

    # re-run H1-H5 with the ERA5 regime, on the same hold-out
    city = pd.read_csv(os.path.join(OUT, "v3_night_city.csv"), parse_dates=["night_date"])
    d = city.merge(m, on="night_date", how="inner")
    d["year"] = d.night_date.dt.year
    ho = d[d.year.isin(C.HOLDOUT_YEARS)]
    cc = ho[ho.regime_era5 == "calm_clear"]
    print(f"\n  hold-out nights {len(ho)}, calm-clear under ERA5 regime {len(cc)}")

    res = []
    h1 = C.block_permutation_spearman(
        cc.city_mean_tmin_c, cc.sd_tmin_c, groups=cc.year)
    res.append({"id": "H1", "regime_source": "ERA5", "estimate": h1["rho"],
                "p_raw": h1["p_block_perm"], "n": h1["n"]})
    h2 = C.partial_spearman(ho, "city_mean_tmin_c", "sd_tmin_c",
                            ["day_rad_wm2", "era5_cc_mean", "night_wind_ms"])
    res.append({"id": "H2", "regime_source": "ERA5 cloud in adjustment set",
                "estimate": h2["r_partial"], "p_raw": h2["p_block_perm"], "n": h2["n"]})
    h3 = C.block_permutation_spearman(
        cc.city_mean_tmin_c, cc.sd_cool_total, groups=cc.year)
    res.append({"id": "H3", "regime_source": "ERA5", "estimate": h3["rho"],
                "p_raw": h3["p_block_perm"], "n": h3["n"]})
    m4 = C.hac_ols(cc.dropna(subset=["city_mean_cool_rate", "city_mean_tmin_c"]),
                   "city_mean_cool_rate ~ city_mean_tmin_c")
    res.append({"id": "H4", "regime_source": "ERA5",
                "estimate": m4.params["city_mean_tmin_c"],
                "p_raw": m4.pvalues["city_mean_tmin_c"], "n": int(m4.nobs)})
    m5 = C.hac_ols(cc.dropna(subset=["sd_cool_rate", "city_mean_tmin_c"]),
                   "sd_cool_rate ~ city_mean_tmin_c")
    res.append({"id": "H5", "regime_source": "ERA5 [declared null]",
                "estimate": m5.params["city_mean_tmin_c"],
                "p_raw": m5.pvalues["city_mean_tmin_c"], "n": int(m5.nobs)})

    r = pd.DataFrame(res)
    r["p_bh"] = C.bh_correct(r.p_raw.tolist())

    orig = pd.read_csv(os.path.join(OUT, "holdout_confirmatory.csv"))[
        ["id", "estimate", "p_bh"]].rename(
        columns={"estimate": "estimate_longwave", "p_bh": "p_bh_longwave"})
    r = r.merge(orig, on="id")
    r["sign_agrees"] = np.sign(r.estimate) == np.sign(r.estimate_longwave)

    print("\n  H1-H5 under ERA5-defined regime vs the longwave-defined original:")
    print(r[["id", "estimate_longwave", "estimate", "p_bh_longwave", "p_bh",
             "sign_agrees"]].round(4).to_string(index=False))
    r.to_csv(os.path.join(OUT, "v3_era5_regime_sensitivity.csv"), index=False)

    print(f"\nwrote -> {OUT}/v3_era5_*.csv")


if __name__ == "__main__":
    main()
