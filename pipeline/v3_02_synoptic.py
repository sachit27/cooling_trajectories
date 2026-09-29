"""
V3 STEP 2 -- Synoptic conditioning with a MEASURED nocturnal clearness index.

The v2 study conditioned on a "calm-clear index" built largely from DAYTIME global
radiation and sunshine. That is a proxy for the wrong time of day: the quantity that
governs nocturnal cooling is the downwelling longwave flux from the sky during the
night itself. Conditioning on a daytime proxy leaves residual confounding, which is
precisely the failure mode that produced the spurious v2 headline.

Here the regime variable is built from MEASURED hourly incoming longwave radiation
(MeteoSwiss SMA/Fluntern, parameter oli000h0) during the night.

Physics
-------
Downwelling longwave  L_in = eps_sky * sigma * T_air^4.
Clear skies have low effective emissivity (eps ~ 0.7-0.8); cloud raises eps toward 1.
So the CLEARNESS of a night is captured by the residual of L_in after removing the
dependence on air temperature and humidity:

    resid = L_in - f(sigma*T^4, RH)          (OLS, night hours only)
    clearness = -resid                        (high = clear sky, low L_in for its T)

This is a measurement-based radiative index: incoming longwave is observed
directly, while sky clearness is the derived regression residual.

A calm-clear composite then combines clearness with night wind (calm) since both
are required for a strong nocturnal heat island (Oke 1982).

Outputs (outputs_v3/):
  v3_night_synoptic.csv        one row per night, all synoptic covariates + regime
  v3_clearness_model.csv       the LWin residualisation model, for transparency
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs_v3")
os.makedirs(OUT, exist_ok=True)

SYNOP = os.path.join(ROOT, "data", "inputs", "official_meteoswiss_hourly_zurich_synoptic.csv")
LW = os.path.join(ROOT, "data_external", "zurich_missing_official", "meteoswiss_smn",
                  "processed", "ogd-smn_sma_hourly_incoming_longwave_2020-2029.csv")

SIGMA = 5.670374419e-8  # Stefan-Boltzmann, W m-2 K-4


def load_longwave() -> pd.DataFrame:
    lw = pd.read_csv(LW)
    lw["ts_utc"] = pd.to_datetime(lw["reference_timestamp"], format="%d.%m.%Y %H:%M", utc=True)
    lw["oli000h0"] = pd.to_numeric(lw["oli000h0"], errors="coerce")
    return lw[["ts_utc", "oli000h0"]].dropna()


def load_synoptic() -> pd.DataFrame:
    d = pd.read_csv(SYNOP)
    d["ts_utc"] = pd.to_datetime(d["timestamp_utc"], utc=True)
    for c in ["tre200h0", "ure200h0", "fkl010h0", "fu3010h0", "rre150h0",
              "gre000h0", "sre000h0"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def build_clearness(sma: pd.DataFrame, lw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit clearness on development summers and apply it to every summer.

    The model uses only external meteorological variables, but fitting its
    coefficients on 2024--2025 would still be transductive feature engineering.
    Keeping the withheld summers out makes the evaluation split literal.
    """
    m = sma.merge(lw, on="ts_utc", how="inner")
    # night hours only (local 20:00-08:00)
    m = m[(m.local_hour >= 20) | (m.local_hour < 8)].copy()
    m = m.dropna(subset=["oli000h0", "tre200h0", "ure200h0"])

    m["sigT4"] = SIGMA * (m.tre200h0 + 273.15) ** 4
    fit_mask = pd.to_datetime(m["night_date"]).dt.year.between(2020, 2023)
    fit_data = m[fit_mask]
    fit = smf.ols("oli000h0 ~ sigT4 + ure200h0", data=fit_data).fit()
    m["lw_resid"] = m["oli000h0"] - fit.predict(m)
    # high clearness == LESS longwave than expected for this T and RH
    m["clearness"] = -m["lw_resid"]

    model = pd.DataFrame({
        "term": fit.params.index,
        "coef": fit.params.values,
        "p_value": fit.pvalues.values,
    })
    model.loc[len(model)] = ["r_squared", fit.rsquared, np.nan]
    model.loc[len(model)] = ["n_hours_fit_2020_2023", len(fit_data), np.nan]
    model.loc[len(model)] = ["n_hours_applied", len(m), np.nan]
    model.loc[len(model)] = ["resid_sd_wm2", fit.resid.std(), np.nan]
    return m, model


def main() -> None:
    print("loading synoptic + longwave ...")
    syn = load_synoptic()
    lw = load_longwave()
    sma = syn[syn.station_abbr == "SMA"].copy()
    print(f"  SMA hourly rows {len(sma):,}; longwave rows {len(lw):,}")

    night_lw, model = build_clearness(sma, lw)
    model.to_csv(os.path.join(OUT, "v3_clearness_model.csv"), index=False)
    print(f"  clearness model R2 = {model.loc[model.term=='r_squared','coef'].iloc[0]:.3f}, "
          f"resid SD = {model.loc[model.term=='resid_sd_wm2','coef'].iloc[0]:.1f} W/m2")

    # ---- aggregate to nights ------------------------------------------------
    lw_night = (night_lw.groupby("night_date")
                .agg(lw_in_wm2_mean=("oli000h0", "mean"),
                     clearness_wm2=("clearness", "mean"),
                     n_lw_hours=("clearness", "size"))
                .reset_index())

    # night-time meteorology, averaged across the three SMN stations
    nightly = syn[(syn.local_hour >= 20) | (syn.local_hour < 8)]
    met_night = (nightly.groupby("night_date")
                 .agg(night_wind_ms=("fkl010h0", "mean"),
                      night_gust_ms=("fu3010h0", "max"),
                      night_rh_pct=("ure200h0", "mean"),
                      night_air_t_c=("tre200h0", "mean"))
                 .reset_index())
    # Precipitation is a station-level accumulation (mm). Summing rre150h0 over
    # every station-hour in a night would multiply-count the ~3 SMN stations, so
    # we sum WITHIN each station-night, then average those totals across stations
    # (areal-mean nightly precipitation in mm).
    night_precip = (nightly.dropna(subset=["rre150h0"])
                    .groupby(["night_date", "station_abbr"]).rre150h0.sum()
                    .groupby("night_date").mean()
                    .rename("night_precip_mm").reset_index())
    met_night = met_night.merge(night_precip, on="night_date", how="left")

    # daytime radiation of the PRECEDING day (heat storage input)
    day = syn[(syn.local_hour >= 8) & (syn.local_hour < 20)]
    met_day = (day.groupby("date_local")
               .agg(day_rad_wm2=("gre000h0", "mean"),
                    day_max_t_c=("tre200h0", "max"))
               .reset_index().rename(columns={"date_local": "night_date"}))
    # Sunshine duration, like precipitation, is a per-station accumulation.
    day_sun = (day.dropna(subset=["sre000h0"])
               .groupby(["date_local", "station_abbr"]).sre000h0.sum()
               .groupby("date_local").mean()
               .rename("day_sun_min").reset_index()
               .rename(columns={"date_local": "night_date"}))
    day_precip = (day.dropna(subset=["rre150h0"])
                  .groupby(["date_local", "station_abbr"]).rre150h0.sum()
                  .groupby("date_local").mean()
                  .rename("day_precip_mm").reset_index()
                  .rename(columns={"date_local": "night_date"}))
    met_day = (met_day.merge(day_sun, on="night_date", how="left")
                      .merge(day_precip, on="night_date", how="left"))

    n = (lw_night.merge(met_night, on="night_date", how="outer")
                 .merge(met_day, on="night_date", how="left"))
    n["night_date"] = pd.to_datetime(n["night_date"])
    n = n[n.night_date.dt.month.isin([6, 7, 8])].copy()
    n = n[n.night_date.dt.year >= 2020]

    # ---- regime definition --------------------------------------------------
    # Calm-clear composite from MEASURED clearness + calm (low wind). All
    # centring/scaling constants and cut points are learned on 2020--2023 only.
    train = n.night_date.dt.year.between(2020, 2023)
    z_train = lambda s: (s - s[train].mean()) / s[train].std()
    n["z_clearness"] = z_train(n.clearness_wm2)
    n["z_calm"] = -z_train(n.night_wind_ms)
    n["calm_clear_index"] = (n.z_clearness + n.z_calm) / 2

    lo, hi = n.loc[train, "calm_clear_index"].quantile([1 / 3, 2 / 3])
    n["regime"] = np.select(
        [n.calm_clear_index >= hi, n.calm_clear_index <= lo],
        ["calm_clear", "cloudy_windy"], default="mixed")

    # legacy daytime-radiation index, retained ONLY for sensitivity comparison
    n["z_dayrad"] = z_train(n.day_rad_wm2)
    n["legacy_day_proxy_index"] = (n.z_dayrad + n.z_calm) / 2

    print(f"\n  nights with synoptic data: {len(n)}  "
          f"({n.night_date.min().date()} -> {n.night_date.max().date()})")
    print(f"  clearness available on {n.clearness_wm2.notna().sum()} nights")
    print("\n  regime counts:")
    print(n.regime.value_counts().to_string())
    print("\n  regime means:")
    print(n.groupby("regime")[["clearness_wm2", "night_wind_ms", "lw_in_wm2_mean",
                               "day_rad_wm2"]].mean().round(2).to_string())

    n.to_csv(os.path.join(OUT, "v3_night_synoptic.csv"), index=False)
    print(f"\nwrote -> {OUT}/v3_night_synoptic.csv")


if __name__ == "__main__":
    main()
