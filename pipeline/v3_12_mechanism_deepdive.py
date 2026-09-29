"""
V3 STEP 12 -- Mechanism deep-dive: from inference to observation.

Closes the four gaps identified in the expert review:

A. TIME-RESOLVED DISPERSION PROFILE. Cross-station SD of temperature at every
   quarter-hour through the night, composited by warmth tercile within the
   calm-clear regime. If spatial inequality is created before the night starts
   and partially erased by nocturnal cooling, the profiles will show it directly
   -- no inference from summary statistics required.

B. SOLAR-TIME CONTROL. Zurich sunset moves ~70 min across JJA and hot nights
   cluster mid-summer, so clock-defined anchors sit at different phases of the
   cooling curve depending on date. The profile is repeated aligned on LOCAL
   SUNSET (NOAA solar geometry), and all anchor quantities are recomputed
   sunset-referenced.

C. ANCHOR SENSITIVITY. The keystone observables used fixed anchors
   (evening 20-22 h; rate 22:00->02:00) that were never varied. Every result
   is recomputed for shifted and sunset-referenced windows.

D. DECOMPOSITION AND EFFECT SIZES. Var(cool_total) decomposed into evening and
   minimum components; all headline associations restated as degC quantities
   with year-block-bootstrap CIs, not only rank correlations.

Outputs (outputs_v3/):
  v3_profile_clock.csv          SD profile by quarter-hour, warmth tercile
  v3_profile_sunset.csv         same, aligned on local sunset
  v3_anchor_sensitivity.csv     rho(warmth, SD) for every anchor variant
  v3_variance_decomposition.csv evening/minimum decomposition + partials
  v3_effect_sizes.csv           degC effect sizes with bootstrap CIs
"""
from __future__ import annotations

import os
import duckdb
import numpy as np
import pandas as pd
from scipy import stats

import v3_core as C

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = C.OUT
RAW = os.path.join(ROOT, "heat", "meteoblue_temperature.parquet")

LAT, LON = 47.3769, 8.5417
SEED = 20260721
N_BOOT = 2000


# ---------------------------------------------------------------------------
# NOAA sunset (accurate to ~1 min, ample for 15-min data)
# ---------------------------------------------------------------------------
def sunset_utc_hours(dates: pd.DatetimeIndex) -> np.ndarray:
    doy = dates.dayofyear.values
    hours = 12.0
    gamma = 2 * np.pi / 365 * (doy - 1 + (hours - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
                       - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma))
    decl = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
            - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
            - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma))
    lat = np.deg2rad(LAT)
    zenith = np.deg2rad(90.833)
    cos_ha = (np.cos(zenith) / (np.cos(lat) * np.cos(decl))
              - np.tan(lat) * np.tan(decl))
    ha = np.rad2deg(np.arccos(np.clip(cos_ha, -1, 1)))
    # NOAA convention, longitude positive EAST: the hour angle is +ha at sunrise
    # and -ha at sunset, so sunset uses (LON - ha).
    sunset_min_utc = 720 - 4 * (LON - ha) - eqtime
    return sunset_min_utc / 60.0


def build_sunset_table(nights: pd.Series) -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.to_datetime(nights.unique()))
    ss_utc = sunset_utc_hours(idx)
    ss = (pd.Series(idx, name="night_date").to_frame()
          .assign(sunset_utc=[d + pd.Timedelta(hours=h) for d, h in zip(idx, ss_utc)]))
    ss["sunset_utc"] = ss["sunset_utc"].dt.tz_localize("UTC")
    ss["sunset_local"] = ss["sunset_utc"].dt.tz_convert("Europe/Zurich").dt.tz_localize(None)
    return ss


# ---------------------------------------------------------------------------
# slot-level panel from raw 15-min data
# ---------------------------------------------------------------------------
SLOT_SQL = f"""
WITH src AS (
    SELECT locationID,
           timezone('Europe/Zurich', CAST(timestamp AS TIMESTAMPTZ)) AS loc_ts_tz,
           TRY_CAST(value AS DOUBLE) AS t_c,
           TRY_CAST(qc_flag AS INTEGER) AS qc
    FROM read_parquet('{RAW}')
),
clean AS (
    SELECT locationID,
           CAST(loc_ts_tz AS TIMESTAMP) AS loc_ts,   -- naive local wall clock
           t_c
    FROM src
    WHERE qc = 0 AND t_c IS NOT NULL AND t_c BETWEEN -30 AND 50
),
nightly AS (
    SELECT locationID, loc_ts, t_c,
           CASE WHEN EXTRACT(hour FROM loc_ts) >= 18
                THEN CAST(loc_ts AS DATE)
                ELSE CAST(loc_ts AS DATE) - INTERVAL 1 DAY END AS night_date
    FROM clean
    WHERE EXTRACT(hour FROM loc_ts) >= 18 OR EXTRACT(hour FROM loc_ts) < 8
)
SELECT night_date, loc_ts, locationID, t_c
FROM nightly
WHERE EXTRACT(month FROM night_date) IN (6,7,8)
  AND EXTRACT(year  FROM night_date) BETWEEN 2020 AND 2025
"""


def load_slots() -> pd.DataFrame:
    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='2300MB'")
    con.execute("PRAGMA threads=4")
    df = con.execute(SLOT_SQL).df()
    con.close()
    df["night_date"] = pd.to_datetime(df["night_date"])
    df["loc_ts"] = pd.to_datetime(df["loc_ts"])
    # Published timestamps mark the end of the preceding 15-minute averaging
    # interval. Associate each value with the interval midpoint.
    df["loc_ts"] = df["loc_ts"] - pd.Timedelta(minutes=7.5)
    return df


# ---------------------------------------------------------------------------
def year_block_ci(nights_df: pd.DataFrame, value_col: str, n_boot=N_BOOT, seed=SEED):
    """Mean of value_col with a CI from resampling whole years."""
    rng = np.random.default_rng(seed)
    years = nights_df.year.unique()
    gsum = nights_df.groupby("year")[value_col].agg(["sum", "size"])
    s_arr, n_arr = gsum["sum"].values, gsum["size"].values.astype(float)
    idx = rng.integers(0, len(years), size=(n_boot, len(years)))
    boots = s_arr[idx].sum(1) / n_arr[idx].sum(1)
    return nights_df[value_col].mean(), *np.percentile(boots, [2.5, 97.5])


def rho_with_ci(x, y, years, n_boot=N_BOOT, seed=SEED):
    """Spearman rho with year-block bootstrap CI."""
    df = pd.DataFrame({"x": x, "y": y, "year": years}).dropna()
    obs = stats.spearmanr(df.x, df.y).statistic
    rng = np.random.default_rng(seed)
    yrs = df.year.unique()
    groups = {y: df[df.year == y][["x", "y"]].values for y in yrs}
    boots = np.empty(n_boot); boots[:] = np.nan
    for i in range(n_boot):
        pick = rng.choice(yrs, size=len(yrs), replace=True)
        arr = np.vstack([groups[y] for y in pick])
        if len(np.unique(arr[:, 0])) > 5:
            boots[i] = stats.spearmanr(arr[:, 0], arr[:, 1]).statistic
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return obs, lo, hi, len(df)


def main() -> None:
    print("loading slot-level panel (18:00-08:00, JJA 2020-2025) ...")
    slots = load_slots()
    print(f"  {len(slots):,} slot observations")

    nights = C.load_nights()          # regime + warmth per night (546 nights)
    cc_nights = nights[nights.regime == "calm_clear"]
    terc = cc_nights.city_mean_tmin_c.quantile([1/3, 2/3])
    cc = cc_nights.assign(tercile=np.select(
        [cc_nights.city_mean_tmin_c >= terc.iloc[1],
         cc_nights.city_mean_tmin_c <= terc.iloc[0]],
        ["hot", "cool"], default="middle"))
    print(f"  calm-clear nights: {len(cc)} "
          f"(hot {sum(cc.tercile=='hot')}, middle {sum(cc.tercile=='middle')}, "
          f"cool {sum(cc.tercile=='cool')})")

    ss = build_sunset_table(nights.night_date)
    print(f"  sunset range: {ss.sunset_local.dt.strftime('%H:%M').min()} - "
          f"{ss.sunset_local.dt.strftime('%H:%M').max()} local")

    # =====================================================================
    # A. clock-aligned profile
    # =====================================================================
    print("\nA. time-resolved dispersion profile (clock-aligned) ...")
    sl = slots.merge(cc[["night_date", "tercile", "year"]], on="night_date", how="inner")
    # offset in hours from 18:00 of the evening date
    base = sl.night_date + pd.Timedelta(hours=18)
    sl["off_h"] = (sl.loc_ts - base).dt.total_seconds() / 3600.0
    sl = sl[(sl.off_h >= 0) & (sl.off_h < 14)]

    per_night = (sl.groupby(["night_date", "tercile", "year", "off_h"])
                 .t_c.agg(sd="std", n="size").reset_index())
    per_night = per_night[per_night.n >= 60]

    rng = np.random.default_rng(SEED)

    def boot_profile(per, off_col, align_label):
        """Vectorised year-block bootstrap of the mean SD per (tercile, offset).

        Resampling whole years and averaging the pooled nights equals the
        n-weighted mean of per-year sums, so per-year (sum, n) suffice."""
        agg = (per.groupby(["tercile", off_col, "year"])
               .sd.agg(s="sum", n="size").reset_index())
        out = []
        for (tc, off), g in agg.groupby(["tercile", off_col]):
            s_arr, n_arr = g.s.values, g.n.values.astype(float)
            k = len(s_arr)
            idx = rng.integers(0, k, size=(800, k))
            boots = s_arr[idx].sum(1) / n_arr[idx].sum(1)
            out.append({"align": align_label, "tercile": tc, "offset_h": off,
                        "sd_mean": s_arr.sum() / n_arr.sum(),
                        "ci_lo": np.percentile(boots, 2.5),
                        "ci_hi": np.percentile(boots, 97.5),
                        "n_nights": int(n_arr.sum())})
        return pd.DataFrame(out)

    prof_clock = boot_profile(per_night, "off_h", "clock")
    prof_clock.to_csv(os.path.join(OUT, "v3_profile_clock.csv"), index=False)

    # =====================================================================
    # B. sunset-aligned profile
    # =====================================================================
    print("B. sunset-aligned profile ...")
    sl2 = sl.merge(ss[["night_date", "sunset_local"]], on="night_date")
    sl2["off_ss"] = ((sl2.loc_ts - sl2.sunset_local).dt.total_seconds() / 3600.0)
    sl2["off_ss"] = (sl2.off_ss * 4).round() / 4          # snap to quarter-hours
    sl2 = sl2[(sl2.off_ss >= -2) & (sl2.off_ss <= 10)]

    per_night2 = (sl2.groupby(["night_date", "tercile", "year", "off_ss"])
                  .t_c.agg(sd="std", n="size").reset_index())
    per_night2 = per_night2[per_night2.n >= 60]
    prof_ss = boot_profile(per_night2, "off_ss", "sunset")
    prof_ss.to_csv(os.path.join(OUT, "v3_profile_sunset.csv"), index=False)

    for lab, pr in [("clock", prof_clock), ("sunset", prof_ss)]:
        hot0 = pr[(pr.tercile == "hot")].sort_values("offset_h")
        cool0 = pr[(pr.tercile == "cool")].sort_values("offset_h")
        print(f"  [{lab}] hot-night SD at first/last offset: "
              f"{hot0.sd_mean.iloc[0]:.2f} -> {hot0.sd_mean.iloc[-1]:.2f} C ; "
              f"cool-night: {cool0.sd_mean.iloc[0]:.2f} -> {cool0.sd_mean.iloc[-1]:.2f} C")

    # =====================================================================
    # C. anchor sensitivity
    # =====================================================================
    print("\nC. anchor sensitivity ...")
    slA = slots.merge(nights[["night_date", "regime", "year",
                              "city_mean_tmin_c"]], on="night_date", how="inner")
    slA = slA.merge(ss[["night_date", "sunset_local"]], on="night_date")
    slA["hour"] = slA.loc_ts.dt.hour + slA.loc_ts.dt.minute / 60.0
    slA["off_ss"] = (slA.loc_ts - slA.sunset_local).dt.total_seconds() / 3600.0

    def station_night_quantity(df, kind, **kw):
        g = df.groupby(["night_date", "locationID"])
        if kind == "cool_total":
            ev = df[(df.hour >= kw["ev0"]) & (df.hour < kw["ev1"])] if "ev0" in kw \
                 else df[(df.off_ss >= kw["ss0"]) & (df.off_ss < kw["ss1"])]
            evm = ev.groupby(["night_date", "locationID"]).t_c.mean()
            # minimum over the full night window 20-08 local
            nightpart = df[(df.hour >= 20) | (df.hour < 8)]
            mn = nightpart.groupby(["night_date", "locationID"]).t_c.min()
            return (evm - mn).rename("q")
        if kind == "cool_rate":
            if "t0" in kw:
                # Values are assigned to 15-minute interval midpoints, so a
                # nominal whole-hour anchor is represented by the adjacent
                # midpoint (for example 21:52:30 for 22:00).
                a = df[np.isclose(df.hour, kw["t0"], atol=0.13)]
                b = df[np.isclose(
                    df.hour,
                    kw["t1"] if kw["t1"] > kw["t0"] else kw["t1"],
                    atol=0.13,
                )]
                # handle wrap past midnight: hours < 8 belong to same night label
            else:
                a = df[np.isclose(df.off_ss, kw["s0"], atol=0.13)]
                b = df[np.isclose(df.off_ss, kw["s1"], atol=0.13)]
            ta = a.groupby(["night_date", "locationID"]).t_c.mean()
            tb = b.groupby(["night_date", "locationID"]).t_c.mean()
            dt = kw.get("dt", 4.0)
            return (-(tb - ta) / dt).rename("q")
        raise ValueError(kind)

    variants = []
    # cool_total with three evening windows + sunset-referenced
    for name, kwargs in [
        ("cool_total ev 20-22 (base)", dict(kind="cool_total", ev0=20, ev1=22)),
        ("cool_total ev 19-21",        dict(kind="cool_total", ev0=19, ev1=21)),
        ("cool_total ev 21-23",        dict(kind="cool_total", ev0=21, ev1=23)),
        ("cool_total ev sunset..+2h",  dict(kind="cool_total", ss0=0, ss1=2)),
    ]:
        variants.append((name, kwargs))
    # cool_rate windows: clock (wrap handled by hour value; 1.0 == 01:00 etc.)
    for name, kwargs in [
        ("cool_rate 22->02 (base)", dict(kind="cool_rate", t0=22.0, t1=2.0, dt=4)),
        ("cool_rate 21->01",        dict(kind="cool_rate", t0=21.0, t1=1.0, dt=4)),
        ("cool_rate 23->03",        dict(kind="cool_rate", t0=23.0, t1=3.0, dt=4)),
        ("cool_rate sunset+1..+5h", dict(kind="cool_rate", s0=1.0, s1=5.0, dt=4)),
    ]:
        variants.append((name, kwargs))

    res = []
    ccA = slA[slA.regime == "calm_clear"]
    for name, kwargs in variants:
        q = station_night_quantity(ccA, **kwargs).reset_index()
        pern = q.groupby("night_date").q.agg(sd="std", n="size").reset_index()
        pern = pern[pern.n >= 60].merge(
            cc[["night_date", "city_mean_tmin_c", "year"]], on="night_date")
        r, lo, hi, n = rho_with_ci(pern.city_mean_tmin_c, pern.sd, pern.year)
        res.append({"variant": name, "sample": "calm_clear",
                    "rho": r, "ci_lo": lo, "ci_hi": hi, "n_nights": n})
        print(f"  {name:28s} rho = {r:+.3f}  [{lo:+.3f}, {hi:+.3f}]  n={n}")
    anch = pd.DataFrame(res)
    anch.to_csv(os.path.join(OUT, "v3_anchor_sensitivity.csv"), index=False)

    # =====================================================================
    # D. decomposition + effect sizes
    # =====================================================================
    print("\nD. variance decomposition and effect sizes ...")
    sn = pd.read_parquet(os.path.join(OUT, "v3_station_night.parquet"))
    # Use the identical complete station set for all three moments. Independent
    # NA deletion makes Var(E-M)=Var(E)+Var(M)-2Cov fail mechanically.
    def complete_moments(x):
        p = x[["t_even_c", "night_min_c"]].dropna()
        ct = p.t_even_c - p.night_min_c
        return pd.Series({
            "sd_t_even": p.t_even_c.std(ddof=1),
            "sd_tmin": p.night_min_c.std(ddof=1),
            "sd_cool_total": ct.std(ddof=1),
            "cov_even_min": p.cov().iloc[0, 1],
        })

    per = sn.groupby("night_date").apply(
        complete_moments, include_groups=False).reset_index()
    m = cc.drop(columns=["sd_cool_total"], errors="ignore").merge(per, on="night_date")

    dec_rows = []
    for y, lab in [("sd_t_even", "SD evening temp (20-22h)"),
                   ("sd_tmin", "SD night minimum"),
                   ("sd_cool_total", "SD total cooling")]:
        r, lo, hi, n = rho_with_ci(m.city_mean_tmin_c, m[y], m.year)
        dec_rows.append({"quantity": lab, "rho_warmth": r, "ci_lo": lo,
                         "ci_hi": hi, "p_value": np.nan, "n": n})
        print(f"  rho(warmth, {lab:26s}) = {r:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    # variance shares: Var(ct) = Var(ev) + Var(min) - 2Cov
    var_ct = (m.sd_cool_total ** 2).mean()
    var_ev = (m.sd_t_even ** 2).mean()
    var_mn = (m.sd_tmin ** 2).mean()
    cov = m.cov_even_min.mean()
    print(f"  mean Var(cool_total) = {var_ct:.3f}; "
          f"Var(even) {var_ev:.3f} + Var(min) {var_mn:.3f} - 2Cov {2*cov:.3f} "
          f"= {var_ev + var_mn - 2*cov:.3f}")
    dec = pd.DataFrame(dec_rows)
    dec.loc[len(dec)] = {"quantity": "mean Var(cool_total)", "rho_warmth": var_ct,
                         "ci_lo": np.nan, "ci_hi": np.nan,
                         "p_value": np.nan, "n": len(m)}
    dec.loc[len(dec)] = {"quantity": "identity check Var(ev)+Var(min)-2Cov",
                         "rho_warmth": var_ev + var_mn - 2 * cov,
                         "ci_lo": np.nan, "ci_hi": np.nan,
                         "p_value": np.nan, "n": len(m)}
    # partial: SD(cool_total) ~ warmth | SD(t_even)  -- is there signal beyond evening?
    p = C.partial_spearman(m.rename(columns={"city_mean_tmin_c": "w"}),
                           "w", "sd_cool_total", ["sd_t_even"])
    print(f"  partial rho(warmth, SD cool_total | SD evening) = "
          f"{p['r_partial']:+.3f} (p={p['p_block_perm']:.3g})")
    dec.loc[len(dec)] = {"quantity": "partial SD(ct)~warmth | SD(even)",
                         "rho_warmth": p["r_partial"], "ci_lo": np.nan,
                         "ci_hi": np.nan, "p_value": p["p_block_perm"],
                         "n": p["n"]}
    dec.to_csv(os.path.join(OUT, "v3_variance_decomposition.csv"), index=False)

    # effect sizes in degC: tercile means with year-block CIs
    eff = []
    for y, lab in [("sd_t_even", "evening SD"), ("sd_tmin", "minimum SD"),
                   ("sd_cool_total", "total-cooling SD")]:
        for tc in ["cool", "middle", "hot"]:
            sub = m[m.tercile == tc]
            mean, lo, hi = year_block_ci(sub, y)
            eff.append({"quantity": lab, "tercile": tc, "mean_c": mean,
                        "ci_lo": lo, "ci_hi": hi, "n_nights": len(sub)})
    eff = pd.DataFrame(eff)
    eff.to_csv(os.path.join(OUT, "v3_effect_sizes.csv"), index=False)
    print("\n  effect sizes (degC), calm-clear terciles:")
    print(eff.round(3).to_string(index=False))

    print(f"\nwrote -> {OUT}/v3_profile_*.csv, v3_anchor_sensitivity.csv, "
          f"v3_variance_decomposition.csv, v3_effect_sizes.csv")


if __name__ == "__main__":
    main()
