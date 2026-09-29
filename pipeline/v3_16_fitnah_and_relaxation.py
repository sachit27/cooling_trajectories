"""
V3 STEP 16 -- FITNAH external validation + a physically-motivated relaxation model.

Two additions requested in senior review:

A. FITNAH EXTERNAL VALIDATION. The Canton Zurich Klimaanalyse (FITNAH physical
   model) simulates the nocturnal temperature and cold-air-drainage field from
   terrain and land cover, without using the meteoblue temperature observations.
   Agreement between our observed Gaussian-process night-heat field and
   FITNAH is therefore an independent, physics-based validation, complementing
   the ERA5 validation of the synoptic regime. We report the cell-level
   agreement and confirm cold-air drainage carries the expected cooling sign.

B. RELAXATION DIAGNOSTIC. After each night's observed dispersion peak, we fit a
   physically motivated decay curve with a floor plus two dynamic parameters:

       SD(t) = SD_dawn + A * exp(-t / tau)

   where t = hours since that night's observed peak, A is peak amplitude above
   the asymptotic floor and tau is the decay timescale. This is explicitly
   exploratory: peak selection and a fit-quality threshold are data dependent,
   and the free-spline functional model in v3_14 remains the primary analysis.

Outputs (outputs_v3/):
  v3_fitnah_validation.csv
  v3_relaxation_pernight.csv
  v3_relaxation_tests.csv
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

import v3_core as C

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = C.OUT
SCREEN = os.path.join(ROOT, "data", "inputs", "citywide_grid_priority_screen.csv")
SEED = 20260721
N_BOOT = 3000


# --------------------------------------------------------------------------
# A. FITNAH external validation
# --------------------------------------------------------------------------
def fitnah_validation():
    s = pd.read_csv(SCREEN)
    # Use identical complete cells for GP and IDW so their external-validation
    # comparison is reproducible and not driven by differing missingness.
    d = s.dropna(subset=["ka_temp_night", "gp_mean_night_min_c",
                         "idw_mean_night_min_c"])
    # spatial-block bootstrap CI (coarse 3x3 grid) so autocorrelation is respected
    e = d.centroid_easting_2056.values
    n = d.centroid_northing_2056.values
    eb = np.clip(np.digitize(e, np.quantile(e, [1/3, 2/3])), 0, 2)
    nb = np.clip(np.digitize(n, np.quantile(n, [1/3, 2/3])), 0, 2)
    blocks = eb * 3 + nb
    ub = np.unique(blocks)
    grp = {b: d[blocks == b] for b in ub}
    # Partial Spearman after removing a smooth second-order coordinate surface
    # from both ranked fields. This checks that raw agreement is not only a
    # shared broad north/south or east/west gradient.
    def detrended_spearman(frame, field):
        ee = ((frame.centroid_easting_2056.values - e.mean()) / e.std())
        nn = ((frame.centroid_northing_2056.values - n.mean()) / n.std())
        x = np.column_stack([np.ones(len(frame)), ee, nn, ee ** 2, nn ** 2, ee * nn])
        a = stats.rankdata(frame.ka_temp_night)
        b = stats.rankdata(frame[field])
        ba = np.linalg.lstsq(x, a, rcond=None)[0]
        bb = np.linalg.lstsq(x, b, rcond=None)[0]
        ra = a - np.einsum("ij,j->i", x, ba)
        rb = b - np.einsum("ij,j->i", x, bb)
        return stats.pearsonr(ra, rb).statistic

    def validate_field(field, seed_offset):
        rho = stats.spearmanr(d.ka_temp_night, d[field]).statistic
        pear = stats.pearsonr(d.ka_temp_night, d[field])[0]
        rho_detrended = detrended_spearman(d, field)
        boots = []
        boots_detrended = []
        local_rng = np.random.default_rng(SEED + seed_offset)
        for _ in range(2000):
            bs = pd.concat([grp[b] for b in local_rng.choice(ub, len(ub))])
            if len(bs) > 50:
                boots.append(stats.spearmanr(bs.ka_temp_night, bs[field]).statistic)
                boots_detrended.append(detrended_spearman(bs, field))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        dlo, dhi = np.percentile(boots_detrended, [2.5, 97.5])
        return rho, pear, lo, hi, rho_detrended, dlo, dhi

    gp = validate_field("gp_mean_night_min_c", 0)
    idw = validate_field("idw_mean_night_min_c", 1)
    rho, pear, lo, hi, rho_detrended, dlo, dhi = gp
    idw_rho, idw_pear, idw_lo, idw_hi, idw_det, idw_dlo, idw_dhi = idw

    dc = s.dropna(subset=["ka_coldair_flow", "gp_mean_night_min_c"])
    rho_cf = stats.spearmanr(dc.ka_coldair_flow, dc.gp_mean_night_min_c).statistic

    out = pd.DataFrame([
        {"quantity": "GP field vs FITNAH night temperature (Spearman)",
         "value": rho, "ci_lo": lo, "ci_hi": hi, "n": len(d)},
        {"quantity": "GP field vs FITNAH night temperature (Pearson)",
         "value": pear, "ci_lo": np.nan, "ci_hi": np.nan, "n": len(d)},
        {"quantity": "GP field vs FITNAH night temperature (coordinate-detrended Spearman)",
         "value": rho_detrended, "ci_lo": dlo, "ci_hi": dhi, "n": len(d)},
        {"quantity": "IDW field vs FITNAH night temperature (Spearman)",
         "value": idw_rho, "ci_lo": idw_lo, "ci_hi": idw_hi, "n": len(d)},
        {"quantity": "IDW field vs FITNAH night temperature (Pearson)",
         "value": idw_pear, "ci_lo": np.nan, "ci_hi": np.nan, "n": len(d)},
        {"quantity": "IDW field vs FITNAH night temperature (coordinate-detrended Spearman)",
         "value": idw_det, "ci_lo": idw_dlo, "ci_hi": idw_dhi, "n": len(d)},
        {"quantity": "GP field vs FITNAH cold-air flow (Spearman)",
         "value": rho_cf, "ci_lo": np.nan, "ci_hi": np.nan, "n": len(dc)},
    ])
    out.to_csv(os.path.join(OUT, "v3_fitnah_validation.csv"), index=False)
    print("A. FITNAH external validation")
    print(f"   GP field vs FITNAH night temp: Spearman {rho:.3f} "
          f"[{lo:.3f},{hi:.3f}] (Pearson {pear:.3f}), n={len(d)}")
    print(f"   coordinate-detrended partial Spearman: {rho_detrended:.3f} "
          f"[{dlo:.3f},{dhi:.3f}]")
    print(f"   IDW field vs FITNAH night temp: Spearman {idw_rho:.3f} "
          f"[{idw_lo:.3f},{idw_hi:.3f}] (Pearson {idw_pear:.3f})")
    print(f"   IDW coordinate-detrended Spearman: {idw_det:.3f} "
          f"[{idw_dlo:.3f},{idw_dhi:.3f}]")
    print(f"   GP field vs FITNAH cold-air flow: Spearman {rho_cf:.3f} "
          f"(cooler where drainage stronger)")
    return rho, lo, hi, pear, rho_detrended, dlo, dhi, rho_cf, idw


# --------------------------------------------------------------------------
# B. relaxation model
# --------------------------------------------------------------------------
def relaxation():
    # per-night sunset-aligned SD curve, calm-clear
    prof = None
    # rebuild the per-night SD curve from the profile step's inputs if present,
    # else recompute quickly from the sunset profile scalars already saved.
    pc = os.path.join(OUT, "v3_traj_pernight.csv")
    # we need the FULL curve, so recompute cross-station SD by hbin per night.
    from v3_14_trajectory_framework import (sunset_local, load_slots, SS_LO, SS_HI)
    nights = C.load_nights()
    ss = sunset_local(nights.night_date)
    cc = nights[nights.regime == "calm_clear"][["night_date", "year", "city_mean_tmin_c"]]
    slots = load_slots().merge(ss, on="night_date").merge(cc, on="night_date")
    slots["hss"] = (slots.loc_ts - slots.sunset_local).dt.total_seconds() / 3600.0
    slots = slots[(slots.hss >= SS_LO) & (slots.hss <= SS_HI)].copy()
    slots["hbin"] = (slots.hss * 4).round() / 4
    persd = (slots.groupby(["night_date", "hbin"]).t_c.agg(sd="std", n="size")
             .reset_index())
    persd = persd[persd.n >= 60]

    def model(t, sd_floor, A, tau):
        return sd_floor + A * np.exp(-t / tau)

    rows = []
    for nd, g in persd.groupby("night_date"):
        g = g.sort_values("hbin")
        t_all, y_all = g.hbin.values, g.sd.values
        peak_idx = int(np.argmax(y_all))
        peak_hss = t_all[peak_idx]
        t = t_all[peak_idx:] - peak_hss
        y = y_all[peak_idx:]
        if len(t) < 8:
            continue
        try:
            p0 = [y.min(), max(y.max() - y.min(), 0.05), 3.0]
            popt, _ = curve_fit(model, t, y, p0=p0,
                                bounds=([0, 0, 0.3], [3, 3, 30]), maxfev=6000)
            resid = y - model(t, *popt)
            r2 = 1 - np.sum(resid**2) / np.sum((y - y.mean())**2)
            rows.append({"night_date": nd, "peak_hss": peak_hss,
                         "sd_floor": popt[0], "amplitude": popt[1],
                         "tau": popt[2], "fit_r2": r2, "n_tail": len(t)})
        except Exception:
            pass
    rel = pd.DataFrame(rows).merge(cc, on="night_date")
    rel = rel.rename(columns={"city_mean_tmin_c": "warmth"})
    rel["year"] = rel.night_date.dt.year
    # Pre-declared diagnostic quality screen; results remain exploratory because
    # the peak itself is selected from the observed curve.
    rel = rel[rel.fit_r2 > 0.3]
    rel.to_csv(os.path.join(OUT, "v3_relaxation_pernight.csv"), index=False)
    print(f"\nB. post-peak relaxation  SD(t)=SD_floor + A*exp(-t/tau)")
    print(f"   fitted on {len(rel)} calm-clear nights (median fit R2="
          f"{rel.fit_r2.median():.2f}); median tau={rel.tau.median():.1f} h")

    def rho_ci(df, col):
        d = df.dropna(subset=[col, "warmth"])
        test = C.block_permutation_spearman(
            d.warmth, d[col], groups=d.year)
        o = test["rho"]
        rng = np.random.default_rng(SEED)
        yrs = d.year.unique()
        group_idx = [np.flatnonzero(d.year.to_numpy() == year) for year in yrs]
        b = []
        for _ in range(N_BOOT):
            sampled = []
            for original in group_idx:
                ng = len(original); local = np.empty(ng, int)
                i = rng.integers(0, ng)
                for j in range(ng):
                    local[j] = i
                    i = (rng.integers(0, ng) if rng.random() < 1 / 14
                         else (i + 1) % ng)
                sampled.append(original[local])
            idx = np.concatenate(sampled)
            b.append(stats.spearmanr(
                d.warmth.to_numpy()[idx], d[col].to_numpy()[idx]).statistic)
        return (o, np.nanpercentile(b, 2.5), np.nanpercentile(b, 97.5),
                test["p_block_perm"], len(d))

    dev = rel[rel.year.isin(C.DEV_YEARS)]
    ho = rel[rel.year.isin(C.HOLDOUT_YEARS)]
    tests = []
    for col, name in [("amplitude", "post-peak amplitude A"),
                      ("tau", "post-peak decay timescale tau")]:
        for lab, dd in [("development", dev), ("withheld", ho)]:
            o, lo, hi, p, n = rho_ci(dd, col)
            tests.append({"parameter": col, "name": name, "status": "exploratory",
                          "sample": lab, "rho": o, "ci_lo": lo, "ci_hi": hi,
                          "shift_p": p, "n": n})
            print(f"   {name:20s} [{lab:11s}] rho(warmth)={o:+.3f} "
                  f"[{lo:+.3f},{hi:+.3f}], shift p={p:.3g}, n={n} (exploratory)")
    pd.DataFrame(tests).to_csv(os.path.join(OUT, "v3_relaxation_tests.csv"), index=False)
    return rel, pd.DataFrame(tests)


def main():
    fitnah_validation()
    relaxation()
    print(f"\nwrote -> {OUT}/v3_fitnah_validation.csv, v3_relaxation_*.csv")


if __name__ == "__main__":
    main()
