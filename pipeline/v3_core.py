"""
V3 CORE -- shared estimation machinery for the dispersion / saturation analysis.

Imported by the development, hold-out and sensitivity scripts so that every sample
is analysed with byte-identical code. No script may define its own estimator.

Inference respects that consecutive summer nights are strongly autocorrelated:
  * summer-stratified circular-shift randomization for correlations
  * year-block (cluster) bootstrap for group differences and slopes
  * Newey-West HAC standard errors for regressions
Nothing here uses a test that assumes independent nights.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs_v3")

RNG_SEED = 20260721
BLOCK_LEN = 14        # nights; > 2x the lag at which autocorrelation dies
N_PERM = 10000
N_BOOT = 5000

DEV_YEARS = (2020, 2021, 2022, 2023)
HOLDOUT_YEARS = (2024, 2025)


# --------------------------------------------------------------------------
# data assembly
# --------------------------------------------------------------------------
def load_nights() -> pd.DataFrame:
    """City-night dispersion metrics joined to measured synoptic regime."""
    city = pd.read_csv(os.path.join(OUT, "v3_night_city.csv"), parse_dates=["night_date"])
    syn = pd.read_csv(os.path.join(OUT, "v3_night_synoptic.csv"), parse_dates=["night_date"])
    d = city.merge(syn, on="night_date", how="inner", suffixes=("", "_syn"))
    d = d[d.night_date.dt.year.between(2020, 2025)].copy()
    d = d.sort_values("night_date").reset_index(drop=True)
    d["year"] = d.night_date.dt.year
    return d


def sample(d: pd.DataFrame, which: str) -> pd.DataFrame:
    if which == "dev":
        return d[d.year.isin(DEV_YEARS)].copy()
    if which == "holdout":
        return d[d.year.isin(HOLDOUT_YEARS)].copy()
    if which == "all":
        return d.copy()
    raise ValueError(which)


# --------------------------------------------------------------------------
# dependence-robust primitives
# --------------------------------------------------------------------------
def _cartesian_sums(arrays):
    """All sums formed by taking one value from each small one-dimensional array."""
    out = np.array([0.0])
    for a in arrays:
        out = np.add.outer(out, a).ravel()
    return out


def circular_shift_null(rx, ry):
    """Correlations for every cyclic rotation of ``ry`` against ``rx``.

    This unstratified helper is retained for diagnostics and single continuous
    series. Paper-facing tests use :func:`circular_shift_p`, which never rotates
    observations across the June--August gaps between summers.
    """
    rx = np.asarray(rx, float)
    ry = np.asarray(ry, float)
    n = len(rx)
    rxc = rx - rx.mean()
    ryc = ry - ry.mean()
    denom = np.sqrt(np.sum(rxc ** 2) * np.sum(ryc ** 2))
    if denom == 0:
        return np.zeros(n)
    # correlation of rxc with every cyclic rotation of ryc == circular
    # cross-correlation / denom (computed directly; n is small here).
    null = np.array([np.dot(rxc, np.roll(ryc, s)) for s in range(n)]) / denom
    return null


def circular_shift_p(rx, ry, groups=None):
    """Exact two-sided circular-shift p-value, optionally stratified by summer.

    With ``groups`` supplied, ``ry`` is rotated independently *within* each group
    and the full Cartesian randomization set is enumerated. This preserves each
    summer's serial structure and its membership while avoiding the artificial
    August-to-next-June adjacency created by concatenating summers. The exact tail
    count is evaluated from two half-set sum distributions, so even tens of
    millions of combinations need not be materialised.

    The test conditions on group membership. It therefore asks whether the paired
    timing carries association beyond any fixed between-summer composition effect.
    As with every shift test, validity requires approximate stationarity within
    each summer.
    """
    rx = np.asarray(rx, float)
    ry = np.asarray(ry, float)
    if len(rx) != len(ry):
        raise ValueError("rx and ry must have equal length")
    if groups is None:
        obs = stats.pearsonr(rx, ry).statistic
        null = circular_shift_null(rx, ry)
        p = float(np.mean(np.abs(null) >= abs(obs) - 1e-9))
        return p, len(null)

    groups = np.asarray(groups)
    if len(groups) != len(rx):
        raise ValueError("groups must have the same length as rx and ry")
    rxc = rx - rx.mean()
    ryc = ry - ry.mean()
    denom = np.sqrt(np.sum(rxc ** 2) * np.sum(ryc ** 2))
    if denom == 0:
        return 1.0, 1

    contributions = []
    # pandas.unique preserves the chronological first-occurrence order.
    for group in pd.unique(groups):
        idx = np.flatnonzero(groups == group)
        contributions.append(np.array([
            np.dot(rxc[idx], np.roll(ryc[idx], shift))
            for shift in range(len(idx))
        ]))

    split = len(contributions) // 2
    left = _cartesian_sums(contributions[:split])
    right = np.sort(_cartesian_sums(contributions[split:]))
    obs = np.dot(rxc, ryc) / denom
    cutoff = max(0.0, (abs(obs) - 1e-12) * denom)
    n_randomizations = int(len(left) * len(right))
    if cutoff == 0:
        return 1.0, n_randomizations
    extreme = 0
    for value in left:
        # The two tails are disjoint whenever cutoff > 0 (all uses here).
        extreme += int(np.searchsorted(right, -cutoff - value, side="right"))
        extreme += int(len(right) - np.searchsorted(
            right, cutoff - value, side="left"))
    return float(extreme / n_randomizations), n_randomizations


def block_permutation_spearman(x, y, block=BLOCK_LEN, n_perm=N_PERM,
                               seed=RNG_SEED, groups=None):
    """
    Spearman rho with an exact circular-shift randomization p-value.

    Replaces the earlier shift+moving-block-bootstrap hybrid, which resampled y
    with replacement and produced spuriously fine-grained p-values.
    If ``groups`` is supplied, rotations are restricted to each summer and the
    exact Cartesian set is used. ``block``, ``n_perm`` and ``seed`` are retained
    for call-signature compatibility but are no longer used.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if groups is not None:
        groups = np.asarray(groups)[ok]
    x, y = x[ok], y[ok]
    n = len(x)
    rho = stats.spearmanr(x, y).statistic
    p, n_rand = circular_shift_p(stats.rankdata(x), stats.rankdata(y), groups)
    return {"rho": rho, "p_block_perm": p, "n": n,
            "n_randomizations": n_rand}


def partial_spearman_point(df: pd.DataFrame, x: str, y: str,
                           covs: list[str], min_n: int = 3):
    """Rank-partial-correlation point estimate used by multiverse scripts.

    This is deliberately inference-free: paper-facing p-values are produced by
    :func:`partial_spearman` with summer-stratified circular shifts.
    """
    sub = df[[x, y] + covs].dropna()
    if len(sub) < min_n:
        return np.nan, len(sub)
    a = sub.rank().to_numpy(float)
    a = (a - a.mean(0)) / a.std(0)
    if not covs:
        return float(stats.pearsonr(a[:, 0], a[:, 1]).statistic), len(sub)
    z = np.column_stack([np.ones(len(a)), a[:, 2:]])
    beta_x = np.linalg.lstsq(z, a[:, 0], rcond=None)[0]
    beta_y = np.linalg.lstsq(z, a[:, 1], rcond=None)[0]
    rx = a[:, 0] - np.einsum("ij,j->i", z, beta_x)
    ry = a[:, 1] - np.einsum("ij,j->i", z, beta_y)
    return float(stats.pearsonr(rx, ry).statistic), len(sub)


def partial_spearman(df: pd.DataFrame, x: str, y: str, covs: list[str],
                     block=BLOCK_LEN, n_perm=N_PERM, seed=RNG_SEED,
                     groups=None):
    """
    Rank-based partial correlation: rank-transform all variables, residualise x and y
    on the covariates by OLS, correlate the residuals. The p-value comes from
    circular shifts of the y residual, stratified by summer when year is present.
    """
    group_values = None
    if groups is None and "year" in df.columns:
        groups = df["year"]
    cols = [x, y] + covs
    r = df[cols].dropna()
    if groups is not None:
        if isinstance(groups, pd.Series):
            group_values = groups.loc[r.index].to_numpy()
        else:
            groups = np.asarray(groups)
            if len(groups) != len(df):
                raise ValueError("groups must align with df")
            group_values = groups[df.index.get_indexer(r.index)]
    a = r.rank().values
    a = (a - a.mean(0)) / a.std(0)
    z = np.column_stack([np.ones(len(a)), a[:, 2:]])
    beta_x = np.linalg.lstsq(z, a[:, 0], rcond=None)[0]
    beta_y = np.linalg.lstsq(z, a[:, 1], rcond=None)[0]
    # np.einsum avoids a macOS Accelerate/NumPy 2.2 matmul warning observed for
    # small Fortran-contiguous design matrices in the pinned environment.
    rx = a[:, 0] - np.einsum("ij,j->i", z, beta_x)
    ry = a[:, 1] - np.einsum("ij,j->i", z, beta_y)

    obs = stats.pearsonr(rx, ry).statistic
    n = len(rx)
    p, n_rand = circular_shift_p(rx, ry, group_values)
    return {"r_partial": obs, "p_block_perm": p, "n": n,
            "n_randomizations": n_rand, "covariates": "+".join(covs)}


def year_block_bootstrap_diff(df: pd.DataFrame, value: str, group: str,
                              g_hi, g_lo, n_boot=N_BOOT, seed=RNG_SEED):
    """
    Difference in means between two groups, with a CI from resampling whole YEARS
    (the natural independent cluster in this design).
    """
    sub = df[df[group].isin([g_hi, g_lo])].dropna(subset=[value])
    obs = (sub.loc[sub[group] == g_hi, value].mean()
           - sub.loc[sub[group] == g_lo, value].mean())
    years = sub.year.unique()
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    boots[:] = np.nan
    for i in range(n_boot):
        pick = rng.choice(years, size=len(years), replace=True)
        parts = [sub[sub.year == y] for y in pick]
        bs = pd.concat(parts)
        hi = bs.loc[bs[group] == g_hi, value]
        lo = bs.loc[bs[group] == g_lo, value]
        if len(hi) and len(lo):
            boots[i] = hi.mean() - lo.mean()
    boots = boots[np.isfinite(boots)]
    lo_ci, hi_ci = np.percentile(boots, [2.5, 97.5])
    # two-sided bootstrap p for H0: diff = 0
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    p = min(max(p, 1 / (len(boots) + 1)), 1.0)
    return {"diff": obs, "ci_lo": lo_ci, "ci_hi": hi_ci, "p_boot": p,
            "n_hi": int((sub[group] == g_hi).sum()), "n_lo": int((sub[group] == g_lo).sum())}


def hac_ols(df: pd.DataFrame, formula: str, maxlags: int = 7):
    """OLS with Newey-West HAC covariance (serial correlation robust)."""
    m = smf.ols(formula, data=df).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return m


def bh_correct(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(order[::-1]):
        k = n - rank
        val = min(prev, p[i] * n / k)
        adj[i] = val
        prev = val
    return list(adj)


# --------------------------------------------------------------------------
# the three headline estimands
# --------------------------------------------------------------------------
def marginal_vs_conditional(d: pd.DataFrame, spread: str = "sd_tmin_c") -> pd.DataFrame:
    """
    E1. The Simpson decomposition.

    Marginal association between city-mean warmth and cross-station dispersion,
    then the same association conditioned on the MEASURED synoptic regime.
    If the marginal estimate is a composition effect, the conditional estimate
    will differ in sign.
    """
    rows = []
    m = block_permutation_spearman(
        d.city_mean_tmin_c, d[spread], groups=d.year)
    rows.append({"estimand": "marginal", "conditioning": "none", **m,
                 "r": m["rho"]})

    for label, covs in [
        ("clearness", ["clearness_wm2"]),
        ("wind", ["night_wind_ms"]),
        ("clearness+wind", ["clearness_wm2", "night_wind_ms"]),
        ("clearness+wind+rh", ["clearness_wm2", "night_wind_ms", "night_rh_pct"]),
    ]:
        p = partial_spearman(d, "city_mean_tmin_c", spread, covs)
        rows.append({"estimand": "conditional", "conditioning": label,
                     "rho": p["r_partial"], "r": p["r_partial"],
                     "p_block_perm": p["p_block_perm"], "n": p["n"]})

    for reg in ["calm_clear", "mixed", "cloudy_windy"]:
        g = d[d.regime == reg]
        if len(g) > 20:
            s = block_permutation_spearman(
                g.city_mean_tmin_c, g[spread], groups=g.year)
            rows.append({"estimand": "within_regime", "conditioning": reg,
                         "r": s["rho"], **s})
    out = pd.DataFrame(rows)
    out["spread_metric"] = spread
    return out


def saturation_within_calm_clear(d: pd.DataFrame) -> pd.DataFrame:
    """
    E2. Within the calm-clear regime -- the regime in which the heat island is
    fully expressed -- does dispersion GROW or SHRINK as nights get hotter?
    """
    cc = d[d.regime == "calm_clear"].copy()
    rows = []
    for metric in ["sd_tmin_c", "p90p10_tmin_c", "iqr_tmin_c",
                   "sd_cool_rate", "p90p10_cool_rate", "sd_cool_total"]:
        if cc[metric].notna().sum() < 20:
            continue
        s = block_permutation_spearman(
            cc.city_mean_tmin_c, cc[metric], groups=cc.year)
        rows.append({"metric": metric, "rho": s["rho"],
                     "p_block_perm": s["p_block_perm"], "n": s["n"]})
    out = pd.DataFrame(rows)
    out["p_bh"] = bh_correct(out.p_block_perm.tolist())
    return out


def cooling_rate_mechanism(d: pd.DataFrame) -> pd.DataFrame:
    """
    E3. Mechanism. If dispersion in night minima compresses on hot calm-clear
    nights, the cooling-RATE distribution should compress too: cool sites lose
    their cooling advantage. Tests dispersion of dT/dt (22:00->02:00).
    """
    cc = d[d.regime == "calm_clear"].copy()
    rows = []
    for y, lab in [("sd_cool_rate", "cooling-rate SD"),
                   ("city_mean_cool_rate", "cooling-rate mean"),
                   ("sd_tmin_c", "Tmin SD")]:
        sub = cc.dropna(subset=[y, "city_mean_tmin_c"])
        if len(sub) < 20:
            continue
        m = hac_ols(sub, f"{y} ~ city_mean_tmin_c")
        rows.append({
            "outcome": y, "label": lab,
            "beta_per_degC": m.params["city_mean_tmin_c"],
            "se_hac": m.bse["city_mean_tmin_c"],
            "p_hac": m.pvalues["city_mean_tmin_c"],
            "n": int(m.nobs), "r2": m.rsquared,
        })
    return pd.DataFrame(rows)
