"""
V3 STEP 13 -- Publication figure set (300 dpi, CIs throughout).

Produces vector manuscript figures in overleaf_submission/figures/. Layouts
are sized for a full manuscript text width.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

import v3_core as C

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = C.OUT
FIG = os.path.join(ROOT, "overleaf_submission", "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "font.size": 9.5,
    "axes.titlesize": 10, "axes.titleweight": "normal",
    "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "font.family": "DejaVu Sans",
})
CB = {"blue": "#2166ac", "red": "#b2182b", "grey": "#5f6b7a",
      "green": "#1b7837", "orange": "#e08214", "purple": "#542788",
      "lightred": "#d6604d", "lightblue": "#4393c3"}
TERC = {"hot": CB["red"], "middle": CB["grey"], "cool": CB["blue"]}

SAVE = dict(bbox_inches="tight", facecolor="white")
PDF_METADATA = {
    "Creator": "Zurich night-heat reproducibility pipeline",
    "Producer": "Matplotlib",
    # Suppress clock-time metadata so repeated builds are byte-identical.
    "CreationDate": None,
    "ModDate": None,
}
def save_pair(fig, stem):
    fig.savefig(os.path.join(FIG, f"{stem}.pdf"), metadata=PDF_METADATA, **SAVE)


def rho_ci(x, y, years, n_boot=2000, seed=20260721):
    df = pd.DataFrame({"x": x, "y": y, "yr": years}).dropna()
    obs = stats.spearmanr(df.x, df.y).statistic
    rng = np.random.default_rng(seed)
    yrs = df.yr.unique()
    boots = np.full(n_boot, np.nan)
    groups = {y: df[df.yr == y] for y in yrs}
    for i in range(n_boot):
        bs = pd.concat([groups[y] for y in rng.choice(yrs, len(yrs))])
        if bs.x.nunique() > 5:
            boots[i] = stats.spearmanr(bs.x, bs.y).statistic
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return obs, lo, hi


def slope_ci(x, y, years, n_boot=2000, seed=20260721):
    df = pd.DataFrame({"x": x, "y": y, "yr": years}).dropna()
    obs = np.polyfit(df.x, df.y, 1)[0]
    rng = np.random.default_rng(seed)
    yrs = df.yr.unique()
    groups = {y: df[df.yr == y] for y in yrs}
    boots = np.full(n_boot, np.nan)
    for i in range(n_boot):
        bs = pd.concat([groups[y] for y in rng.choice(yrs, len(yrs))])
        if bs.x.nunique() > 5:
            boots[i] = np.polyfit(bs.x, bs.y, 1)[0]
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return obs, lo, hi


# ------------------------------------------------------------------ F1
def f1():
    sc = pd.read_csv(os.path.join(OUT, "v3_specification_curve.csv"))
    sc = sc.sort_values("rho").reset_index(drop=True)
    fig = plt.figure(figsize=(7.4, 8.9))
    # Give the legend its own row so that it cannot obscure the estimates.
    gs = fig.add_gridspec(3, 1, height_ratios=[0.22, 1.35, 3.7], hspace=0.42)

    axleg = fig.add_subplot(gs[0])
    axleg.axis("off")
    ax = fig.add_subplot(gs[1])
    ax.scatter(np.arange(len(sc)), sc.rho, s=2.5,
               c=np.where(sc.rho > 0, CB["lightred"], CB["lightblue"]),
               alpha=0.6, linewidths=0, rasterized=True)
    ax.axhline(0, color="k", lw=1.0)
    med = sc.rho.median()
    ax.axhline(med, color=CB["green"], lw=1.4, ls="--",
               label=f"median = {med:+.2f}")
    ax.fill_between([0, len(sc)], sc.rho.quantile(.25), sc.rho.quantile(.75),
                    color=CB["green"], alpha=0.10,
                    label=f"IQR [{sc.rho.quantile(.25):+.2f}, {sc.rho.quantile(.75):+.2f}]")
    conv = sc.loc[sc.adjustment_set == "unadjusted", "rho"].median()
    ax.axhline(conv, color=CB["orange"], lw=1.3, ls=":",
               label=f"unadjusted (conventional) = {conv:+.2f}")
    ax.set_xlim(0, len(sc))
    ax.set_ylabel(r"$\rho$(city-mean $T_{\min}$, dispersion)")
    ax.set_xlabel(f"specification, ranked (n = {len(sc):,})")
    ax.set_title("(a) Night-minimum dispersion vs. warmth, all specifications",
                 loc="left")
    handles, legend_labels = ax.get_legend_handles_labels()
    axleg.legend(handles, legend_labels, frameon=False, loc="center",
                 ncol=3, columnspacing=1.6, handlelength=2.5)
    ax.text(0.99, 0.05, f"{(sc.rho > 0).mean():.0%} positive, "
            f"{(sc.rho <= 0).mean():.0%} reverse sign",
            transform=ax.transAxes, ha="right", fontsize=7.5, color=CB["grey"])

    ax2 = fig.add_subplot(gs[2])
    from matplotlib.lines import Line2D
    NICE = {"day_radiation": "daytime radiation", "clearness+wind": "clearness + wind",
            "dayrad+wind": "day rad + wind",
            "full(dayrad+clear+wind)": "full (day rad+clear+wind)",
            "full+precip": "full + precip", "sd_tmin_c": "SD", "p90p10_tmin_c": "P90-P10",
            "iqr_tmin_c": "IQR", "all_nights": "all nights",
            "calm_clear_only": "calm-clear only", "all": "all years",
            "dev": "development", "holdout": "withheld",
            "unadjusted": "unadjusted", "clearness": "clearness", "wind": "wind"}
    # Six factor groups, each shown as its own labelled block with an explicit
    # level order. Within a level: dot = median, thick bar = interquartile range,
    # thin line = 10th-90th percentile. Adjustment sets are highlighted because
    # they account for the largest variation.
    GROUPS = [
        ("adjustment set", "adjustment_set", True,
         ["unadjusted", "day_radiation", "clearness", "wind",
          "dayrad+wind", "clearness+wind", "full(dayrad+clear+wind)", "full+precip"]),
        ("dispersion metric", "spread_metric", False,
         ["sd_tmin_c", "p90p10_tmin_c", "iqr_tmin_c"]),
        ("night sample", "sample", False, ["all_nights", "calm_clear_only"]),
        ("night window", "night_window", False, ["20-08", "21-07", "22-06"]),
        ("regime threshold", "regime_cut", False, ["33/67", "25/75", "20/80"]),
        ("study period", "years", False, ["all", "dev", "holdout"]),
    ]
    y = 0.0
    yticks, yticklabels, header_idx = [], [], []
    for gname, factor, is_adj, order in GROUPS:
        yticks.append(y); yticklabels.append(gname); header_idx.append(len(yticks) - 1)
        y += 1.0
        col = CB["purple"] if is_adj else "#7f8792"
        for lvl in order:
            g = sc[sc[factor] == lvl]
            if not len(g):
                continue
            med = g.rho.median()
            q1, q3 = g.rho.quantile([.25, .75])
            p10, p90 = g.rho.quantile([.10, .90])
            ax2.plot([p10, p90], [y, y], color=col, lw=1.3, alpha=0.55,
                     solid_capstyle="round", zorder=2)
            ax2.plot([q1, q3], [y, y], color=col, lw=4.6, alpha=0.9,
                     solid_capstyle="round", zorder=3)
            ax2.plot([med], [y], "o", color=col, ms=5.2, mec="white", mew=0.8, zorder=4)
            yticks.append(y); yticklabels.append("   " + NICE.get(lvl, lvl))
            y += 1.0
        y += 0.9
    ax2.axvline(0, color="k", lw=1.0, zorder=1)
    ax2.set_ylim(-1.0, y - 0.4); ax2.invert_yaxis()
    ax2.set_yticks(yticks); ax2.set_yticklabels(yticklabels, fontsize=7.4)
    for i, lab in enumerate(ax2.get_yticklabels()):
        if i in header_idx:
            lab.set_fontweight("bold"); lab.set_fontsize(7.9)
    ax2.tick_params(axis="y", length=0)
    ax2.set_xlabel(
        "Spearman $\\rho$: city-wide night warmth vs. cross-station dispersion\n"
        "$\\leftarrow$ warmer nights more spatially uniform"
        "        warmer nights more spatially unequal $\\rightarrow$",
        fontsize=8.5)
    leg = [Line2D([0], [0], color="#7f8792", lw=4.6, label="interquartile range"),
           Line2D([0], [0], color="#7f8792", lw=1.3, alpha=0.6,
                  label="10th to 90th percentile"),
           Line2D([0], [0], color="#7f8792", marker="o", lw=0, mfc="#7f8792",
                  mec="white", ms=5.2, label="median")]
    # Place the legend below the x-axis label as a single horizontal row so it
    # never overlaps the intervals (the tight bounding box keeps it visible).
    ax2.legend(handles=leg, frameon=False, fontsize=6.8, ncol=3,
               handlelength=1.8, columnspacing=1.8,
               loc="upper center", bbox_to_anchor=(0.5, -0.205),
               borderaxespad=0.0)
    ax2.set_title("(b) Distribution of $\\rho$ by analytic choice "
                  "(purple: adjustment sets)", loc="left")
    save_pair(fig, "fig1_specification_curve")
    plt.close(fig); print("  F1 done")


# ------------------------------------------------------------------ F2
def f2():
    d = C.load_nights(); dev = C.sample(d, "dev")
    fig = plt.figure(figsize=(7.2, 7.4))
    gs = fig.add_gridspec(
        2, 3, height_ratios=[1.35, 1.0], width_ratios=[1.0, 1.0, 0.06],
        hspace=0.50, wspace=0.32)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, :2])]
    colour_ax = fig.add_subplot(gs[0, 2])
    titles = {"a": "(a) Daytime radiation", "b": "(b) Nocturnal clearness"}
    colour_norm = matplotlib.colors.Normalize(
        vmin=dev.sd_tmin_c.min(), vmax=dev.sd_tmin_c.max())
    colour_mappable = None
    for ax, (v, lab, tag) in zip(axes[:2], [
            ("day_rad_wm2", "daytime global radiation (W m$^{-2}$)", "a"),
            ("clearness_wm2", "measured nocturnal clearness (W m$^{-2}$)", "b")]):
        s = dev[[v, "city_mean_tmin_c", "sd_tmin_c"]].dropna()
        colour_mappable = ax.scatter(
            s[v], s.city_mean_tmin_c, c=s.sd_tmin_c, s=16,
            cmap="viridis", norm=colour_norm, alpha=0.85, linewidths=0,
            rasterized=True)
        r = stats.spearmanr(s[v], s.city_mean_tmin_c).statistic
        rs_ = stats.spearmanr(s[v], s.sd_tmin_c).statistic
        ax.set_xlabel(lab); ax.set_ylabel("city-mean $T_{\\min}$ (°C)")
        ax.set_title(titles[tag], loc="left")
        txt = (rf"$\rho_{{\mathrm{{warmth}}}}={r:+.2f}$" + "\n"
               rf"$\rho_{{\mathrm{{disp}}}}={rs_:+.2f}$")
        ax.text(0.04, 0.96, txt, transform=ax.transAxes, ha="left", va="top",
                fontsize=8.5, color="#20252b",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.75))
    cb = fig.colorbar(colour_mappable, cax=colour_ax)
    cb.set_label("cross-station dispersion (°C)", fontsize=8.5)
    cb.ax.tick_params(labelsize=8)
    ax = axes[2]
    sets = ["unadjusted", "clearness", "day_radiation", "wind", "clearness+wind",
            "dayrad+wind", "full(dayrad+clear+wind)", "full+precip"]
    sc = pd.read_csv(os.path.join(OUT, "v3_specification_curve.csv"))
    sub = sc[(sc.years == "dev") & (sc["sample"] == "all_nights")
             & (sc.spread_metric == "sd_tmin_c") & (sc.night_window == "20-08")
             & (sc.regime_cut == "33/67")]
    vals = [sub.loc[sub.adjustment_set == s, "rho"].mean() for s in sets]
    ax.barh(range(len(sets)), vals,
            color=[CB["lightred"] if v > 0 else CB["lightblue"] for v in vals])
    ax.set_yticks(range(len(sets)))
    NICEc = {"day_radiation": "daytime radiation", "clearness+wind": "clearness + wind",
             "dayrad+wind": "day rad + wind", "full(dayrad+clear+wind)": "full set",
             "full+precip": "full + precip"}
    ax.set_yticklabels([NICEc.get(s, s) for s in sets], fontsize=8.5)
    ax.invert_yaxis(); ax.axvline(0, color="k", lw=1.0)
    ax.set_xlabel(r"$\rho$(warmth, dispersion)")
    ax.set_title("(c) Estimates for a fixed development-sample specification",
                 loc="left")
    fig.subplots_adjust(left=0.12, right=0.94, bottom=0.08, top=0.95)
    save_pair(fig, "fig2_confounders")
    plt.close(fig); print("  F2 done")


def f2_specification_sensitivity():
    """Main Figure 2: ranked estimates and expanded analytical choices."""
    sc = pd.read_csv(os.path.join(OUT, "v3_specification_curve.csv"))
    ranked = sc.sort_values("rho").reset_index(drop=True)

    fig = plt.figure(figsize=(6.7, 7.55), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[0.92, 3.15], hspace=0.10)
    ax_rank = fig.add_subplot(gs[0])
    ax_choice = fig.add_subplot(gs[1])

    xrank = np.arange(len(ranked))
    ax_rank.scatter(
        xrank, ranked.rho, s=4.5,
        c=np.where(ranked.rho > 0, CB["lightred"], CB["lightblue"]),
        alpha=0.72, linewidths=0, rasterized=True)
    ax_rank.axhline(0, color="k", lw=1.0)
    med = ranked.rho.median()
    q1, q3 = ranked.rho.quantile([0.25, 0.75])
    conv = ranked.loc[ranked.adjustment_set == "unadjusted", "rho"].median()
    ax_rank.axhline(med, color=CB["green"], lw=1.4, ls="--",
                    label=f"median {med:+.2f}")
    ax_rank.fill_between([0, len(ranked)], q1, q3, color=CB["green"], alpha=0.10,
                         label=f"IQR [{q1:+.2f}, {q3:+.2f}]")
    ax_rank.axhline(conv, color=CB["orange"], lw=1.4, ls=":",
                    label=f"unadjusted median {conv:+.2f}")
    ax_rank.set_xlim(0, len(ranked))
    ax_rank.set_ylabel(r"Spearman $\rho$", fontsize=9.5)
    ax_rank.set_xlabel(f"Specification, ranked by estimate ($n={len(ranked):,}$)",
                       fontsize=9.5)
    ax_rank.set_title("(a) Association between city warmth and night-minimum dispersion",
                      loc="left", fontsize=10.5)
    ax_rank.legend(frameon=False, ncol=3, loc="upper left", fontsize=8.2,
                   handlelength=2.2, columnspacing=1.2)
    ax_rank.text(
        0.99, 0.06,
        f"{(ranked.rho > 0).mean():.1%} positive; {(ranked.rho < 0).mean():.1%} negative",
        transform=ax_rank.transAxes, ha="right", fontsize=8.3, color=CB["grey"])
    ax_rank.tick_params(labelsize=8.5)

    factors = [
        ("Adjustment set", "adjustment_set", [
            ("unadjusted", "unadjusted"),
            ("day_radiation", "daytime radiation"),
            ("clearness", "nocturnal clearness"),
            ("wind", "night wind"),
            ("dayrad+wind", "daytime radiation + wind"),
            ("clearness+wind", "clearness + wind"),
            ("full(dayrad+clear+wind)", "radiation + clearness + wind"),
            ("full+precip", "full set + precipitation"),
        ]),
        ("Dispersion measure", "spread_metric", [
            ("sd_tmin_c", "station standard deviation"),
            ("p90p10_tmin_c", "90th minus 10th percentile"),
            ("iqr_tmin_c", "interquartile range"),
        ]),
        ("Night sample", "sample", [
            ("all_nights", "all nights"),
            ("calm_clear_only", "calm-clear nights"),
        ]),
        ("Night window", "night_window", [
            ("20-08", "20:00 to 08:00"),
            ("21-07", "21:00 to 07:00"),
            ("22-06", "22:00 to 06:00"),
        ]),
        ("Regime thresholds", "regime_cut", [
            ("20/80", "20th and 80th percentiles"),
            ("25/75", "25th and 75th percentiles"),
            ("33/67", "33rd and 67th percentiles"),
        ]),
        ("Study period", "years", [
            ("all", "all summers"),
            ("dev", "development, 2020 to 2023"),
            ("holdout", "withheld, 2024 to 2025"),
        ]),
    ]
    y = 0.0
    yrows, ticklabels, group_headers, separators = [], [], [], []
    for group, column, levels in factors:
        group_headers.append((y, group))
        y += 0.72
        for value, label in levels:
            d = sc.loc[sc[column] == value, "rho"]
            q10, q25, q50, q75, q90 = d.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
            colour = CB["purple"] if group == "Adjustment set" else CB["grey"]
            ax_choice.plot([q10, q90], [y, y], color=colour, lw=1.1,
                           solid_capstyle="round")
            ax_choice.plot([q25, q75], [y, y], color=colour, lw=4.2,
                           solid_capstyle="round")
            ax_choice.scatter(q50, y, s=34, color=colour, edgecolor="white",
                              linewidth=0.6, zorder=3)
            yrows.append(y); ticklabels.append(label); y += 0.86
        separators.append(y - 0.25)
        y += 0.34
    ax_choice.axvline(0, color="k", lw=1.0)
    for ypos in separators[:-1]:
        ax_choice.axhline(ypos, color="#dddddd", lw=0.7)
    for ypos, group in group_headers:
        ax_choice.text(-0.40, ypos, group.upper(), fontsize=8.5,
                       fontweight="bold", color="#30343b", va="center")
    ax_choice.set_yticks(yrows)
    ax_choice.set_yticklabels(ticklabels, fontsize=8.1)
    # Reserve a clear band above the first group for the interval key. Keeping
    # the key inside that band prevents it from covering the group heading or
    # the first adjustment-set row after the figure is scaled in LaTeX.
    ax_choice.set_ylim(y + 0.05, -1.55)
    ax_choice.set_xlim(-0.40, 0.49)
    ax_choice.set_xticks(np.arange(-0.4, 0.5, 0.1))
    ax_choice.set_xlabel(r"Spearman $\rho$ across specifications containing each level",
                         fontsize=9.5)
    ax_choice.set_title("(b) Distribution associated with each analytical choice",
                        loc="left", fontsize=10.5)
    ax_choice.tick_params(axis="x", labelsize=8.5)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=CB["grey"], lw=1.1, label="10th to 90th percentile"),
        Line2D([0], [0], color=CB["grey"], lw=4.2, label="interquartile range"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=CB["grey"],
               markeredgecolor=CB["grey"], markersize=5.5, label="median"),
    ]
    ax_choice.legend(handles=handles, frameon=False, ncol=3,
                     loc="upper center", bbox_to_anchor=(0.5, 0.995),
                     fontsize=7.8, handlelength=2.1, columnspacing=1.4,
                     borderaxespad=0.0)
    ax_choice.text(0.99, 0.012,
                   "Specification distributions; CIs reported separately",
                   transform=ax_choice.transAxes, ha="right", va="bottom",
                   fontsize=7.8, color=CB["grey"])

    save_pair(fig, "fig_main_02_specification_sensitivity")
    plt.close(fig); print("  Main F2 specification sensitivity done")


def f3_meteorological_basis():
    """Main Figure 3: readable meteorological basis for adjustment sensitivity."""
    dev = C.sample(C.load_nights(), "dev")
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.55), sharey=True,
                             constrained_layout=True)
    norm = matplotlib.colors.Normalize(vmin=dev.sd_tmin_c.min(),
                                       vmax=dev.sd_tmin_c.max())
    specs = [
        (axes[0], "day_rad_wm2", "Daytime global radiation (W m$^{-2}$)",
         "(a) Daytime radiation"),
        (axes[1], "clearness_wm2", "Nocturnal clearness (W m$^{-2}$)",
         "(b) Nocturnal clearness"),
    ]
    mappable = None
    for ax, variable, xlabel, title in specs:
        dat = dev[[variable, "city_mean_tmin_c", "sd_tmin_c"]].dropna()
        mappable = ax.scatter(dat[variable], dat.city_mean_tmin_c,
                              c=dat.sd_tmin_c, s=18, cmap="viridis", norm=norm,
                              alpha=0.78, linewidths=0, rasterized=True)
        rw = stats.spearmanr(dat[variable], dat.city_mean_tmin_c).statistic
        rd = stats.spearmanr(dat[variable], dat.sd_tmin_c).statistic
        ax.text(0.04, 0.96,
                rf"with city warmth: $\rho={rw:+.2f}$" + "\n" +
                rf"with spatial dispersion: $\rho={rd:+.2f}$",
                transform=ax.transAxes, ha="left", va="top", fontsize=8.4,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#dddddd",
                          alpha=0.90))
        ax.set_xlabel(xlabel, fontsize=9.2)
        ax.set_title(title, loc="left", fontsize=10.2)
        ax.tick_params(labelsize=8.4)
    axes[0].set_ylabel(r"City-mean night $T_{\min}$ (°C)", fontsize=9.2)
    cb = fig.colorbar(mappable, ax=axes, fraction=0.035, pad=0.025)
    cb.set_label("Cross-station SD of night minimum (°C)", fontsize=8.5)
    cb.ax.tick_params(labelsize=7.8)
    save_pair(fig, "fig_main_03_meteorological_basis")
    plt.close(fig); print("  Main F3 meteorological basis done")


# ------------------------------------------------------------------ F3
def f3():
    sc = pd.read_csv(os.path.join(OUT, "v3_observable_stability_curve.csv"))
    s = pd.read_csv(os.path.join(OUT, "v3_observable_stability_summary.csv"))
    order = ["sd_tmin_c", "city_mean_cool_rate", "sd_cool_rate", "sd_cool_total"]
    labels = {"sd_tmin_c": "night-minimum dispersion (endpoint)",
              "city_mean_cool_rate": "mean cooling rate (level)",
              "sd_cool_rate": "cooling-rate dispersion (trajectory)",
              "sd_cool_total": "total-cooling dispersion (trajectory)"}
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.2, 6.2),
                                  gridspec_kw={"height_ratios": [1.15, 1.0]})
    rng = np.random.default_rng(7)
    for i, o in enumerate(order):
        g = sc[sc.outcome == o]
        y = i + rng.uniform(-0.20, 0.20, len(g))
        ax.scatter(g.rho, y, s=10,
                   c=np.where(g.rho > 0, CB["lightred"], CB["lightblue"]),
                   alpha=0.65, linewidths=0, rasterized=True)
        med = g.rho.median()
        ax.plot([med, med], [i - 0.30, i + 0.30], color="k", lw=1.8, zorder=5)
        share = (g.rho > 0).mean()
        ax.text(0.62, i, f"{share:.0%} positive", va="center", fontsize=8,
                color=CB["green"] if share == 1 else CB["grey"])
    ax.axvline(0, color="k", lw=1.1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([labels[o] for o in order], fontsize=8)
    ax.set_xlim(-0.45, 0.78)
    ax.set_xlabel(r"$\rho$(city-mean $T_{\min}$, quantity), 144 specifications each"
                  "\n(black tick = median)")
    ax.set_title("(a) Estimate distributions for four night-time observables", loc="left")

    OFFS = {"sd_cool_total": (0, 12, "center"),
            "sd_cool_rate": (0, 10, "center"),
            "sd_tmin_c": (10, 8, "left"),
            "city_mean_cool_rate": (-10, -18, "right")}
    for _, r in s.iterrows():
        c = CB["green"] if r.outcome == "sd_cool_total" else CB["grey"]
        ax2.scatter(r.adj_range, 100 * r.sign_consistency, s=80, color=c, zorder=3)
        dx, dy, ha = OFFS[r.outcome]
        short = {"sd_tmin_c": "night-min disp.", "city_mean_cool_rate": "mean rate",
                 "sd_cool_rate": "rate disp.", "sd_cool_total": "total-cooling disp."}
        ax2.annotate(short[r.outcome], (r.adj_range, 100 * r.sign_consistency),
                     textcoords="offset points", xytext=(dx, dy), fontsize=8, ha=ha)
    ax2.axhline(100, color=CB["green"], lw=0.8, ls=":")
    ax2.set_xlabel("range of median $\\rho$ across adjustment sets")
    ax2.set_ylabel("sign consistency (%)")
    ax2.set_ylim(62, 106); ax2.set_xlim(0.24, 0.445)
    ax2.set_title("(b) Sign consistency across adjustment sets", loc="left")
    fig.tight_layout()
    save_pair(fig, "fig_s1_observable_comparison")
    plt.close(fig); print("  F3 done")


# ------------------------------------------------------------------ F4
def f4():
    pc = pd.read_csv(os.path.join(OUT, "v3_profile_clock.csv"))
    ps = pd.read_csv(os.path.join(OUT, "v3_profile_sunset.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.7), sharey=True)
    for ax, pr, tag, xlab, xline in [
            (axes[0], pc, "a", "local time", 2.0),
            (axes[1], ps, "b", "hours after local sunset", 0.0)]:
        for tc in ["hot", "middle", "cool"]:
            g = pr[pr.tercile == tc].sort_values("offset_h")
            ax.plot(g.offset_h, g.sd_mean, color=TERC[tc], lw=1.8,
                    label=f"{tc} tercile")
            ax.fill_between(g.offset_h, g.ci_lo, g.ci_hi, color=TERC[tc], alpha=0.16,
                            linewidth=0)
        ax.axvline(xline, color="k", lw=0.8, ls=":")
        note = "20:00" if tag == "a" else "sunset"
        ax.text(xline, ax.get_ylim()[1] * 0.98, f" {note}", fontsize=7,
                va="top", color="k")
        ax.set_xlabel(xlab)
        ax.set_title(f"({tag}) {'Clock-aligned' if tag=='a' else 'Sunset-aligned'}",
                     loc="left")
        if tag == "a":
            ax.set_xticks(np.arange(0, 15, 2))
            ax.set_xticklabels(["18:00", "20:00", "22:00", "00:00",
                                "02:00", "04:00", "06:00", "08:00"])
    axes[0].set_ylabel("cross-station SD of temperature (°C)")
    axes[0].legend(frameon=False, loc="upper right", title="calm-clear nights,\nwarmth tercile",
                   title_fontsize=7)
    fig.suptitle("Cross-station temperature SD on 183 calm-clear nights",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    save_pair(fig, "fig4_night_profile")
    plt.close(fig); print("  F4 done")


# The former confirmatory forest plot was removed. It used whole-summer
# resampling for a two-summer holdout, which produced degenerate intervals.
# The manuscript now reports complete-calendar moving-block intervals in
# f4_nocturnal_trajectory_combined.


# ------------------------------------------------------------------ F6
def f6():
    an = pd.read_csv(os.path.join(OUT, "v3_anchor_sensitivity.csv"))
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    an = an.iloc[::-1].reset_index(drop=True)
    def nice(v):
        return (v.replace("cool_total", "total cooling,").replace("cool_rate", "cooling rate,")
                 .replace(" ev ", " ").replace("->", "\u2192").replace("..", " to "))
    for i, r in an.iterrows():
        c = CB["green"] if "base" in r.variant else CB["grey"]
        ax.plot([r.ci_lo, r.ci_hi], [i, i], color=c, lw=1.6)
        ax.scatter(r.rho, i, color=c, s=34, zorder=5,
                   marker="s" if "sunset" in r.variant else "o")
    ax.axvline(0, color="k", lw=1.0)
    ax.axhline(3.5, color="#cccccc", lw=0.8, ls="-")
    ax.set_yticks(range(len(an)))
    ax.set_yticklabels([nice(v) for v in an.variant], fontsize=7.5)
    ax.set_xlabel(r"$\rho$(warmth, cross-station SD), calm-clear nights"
                  "\n(95% year-block CI; squares = sunset-referenced)")
    ax.set_title("Sensitivity to evening and cooling-rate anchors", loc="left")
    fig.tight_layout()
    save_pair(fig, "fig_s2_anchor_sensitivity")
    plt.close(fig); print("  F6 done")


# ------------------------------------------------------------------ F7-F9: restyled reuse
def f7():
    m = pd.read_csv(os.path.join(OUT, "v3_night_synoptic_era5.csv"),
                    parse_dates=["night_date"])
    sens = pd.read_csv(os.path.join(OUT, "v3_era5_regime_sensitivity.csv"))
    fig = plt.figure(figsize=(7.2, 6.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, hspace=0.20, wspace=0.34)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    ax = axes[0]
    s = m[["clearness_wm2", "era5_cc_mean", "regime"]].dropna()
    cols = {"calm_clear": CB["red"], "mixed": CB["grey"], "cloudy_windy": CB["blue"]}
    for reg, g in s.groupby("regime"):
        ax.scatter(g.era5_cc_mean, g.clearness_wm2, s=9, alpha=0.6, c=cols[reg],
                   label=reg.replace("_", "-"), linewidths=0, rasterized=True)
    r = stats.spearmanr(s.era5_cc_mean, s.clearness_wm2).statistic
    ax.set_xlabel("model-pinned ERA5 night cloud cover (%)")
    ax.set_ylabel("longwave clearness (W m$^{-2}$)")
    ax.set_title("(a) Longwave clearness and ERA5 cloud", loc="left")
    ax.text(0.03, 0.04, rf"Spearman $\rho={r:+.2f}$", transform=ax.transAxes,
            fontsize=8.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.88,
              loc="upper right")
    ax = axes[1]
    between = m.era5_cc_mean.std()
    ax.hist(m.era5_cc_sd_spatial.dropna(), bins=30, color=CB["purple"], alpha=0.8)
    ax.axvline(between, color=CB["red"], lw=1.5, ls="--")
    ax.text(between * 0.97, ax.get_ylim()[1] * 0.80,
            f"night-to-night\nSD = {between:.0f} pp",
            ha="right", fontsize=7, color=CB["red"])
    ax.set_xlabel("night cloud SD between 2 ERA5 cells (pp)")
    ax.set_ylabel("nights")
    ax.set_title("(b) Cloud variation across two ERA5 cells", loc="left")
    ax.text(0.97, 0.97,
            f"median two-cell SD = {m.era5_cc_sd_spatial.median():.1f} pp",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.82))

    # Keep correlations and rate slopes on separate axes because their units differ.
    corr = sens.iloc[:3].copy()
    rate = sens.iloc[3:].copy()
    corr_labels = ["night-minimum\ndispersion", "adjusted night-minimum\ndispersion",
                   "total-cooling\ndispersion"]
    rate_labels = ["mean cooling\nrate", "cooling-rate\ndispersion"]
    for ax, sub, labels, tag, xlabel in [
            (axes[2], corr, corr_labels, "c", "Spearman or partial Spearman $\\rho$"),
            (axes[3], rate, rate_labels, "d", "slope (°C h$^{-1}$ per °C warmth)")]:
        yy = np.arange(len(sub)); off = 0.10
        ax.scatter(sub.estimate_longwave, yy + off, color=CB["grey"], marker="o",
                   s=38, label="longwave regime")
        ax.scatter(sub.estimate, yy - off, color=CB["green"], marker="s", s=38,
                   label="ERA5 regime")
        ax.axvline(0, color="k", lw=0.9)
        ax.set_yticks(yy); ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_title(f"({tag}) Sensitivity to regime definition", loc="left")
    handles, labels = axes[2].get_legend_handles_labels()
    axes[3].legend(handles, labels, frameon=False, loc="upper center")
    save_pair(fig, "fig7_era5_validation")
    plt.close(fig); print("  F7 done")


def f8():
    g = pd.read_csv(os.path.join(OUT, "v3_priority_screen.csv"))
    x, y = g.centroid_easting_2056.values, g.centroid_northing_2056.values
    exposure_panels = [
        ("gp_mean_night_min_c", "mean night $T_{\\min}$ (°C)\nGaussian-process field", "inferno"),
        ("gp_mean_night_min_c_sd", "predictive SD (°C)", "viridis"),
        ("p_highest_tier", "P(highest-priority tier)", "magma"),
    ]
    context_panels = [
        ("est65_qmargin", "residents 65+ per cell", "YlGnBu"),
        ("routed_refuge_distance_m", "routed walk to retained\ncanopy refuge (m)", "cividis"),
        ("canopy_cover_250m_direct", "canopy fraction (250 m)", "Greens"),
    ]

    def draw_map_set(panels, stem, heading):
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.6))
        for j, (ax, (col, title, cmap)) in enumerate(zip(axes, panels)):
            v = g[col].values; ok = np.isfinite(v)
            vmin, vmax = np.percentile(v[ok], [1, 99])
            h = ax.scatter(x[ok], y[ok], c=v[ok], s=2.7, cmap=cmap, marker="s",
                           linewidths=0, vmin=vmin, vmax=vmax, rasterized=True)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(False)
            ax.set_title(f"({'abc'[j]}) {title}", fontsize=8.5, loc="left")
            cb = plt.colorbar(h, ax=ax, fraction=0.045, pad=0.02, extend="both")
            cb.ax.tick_params(labelsize=7)
            x0, y0 = x.min() + 300, y.min() + 400
            ax.plot([x0, x0 + 2000], [y0, y0], color="k", lw=1.6)
            ax.text(x0 + 1000, y0 + 190, "2 km", ha="center", fontsize=7)
            ax.annotate("N", xy=(0.91, 0.87), xytext=(0.91, 0.70),
                        xycoords="axes fraction", textcoords="axes fraction",
                        ha="center", va="center", fontsize=8,
                        arrowprops=dict(arrowstyle="-|>", color="k", lw=1.0))
        fig.suptitle(heading, fontsize=10, y=0.99)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        save_pair(fig, stem)
        plt.close(fig)

    draw_map_set(exposure_panels, "fig8a_city_maps_exposure",
                 "Outdoor night-heat field and classification uncertainty")
    draw_map_set(context_panels, "fig8b_city_maps_context",
                 "Population and adaptation context")
    print("  F8a/F8b done")


def f9():
    g = pd.read_csv(os.path.join(OUT, "v3_priority_screen.csv"))
    decision = pd.read_csv(os.path.join(
        OUT, "v3_priority_decision_multiverse_cell.csv"))
    decision_summary = pd.read_csv(os.path.join(
        OUT, "v3_priority_decision_multiverse_summary.csv"))
    fig = plt.figure(figsize=(7.2, 6.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.9])
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, :])]
    ax = axes[0]
    order = ["baseline_monitoring", "elevated_monitoring", "high_priority",
             "highest_priority"]
    short = ["baseline", "elevated", "high", "highest"]
    tab = pd.crosstab(g.tier_idw, g.tier_gp).reindex(index=order, columns=order,
                                                     fill_value=0)
    im = ax.imshow(tab.values, cmap="Blues")
    for iy in range(4):
        for ix in range(4):
            val = int(tab.iloc[iy, ix])
            ax.text(ix, iy, f"{val}", ha="center", va="center", fontsize=8,
                    color="white" if val > tab.values.max() * 0.45 else "black")
    ax.set_xticks(range(4)); ax.set_xticklabels(short, rotation=35, ha="right", fontsize=7)
    ax.set_yticks(range(4)); ax.set_yticklabels(short, fontsize=7)
    ax.set_xlabel("GP tier"); ax.set_ylabel("IDW tier")
    ax.set_title("(a) Tier transitions after changing\nheat-field interpolation", loc="left")
    ax = axes[1]
    scopes = [
        ("all_heat_fields_x_weights_x_population_lenses",
         "all decision choices", CB["purple"], "o"),
        ("weights_at_qmargin", "indicator weights", CB["green"], "s"),
        ("population_lenses_at_baseline_weights",
         "population estimates", CB["red"], "^"),
    ]
    for scope, label, colour, marker in scopes:
        sub = decision_summary[decision_summary.scope == scope].sort_values(
            "top_percent")
        stable_share = (
            sub.n_baseline_frequency_ge_0_90 / sub.n_baseline_selected
        )
        ax.plot(sub.top_percent, stable_share, marker=marker, color=colour,
                label=label)
    ax.set_xlabel("illustrative share of cells prioritised (%)")
    ax.set_ylabel(r"baseline cells retained in $\geq$90% of scenarios")
    ax.set_ylim(0, 1)
    ax.set_xticks([5, 10, 15, 20])
    ax.legend(frameon=False, fontsize=7.0)
    ax.set_title("(b) Priority retention by monitoring share", loc="left")
    ax = axes[2]
    hi = decision[decision.baseline_qmargin_top10 == 1]
    ax.hist(hi.decision_frequency_top10_all, bins=np.linspace(0, 1, 21),
            color=CB["green"], alpha=0.88)
    ax.axvline(0.90, color=CB["red"], lw=1.4, ls="--")
    n90 = int((hi.decision_frequency_top10_all >= 0.90).sum())
    n_specs = int(decision_summary.loc[
        (decision_summary.top_percent == 10)
        & decision_summary.scope.str.startswith("all_heat_fields"),
        "n_specifications"].iloc[0])
    ax.set_xlabel(f"retention frequency across {n_specs} decision specifications")
    ax.set_ylabel("baseline top-decile cells")
    ax.set_title("(c) Sensitivity to field, weights and population data", loc="left")
    ax.text(0.03, 0.94, rf"{n90} of {len(hi)} cells at frequency $\geq0.90$",
            transform=ax.transAxes, va="top", fontsize=8)
    save_pair(fig, "fig_main_08_decision_sensitivity")
    plt.close(fig); print("  F9 done")


# ------------------------------------------------------------------ F4b (functional)
def f4b_trajectory_framework():
    """New centrepiece: functional dispersion curve by warmth level + the three
    trajectory scalars confirmed on withheld data."""
    fit = pd.read_csv(os.path.join(OUT, "v3_traj_functional_fit.csv"))
    conf = pd.read_csv(os.path.join(OUT, "v3_traj_confirmatory.csv"))
    calendar = pd.read_csv(os.path.join(os.path.dirname(OUT), "outputs_robust",
                                        "v3_17_dusk_associations.csv"))
    trajectory_calendar = pd.read_csv(os.path.join(
        os.path.dirname(OUT), "outputs_robust",
        "v3_17_trajectory_associations.csv"))
    lvlcol = {"cool (10th pct)": CB["blue"], "mean warmth": CB["grey"],
              "hot (90th pct)": CB["red"]}
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.2, 6.3),
                                  gridspec_kw={"height_ratios": [1.1, 1.0]})
    # (a) fitted curve
    for lvl, c in lvlcol.items():
        g = fit[fit.warmth_level == lvl].sort_values("hss")
        ax.plot(g.hss, g.sd_fit, color=c, lw=2.0, label=lvl.replace(" (", "\n("))
        ax.fill_between(g.hss, g.ci_lo, g.ci_hi, color=c, alpha=0.15, linewidths=0)
    ax.set_xlabel("hours after local sunset")
    ax.set_ylabel("modelled cross-station SD (°C)")
    ax.set_title("(a) Functional dispersion curve by city warmth", loc="left")
    ax.legend(frameon=False, fontsize=7, title="calm-clear\nnights", title_fontsize=7)
    ax.text(0.02, 0.03, "Descriptive curves; pointwise calendar-block intervals",
            transform=ax.transAxes, fontsize=6.8, color=CB["grey"], va="bottom")

    # (b) trajectory scalars: dev vs withheld rho with CIs
    order = ["dusk_sd", "integ_sd", "decay"]
    lab = {"dusk_sd": "dusk dispersion", "integ_sd": "integrated dispersion (°C·h)",
           "decay": "overnight change in dispersion"}
    for i, sc in enumerate(order[::-1]):
        for smp, dy, c, mk in [("development", +0.13, CB["grey"], "o"),
                               ("withheld", -0.13, CB["green"], "s")]:
            r = conf[(conf.scalar == sc) & (conf["sample"] == smp)].iloc[0]
            ax2.plot([r.ci_lo, r.ci_hi], [i + dy] * 2, color=c, lw=1.6)
            ax2.scatter(r.rho, i + dy, color=c, marker=mk, s=34, zorder=5)
        ax2.axhline(i - 0.5, color="#eee", lw=0.6, zorder=0)
    ax2.axvline(0, color="k", lw=1.0)
    ax2.set_yticks(range(3))
    ax2.set_yticklabels([lab[s] for s in order[::-1]], fontsize=8.5)
    ax2.set_xlabel(r"$\rho$ with city warmth (95% CI)")
    ax2.set_title("(b) Trajectory scalars", loc="left")
    h1 = plt.Line2D([], [], color=CB["grey"], marker="o", ls="-", label="development")
    h2 = plt.Line2D([], [], color=CB["green"], marker="s", ls="-", label="withheld")
    ax2.legend(handles=[h1, h2], frameon=False, fontsize=8, loc="lower right",
               title="calm-clear nights\n123 development, 60 withheld",
               title_fontsize=8)
    fig.tight_layout()
    save_pair(fig, "fig4b_trajectory_framework")
    plt.close(fig); print("  F4b done")


# ------------------------------------------------------------------ F10 (station traits)
def f10_station_traits():
    """Repeated spatial prediction and leave-one-variable-out importance."""
    m = pd.read_csv(os.path.join(OUT, "v3_station_trait_models.csv"))
    st = pd.read_csv(os.path.join(OUT, "v3_station_traits.csv"))
    targets = ["dusk_anom", "integ_anom", "decay_tend"]
    tlab = {"dusk_anom": "dusk anomaly", "integ_anom": "integrated anomaly",
            "decay_tend": "overnight change tendency"}
    plab = {"elevation": "elevation", "bldg_frac": "building fraction",
            "canopy": "canopy cover"}
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4))
    axes = axes.ravel()
    preds = ["elevation", "bldg_frac", "canopy"]
    for i, (ax, tgt) in enumerate(zip(axes[:3], targets)):
        g = m[m.target == tgt].set_index("predictor").reindex(preds[::-1])
        value = g.delta_cv_r2_drop.to_numpy()
        lo = value - g.delta_cv_r2_q05.to_numpy()
        hi = g.delta_cv_r2_q95.to_numpy() - value
        cols = [CB["green"] if v > 0 else "#9aa1aa" for v in value]
        ax.barh(range(len(g)), value, color=cols, alpha=0.9)
        ax.errorbar(value, range(len(g)), xerr=np.vstack([lo, hi]), fmt="none",
                    ecolor="#20252b", elinewidth=0.8, capsize=2, zorder=4)
        ax.set_yticks(range(len(g)))
        ax.set_yticklabels([plab[p] for p in g.index], fontsize=8)
        ax.axvline(0, color="k", lw=1.0)
        ax.set_xlim(-0.065, 0.36)
        ax.set_xlabel(r"drop in out-of-block $R^2$ when omitted")
        ax.set_title(
            f"({'abc'[i]}) {tlab[tgt]}\n"
            f"median spatial CV $R^2$={g.cv_r2.iloc[0]:.2f} "
            f"[{g.cv_r2_q05.iloc[0]:.2f}, {g.cv_r2_q95.iloc[0]:.2f}]",
            loc="left")
    # 4th panel: the signature scatter -- building fraction vs dusk anomaly
    ax = axes[3]
    h = ax.scatter(st.bldg_frac, st.dusk_anom, c=st.elevation, cmap="viridis",
                   s=22, linewidths=0)
    ax.set_xlabel("building fraction (250 m)")
    ax.set_ylabel("station dusk anomaly (°C)")
    ax.set_title("(d) Building fraction and station dusk anomaly", loc="left")
    cb = plt.colorbar(h, ax=ax, pad=0.02); cb.set_label("elevation (m)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig.suptitle("Repeated geographic holdouts for station trajectory traits "
                 "($n=93$; bars are medians, lines are 5th--95th percentiles)",
                 fontsize=10, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_pair(fig, "fig_main_05_station_traits")
    plt.close(fig); print("  F10 done")


# ------------------------------------------------------------------ F0 (DAG)
def f0_dag():
    """Causal diagram for the adjustment sensitivity of night minima."""
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.colors import to_rgba
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.set_xlim(0, 12.6); ax.set_ylim(0, 7.0); ax.axis("off")

    nodes = {
        # key: x, y, width, height, label, pathway colour
        "dayrad": (1.45, 5.55, 2.20, 0.92, "daytime\nradiation", CB["orange"]),
        "clear":  (1.45, 1.45, 2.20, 0.92, "nocturnal\nclearness", CB["blue"]),
        "warmth": (4.50, 3.50, 2.25, 0.92, "city-mean\nnight warmth", CB["grey"]),
        "dusk":   (7.80, 5.55, 2.45, 0.92, "evening spatial\ncontrast", CB["orange"]),
        "cool":   (7.80, 1.45, 2.45, 0.92, "differential\novernight cooling", CB["blue"]),
        "disp":   (11.30, 3.50, 2.35, 0.92, "night-minimum\ndispersion", CB["grey"]),
    }
    patch = {}
    for key, (x, y, w, h, label, colour) in nodes.items():
        neutral = key in {"warmth", "disp"}
        edge = "#3f4752" if neutral else colour
        fill = "#f1f2f4" if neutral else to_rgba(colour, 0.13)
        p = mpatches.FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.06", fc=fill, ec=edge, lw=1.3, zorder=3)
        ax.add_patch(p); patch[key] = p
        ax.text(x, y, label, ha="center", va="center", fontsize=8.4,
                color="#20252b", zorder=4)

    def arrow(a, b, colour, rad=0.0, dashed=False, lw=1.5, z=2):
        p = FancyArrowPatch(nodes[a][:2], nodes[b][:2],
                            connectionstyle=f"arc3,rad={rad}",
                            arrowstyle="-|>", mutation_scale=13, lw=lw,
                            linestyle=(0, (6, 3)) if dashed else "-",
                            color=colour, patchA=patch[a], patchB=patch[b],
                            shrinkA=1.5, shrinkB=1.5, zorder=z)
        ax.add_patch(p)

    # Separate daytime/evening and nocturnal process lanes.
    arrow("dayrad", "dusk", CB["orange"])
    arrow("clear", "cool", CB["blue"])
    arrow("dusk", "disp", CB["orange"])
    arrow("cool", "disp", CB["blue"])
    # The two meteorological variables are common causes of warmth and dispersion.
    arrow("dayrad", "warmth", CB["orange"])
    arrow("clear", "warmth", CB["blue"])
    # The target arrow is isolated between the two conventional summaries.
    arrow("warmth", "disp", "#20252b", dashed=True, lw=1.7, z=5)

    ax.text(3.25, 4.53, "$+$", color=CB["orange"], fontsize=13,
            ha="center", va="center", zorder=6)
    ax.text(3.25, 2.47, "$-$", color=CB["blue"], fontsize=13,
            ha="center", va="center", zorder=6)
    ax.text(7.90, 3.93, "$\\theta$", fontsize=13, style="italic",
            ha="center", va="center", color="#20252b", zorder=6)

    ax.text(1.45, 6.60, "METEOROLOGICAL\nDRIVERS",
            ha="center", va="center", fontsize=7.3, color=CB["grey"])
    ax.text(4.50, 6.60, "CITY CONDITION",
            ha="center", va="center", fontsize=7.3, color=CB["grey"])
    ax.text(7.80, 6.60, "COOLING-TRAJECTORY\nCOMPONENTS",
            ha="center", va="center", fontsize=7.3, color=CB["grey"])
    ax.text(11.30, 6.60, "NIGHT-MINIMUM\nENDPOINT",
            ha="center", va="center", fontsize=7.3, color=CB["grey"])

    ax.plot([2.05, 2.65], [0.35, 0.35], color="#3f4752", lw=1.5)
    ax.text(2.80, 0.35, "hypothesised process path", va="center", fontsize=7.4,
            color=CB["grey"])
    ax.plot([7.05, 7.65], [0.35, 0.35], color="#20252b", lw=1.7, ls=(0, (6, 3)))
    ax.text(7.80, 0.35, "target effect $\\theta$ (not identified)",
            va="center", fontsize=7.4, color=CB["grey"])

    fig.tight_layout(pad=0.3)
    save_pair(fig, "fig_main_01_dag")
    plt.close(fig); print("  F0 (DAG) done")


# ------------------------------------------------------------------ F7b (FITNAH)
def f7b_fitnah():
    import pandas as pd
    s = pd.read_csv(os.path.join(os.path.dirname(OUT), "outputs_robust",
                                 "citywide_grid_priority_screen.csv"))
    fv = pd.read_csv(os.path.join(OUT, "v3_fitnah_validation.csv"))
    d = s.dropna(subset=["ka_temp_night", "gp_mean_night_min_c"])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.5),
                                  gridspec_kw={"width_ratios": [1.0, 1.25]})
    # (a) scatter GP vs FITNAH
    hb = ax.hexbin(d.ka_temp_night, d.gp_mean_night_min_c, gridsize=32,
                   cmap="viridis", mincnt=1, linewidths=0)
    raw = fv[fv.quantity.str.contains("Spearman", regex=False) &
             ~fv.quantity.str.contains("detrended", case=False)].iloc[0]
    det = fv[fv.quantity.str.contains("detrended", case=False)].iloc[0]
    rho, lo, hi = raw["value"], raw["ci_lo"], raw["ci_hi"]
    ax.set_xlabel("FITNAH modelled night temperature (°C)")
    ax.set_ylabel("observed GP night $T_{\\min}$ (°C)")
    ax.set_title(rf"(a) $\rho$ = {rho:.2f} [{lo:.2f}, {hi:.2f}]", loc="left")
    ax.text(0.03, 0.97, rf"coordinate-detrended $\rho$={det['value']:.2f}",
            transform=ax.transAxes, va="top", fontsize=8, color="#20252b",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.82))
    cb = plt.colorbar(hb, ax=ax, pad=0.02); cb.set_label("cells", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    # (b) cold-air drainage vs observed field
    dc = s.dropna(subset=["ka_coldair_flow", "gp_mean_night_min_c"])
    quality = pd.read_csv(os.path.join(os.path.dirname(OUT), "data_external",
                                      "fitnah_coldflow_extraction_quality.csv"))
    affected = quality.loc[quality.negative_code_pixel_count > 0, "recordid"]
    dc = dc[(dc.ka_coldair_flow >= 0) & ~dc.recordid.isin(affected)]
    hb2 = ax2.hexbin(np.log1p(dc.ka_coldair_flow), dc.gp_mean_night_min_c, gridsize=30,
                    cmap="cividis", mincnt=1, linewidths=0)
    rc = fv[fv.quantity.str.contains("cold-air", case=False)].iloc[0]["value"]
    ax2.set_xlabel("FITNAH cold-air drainage flow  (log$_{1+x}$)")
    ax2.set_ylabel("observed GP night $T_{\\min}$ (°C)")
    ax2.set_title(rf"(b) Cold-air drainage and observed temperature ($\rho$={rc:.2f})",
                  loc="left")
    cb2 = plt.colorbar(hb2, ax=ax2, pad=0.02); cb2.set_label("cells", fontsize=8)
    cb2.ax.tick_params(labelsize=7)
    fig.suptitle("Spatial association with the retained Klimaanalyse (FITNAH) model",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    save_pair(fig, "fig7b_fitnah_validation")
    plt.close(fig); print("  F7b (FITNAH) done")


def f4_nocturnal_trajectory_combined():
    """Main Figure 4: modelled, empirical and withheld trajectory evidence."""
    fit = pd.read_csv(os.path.join(OUT, "v3_traj_functional_fit.csv"))
    conf = pd.read_csv(os.path.join(OUT, "v3_traj_confirmatory.csv"))
    calendar = pd.read_csv(os.path.join(os.path.dirname(OUT), "outputs_robust",
                                        "v3_17_dusk_associations.csv"))
    pc = pd.read_csv(os.path.join(OUT, "v3_profile_clock.csv"))
    ps = pd.read_csv(os.path.join(OUT, "v3_profile_sunset.csv"))
    trajectory_calendar = pd.read_csv(os.path.join(
        os.path.dirname(OUT), 'outputs_robust', 'v3_17_trajectory_associations.csv'))

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.3), constrained_layout=True)
    ax_fit, ax_sunset, ax_clock, ax_holdout = axes.ravel()
    lvlcol = {"cool (10th pct)": CB["blue"], "mean warmth": CB["grey"],
              "hot (90th pct)": CB["red"]}
    for level, colour in lvlcol.items():
        dat = fit[fit.warmth_level == level].sort_values("hss")
        ax_fit.plot(dat.hss, dat.sd_fit, color=colour, lw=1.9,
                    label=level.replace(" (", "\n("))
        ax_fit.fill_between(dat.hss, dat.ci_lo, dat.ci_hi, color=colour,
                            alpha=0.15, linewidths=0)
    ax_fit.set_xlabel("hours after local sunset")
    ax_fit.set_ylabel("modelled cross-station SD (°C)")
    ax_fit.set_title("(a) Fitted curves by city warmth", loc="left")
    ax_fit.legend(frameon=False, fontsize=6.8, loc="upper right",
                  title="city warmth", title_fontsize=6.8)
    ax_fit.axvspan(8.05, 9.0, color="#f2c98d", alpha=0.22, linewidth=0)

    def profile_panel(ax, profile, tag, title, xlabel, line_x):
        for tercile in ["hot", "middle", "cool"]:
            dat = profile[profile.tercile == tercile].sort_values("offset_h")
            ax.plot(dat.offset_h, dat.sd_mean, color=TERC[tercile], lw=1.7)
            ax.fill_between(dat.offset_h, dat.ci_lo, dat.ci_hi,
                            color=TERC[tercile], alpha=0.15, linewidths=0)
        ax.axvline(line_x, color="k", lw=0.8, ls=":")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("empirical cross-station SD (°C)")
        ax.set_title(f"({tag}) {title}", loc="left")

    profile_panel(ax_sunset, ps, "b", "Sunset-aligned profiles by warmth tercile",
                  "hours after local sunset", 0.0)
    ax_sunset.axvspan(8.05, float(ps.offset_h.max()),
                      color="#f2c98d", alpha=0.22, linewidth=0)
    ax_sunset.text(8.02, 0.04, "earliest\napparent sunrise",
                   transform=ax_sunset.get_xaxis_transform(), ha="right",
                   va="bottom", fontsize=6.1, color=CB["grey"])
    ax_sunset.text(0.02, 0.96, "sunset", transform=ax_sunset.transAxes,
                   va="top", fontsize=6.8)
    profile_panel(ax_clock, pc, "c", "Clock-aligned profiles by warmth tercile",
                  "local time", 2.0)
    ax_clock.set_xticks(np.arange(0, 15, 2))
    ax_clock.set_xticklabels(["18:00", "20:00", "22:00", "00:00",
                              "02:00", "04:00", "06:00", "08:00"],
                             fontsize=6.8)
    ax_clock.text(0.14, 0.96, "20:00", transform=ax_clock.transAxes,
                  va="top", fontsize=6.8)

    scalar_order = ["decay", "integ_sd", "dusk_sd"]
    scalar_labels = {"dusk_sd": "dusk dispersion",
                     "integ_sd": "integrated dispersion (°C h)",
                     "decay": "post-peak change"}
    for i, scalar in enumerate(scalar_order):
        for sample, dy, colour, marker in [
                ("development", 0.13, CB["grey"], "o"),
                ("withheld", -0.13, CB["green"], "s")]:
            row = conf[(conf.scalar == scalar) & (conf["sample"] == sample)].iloc[0]
            if scalar == "dusk_sd":
                scope = ("development" if sample == "development"
                         else "internal_temporal_evaluation")
                row = calendar[
                    (calendar.analysis_scope == scope)
                    & (calendar.station_set == "dynamic_all")
                    & (calendar.timestamp_convention == "interval_midpoint")
                    & (calendar.radiation_correction_policy ==
                       "published_corrected_values")
                    & (calendar.estimand == "spearman")
                ].rename(columns={"ci_low": "ci_lo", "ci_high": "ci_hi"}).iloc[0]
            else:
                metric = ("integrated_0_9h" if scalar == "integ_sd"
                          else "post_peak_slope_0_9h")
                scope = ("development" if sample == "development"
                         else "internal_temporal_evaluation")
                row = trajectory_calendar[
                    (trajectory_calendar.analysis_scope == scope) &
                    (trajectory_calendar.metric == metric)
                ].rename(columns={"ci_low": "ci_lo", "ci_high": "ci_hi"}).iloc[0]
            ax_holdout.plot([row.ci_lo, row.ci_hi], [i + dy, i + dy],
                            color=colour, lw=1.5)
            ax_holdout.scatter(row.rho, i + dy, color=colour, marker=marker,
                               s=28, zorder=4)
        ax_holdout.axhline(i - 0.5, color="#eeeeee", lw=0.6, zorder=0)
    ax_holdout.axvline(0, color="k", lw=0.9)
    ax_holdout.set_yticks(range(3))
    ax_holdout.set_yticklabels([scalar_labels[s] for s in scalar_order], fontsize=7.2)
    ax_holdout.set_xlabel(r"$\rho$ with city warmth (95% CI)")
    ax_holdout.set_title("(d) Development and internal temporal evaluation",
                         loc="left")
    dev_handle = plt.Line2D([], [], color=CB["grey"], marker="o", ls="-",
                            label="development")
    held_handle = plt.Line2D([], [], color=CB["green"], marker="s", ls="-",
                            label="internal evaluation")
    ax_holdout.legend(handles=[dev_handle, held_handle], frameon=False,
                      fontsize=7, loc="lower right")

    save_pair(fig, "fig_main_04_nocturnal_trajectory")
    plt.close(fig); print("  Main F4 trajectory done")


def f6_fitnah_validation():
    """Main Figure 6: spatial comparison with the retained FITNAH fields."""
    screen = pd.read_csv(os.path.join(os.path.dirname(OUT), "outputs_robust",
                                      "v3_spatial_rebuild_grid.csv"))
    comparisons = pd.read_csv(os.path.join(os.path.dirname(OUT), "outputs_robust",
                                            "v3_spatial_rebuild_comparisons.csv"))
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.5, 3.55),
                                     constrained_layout=True)

    field = "gp_adjusted_mean_night_min_c"
    dat = screen.dropna(subset=["ka_temp_night", field])
    hb = ax_a.hexbin(dat.ka_temp_night, dat[field], gridsize=29,
                     cmap="viridis", mincnt=1, linewidths=0,
                     edgecolors="none", rasterized=True)
    raw = comparisons[
        comparisons.comparison == "rebuilt adjusted GP vs retained FITNAH"].iloc[0]
    detrended = comparisons[
        comparisons.comparison ==
        "rebuilt adjusted GP vs retained FITNAH (quadratic-coordinate detrended)"
    ].iloc[0]
    ax_a.set_xlabel("FITNAH night temperature (°C)", fontsize=9.0)
    ax_a.set_ylabel(r"Coverage-adjusted GP night $T_{\min}$ (°C)", fontsize=9.0)
    ax_a.set_title(
        rf"(a) Temperature field: $\rho={raw['spearman_rho']:.2f}$" + "\n" +
        rf"Coordinate-detrended: $\rho={detrended['spearman_rho']:.2f}$",
        loc="left", fontsize=8.9, linespacing=1.35)
    cb = fig.colorbar(hb, ax=ax_a, pad=0.015, fraction=0.05)
    cb.set_label("Grid cells", fontsize=8.0); cb.ax.tick_params(labelsize=7.5)

    cold = screen.dropna(subset=["ka_coldair_flow", field])
    quality = pd.read_csv(os.path.join(os.path.dirname(OUT), "data_external",
                                      "fitnah_coldflow_extraction_quality.csv"))
    affected = quality.loc[quality.negative_code_pixel_count > 0, "recordid"]
    cold = cold[(cold.ka_coldair_flow >= 0) & ~cold.recordid.isin(affected)]
    hb2 = ax_b.hexbin(np.log1p(cold.ka_coldair_flow), cold[field],
                      gridsize=28, cmap="cividis", mincnt=1, linewidths=0,
                      edgecolors="none", rasterized=True)
    rcold = stats.spearmanr(cold.ka_coldair_flow, cold[field]).statistic
    ax_b.set_xlabel("FITNAH cold-air drainage flow, log(1 + x)", fontsize=9.0)
    ax_b.set_ylabel(r"Coverage-adjusted GP night $T_{\min}$ (°C)", fontsize=9.0)
    ax_b.set_title(rf"(b) Cold-air drainage: $\rho={rcold:.2f}$" + "\n ",
                   loc="left", fontsize=8.9, linespacing=1.35)
    cb2 = fig.colorbar(hb2, ax=ax_b, pad=0.015, fraction=0.05)
    cb2.set_label("Grid cells", fontsize=8.0); cb2.ax.tick_params(labelsize=7.5)
    for ax in (ax_a, ax_b):
        ax.tick_params(labelsize=8.2)

    save_pair(fig, "fig_main_06_fitnah_validation")
    plt.close(fig); print("  Main F6 FITNAH validation done")


def fs3_era5_sensitivity():
    """Supplementary Figure S3: ERA5 temporal sensitivity checks."""
    era = pd.read_csv(os.path.join(OUT, "v3_night_synoptic_era5.csv"),
                      parse_dates=["night_date"])
    sensitivity = pd.read_csv(os.path.join(OUT, "v3_era5_regime_sensitivity.csv"))
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 6.6))
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.08, top=0.91,
                        hspace=0.78, wspace=0.58)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    scat = era[["clearness_wm2", "era5_cc_mean", "regime"]].dropna()
    regime_colours = {"calm_clear": CB["red"], "cloudy_windy": CB["blue"],
                      "mixed": "#9ba8b6"}
    regime_labels = {"calm_clear": "Calm-clear nights",
                     "cloudy_windy": "Cloudy-windy nights",
                     "mixed": "Mixed nights"}
    for regime in ["calm_clear", "cloudy_windy", "mixed"]:
        group = scat[scat.regime == regime]
        ax_a.scatter(group.era5_cc_mean, group.clearness_wm2, s=13, alpha=0.68,
                     c=regime_colours[regime], label=regime_labels[regime],
                     linewidths=0, rasterized=True)
    rera = stats.spearmanr(scat.era5_cc_mean, scat.clearness_wm2).statistic
    ax_a.set_xlabel("ERA5 night cloud cover (%)", fontsize=9.0)
    ax_a.set_ylabel("Measured longwave clearness (W m$^{-2}$)", fontsize=9.0)
    ax_a.set_title(rf"(a) Measured and ERA5 indicators: $\rho={rera:.2f}$",
                   loc="left", fontsize=9.4)
    regime_legend = ax_a.legend(
        title="Station-defined regime", frameon=True, fontsize=7.2,
        title_fontsize=7.6, loc="upper right", borderpad=0.45,
        handletextpad=0.45, labelspacing=0.35, framealpha=0.94)
    regime_legend.get_frame().set_edgecolor("#c7ccd1")
    regime_legend.get_frame().set_linewidth(0.6)

    spatial_sd = era.era5_cc_sd_spatial.median()
    temporal_sd = era.era5_cc_mean.std()
    labels = ["Between-cell variation", "Between-night variation"]
    vals = [spatial_sd, temporal_sd]
    bars = ax_b.barh([0, 1], vals, color=[CB["purple"], CB["orange"]], height=0.55)
    ax_b.set_yticks([])
    ax_b.invert_yaxis()
    ax_b.set_xlabel("Cloud-cover variation (percentage points)", fontsize=9.0)
    ax_b.set_title("(b) Spatial and temporal ERA5 variation", loc="left", fontsize=9.4)
    ax_b.set_xlim(0, max(vals) * 1.18)
    for ypos, label in enumerate(labels):
        ax_b.text(0.02, ypos - 0.31, label,
                  transform=ax_b.get_yaxis_transform(), ha="left", va="bottom",
                  fontsize=8.0)
    for bar, value in zip(bars, vals):
        ax_b.text(value + 0.7, bar.get_y() + bar.get_height() / 2,
                  f"{value:.1f} pp", va="center", fontsize=8.5)

    def paired_panel(ax, subset, labels, title, xlabel, xlim):
        yy = np.arange(len(subset))
        for y, (_, row) in zip(yy, subset.iterrows()):
            ax.plot([row.estimate_longwave, row.estimate], [y, y],
                    color="#aeb5bd", lw=1.5, zorder=1)
            ax.scatter(row.estimate_longwave, y, color=CB["grey"], marker="o",
                       s=38, zorder=3)
            ax.scatter(row.estimate, y, color=CB["green"], marker="s",
                       s=38, zorder=3)
        ax.axvline(0, color="k", lw=0.9)
        ax.set_yticks(yy); ax.set_yticklabels(labels, fontsize=8.0)
        ax.set_ylim(len(subset) - 0.5, -0.5)
        ax.set_xlim(*xlim)
        ax.set_xlabel(xlabel, fontsize=9.0)
        ax.set_title(title, loc="left", fontsize=9.4)
        ax.tick_params(axis="x", labelsize=8.0)

    corr = sensitivity[sensitivity.id.isin(["H1", "H2", "H3"])].copy()
    rate = sensitivity[sensitivity.id.isin(["H4", "H5"])].copy()
    paired_panel(ax_c, corr,
                 ["Night-minimum dispersion", "Adjusted minimum dispersion",
                  "Total-cooling dispersion"],
                 "(c) Correlations under the two definitions",
                 r"Spearman or partial $\rho$", (-0.03, 0.61))
    paired_panel(ax_d, rate,
                 ["Mean cooling rate", "Cooling-rate dispersion"],
                 "(d) Cooling-rate slopes under the two definitions",
                 "Slope (°C h$^{-1}$ per °C)", (-0.004, 0.046))
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=CB["grey"],
               markeredgecolor="none", label="station longwave definition"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=CB["green"],
               markeredgecolor="none", label="ERA5 cloud definition"),
    ]
    comparison_legend = fig.legend(
        handles=handles, title="Definitions compared in panels (c) and (d)",
        frameon=False, ncol=2, loc="center",
        bbox_to_anchor=(0.55, 0.49), fontsize=8.0,
        title_fontsize=8.1, columnspacing=1.8, handletextpad=0.45)
    comparison_legend._legend_box.align = "center"
    for ax in (ax_a, ax_b):
        ax.tick_params(labelsize=8.0)
    save_pair(fig, "fig_s3_era5_sensitivity")
    plt.close(fig); print("  Supplementary F3 ERA5 sensitivity done")


def f7_spatial_context_combined():
    """Main Figure 7: all six validated heat, uncertainty and context layers."""
    grid = pd.read_csv(os.path.join(os.path.dirname(OUT), "outputs_robust",
                                    "v3_spatial_rebuild_grid.csv"))
    decision = pd.read_csv(os.path.join(
        OUT, "v3_priority_decision_multiverse_cell.csv"),
        usecols=["recordid", "decision_frequency_top10_all"])
    grid = grid.merge(decision, on="recordid", how="left", validate="one_to_one")
    x = grid.centroid_easting_2056.values
    y = grid.centroid_northing_2056.values
    panels = [
        ("gp_adjusted_mean_night_min_c",
         r"coverage-adjusted mean night $T_{\min}$ (°C)", "inferno"),
        ("gp_adjusted_mean_night_min_c_sd",
         "uncertainty in temperature estimate\n(standard deviation, °C)",
         "viridis"),
        ("decision_frequency_top10_all",
         "selection frequency across decision scenarios", "magma"),
        ("est65_qmargin", "residents aged 65+ per cell", "YlGnBu"),
        ("bldg_footprint_frac_250m_direct", "building fraction (250 m)", "cividis"),
        ("canopy_cover_250m_direct", "canopy fraction (250 m)", "Greens"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.5, 6.3), constrained_layout=True)
    for idx, (ax, (column, title, cmap)) in enumerate(zip(axes.ravel(), panels)):
        values = grid[column].values
        valid = np.isfinite(values)
        vmin, vmax = np.percentile(values[valid], [1, 99])
        artist = ax.scatter(x[valid], y[valid], c=values[valid], s=2.8, cmap=cmap,
                            marker="s", linewidths=0, vmin=vmin, vmax=vmax,
                            rasterized=True)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"({'abcdef'[idx]}) {title}", fontsize=8.2, loc="left")
        cb = fig.colorbar(artist, ax=ax, fraction=0.045, pad=0.018, extend="both")
        cb.ax.tick_params(labelsize=6.3)
        x0, y0 = x.min() + 300, y.min() + 400
        ax.plot([x0, x0 + 2000], [y0, y0], color="k", lw=1.4)
        ax.text(x0 + 1000, y0 + 190, "2 km", ha="center", fontsize=6.5)
        ax.annotate("N", xy=(0.91, 0.87), xytext=(0.91, 0.70),
                    xycoords="axes fraction", textcoords="axes fraction",
                    ha="center", va="center", fontsize=7,
                    arrowprops=dict(arrowstyle="-|>", color="k", lw=0.9))
    save_pair(fig, "fig_main_07_spatial_context")
    plt.close(fig); print("  Main F7 spatial context done")


if __name__ == "__main__":
    print("publication figures ->", FIG)
    f2_specification_sensitivity()
    f3_meteorological_basis()
    f4_nocturnal_trajectory_combined()
    f10_station_traits()
    f7_spatial_context_combined()
    f9()
    f3(); f6(); fs3_era5_sensitivity()
    print("done")
