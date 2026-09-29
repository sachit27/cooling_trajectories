"""Revision figures (September 2026).

All maps use CH1903+/LV95 (EPSG:2056), a common base (neighbouring
municipalities, Lake Zurich, city districts and boundary), classed sequential
or diverging colour schemes with a neutral midpoint, one north arrow and one
scale bar per figure, and horizontal class legends.  Charts use one inference
method throughout (summer-stratified 14-day calendar blocks).
"""
from __future__ import annotations

import json
import os
import sys

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch, Polygon as MplPolygon, Rectangle
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V3 = os.path.join(ROOT, "outputs_v3")
RB = os.path.join(ROOT, "outputs_robust")
RV = os.path.join(ROOT, "outputs_revision")
GIS = os.path.join(ROOT, "gis")
FIG = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

# ---------------------------------------------------------------- style
INK, INK2, MUTED, GRIDC = "#1f1f1f", "#4a4a48", "#8a8a86", "#e6e5e1"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
WARM, MID, COOL = "#d4502c", "#8a8a86", "#2a78d6"
DEV_C, EVAL_C = "#6f6f6b", "#2a78d6"
LAKE, LAND_OUT, LAND_EDGE = "#d6e7f3", "#f3f2ee", "#d7d5cf"
mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelsize": 7.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "legend.fontsize": 7, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
    "savefig.dpi": 300, "figure.dpi": 150, "lines.linewidth": 1.4,
})
W = 7.3  # inches, full text width


def save(fig, stem):
    fig.savefig(os.path.join(FIG, stem + ".pdf"), bbox_inches="tight", facecolor="white",
                metadata={"Creator": "v3_28_revision_figures.py", "Producer": None,
                          "CreationDate": None})
    fig.savefig(os.path.join(FIG, stem + ".png"), bbox_inches="tight", facecolor="white", dpi=200)
    # editable vector copy: text kept as text (svg.fonttype = none)
    with mpl.rc_context({"svg.fonttype": "none"}):
        fig.savefig(os.path.join(FIG, stem + ".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def grid_axes(ax, x=True):
    ax.grid(axis="y" if x else "x", color=GRIDC, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- base map
kreise = gpd.read_file(os.path.join(GIS, "stadtkreise.gpkg"))
city = gpd.read_file(os.path.join(GIS, "city_boundary.gpkg"))
lake_city = gpd.read_file(os.path.join(GIS, "lakezurich.gpkg"))
muni = gpd.read_file(os.path.join(GIS, "ch_municipalities.gpkg"))
chlakes = gpd.read_file(os.path.join(GIS, "ch_lakes.gpkg"))
country = gpd.read_file(os.path.join(GIS, "ch_country.gpkg"))
cantons = gpd.read_file(os.path.join(GIS, "ch_cantons.gpkg"))
EXT = (2675600, 2690300, 1240700, 1255000)
land = city.geometry.iloc[0].difference(lake_city.geometry.union_all())
lake_all = chlakes[chlakes.intersects(city.geometry.iloc[0].buffer(3000))].geometry.union_all()
lake_all = lake_all.difference(city.geometry.iloc[0]).union(lake_city.geometry.union_all())
KB = gpd.GeoSeries([kreise.boundary.union_all().intersection(land.buffer(-1))], crs=2056)
KREIS_LABEL = {}
for _, r in kreise.iterrows():
    p = r.geometry.difference(lake_city.geometry.union_all()).representative_point()
    KREIS_LABEL[r.Kreisnummer] = (p.x, p.y)


def basemap(ax, ext=EXT, districts=True, lake_label=True):
    ax.set_facecolor("white")
    m = muni[muni.intersects(gpd.GeoSeries([city.geometry.iloc[0].buffer(6000)], crs=2056).iloc[0])]
    m = m[m.id != 261]
    m.plot(ax=ax, color=LAND_OUT, edgecolor=LAND_EDGE, lw=0.35, zorder=0)
    gpd.GeoSeries([land], crs=2056).plot(ax=ax, color="white", edgecolor="none", zorder=0.5)
    gpd.GeoSeries([lake_all], crs=2056).plot(ax=ax, color=LAKE, edgecolor="none", zorder=0.6)
    if lake_label:
        ax.text(2684150, 1242900, "Lake\nZurich", color="#5b86a6", fontsize=6.5, style="italic",
                ha="center", va="center", zorder=6, rotation=-52)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def overlay(ax, districts=True):
    if districts:
        KB.plot(ax=ax, color="#8f8d88", lw=0.35, zorder=4)
    gpd.GeoSeries([land], crs=2056).boundary.plot(ax=ax, color=INK, lw=0.7, zorder=4.5)


def north_scale(ax, x0=2676300, y0=1241400, km=2):
    ax.add_patch(Rectangle((x0, y0), km * 500, 130, fc=INK, ec=INK, lw=0.4, zorder=7))
    ax.add_patch(Rectangle((x0 + km * 500, y0), km * 500, 130, fc="white", ec=INK, lw=0.4, zorder=7))
    for i, lab in enumerate(["0", f"{km/2:g}", f"{km:g} km"]):
        ax.text(x0 + i * km * 500, y0 + 260, lab, ha="center" if i < 2 else "left", va="bottom",
                fontsize=6, color=INK, zorder=7)
    xa, ya = x0 + 250, y0 + 1500
    ax.add_patch(MplPolygon([[xa, ya + 700], [xa - 230, ya], [xa, ya + 180]], closed=True,
                            fc=INK, ec=INK, lw=0.4, zorder=7))
    ax.add_patch(MplPolygon([[xa, ya + 700], [xa + 230, ya], [xa, ya + 180]], closed=True,
                            fc="white", ec=INK, lw=0.4, zorder=7))
    ax.text(xa, ya + 800, "N", ha="center", va="bottom", fontsize=7, fontweight="bold", zorder=7)


def raster(values, e, n, res=100.0):
    x0, y0 = e.min() - res / 2, n.min() - res / 2
    nx = int(round((e.max() - e.min()) / res)) + 1
    ny = int(round((n.max() - n.min()) / res)) + 1
    arr = np.full((ny, nx), np.nan)
    ix = np.round((e - e.min()) / res).astype(int)
    iy = np.round((n - n.min()) / res).astype(int)
    arr[iy, ix] = values
    return arr, (x0, x0 + nx * res, y0, y0 + ny * res)


def classed(cmap_name, bounds, extend="both"):
    base = plt.get_cmap(cmap_name)
    n = len(bounds) - 1 + (extend in ("both", "min")) + (extend in ("both", "max"))
    cols = base(np.linspace(0.12, 0.95, n))
    cmap = ListedColormap(cols)
    return cmap, BoundaryNorm(bounds, cmap.N, extend=extend)


def cell_map(ax, vals, e, n, cmap, norm):
    arr, ex = raster(vals, e, n)
    return ax.imshow(np.ma.masked_invalid(arr), origin="lower", extent=ex, cmap=cmap, norm=norm,
                     interpolation="nearest", zorder=2)


def hcbar(fig, ax, mappable, label, ticks=None, fmt=None, extend=None):
    cax = ax.inset_axes([0.08, -0.07, 0.84, 0.035])
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal", ticks=ticks, format=fmt,
                      spacing="uniform", extendfrac=0.06, **({"extend": extend} if extend else {}))
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=6.3, length=2, width=0.4)
    cb.set_label(label, fontsize=7, labelpad=2)
    return cb


# ---------------------------------------------------------------- data
st = pd.read_csv(os.path.join(RB, "v3_spatial_rebuild_station_targets.csv"))
cov = pd.read_csv(os.path.join(RB, "v3_17_station_coverage_audit.csv"))
st = st.merge(cov[["locationID", "fraction_of_observed_city_nights"]], on="locationID")
SMA_XY = (2685089, 1248065)
KOLLER = st[st.locationID.str.contains("Kollerwiese")][["EKoord", "NKoord"]].mean().to_numpy()
grid = pd.read_csv(os.path.join(ROOT, "data", "inputs", "citywide_grid_priority_screen.csv")).merge(
    pd.read_csv(os.path.join(RB, "v3_spatial_rebuild_grid.csv"))[
        ["recordid", "gp_adjusted_mean_night_min_c", "gp_adjusted_mean_night_min_c_sd",
         "idw2_adjusted_mean_night_min_c"]], on="recordid")
grid = grid.merge(pd.read_csv(os.path.join(RV, "rev_screen_cells.csv")).drop(
    columns=["centroid_easting_2056", "centroid_northing_2056"]), on="recordid")
grid = grid.merge(pd.read_csv(os.path.join(RV, "rev_dusk_field_grid.csv")), on="recordid")
GE, GN = grid.centroid_easting_2056.to_numpy(), grid.centroid_northing_2056.to_numpy()
summary = json.load(open(os.path.join(RV, "rev_summary.json")))


# ================================================================ Fig 1
def fig1():
    fig = plt.figure(figsize=(W, 4.3))
    ax = fig.add_axes([0.0, 0.15, 0.50, 0.80])
    basemap(ax)
    # inhabited 100 m cells as context
    arr, ex = raster(np.ones(len(grid)), GE, GN)
    ax.imshow(np.ma.masked_invalid(arr), origin="lower", extent=ex,
              cmap=ListedColormap(["#e4e1da"]), interpolation="nearest", zorder=1)
    overlay(ax)
    bounds = [400, 420, 440, 460, 480, 500, 550, 640]
    base_ = plt.get_cmap("YlGnBu")
    cmap = ListedColormap(base_(np.linspace(0.25, 0.97, len(bounds) - 1)))
    norm = BoundaryNorm(bounds, cmap.N)
    sc = ax.scatter(st.EKoord, st.NKoord, c=st.masl, cmap=cmap, norm=norm, s=16,
                    edgecolor=INK, linewidth=0.45, zorder=8)
    ax.scatter(*SMA_XY, marker="*", s=95, c="#eb6834", edgecolor=INK, lw=0.5, zorder=9)
    halo = [pe.withStroke(linewidth=2.2, foreground="white")]
    ax.annotate("Zurich-Fluntern\nMeteoSwiss reference", SMA_XY, xytext=(2687500, 1252500),
                fontsize=6, color=INK, ha="center", zorder=9, path_effects=halo,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=INK2))
    ax.scatter(*KOLLER, marker="D", s=22, facecolor="none", edgecolor="#d4502c", lw=1.0, zorder=9)
    ax.annotate("Kollerwiese pair\n(22 m apart)", KOLLER, xytext=(2676400, 1245200), fontsize=6,
                ha="left", path_effects=halo, arrowprops=dict(arrowstyle="-", lw=0.5, color=INK2), zorder=9)
    for name, xy in {"Zürichberg": (2686300, 1249200), "Käferberg": (2680300, 1251900),
                     "Uetliberg": (2679000, 1244000)}.items():
        ax.text(*xy, name, fontsize=6, style="italic", color="#57544e", ha="center", zorder=6,
                path_effects=halo)
    north_scale(ax)
    cb = hcbar(fig, ax, sc, "Station elevation (m a.s.l.)", ticks=bounds)
    ax.set_title("(a) Study area and monitoring network", pad=3)
    # locator inset
    ia = ax.inset_axes([0.69, 0.0, 0.31, 0.24])
    cantons.plot(ax=ia, color="#efede8", edgecolor="#c9c6bf", lw=0.25)
    chlakes.plot(ax=ia, color=LAKE, edgecolor="none")
    country.boundary.plot(ax=ia, color=INK2, lw=0.4)
    ia.scatter([2683000], [1247500], s=14, c="#d4502c", edgecolor=INK, lw=0.4, zorder=5)
    ia.set_aspect("equal"); ia.set_xticks([]); ia.set_yticks([])
    for s in ia.spines.values():
        s.set_linewidth(0.4); s.set_color(MUTED)
    ia.text(0.03, 0.05, "Switzerland", transform=ia.transAxes, fontsize=5.5, color=INK2)
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", mfc="#7fcdbb", mec=INK, mew=0.45, ms=4.5, label="Network station (n = 93)"),
        Patch(fc="#e4e1da", ec="none", label="Inhabited 100 m cell"),
        Line2D([], [], color="#8f8d88", lw=0.6, label="City district (Kreis)")],
        loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=6, handlelength=1.4,
        borderaxespad=0.2, frameon=True, facecolor="white", edgecolor="#c9c6bf",
        framealpha=1.0, fancybox=False).set_zorder(20)

    # --- workflow: plain flowchart
    bx = fig.add_axes([0.54, 0.10, 0.46, 0.85]); bx.set_xlim(0, 1); bx.set_ylim(0, 1); bx.axis("off")
    bx.set_title("(b) Workflow", pad=3)

    def rect(x, y, w, h, text, fc="white", bold=False, fs=6.4):
        bx.add_patch(Rectangle((x, y), w, h, fc=fc, ec=INK2, lw=0.6))
        bx.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
                fontweight="bold" if bold else "normal", linespacing=1.3)

    def arrow(x0, y0, x1, y1):
        bx.annotate("", (x1, y1), (x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=0.6, color=INK2, shrinkA=0, shrinkB=0))

    grey = "#efeeea"
    rect(0.05, 0.86, 0.90, 0.11, "Air temperature, 93 stations, 15 min\nJune–August 2020–2025", grey)
    rect(0.05, 0.69, 0.90, 0.10, "Quality control and night selection\n546 nights, 45,835 station-nights")
    arrow(0.5, 0.86, 0.5, 0.79)
    xs = [0.02, 0.35, 0.68]; w = 0.30
    heads = ["Night-minimum\nsensitivity", "Sunset-aligned\nprofiles", "Spatial field\nand screen"]
    methods = ["1,296 combinations\nof window, spread\nmeasure, threshold,\nweather adjustment",
               "Spread across\nstations by hour;\nstation traits;\ncooling delay",
               "Gaussian process;\nscreen with heat,\nolder residents,\ncanopy, buildings"]
    outs = ["Range of\ncorrelations", "Timing and size of\nlocal differences", "Uncertainty and\ncandidate areas"]
    for x, h_, m, o in zip(xs, heads, methods, outs):
        arrow(0.5 if x == 0.35 else x + w / 2, 0.69, x + w / 2, 0.60) if x == 0.35 else None
        rect(x, 0.50, w, 0.10, h_, bold=True)
        rect(x, 0.25, w, 0.20, m)
        rect(x, 0.04, w, 0.14, o, grey)
        arrow(x + w / 2, 0.50, x + w / 2, 0.45)
        arrow(x + w / 2, 0.25, x + w / 2, 0.18)
    # branch lines from the quality-control box
    bx.plot([0.5, 0.5], [0.69, 0.645], color=INK2, lw=0.6)
    bx.plot([xs[0] + w / 2, xs[2] + w / 2], [0.645, 0.645], color=INK2, lw=0.6)
    for x in (xs[0], xs[2]):
        arrow(x + w / 2, 0.645, x + w / 2, 0.60)
    save(fig, "fig01_study_area_workflow")


# ================================================================ Fig 2
def fig2():
    sc = pd.read_csv(os.path.join(V3, "v3_specification_curve.csv"))
    sc = sc.sort_values("rho").reset_index(drop=True)
    fig = plt.figure(figsize=(W, 4.9))
    ax = fig.add_axes([0.07, 0.72, 0.9, 0.25])
    pos = sc.rho > 0
    ax.scatter(np.arange(len(sc))[pos], sc.rho[pos], s=2.5, c=ORANGE, lw=0, label="Positive")
    ax.scatter(np.arange(len(sc))[~pos], sc.rho[~pos], s=2.5, c=BLUE, lw=0, label="Negative")
    q1, med, q3 = sc.rho.quantile([.25, .5, .75])
    ax.axhspan(q1, q3, color="#f0efeb", zorder=0)
    ax.axhline(0, color=INK, lw=0.6)
    ax.axhline(med, color=INK2, lw=0.8, ls="--")
    unadj = sc[sc.adjustment_set == "unadjusted"].rho.median()
    ax.text(len(sc) * 0.02, med + 0.02, f"median {med:+.2f} (IQR {q1:+.2f} to {q3:+.2f})", fontsize=6.5, color=INK2)
    ax.set_xlim(-5, len(sc) + 5); ax.set_ylabel("Spearman ρ")
    ax.set_xlabel(f"Specification, ranked by estimate (n = {len(sc):,})")
    ax.text(0.99, 0.06, f"{pos.mean()*100:.1f}% positive", transform=ax.transAxes, ha="right", fontsize=6.5, color=INK2)
    ax.set_title("(a) Correlation between city warmth and the spread of night minima", pad=4)
    grid_axes(ax)
    # panel b: factor summaries
    ax2 = fig.add_axes([0.30, 0.05, 0.67, 0.56])
    labels = {"unadjusted": "Unadjusted", "day_radiation": "Daytime radiation", "clearness": "Night clearness",
              "wind": "Night wind", "dayrad+wind": "Radiation + wind", "clearness+wind": "Clearness + wind",
              "full(dayrad+clear+wind)": "Radiation + clearness + wind",
              "full+precip": "All three + precipitation",
              "sd_tmin_c": "Standard deviation", "p90p10_tmin_c": "90th–10th percentile",
              "iqr_tmin_c": "Interquartile range", "all_nights": "All nights",
              "calm_clear_only": "Calm–clear nights", "20-08": "20:00–08:00", "21-07": "21:00–07:00",
              "22-06": "22:00–06:00", "20/80": "20th/80th percentile", "25/75": "25th/75th percentile",
              "33/67": "33rd/67th percentile", "all": "2020–2025", "dev": "2020–2023",
              "holdout": "2024–2025"}
    groups = [("Weather adjustment", "adjustment_set"), ("Spread measure", "spread_metric"),
              ("Nights", "sample"), ("Night window", "night_window"),
              ("Calm–clear threshold", "regime_cut"), ("Period", "years")]
    y = 0; yt, yl = [], []
    for gname, col in groups:
        order = {"sample": ["all_nights", "calm_clear_only"], "night_window": ["20-08", "21-07", "22-06"],
                 "regime_cut": ["33/67", "25/75", "20/80"], "years": ["all", "dev", "holdout"],
                 "spread_metric": ["sd_tmin_c", "p90p10_tmin_c", "iqr_tmin_c"]}
        levels = order.get(col, list(dict.fromkeys(sc[col])))
        if col == "adjustment_set":
            levels = ["unadjusted", "day_radiation", "clearness", "wind", "dayrad+wind", "clearness+wind",
                      "full(dayrad+clear+wind)", "full+precip"]
            levels = [l for l in levels if l in set(sc[col])] + [l for l in set(sc[col]) if l not in levels]
        ax2.text(-0.44, -y + 0.2, gname, fontsize=7, fontweight="bold", color=INK, ha="left",
                 transform=ax2.get_yaxis_transform())
        y += 1
        for lv in levels:
            r = sc[sc[col] == lv].rho
            p10, p25, p50, p75, p90 = r.quantile([.1, .25, .5, .75, .9])
            c = BLUE if col == "adjustment_set" else INK2
            ax2.plot([p10, p90], [-y, -y], color=c, lw=0.8, solid_capstyle="round")
            ax2.plot([p25, p75], [-y, -y], color=c, lw=3.2, solid_capstyle="round")
            ax2.plot(p50, -y, "o", ms=3.8, mfc="white", mec=c, mew=1.0)
            yt.append(-y); yl.append(labels.get(lv, lv)); y += 1
        y += 0.4
    ax2.set_yticks(yt); ax2.set_yticklabels(yl, fontsize=6.6)
    ax2.axvline(0, color=INK, lw=0.6)
    ax2.set_xlim(-0.4, 0.5); ax2.set_ylim(-y + 0.5, 0.9)
    ax2.set_xlabel("Spearman ρ across the specifications that share each choice")
    ax2.tick_params(axis="y", length=0)
    ax2.grid(axis="x", color=GRIDC, lw=0.5); ax2.set_axisbelow(True)
    ax2.set_title("(b) Estimates grouped by analytical choice", x=-0.44, pad=4)
    ax2.legend(handles=[Line2D([], [], color=INK2, lw=0.8, label="10th–90th percentile"),
                        Line2D([], [], color=INK2, lw=3.2, label="Interquartile range"),
                        Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, label="Median")],
               loc="lower left", fontsize=6.3)
    save(fig, "fig02_specification_sensitivity")


# ================================================================ Fig 3
def fig3():
    n = pd.read_csv(os.path.join(V3, "v3_night_synoptic.csv"), parse_dates=["night_date"]).merge(
        pd.read_csv(os.path.join(V3, "v3_night_city.csv"), parse_dates=["night_date"]), on="night_date")
    n = n[n.night_date.dt.year.between(2020, 2023) & n.night_date.dt.month.isin([6, 7, 8])]
    n = n[n.n_stations >= 70]
    fig, axs = plt.subplots(2, 2, figsize=(W * 0.78, 4.4), constrained_layout=True)
    xs = [("day_rad_wm2", "Daytime global radiation (W m$^{-2}$)"),
          ("clearness_wm2", "Night-time clearness (W m$^{-2}$)")]
    ys = [("city_mean_tmin_c", "City-mean night minimum (°C)"),
          ("sd_tmin_c", "Cross-station SD of night minima (°C)")]
    k = 0
    for i, (yc, yl) in enumerate(ys):
        for j, (xc, xl) in enumerate(xs):
            a = axs[i, j]
            a.scatter(n[xc], n[yc], s=6, c=INK2, alpha=0.45, lw=0)
            r = stats.spearmanr(n[xc], n[yc]).statistic
            # lowess-like binned medians
            b = pd.qcut(n[xc], 8)
            med = n.groupby(b, observed=True)[[xc, yc]].median()
            a.plot(med[xc], med[yc], color=ORANGE if r > 0 else BLUE, lw=1.6, marker="o", ms=3)
            a.text(0.03, 0.95, f"ρ = {r:+.2f}".replace("-", "−"), transform=a.transAxes, va="top", fontsize=7.2,
                   fontweight="bold", color=INK)
            a.set_title(f"({'abcd'[k]})", pad=2); k += 1
            if i == 1: a.set_xlabel(xl)
            if j == 0: a.set_ylabel(yl)
            grid_axes(a)
    fig.text(0.5, -0.03, "Points: nights, 2020–2023 (n = %d). Line: medians of eight equal-count bins." % len(n),
             ha="center", fontsize=6.5, color=INK2)
    save(fig, "fig03_meteorological_basis")


# ================================================================ Fig 4
def fig4():
    pr = pd.read_csv(os.path.join(RV, "rev_profile_sunset.csv"))
    fit = pd.read_csv(os.path.join(RV, "rev_functional_fit.csv"))
    ass = pd.read_csv(os.path.join(RV, "rev_dusk_window_associations.csv"))
    prim = pd.read_csv(os.path.join(RB, "v3_17_dusk_associations.csv"))
    fig = plt.figure(figsize=(W, 2.75))
    a1 = fig.add_axes([0.065, 0.16, 0.245, 0.72]); a2 = fig.add_axes([0.355, 0.16, 0.245, 0.72], sharey=a1)
    a3 = fig.add_axes([0.815, 0.16, 0.18, 0.72])
    for a in (a1, a2):
        a.axvspan(8.05, 9.0, color="#f6ead2", zorder=0, lw=0)
        a.axvline(0, color=INK, lw=0.6, ls=":")
        a.text(-0.1, 0.8, "sunset", fontsize=6.3, color=INK2, va="bottom", ha="right", rotation=90)
        a.text(8.52, 2.08, "earliest\nsunrise", fontsize=5.8, color="#9a6b1c", ha="center", va="top")
        a.set_xlim(-2, 9); a.set_xticks(range(-2, 10, 2)); grid_axes(a)
        a.set_xlabel("Hours relative to sunset")
    for t, c, lab in [("warm", WARM, "Warmest third"), ("middle", MID, "Middle third"), ("cool", COOL, "Coolest third")]:
        p = pr[pr.tercile == t].sort_values("offset_h")
        a1.fill_between(p.offset_h, p.ci_lo, p.ci_hi, color=c, alpha=0.18, lw=0)
        a1.plot(p.offset_h, p.sd_mean, color=c, lw=1.5, label=lab + " of nights")
    a1.set_ylabel("Cross-station SD of air temperature (°C)")
    a1.set_ylim(0.75, 2.1)
    a1.set_title("(a) Observed profiles", pad=3)
    a1.legend(loc="upper right", bbox_to_anchor=(0.86, 1.0), fontsize=6.2, handlelength=1.3)
    for lab, c, short in [("warm (90th pct)", WARM, "90th pct."), ("mean warmth", MID, "Mean"),
                          ("cool (10th pct)", COOL, "10th pct.")]:
        f = fit[fit.warmth_level == lab]
        a2.fill_between(f.hss, f.ci_lo, f.ci_hi, color=c, alpha=0.18, lw=0)
        a2.plot(f.hss, f.sd_fit, color=c, lw=1.5)
        a2.plot([], [], color=c, lw=1.5, label={"90th pct.": "Warmth, 90th percentile", "Mean": "Mean warmth", "10th pct.": "Warmth, 10th percentile"}[short])
    plt.setp(a2.get_yticklabels(), visible=False)
    a2.legend(loc="upper right", bbox_to_anchor=(0.86, 1.0), fontsize=6.2, handlelength=1.3)
    a2.set_title("(b) Spline model", pad=3)
    # (c) forest
    rows = []
    names = [("pre_m1_0", "none", "1 h before sunset"), ("dusk_0_1", "none", "First hour after sunset"),
             ("w_1_2", "none", "1–2 h after sunset"), ("w_3_5", "none", "3–5 h after sunset"),
             ("dusk_0_1", "day_radiation", "First hour, adjusted\nfor daytime radiation")]
    for i, (w, adj, lab) in enumerate(names):
        for s, c, mk, off in [("development", DEV_C, "o", 0.16), ("evaluation", EVAL_C, "s", -0.16)]:
            r = ass[(ass.scope == s) & (ass.window == w) & (ass.adjust == adj) &
                    (ass.analysis == ("window" if adj == "none" else "dusk_partial"))].iloc[0]
            rho, lo, hi = r.rho, r.ci_low, r.ci_high
            if w == "dusk_0_1" and adj == "none":
                q = prim[(prim.analysis_scope == ("development" if s == "development" else "internal_temporal_evaluation")) &
                         (prim.estimand == "spearman") & (prim.station_set == "dynamic_all") &
                         (prim.radiation_correction_policy == "published_corrected_values") &
                         (prim.timestamp_convention == "interval_midpoint")].iloc[0]
                rho, lo, hi = q.rho, q.ci_low, q.ci_high
            y = -i + off
            a3.plot([lo, hi], [y, y], color=c, lw=1.0)
            a3.plot(rho, y, mk, ms=4, color=c, mec="white", mew=0.5)
    a3.axvline(0, color=INK, lw=0.6)
    a3.set_yticks([-i for i in range(len(names))]); a3.set_yticklabels([n[2] for n in names], fontsize=6.4)
    a3.tick_params(axis="y", length=0)
    a3.set_xlim(-0.05, 0.9); a3.set_xlabel("Spearman ρ with city warmth")
    a3.axhline(-3.5, color=GRIDC, lw=0.6)
    a3.grid(axis="x", color=GRIDC, lw=0.5); a3.set_axisbelow(True)
    a3.set_title("(c) Association with warmth", x=-1.05, pad=3)
    a3.set_ylim(-5.3, 0.5)
    a3.legend(handles=[Line2D([], [], marker="o", color=DEV_C, lw=1, ms=4, label="2020–2023"),
                       Line2D([], [], marker="s", color=EVAL_C, lw=1, ms=4, label="2024–2025")],
              loc="lower right", fontsize=6.2, handlelength=1.2)
    save(fig, "fig04_sunset_profiles")


# ================================================================ Fig 5
def fig5():
    tr = pd.read_csv(os.path.join(V3, "v3_station_traits.csv"))
    tm = pd.read_csv(os.path.join(RV, "rev_station_trait_models.csv"))
    tm = tm[tm.model == "three"]
    fig = plt.figure(figsize=(W, 5.9))
    ax = fig.add_axes([0.0, 0.44, 0.46, 0.54])
    basemap(ax); overlay(ax)
    lim = 3.0
    bounds = [-3, -2, -1, -0.5, 0.5, 1, 2, 3]
    cols = plt.get_cmap("RdBu_r")(np.array([0.08, 0.2, 0.33, 0.5, 0.67, 0.8, 0.92]))
    cols[3] = [0.93, 0.93, 0.91, 1]
    cmap = ListedColormap(cols); norm = BoundaryNorm(bounds, cmap.N)
    cmap.set_under(plt.get_cmap("RdBu_r")(0.0)); cmap.set_over(plt.get_cmap("RdBu_r")(1.0))
    trm = tr.merge(st[["locationID"]], on="locationID")
    sc = ax.scatter(tr.EKoord, tr.NKoord, c=tr.dusk_anom, cmap=cmap, norm=norm, s=24,
                    edgecolor=INK, lw=0.45, zorder=8)
    north_scale(ax)
    hcbar(fig, ax, sc, "Station minus city mean, first hour after sunset (°C)", ticks=bounds, extend="both")
    ax.set_title("(a) Dusk anomaly on 183 calm–clear nights", pad=3)
    # (b) delta R2
    b = fig.add_axes([0.58, 0.52, 0.40, 0.42])
    targets = [("dusk_anom", "Dusk anomaly\n(0 to 1 h)"), ("integ_anom", "Integrated anomaly\n(0 to 9 h)"),
               ("decay_tend", "Overnight change\n(1 to 9 h)")]
    preds = [("bldg_frac", "Building fraction", "#6b4c9a", "s"), ("elevation", "Elevation", BLUE, "o"),
             ("canopy", "Tree canopy", AQUA, "^")]
    for i, (t, tl) in enumerate(targets):
        for j, (p, pl, c, mk) in enumerate(preds):
            r = tm[(tm.target == t) & (tm.predictor == p)].iloc[0]
            y = -i - (j - 1) * 0.22
            b.plot([r.delta_r2_q05, r.delta_r2_q95], [y, y], color=c, lw=1.0)
            b.plot(r.delta_r2_median, y, mk, color=c, ms=4.3, mec="white", mew=0.4)
    r2s = [tm[tm.target == t].iloc[0].cv_r2_median for t, _ in targets]
    b.set_yticks([0, -1, -2])
    b.set_yticklabels([f"{t[1]}\nheld-out R² {r:.2f}" for t, r in zip(targets, r2s)], fontsize=6.4)
    b.tick_params(axis="y", length=0); b.axvline(0, color=INK, lw=0.6)
    b.set_xlim(-0.06, 0.37); b.set_ylim(-2.5, 0.5)
    b.set_xlabel("Loss in held-out R² when the predictor is removed")
    b.grid(axis="x", color=GRIDC, lw=0.5); b.set_axisbelow(True)
    b.legend(handles=[Line2D([], [], marker=m, color=c, lw=1, ms=4, label=l) for _, l, c, m in preds],
             loc="upper right", fontsize=6.3)
    b.set_title("(b) Predictor importance, 50 geographic splits", pad=3, x=-0.3)

    # (c)(d) partial residual plots
    def partial(ax_, target, pred, others, xl, yl, title, color):
        d = tr.dropna(subset=[target, pred] + others)
        Xo = np.column_stack([np.ones(len(d))] + [d[o] for o in others])
        ry = d[target] - Xo @ np.linalg.lstsq(Xo, d[target], rcond=None)[0]
        rx = d[pred] - Xo @ np.linalg.lstsq(Xo, d[pred], rcond=None)[0]
        ax_.scatter(rx * (100 if pred != "elevation" else 1), ry, s=10, c=color, alpha=0.75, lw=0.3, ec="white")
        sl = np.polyfit(rx, ry, 1)
        xx = np.linspace(rx.min(), rx.max(), 10)
        ax_.plot(xx * (100 if pred != "elevation" else 1), np.polyval(sl, xx), color=INK, lw=1.1)
        r = stats.spearmanr(rx, ry).statistic
        ax_.text(0.03, 0.95, f"partial ρ = {r:+.2f}", transform=ax_.transAxes, va="top", fontsize=6.8)
        ax_.set_xlabel(xl); ax_.set_ylabel(yl); ax_.set_title(title, pad=3); grid_axes(ax_)
    c_ = fig.add_axes([0.08, 0.06, 0.37, 0.23]); d_ = fig.add_axes([0.60, 0.06, 0.37, 0.23])
    partial(c_, "dusk_anom", "bldg_frac", ["elevation", "canopy"],
            "Building fraction, adjusted (percentage points)", "Dusk anomaly,\nadjusted (°C)",
            "(c) Building fraction and dusk anomaly", "#6b4c9a")
    partial(d_, "decay_tend", "canopy", ["elevation", "bldg_frac"],
            "Tree canopy, adjusted (percentage points)", "Overnight change,\nadjusted (°C h$^{-1}$)",
            "(d) Tree canopy and overnight change", AQUA)
    save(fig, "fig05_station_traits")


# ================================================================ Fig 6
def fig6():
    v = pd.read_csv(os.path.join(RV, "rev_ventilation_station_22c.csv"))
    v = v.merge(st[["locationID", "EKoord", "NKoord"]], on="locationID")
    fig = plt.figure(figsize=(W, 3.2))
    ax = fig.add_axes([0.0, 0.10, 0.46, 0.84])
    basemap(ax); overlay(ax)
    bounds = [1, 2, 3, 4, 5, 6, 8]
    cmap, norm = classed("YlOrRd", bounds, extend="both")
    sc = ax.scatter(v.EKoord, v.NKoord, c=v.median_delay, cmap=cmap, norm=norm, s=24,
                    edgecolor=INK, lw=0.45, zorder=8)
    north_scale(ax)
    hcbar(fig, ax, sc, "Median hours after sunset until air < 22 °C", ticks=bounds)
    ax.set_title(f"(a) Hot evenings (n = {summary['vent_hot_evenings_n']})", pad=3)
    b = fig.add_axes([0.58, 0.20, 0.40, 0.70])
    b.scatter(v.bldg_frac * 100, v.median_delay, s=14, c=ORANGE, alpha=0.8, lw=0.3, ec="white")
    r = stats.spearmanr(v.bldg_frac, v.median_delay).statistic
    lo = np.polyfit(v.bldg_frac * 100, v.median_delay, 1)
    xx = np.linspace(0, 55, 10); b.plot(xx, np.polyval(lo, xx), color=INK, lw=1.0)
    b.text(0.03, 0.95, f"ρ = {r:+.2f}, n = {len(v)} stations", transform=b.transAxes, va="top", fontsize=6.8)
    b.set_xlabel("Building footprint fraction within 250 m (%)")
    b.set_ylabel("Median hours after sunset until < 22 °C")
    b.set_title("(b) Delay and building density", pad=3); grid_axes(b)
    save(fig, "fig06_evening_ventilation")


# ================================================================ Fig 7
def fig7():
    fig, axs = plt.subplots(2, 3, figsize=(W, 5.6))
    plt.subplots_adjust(left=0.0, right=1.0, top=0.95, bottom=0.08, wspace=0.03, hspace=0.30)
    specs = [
        ("gp_adjusted_mean_night_min_c", "(a) Mean summer night minimum", "°C",
         [15.0, 15.5, 16.0, 16.5, 17.0, 17.5], "OrRd", None, True),
        ("gp_adjusted_mean_night_min_c_sd", "(b) Uncertainty of (a), 1 SD", "°C",
         [0.40, 0.42, 0.44, 0.46, 0.48, 0.50], "Purples", None, True),
        ("freq54", "(c) Selection frequency, 54 specifications", "share of specifications",
         [0.1, 0.3, 0.5, 0.7, 0.9], "YlGnBu", "%.1f", False),
        ("est65_qmargin", "(d) Residents aged 65+", "residents per cell",
         [5, 10, 20, 30, 40], "PuBu", None, False),
        ("bldg_footprint_frac_250m_direct", "(e) Building footprint, 250 m", "fraction",
         [0.10, 0.15, 0.20, 0.25, 0.30, 0.35], "Greys", "%.2f", False),
        ("canopy_cover_250m_direct", "(f) Tree canopy, 250 m", "fraction",
         [0.10, 0.15, 0.20, 0.30, 0.40, 0.50], "Greens", "%.2f", False)]
    for ax, (col, title, unit, bounds, cm, fmt, show_st) in zip(axs.ravel(), specs):
        basemap(ax, lake_label=(col == "gp_adjusted_mean_night_min_c"))
        vals = grid[col].to_numpy(float)
        if col == "freq54":
            vals = np.where(vals <= 0, np.nan, vals)
        cmap, norm = classed(cm, bounds, extend="min" if col == "freq54" else "both")
        im = cell_map(ax, vals, GE, GN, cmap, norm)
        overlay(ax)
        if show_st:
            ax.scatter(st.EKoord, st.NKoord, s=3.5, c=INK, lw=0, zorder=8)
        if col == "freq54":
            s81 = grid[grid.stable90]
            ax.scatter(s81.centroid_easting_2056, s81.centroid_northing_2056, s=4, marker="s",
                       facecolor="none", edgecolor=ORANGE, lw=0.6, zorder=8)
        ax.set_title(title, pad=2, fontsize=7.4)
        hcbar(fig, ax, im, unit, ticks=bounds, fmt=fmt)
    north_scale(axs[0, 0])
    axs[0, 0].legend(handles=[Line2D([], [], marker="o", ls="", ms=2.5, color=INK, label="Station")],
                     loc="upper left", fontsize=6, borderaxespad=0.1)
    axs[0, 2].legend(handles=[Line2D([], [], marker="s", ls="", ms=3.5, mfc="none", mec=ORANGE, mew=0.8,
                                     label="Selected in ≥90% (81 cells)")],
                     loc="upper left", fontsize=6, borderaxespad=0.1)
    save(fig, "fig07_spatial_indicators")


# ================================================================ Fig 8
def fig8():
    import v3_06_priority_screen as screen
    cfg = pd.read_csv(os.path.join(V3, "v3_priority_decision_multiverse_summary.csv"))
    gp = screen.score(screen.components(grid, "rebuilt_gp", screen.POPULATION_LENSES["qmargin"]), screen.W_MAIN)
    idw = screen.score(screen.components(grid, "rebuilt_idw", screen.POPULATION_LENSES["qmargin"]), screen.W_MAIN)
    fig = plt.figure(figsize=(W, 4.6))
    a = fig.add_axes([0.07, 0.58, 0.36, 0.36])
    tg, ti = np.quantile(gp, 0.9), np.quantile(idw, 0.9)
    both = (gp >= tg) & (idw >= ti); one = (gp >= tg) ^ (idw >= ti)
    a.scatter(gp[~both & ~one], idw[~both & ~one], s=2, c="#c9c7c1", lw=0, label="Neither top 10%")
    a.scatter(gp[one], idw[one], s=4, c=ORANGE, lw=0, label=f"One method only ({int(one.sum()//2)} per method)")
    a.scatter(gp[both], idw[both], s=4, c=BLUE, lw=0, label=f"Both methods ({int(both.sum())})")
    a.axvline(tg, color=INK2, lw=0.6, ls="--"); a.axhline(ti, color=INK2, lw=0.6, ls="--")
    r = stats.spearmanr(gp, idw).statistic
    a.text(0.03, 0.95, f"ρ = {r:.3f}", transform=a.transAxes, va="top", fontsize=6.8)
    a.set_xlabel("Score with Gaussian-process heat layer"); a.set_ylabel("Score with IDW heat layer")
    a.legend(loc="lower right", fontsize=6, markerscale=2, handletextpad=0.3)
    a.set_title("(a) Interpolation method", pad=3); grid_axes(a)
    b = fig.add_axes([0.57, 0.58, 0.41, 0.36])
    lab = {"all_heat_fields_x_weights_x_population_lenses": ("All 54 combinations", BLUE, "o"),
           "weights_at_qmargin": ("Weights only (9)", ORANGE, "s"),
           "population_lenses_at_baseline_weights": ("Population data only (3)", AQUA, "^")}
    for scope, (l, c, m) in lab.items():
        d = cfg[cfg.scope == scope].sort_values("top_percent")
        if d.empty:
            continue
        b.plot(d.top_percent, d.n_baseline_frequency_ge_0_90 / d.n_baseline_selected, marker=m, color=c,
               ms=4, lw=1.2, label=l)
    b.set_ylim(0, 0.8); b.set_xticks([5, 10, 15, 20])
    b.set_xlabel("Share of cells selected (%)"); b.set_ylabel("Share of baseline cells kept\nin ≥90% of combinations")
    b.legend(loc="upper left", fontsize=6.3); b.set_title("(b) Stability by screening share", pad=3); grid_axes(b)
    c = fig.add_axes([0.07, 0.09, 0.91, 0.33])
    f = grid.loc[grid.baseline_top10, "freq54"]
    bins = np.arange(0, 1.0001 + 1 / 54, 1 / 54 * 3)
    h, e = np.histogram(f, bins=np.r_[np.arange(0, 0.9, 0.05), 0.9, 0.95, 1.0001])
    for i in range(len(h)):
        c.bar((e[i] + e[i + 1]) / 2, h[i], width=(e[1] - e[0]) * 0.92,
              color=BLUE if e[i] >= 0.899 else "#b9c9dc", lw=0)
    c.axvline(0.9, color=INK, lw=0.7, ls="--")
    c.set_ylim(0, h.max() * 1.3)
    c.text(0.893, h.max() * 1.27, f"{int(summary['screen_stable81'])} of 422 cells\nselected in ≥90% →",
           fontsize=6.6, va="top", ha="right")
    c.set_xlabel("Selection frequency across 54 combinations of heat layer, weights and population data")
    c.set_ylabel("Baseline top-10% cells"); c.set_xlim(0, 1.02)
    c.set_title("(c) How often each baseline cell is selected", pad=3); grid_axes(c)
    save(fig, "fig08_decision_sensitivity")


# ================================================================ Supplementary
def figS_era5():
    e = pd.read_csv(os.path.join(V3, "v3_night_synoptic_era5.csv"), parse_dates=["night_date"])
    fig, axs = plt.subplots(1, 2, figsize=(W * 0.85, 2.7), constrained_layout=True)
    a = axs[0]
    cols = {"calm_clear": BLUE, "mixed": "#b5b3ad", "cloudy_windy": ORANGE}
    names = {"calm_clear": "Calm–clear", "mixed": "Mixed", "cloudy_windy": "Cloudy–windy"}
    for k, c in cols.items():
        d = e[e.regime == k]
        a.scatter(d.era5_cc_mean, d.clearness_wm2, s=6, c=c, lw=0, alpha=0.8, label=names[k])
    r = stats.spearmanr(e.era5_cc_mean, e.clearness_wm2, nan_policy="omit").statistic
    a.text(0.97, 0.95, f"ρ = {r:.2f}, n = {e.era5_cc_mean.notna().sum()}".replace("-", "−"), transform=a.transAxes,
           ha="right", va="top", fontsize=6.8)
    a.set_xlabel("ERA5 night cloud cover (%)"); a.set_ylabel("Longwave clearness (W m$^{-2}$)")
    a.legend(loc="lower left", fontsize=6.2, markerscale=1.5, title="Night type", title_fontsize=6.2)
    a.set_title("(a) Two cloud indicators", pad=3); grid_axes(a)
    b = axs[1]
    sp = e.era5_cc_sd_spatial.median(); tm = e.era5_cc_mean.std()
    b.barh([1, 0], [sp, tm], color=[MID, BLUE], height=0.5)
    for y, v in [(1, sp), (0, tm)]:
        b.text(v + 0.6, y, f"{v:.1f} pp", va="center", fontsize=6.8)
    b.set_yticks([1, 0]); b.set_yticklabels(["Between ERA5 cells\n(median SD per night)", "Between nights\n(SD of city mean)"])
    b.set_xlabel("Cloud-cover variation (percentage points)"); b.set_xlim(0, 38)
    b.tick_params(axis="y", length=0)
    b.set_title("(b) Spatial vs temporal variation", pad=3); b.grid(axis="x", color=GRIDC, lw=0.5)
    save(fig, "figS1_era5_cloud")


def figS_fluntern():
    h = pd.read_csv(os.path.join(RV, "rev_fluntern_hour_of_day.csv"))
    s = pd.read_csv(os.path.join(RV, "rev_fluntern_shift.csv"))
    fig, axs = plt.subplots(1, 2, figsize=(W * 0.85, 2.6), constrained_layout=True)
    a = axs[0]
    a.bar(h.local_hour, h["mean"], color=[ORANGE if v > 0 else BLUE for v in h["mean"]], width=0.75)
    a.axhline(0, color=INK, lw=0.6)
    a.set_xlabel("Local hour (CEST, interval midpoint)"); a.set_ylabel("Network minus MeteoSwiss (°C)")
    a.set_xticks(range(0, 24, 3)); a.set_title("(a) Mean hourly difference, June–August", pad=3); grid_axes(a)
    b = axs[1]
    b.plot(s.shift_min, s.rmse_all, color=MID, lw=1.3, label="All hours")
    b.plot(s.shift_min, s.rmse_17_22, color=BLUE, lw=1.3, label="17:00–22:59")
    for col, c in [("rmse_all", MID), ("rmse_17_22", BLUE)]:
        k = s[col].idxmin(); b.plot(s.shift_min[k], s[col][k], "o", color=c, ms=4)
    b.axvline(0, color=INK, lw=0.6, ls=":")
    b.set_xlabel("Time shift of network record (min)")
    b.set_ylabel("RMSE (°C)"); b.legend(fontsize=6.3)
    b.set_title("(b) Agreement after a time shift", pad=3); grid_axes(b)
    save(fig, "figS2_fluntern_timing")


def figS_clock():
    pr = pd.read_csv(os.path.join(RV, "rev_profile_clock.csv"))
    fig, a = plt.subplots(figsize=(W * 0.6, 2.6), constrained_layout=True)
    a.axvspan(2 + 8 / 60, 3 + 26 / 60, color="#eeeeea", lw=0, zorder=0)
    a.text(2.78, 1.95, "range of\nsunset times", ha="center", va="top", fontsize=6, color=INK2)
    for t, c, lab in [("warm", WARM, "Warmest third"), ("middle", MID, "Middle third"), ("cool", COOL, "Coolest third")]:
        p = pr[pr.tercile == t].sort_values("offset_h")
        a.fill_between(p.offset_h, p.ci_lo, p.ci_hi, color=c, alpha=0.18, lw=0)
        a.plot(p.offset_h, p.sd_mean, color=c, lw=1.4, label=lab)
    a.set_xticks(range(0, 15, 2)); a.set_xticklabels([f"{(18+h)%24:02d}:00" for h in range(0, 15, 2)])
    a.set_xlabel("Local clock time (CEST)"); a.set_ylabel("Cross-station SD (°C)")
    a.legend(fontsize=6.3, loc="upper right"); grid_axes(a)
    save(fig, "figS3_clock_profiles")


def figS_observables():
    oc = pd.read_csv(os.path.join(V3, "v3_observable_stability_curve.csv"))
    names = {"sd_tmin_c": "Spread of night minima", "city_mean_cool_rate": "Mean cooling rate",
             "sd_cool_rate": "Spread of cooling rate", "sd_cool_total": "Spread of total cooling"}
    fig, a = plt.subplots(figsize=(W * 0.7, 2.4), constrained_layout=True)
    rng = np.random.default_rng(1)
    for i, (k, lab) in enumerate(names.items()):
        r = oc[oc.outcome == k].rho.to_numpy()
        y = -i + rng.uniform(-0.18, 0.18, len(r))
        a.scatter(r, y, s=5, c=np.where(r > 0, ORANGE, BLUE), lw=0, alpha=0.8)
        a.plot([np.median(r)] * 2, [-i - 0.3, -i + 0.3], color=INK, lw=1.3)
        a.text(0.68, -i, f"{(r > 0).mean()*100:.0f}% positive", va="center", fontsize=6.5, color=INK2)
    a.axvline(0, color=INK, lw=0.6)
    a.set_yticks([-i for i in range(4)]); a.set_yticklabels(list(names.values())); a.tick_params(axis="y", length=0)
    a.set_xlim(-0.45, 0.85); a.set_xlabel("Spearman ρ with city warmth (144 matched specifications each; bar = median)")
    a.grid(axis="x", color=GRIDC, lw=0.5); a.set_axisbelow(True)
    save(fig, "figS4_observable_comparison")


def figS_anchor():
    an = pd.read_csv(os.path.join(RV, "rev_anchor_sensitivity.csv"))
    fig, a = plt.subplots(figsize=(W * 0.62, 2.7), constrained_layout=True)
    for i, r in an.iterrows():
        prim = "primary" in r.variant
        ss = "sunset" in r.variant
        c = BLUE if prim else INK2
        a.plot([r.ci_low, r.ci_high], [-i, -i], color=c, lw=1.1)
        a.plot(r.rho, -i, "s" if ss else "o", color=c, ms=4.5, mec="white", mew=0.4)
    a.axvline(0, color=INK, lw=0.6); a.axhline(-3.5, color=GRIDC, lw=0.6)
    a.set_yticks([-i for i in range(len(an))])
    a.set_yticklabels([v.replace(" (primary)", " (primary)") for v in an.variant], fontsize=6.4)
    a.tick_params(axis="y", length=0)
    a.set_xlabel("Spearman ρ between city warmth and cross-station SD")
    a.grid(axis="x", color=GRIDC, lw=0.5); a.set_axisbelow(True)
    save(fig, "figS5_anchor_sensitivity")


def figS_duskfield():
    fig, axs = plt.subplots(1, 2, figsize=(W * 0.82, 3.1))
    plt.subplots_adjust(left=0, right=1, top=0.92, bottom=0.12, wspace=0.04)
    a = axs[0]; basemap(a)
    bounds = [-1.5, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 1.5]
    cols = plt.get_cmap("RdBu_r")(np.array([0.08, 0.2, 0.33, 0.5, 0.67, 0.8, 0.92]))
    cols[3] = [0.93, 0.93, 0.91, 1]
    cmap = ListedColormap(cols); norm = BoundaryNorm(bounds, cmap.N)
    cmap.set_under(plt.get_cmap("RdBu_r")(0.0)); cmap.set_over(plt.get_cmap("RdBu_r")(1.0))
    im = cell_map(a, grid.gp_dusk_anom_c.to_numpy(float), GE, GN, cmap, norm)
    overlay(a); a.scatter(st.EKoord, st.NKoord, s=3, c=INK, lw=0, zorder=8)
    north_scale(a); hcbar(fig, a, im, "Dusk anomaly (°C)", ticks=bounds, extend="both")
    a.set_title("(a) Interpolated dusk anomaly", pad=2)
    b = axs[1]; basemap(b, lake_label=False)
    cat = np.full(len(grid), np.nan)
    cat[grid.baseline_top10 & grid.dusk_baseline_top10] = 2
    cat[grid.baseline_top10 & ~grid.dusk_baseline_top10] = 1
    cat[~grid.baseline_top10 & grid.dusk_baseline_top10] = 0
    cm = ListedColormap([ORANGE, "#8f5bb6", BLUE]); nm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3)
    cell_map(b, cat, GE, GN, cm, nm); overlay(b)
    b.legend(handles=[Patch(fc=BLUE, label=f"Top 10% with both heat layers ({int((cat==2).sum())})"),
                      Patch(fc="#8f5bb6", label=f"Night-minimum layer only ({int((cat==1).sum())})"),
                      Patch(fc=ORANGE, label=f"Dusk layer only ({int((cat==0).sum())})")],
             loc="lower left", bbox_to_anchor=(0.0, -0.16), fontsize=6.2, ncol=1)
    b.set_title("(b) Baseline screen with each heat layer", pad=2)
    save(fig, "figS6_dusk_field_screen")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    which = sys.argv[2:] or ["fig1", "fig2", "fig3", "fig4", "fig5", "fig6", "fig7", "fig8",
                             "figS_era5", "figS_fluntern", "figS_clock", "figS_observables",
                             "figS_anchor", "figS_duskfield"]
    for w in which:
        print("drawing", w, flush=True)
        globals()[w]()
