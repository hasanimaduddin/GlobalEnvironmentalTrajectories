# -*- coding: utf-8 -*-
"""
Panel export + 3 outputs:
(1) Annual mean distance (overall, top40% far, bottom10% near)  [no inversion]
(2) Inequality on attainment (DMAX - distance): Gini & Palma     [higher = better]
(3) Lorenz curve per year on attainment (with perfect equality line)

Inputs:
  BASE/cluster_prep/panel_scaled.xlsx (sheet: panel_scaled_pm1)

Outputs (under BASE/temporal_outputs/fairness):
  - panel_distance_to_ideal.xlsx
      * distance_panel
      * summary_means_distance
      * inequality_attainment
      * params
  - means_top40_bottom10_distance.png
  - inequality_gini_palma_attainment.png
  - lorenz/lorenz_YYYY.png   (one PNG per year)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========= CONFIG =========
BASE = r"C:\Users\LEGION\Documents\Paper3 Intergenerational Environmental Efficiency\Analysis102025"
PANEL_XLSX  = os.path.join(BASE, "cluster_prep", "panel_scaled.xlsx")
PANEL_SHEET = "panel_scaled_pm1"

OUT_DIR  = os.path.join(BASE, "temporal_outputs", "fairness")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_XLSX = os.path.join(OUT_DIR, "panel_distance_to_ideal.xlsx")
FIG_MEANS = os.path.join(OUT_DIR, "means_top40_bottom10_distance.png")
FIG_INEQ  = os.path.join(OUT_DIR, "inequality_gini_palma_attainment.png")
LORENZ_DIR = os.path.join(OUT_DIR, "lorenz")
os.makedirs(LORENZ_DIR, exist_ok=True)

ID_COL   = "Country Code"
NAME_COL = "Country Name"
TIME_COL = "Year"
DIM_COLS = ["PC_scaled_pm1","DM_scaled_pm1","RE_scaled_pm1","DDR_scaled_pm1","RDP_scaled_pm1"]

# Optional year window (inclusive)
YEAR_MIN = None
YEAR_MAX = None

# Percentile cuts for the distance means figure
TOP_FAR_PCT     = 0.40   # farthest 40% (distance largest)
BOTTOM_NEAR_PCT = 0.10   # closest 10%  (distance smallest)

# Palma tails for attainment inequality (classic: top10 / bottom40)
PALMA_TOP_PCT = 0.10
PALMA_BOT_PCT = 0.40

# DMAX for 5-D [-1,1] with ideal z*=(1,...,1)
DMAX = float(np.sqrt(20.0))


# ---- Plot window & y-scale knobs ----
X_START = 1995
X_END   = 2024
GINI_YLIM  = (0, 0.5)        # e.g., (0.08, 0.13); set to None for auto
PALMA_YLIM = (1, 2)        # e.g., (1.40, 1.62); set to None for auto


# ========= HELPERS =========
def _need(df, cols, where=""):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise KeyError(f"Missing columns {miss} in {where or 'input'}.")

def load_panel():
    df = pd.read_excel(PANEL_XLSX, sheet_name=PANEL_SHEET, engine="openpyxl")
    _need(df, [NAME_COL, ID_COL, TIME_COL] + DIM_COLS, f"{PANEL_XLSX}:{PANEL_SHEET}")
    df = df[[NAME_COL, ID_COL, TIME_COL] + DIM_COLS].dropna(subset=[ID_COL, TIME_COL]).copy()
    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce").astype(int)
    if YEAR_MIN is not None: df = df[df[TIME_COL] >= YEAR_MIN]
    if YEAR_MAX is not None: df = df[df[TIME_COL] <= YEAR_MAX]
    return df.sort_values([ID_COL, TIME_COL]).reset_index(drop=True)

def compute_distance_panel(df: pd.DataFrame) -> pd.DataFrame:
    z = np.ones(len(DIM_COLS), dtype=float)
    X = df[DIM_COLS].to_numpy(dtype=float)
    dist = np.sqrt(((X - z)**2).sum(axis=1))   # lower = better
    out = df[[NAME_COL, ID_COL, TIME_COL]].copy()
    out["dist_ideal"] = dist.astype(float)
    return out

def summarize_distance_means(dist_panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for y, g in dist_panel.groupby(TIME_COL):
        x = np.sort(g["dist_ideal"].to_numpy(dtype=float))  # ascending: near ... far
        n = x.size
        if n == 0: 
            continue
        k_top = max(1, int(np.ceil(TOP_FAR_PCT * n)))
        k_bot = max(1, int(np.ceil(BOTTOM_NEAR_PCT * n)))
        rows.append({
            "Year": int(y),
            "n_countries": int(n),
            "mean_dist": float(x.mean()),
            "top40_mean_dist": float(x[-k_top:].mean()),    # farthest 40%
            "bottom10_mean_dist": float(x[:k_bot].mean()),  # closest 10%
        })
    return pd.DataFrame(rows).sort_values("Year")

def plot_distance_means(means_df: pd.DataFrame, out_path: str):
    x = means_df["Year"].values
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, means_df["mean_dist"].values,           lw=2.0, label="Overall mean")
    ax.plot(x, means_df["top40_mean_dist"].values,     lw=2.0, label="Top 40% (farthest)")
    ax.plot(x, means_df["bottom10_mean_dist"].values,  lw=2.0, label="Bottom 10% (closest)")
    ax.set_xlabel("Year"); ax.set_ylabel("Distance to ideal (L2)")
    ax.set_title("Annual mean distance to ideal — overall vs tails")
    ax.grid(True, alpha=0.25); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out_path, dpi=300, bbox_inches="tight"); plt.close(fig)

def gini_unweighted(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    if np.all(x == 0):
        return 0.0
    xmin = x.min()
    if xmin < 0:  # safeguard
        x = x - xmin
    xs = np.sort(x)
    n = xs.size
    i = np.arange(1, n + 1, dtype=float)
    return 2.0 * np.sum(i * xs) / (n * np.sum(xs)) - (n + 1.0) / n

def palma_ratio_attainment(x: np.ndarray, p_top=0.10, p_bot=0.40) -> float:
    """Palma on attainment (higher = better): mean(top p_top) / mean(bottom p_bot)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 10:
        return np.nan
    xs = np.sort(x)  # ascending: worst ... best
    k_top = max(1, int(np.ceil(p_top * n)))
    k_bot = max(1, int(np.ceil(p_bot * n)))
    num = xs[-k_top:].mean()
    den = xs[:k_bot].mean()
    return np.nan if den == 0 else num / den

def inequality_on_attainment(dist_panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for y, g in dist_panel.groupby(TIME_COL):
        dist = g["dist_ideal"].to_numpy(dtype=float)
        attain = np.clip(DMAX - dist, 0.0, DMAX)  # higher = closer/better
        rows.append({
            "Year": int(y),
            "n_countries": int(np.isfinite(attain).sum()),
            "Gini_attain": float(gini_unweighted(attain)),
            "Palma_attain": float(palma_ratio_attainment(attain, PALMA_TOP_PCT, PALMA_BOT_PCT)),
        })
    return pd.DataFrame(rows).sort_values("Year")

def plot_gini_palma(per_year: pd.DataFrame,
                    out_path: str,
                    x_start: int = X_START,
                    x_end: int   = X_END,
                    gini_ylim: tuple | None = GINI_YLIM,
                    palma_ylim: tuple | None = PALMA_YLIM,
                    slope_fmt: str = "{:+.4f}"):   # << control slope text
    """
    Plot Gini (left y) & Palma (right y) with linear trend lines.
    Legend shows '... trend (slope: ±xxxx.xxxx)' using slope_fmt.
    """

    # Data
    x = per_year["Year"].to_numpy(dtype=float)
    g = per_year["Gini_attain"].to_numpy(dtype=float)
    p = per_year["Palma_attain"].to_numpy(dtype=float)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    # Base series
    line_g, = ax1.plot(x, g, lw=2.2, label="Gini")
    line_p, = ax2.plot(x, p, lw=2.2, color="tab:red", label="Palma")

    # Fit helper
    def _fit_line(xv, yv):
        m = np.isfinite(xv) & np.isfinite(yv)
        if m.sum() < 2:
            return None
        slope, intercept = np.polyfit(xv[m], yv[m], 1)
        xx = np.linspace(x_start, x_end, 200)
        yy = intercept + slope * xx
        return slope, xx, yy

    # Gini trend
    fg = _fit_line(x, g)
    if fg is not None:
        slope_g, xxg, yyg = fg
        ax1.plot(xxg, yyg, ls="--", lw=1.6, color=line_g.get_color(),
                 label=f"Gini trend (slope: {slope_fmt.format(slope_g)})")

    # Palma trend
    fp = _fit_line(x, p)
    if fp is not None:
        slope_p, xxp, yyp = fp
        ax2.plot(xxp, yyp, ls="--", lw=1.6, color=line_p.get_color(),
                 label=f"Palma trend (slope: {slope_fmt.format(slope_p)})")

    # Labels & axes formatting
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Gini")
    ax2.set_ylabel(f"Palma (top{int(PALMA_TOP_PCT*100)} / bottom{int(PALMA_BOT_PCT*100)})")

    ax1.set_xlim(x_start, x_end)
    ax1.set_xticks(np.arange(x_start, x_end + 1, 5))
    ax1.set_xticks(np.arange(x_start, x_end + 1, 1), minor=True)
    ax1.tick_params(axis="x", which="minor", length=3)

    if gini_ylim is not None:
        ax1.set_ylim(*gini_ylim)
    if palma_ylim is not None:
        ax2.set_ylim(*palma_ylim)

    # Combined legend
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, frameon=False, loc="upper left")

    ax1.grid(True, alpha=0.25)
    fig.suptitle("Inequality in inverted distance to ideal — Gini & Palma", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ----- LORENZ -----
def lorenz_points(attain: np.ndarray):
    """Return population shares (p) and cumulative attainment shares (L(p))."""
    x = np.asarray(attain, dtype=float)
    x = x[np.isfinite(x)]
    x = np.clip(x, 0.0, None)       # attainment must be nonnegative
    n = x.size
    if n == 0:
        return np.array([0,1]), np.array([0,1])
    xs = np.sort(x)                 # ascending: worst ... best
    cumx = np.cumsum(xs)
    total = cumx[-1]
    p = np.arange(0, n + 1) / n     # 0..1 population share
    if total == 0:
        L = np.zeros_like(p)        # degenerate: everyone zero
    else:
        L = np.concatenate(([0.0], cumx / total))
    return p, L

def plot_lorenz_per_year(dist_panel: pd.DataFrame, out_dir: str):
    for y, g in dist_panel.groupby(TIME_COL):
        dist = g["dist_ideal"].to_numpy(dtype=float)
        attain = np.clip(DMAX - dist, 0.0, DMAX)
        p, L = lorenz_points(attain)
        G = gini_unweighted(attain)

        fig, ax = plt.subplots(figsize=(6.2, 6.2))
        ax.plot([0,1], [0,1], '--', lw=1.2, color='gray', label="Perfect equality")  # 45°
        ax.plot(p, L, lw=2.2, color='tab:blue', label="Lorenz (attainment)")
        ax.set_xlabel("Cumulative share of countries")
        ax.set_ylabel("Cumulative share of attainment")
        ax.set_title(f"Lorenz curve — attainment, Year {int(y)}\nGini = {G:.3f}")
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, loc="lower right")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"lorenz_{int(y)}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

# ========= MAIN =========
def main():
    panel = load_panel()
    dist_panel = compute_distance_panel(panel)

    means_df = summarize_distance_means(dist_panel)
    ineq_df  = inequality_on_attainment(dist_panel)

    # Write Excel
    try:
        with pd.ExcelWriter(OUT_XLSX, engine="xlsxwriter") as xw:
            dist_panel.to_excel(xw, sheet_name="distance_panel", index=False)
            means_df.to_excel(xw, sheet_name="summary_means_distance", index=False)
            ineq_df.to_excel(xw, sheet_name="inequality_attainment", index=False)
            pd.DataFrame({
                "panel_sheet":[PANEL_SHEET],
                "dim_cols":[", ".join(DIM_COLS)],
                "year_min":[YEAR_MIN],
                "year_max":[YEAR_MAX],
                "dmax":[DMAX],
                "top_far_pct":[TOP_FAR_PCT],
                "bottom_near_pct":[BOTTOM_NEAR_PCT],
                "palma_top_pct":[PALMA_TOP_PCT],
                "palma_bot_pct":[PALMA_BOT_PCT],
                "notes":[
                    "Means on distance; Gini/Palma on attainment = DMAX - distance (no normalization)."
                ],
            }).to_excel(xw, sheet_name="params", index=False)
    except Exception:
        with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as xw:
            dist_panel.to_excel(xw, sheet_name="distance_panel", index=False)
            means_df.to_excel(xw, sheet_name="summary_means_distance", index=False)
            ineq_df.to_excel(xw, sheet_name="inequality_attainment", index=False)
            pd.DataFrame({
                "panel_sheet":[PANEL_SHEET],
                "dim_cols":[", ".join(DIM_COLS)],
                "year_min":[YEAR_MIN],
                "year_max":[YEAR_MAX],
                "dmax":[DMAX],
                "top_far_pct":[TOP_FAR_PCT],
                "bottom_near_pct":[BOTTOM_NEAR_PCT],
                "palma_top_pct":[PALMA_TOP_PCT],
                "palma_bot_pct":[PALMA_BOT_PCT],
                "notes":[
                    "Means on distance; Gini/Palma on attainment = DMAX - distance (no normalization)."
                ],
            }).to_excel(xw, sheet_name="params", index=False)

    # Figures
    plot_distance_means(means_df, FIG_MEANS)
    plot_gini_palma(ineq_df, FIG_INEQ)
    plot_lorenz_per_year(dist_panel, LORENZ_DIR)

    print("Saved Excel:", OUT_XLSX)
    print("Saved figures:")
    print(" -", FIG_MEANS)
    print(" -", FIG_INEQ)
    print(" - Lorenz per year ->", LORENZ_DIR)

if __name__ == "__main__":
    main()
