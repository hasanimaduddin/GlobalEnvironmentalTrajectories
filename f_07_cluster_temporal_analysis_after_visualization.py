# -*- coding: utf-8 -*-
"""
Temporal migration analysis for Hasan

Outputs:
- Cluster share over time (stacked area, % on y-axis)
- NEW: Within-cluster distance to centroid (mean WCD) by year, per cluster (lines)
- Total movement (Σ ||Δz||₂)
- Migration counts (up vs down)
- Permanence (per-year forward-looking 5y, plus cumulative)

Notes:
- The WCD panel will try to read an existing table first:
    1) BASE/temporal_outputs/wcd_by_year_k4.csv  (from your earlier script)
    2) BASE/cluster_prep/clusters_kmeans.xlsx, sheet 'wcd_by_year_cluster' (if present)
  If neither exists, it will compute WCD by refitting KMeans within each year (k=4).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.cluster import KMeans

# ----------------
# CONFIG
# ----------------
BASE = r"C:/Users/LEGION/Documents/Paper3 Intergenerational Environmental Efficiency/Analysis102025"

# inputs
PANEL_XLSX  = os.path.join(BASE, r"cluster_prep/panel_scaled.xlsx")
PANEL_SHEET = "panel_scaled_pm1"
CLUST_XLSX  = os.path.join(BASE, r"cluster_prep/clusters_kmeans.xlsx")
CLUST_SHEET = "cluster_assignments"

# outputs
OUT_DIR     = os.path.join(BASE, r"temporal_outputs")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FIG     = os.path.join(OUT_DIR, "five_stack_overview.png")

# columns / params
ID_COL   = "Country Code"
NAME_COL = "Country Name"
TIME_COL = "Year"
CL_COL   = "cluster"
DIM_COLS = ["PC_scaled_pm1", "DM_scaled_pm1", "RE_scaled_pm1", "DDR_scaled_pm1", "RDP_scaled_pm1"]

K = 4
PERM_WINDOW = 5
SHOW_CUMULATIVE_PERMANENCE = True

# X-axis formatting
X_START = 1995
X_END   = 2024
MAJOR_TICKS = [1995, 2000, 2005, 2010, 2015, 2020, 2024]

# KMeans settings (only used if we need to compute WCD from scratch)
RANDOM_STATE = 0
N_INIT       = 100
MAX_ITER     = 500

# ----------------
# LOAD DATA (panel + cluster labels)
# ----------------
panel = (
    pd.read_excel(PANEL_XLSX, sheet_name=PANEL_SHEET, engine="openpyxl")
      [[NAME_COL, ID_COL, TIME_COL] + DIM_COLS]
      .dropna(subset=[ID_COL, TIME_COL])
      .assign(Year=lambda d: d[TIME_COL].astype(int))
      .sort_values([ID_COL, TIME_COL])
)

clusters = (
    pd.read_excel(CLUST_XLSX, sheet_name=CLUST_SHEET, engine="openpyxl")
      [[NAME_COL, ID_COL, TIME_COL, CL_COL]]
      .dropna(subset=[ID_COL, TIME_COL, CL_COL])
      .assign(Year=lambda d: d[TIME_COL].astype(int),
              cluster=lambda d: d[CL_COL].astype(int))
      .sort_values([ID_COL, TIME_COL])
)

# ----------------
# HELPERS
# ----------------
def compute_distance_movement(df):
    df = df.sort_values([ID_COL, TIME_COL]).copy()
    for col in DIM_COLS:
        df[f"d_{col}"] = df.groupby(ID_COL)[col].diff()
    dcols = [f"d_{c}" for c in DIM_COLS]
    df["dist"] = np.sqrt(np.sum(np.square(df[dcols]), axis=1))
    return df

def compute_migration_counts(cldf):
    cldf = cldf.sort_values([ID_COL, TIME_COL]).copy()
    cldf["d_cl"] = cldf.groupby(ID_COL)[CL_COL].diff()
    tmp = cldf.dropna(subset=["d_cl"]).copy()
    tmp["up"]   = (tmp["d_cl"] > 0).astype(int)
    tmp["down"] = (tmp["d_cl"] < 0).astype(int)
    return tmp.groupby(TIME_COL).agg(up=("up","sum"),
                                     down=("down","sum")).assign(net=lambda d: d["up"]-d["down"])

def compute_permanence_forward_5y(cldf, window=5):
    cldf = cldf.sort_values([ID_COL, TIME_COL]).copy()
    years = sorted(cldf[TIME_COL].unique())
    piv = cldf.pivot(index=ID_COL, columns=TIME_COL, values=CL_COL).sort_index()
    per_up = pd.Series(0, index=pd.Index(years, name=TIME_COL), dtype=int)
    per_dn = pd.Series(0, index=pd.Index(years, name=TIME_COL), dtype=int)

    for _, row in piv.iterrows():
        vals = row.dropna(); ts = vals.index.tolist()
        for k in range(1, len(ts)):
            t, prev = ts[k], ts[k-1]
            d = vals.loc[t] - vals.loc[prev]
            if d == 0:
                continue
            new_rank = vals.loc[t]
            horizon = [yr for yr in ts if (yr >= t) and (yr <= t + (window - 1))]
            if len(horizon) < window:
                continue
            future_vals = vals.loc[horizon]
            if d > 0 and (future_vals >= new_rank).all():
                per_up.loc[t] += 1
            if d < 0 and (future_vals <= new_rank).all():
                per_dn.loc[t] += 1

    out = pd.DataFrame({"perm_up_per_year": per_up,
                        "perm_down_per_year": per_dn})
    if SHOW_CUMULATIVE_PERMANENCE:
        out["perm_up_cum"]   = out["perm_up_per_year"].cumsum()
        out["perm_down_cum"] = out["perm_down_per_year"].cumsum()
    return out

def compute_cluster_sizes_over_time(cldf, K=4):
    counts = cldf.groupby([TIME_COL, CL_COL])[ID_COL].nunique().unstack(CL_COL).fillna(0).astype(int)
    for c in range(1, K+1):
        if c not in counts.columns:
            counts[c] = 0
    counts = counts[range(1, K+1)]
    counts.columns = [f"C{c}" for c in range(1, K+1)]
    return counts

def order_labels_by_composite(centers: np.ndarray) -> dict:
    comp = centers.mean(axis=1)
    order_old = np.argsort(comp)
    return {int(old): int(new) for new, old in enumerate(order_old, start=1)}  # old -> 1..k

def get_wcd_by_year_cluster():
    """
    Try to load WCD summary; if not available, compute it by refitting KMeans within each year.
    Returns a DataFrame with columns: Year, cluster (1..K per year, low→high), wcd_mean, n
    """
    # 1) CSV from earlier script
    csv_path = os.path.join(OUT_DIR, "wcd_by_year_k4.csv")
    if os.path.exists(csv_path):
        w = pd.read_csv(csv_path)
        if {"Year","cluster","wcd_mean"}.issubset(w.columns):
            w["Year"] = w["Year"].astype(int)
            w["cluster"] = w["cluster"].astype(int)
            return w.sort_values(["Year","cluster"])

    # 2) Excel sheet from enhanced clustering export
    xlsx_alt = os.path.join(BASE, "cluster_prep", "clusters_kmeans.xlsx")
    if os.path.exists(xlsx_alt):
        try:
            w = pd.read_excel(xlsx_alt, sheet_name="wcd_by_year_cluster", engine="openpyxl")
            if {"Year","cluster","wcd_mean"}.issubset(w.columns):
                w["Year"] = w["Year"].astype(int)
                w["cluster"] = w["cluster"].astype(int)
                return w.sort_values(["Year","cluster"])
        except Exception:
            pass

    # 3) Compute from scratch: refit KMeans by year and compute mean distance to own centroid
    print("[INFO] WCD file not found; computing WCD by refitting KMeans each year (k=4).")
    years = sorted(panel["Year"].unique().tolist())
    rows = []
    for y in years:
        dy = panel[panel["Year"] == y].dropna(subset=DIM_COLS)
        X = dy[DIM_COLS].to_numpy(dtype=float)
        if X.shape[0] < K:
            continue
        km = KMeans(n_clusters=K, n_init=N_INIT, max_iter=MAX_ITER, random_state=RANDOM_STATE)
        labels0 = km.fit_predict(X)
        centers = km.cluster_centers_
        m = order_labels_by_composite(centers)
        own_centers = centers[labels0]
        dists = np.linalg.norm(X - own_centers, axis=1)
        labels_ord = np.array([m[int(l)] for l in labels0], dtype=int)
        for c in range(1, K+1):
            mask = labels_ord == c
            if not np.any(mask):
                continue
            rows.append({
                "Year": int(y),
                "cluster": int(c),
                "wcd_mean": float(np.mean(dists[mask])),
                "n": int(mask.sum())
            })
    return pd.DataFrame(rows).sort_values(["Year","cluster"])

# ----------------
# PREP METRICS
# ----------------
panel_mov = compute_distance_movement(panel.copy())
total_movement = (
    panel_mov.dropna(subset=["dist"])
             .groupby(TIME_COL)["dist"]
             .sum()
             .rename("total_movement")
)

migration_df = compute_migration_counts(clusters)
permanence_df = compute_permanence_forward_5y(clusters, window=PERM_WINDOW)
cluster_sizes = compute_cluster_sizes_over_time(clusters, K=K)
wcd_df = get_wcd_by_year_cluster()

# ----------------
# PLOT: FIVE-STACK (WCD panel inserted as #2)
# ----------------
def five_stack_plot(cluster_sizes, wcd_df, total_movement, migration_df, permanence_df,
                    title="Temporal Migration — Five-Stack Overview", outpath=None):

    years = sorted(set(cluster_sizes.index)
                   | set(total_movement.index)
                   | set(migration_df.index)
                   | set(permanence_df.index)
                   | set(wcd_df["Year"].unique()))
    idx = pd.Index(years, name=TIME_COL)

    cs = cluster_sizes.reindex(idx).fillna(0)
    tm = total_movement.reindex(idx).fillna(0.0)
    mg = migration_df.reindex(idx).fillna(0)
    pm = permanence_df.reindex(idx).fillna(0)

    # convert cluster sizes to shares
    denom = cs.sum(axis=1).replace(0, np.nan)
    cs_share = cs.div(denom, axis=0).fillna(0.0)

    fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True)

    # 1) Cluster shares (%)
    axes[0].stackplot(idx, cs_share.T.values, labels=cs_share.columns)
    axes[0].legend(loc="upper left", ncols=min(len(cs_share.columns), 6), frameon=False)
    axes[0].set_ylabel("Cluster share (%)")
    axes[0].set_title(title)
    axes[0].set_ylim(0, 1)
    axes[0].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
    axes[0].grid(True, alpha=0.2)

    # 2) WCD lines (mean distance to year-centroid by ordered cluster)
    palette = ["#1f77b4","#ff7f0e","#2ca02c","#d62728"]
    for c in sorted(wcd_df["cluster"].unique()):
        gc = wcd_df[wcd_df["cluster"] == c]
        axes[1].plot(gc["Year"], gc["wcd_mean"], linewidth=2, label=f"C{c}",
                     color=palette[(int(c)-1) % len(palette)])
    axes[1].set_ylabel("Mean WCD")
    axes[1].set_title("Mean distance to centroid (per year, per ordered cluster)")
    axes[1].legend(title="Cluster", frameon=False, ncol=4)
    axes[1].grid(True, alpha=0.3)

    # 3) Total movement
    axes[2].plot(idx, tm.values, linewidth=2)
    axes[2].set_ylabel("Total movement\nΣ ||Δz||₂")
    axes[2].grid(True, alpha=0.3)

    # 4) Migration counts
    axes[3].plot(idx, mg["up"].values, linewidth=2, label="Up")
    axes[3].plot(idx, mg["down"].values, linewidth=2, label="Down")
    axes[3].set_ylabel("Migration counts")
    axes[3].legend(loc="upper left", frameon=False)
    axes[3].grid(True, alpha=0.3)

    # 5) Permanence
    axes[4].plot(idx, pm["perm_up_per_year"].values, linewidth=2,
                 label=f"Permanent-up per year (≥{PERM_WINDOW}y)")
    axes[4].plot(idx, pm["perm_down_per_year"].values, linewidth=2,
                 label=f"Permanent-down per year (≥{PERM_WINDOW}y)")
    if SHOW_CUMULATIVE_PERMANENCE:
        axes[4].plot(idx, pm["perm_up_cum"].values, "--", linewidth=1.5, label="Permanent-up (cum.)")
        axes[4].plot(idx, pm["perm_down_cum"].values, "--", linewidth=1.5, label="Permanent-down (cum.)")
    axes[4].set_ylabel("Permanence")
    axes[4].set_xlabel("Year")
    axes[4].legend(loc="upper left", frameon=False)
    axes[4].grid(True, alpha=0.3)

    # X-axis formatting for all panels
    for ax in axes:
        ax.set_xlim(X_START, X_END)
        ax.set_xticks(MAJOR_TICKS)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.tick_params(axis='x', which='minor', length=3)

    fig.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=300, bbox_inches="tight")
    return fig, axes

fig, axes = five_stack_plot(
    cluster_sizes=cluster_sizes,
    wcd_df=wcd_df,
    total_movement=total_movement,
    migration_df=migration_df,
    permanence_df=permanence_df,
    title="Temporal Migration — Five-Stack Overview",
    outpath=OUT_FIG
)

print("Saved figure to:", OUT_FIG)
print("All outputs in  :", OUT_DIR)
