# -*- coding: utf-8 -*-
"""
PCA visualizations on five-lens panel (global clustering setup)
- Consistent, equal axes across all plots (centered at 0,0)
- Origin cross more visible
- Exports PCA loadings to Excel for interpretation
- NEW: Biplot-style **loadings arrows** overlaid on the cluster-assignment scatters
  (global and year-by-year). Not drawn on the centroid-trajectory figure.

Inputs:
  - BASE/cluster_prep/panel_scaled.xlsx  (sheet: panel_scaled_pm1)
  - BASE/cluster_prep/clusters_kmeans.xlsx (sheet: cluster_assignments)

Outputs (under BASE/cluster_prep/pca_figs):
  - pca_global_scatter.png
  - pca_by_year/YYYY_pca.png
  - pca_cluster_centroid_trajectories.png
  - pca_scores.csv
  - pca_outputs.xlsx  (sheets: 'pca_loadings', 'explained_variance', 'plot_params')
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ========================
# CONFIG
# ========================
BASE = r"C:\Users\LEGION\Documents\Paper3 Intergenerational Environmental Efficiency\Analysis102025"
PANEL_XLSX   = os.path.join(BASE, "cluster_prep", "panel_scaled.xlsx")
PANEL_SHEET  = "panel_scaled_pm1"   # [-1,1] features for analysis
CLUST_XLSX   = os.path.join(BASE, "cluster_prep", "clusters_kmeans.xlsx")
CLUST_SHEET  = "cluster_assignments"

OUT_DIR      = os.path.join(BASE, "cluster_prep", "pca_figs")
OUT_BY_YEAR  = os.path.join(OUT_DIR, "by_year")
OUT_XLSX     = os.path.join(OUT_DIR, "pca_outputs.xlsx")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(OUT_BY_YEAR, exist_ok=True)

# Feature set (must match what you clustered on)
FEATURE_COLS = ["PC_scaled_pm1","DM_scaled_pm1","RE_scaled_pm1","DDR_scaled_pm1","RDP_scaled_pm1"]
FEATURE_NICE = ["PC","DM","RE","DDR","RDP"]  # for loadings labels & Excel

# Year window to plot (None = all)
START_YEAR = 1972
END_YEAR   = None   # e.g., 2024

# Plot aesthetics (matplotlib only; no seaborn)
POINT_ALPHA_GLOBAL = 0.25
POINT_SIZE_GLOBAL  = 10
POINT_SIZE_YEAR    = 18
CENTROID_SIZE      = 140
LINE_WIDTH         = 2.0
AX_CROSS_COLOR     = "lightgray"
AX_CROSS_WIDTH     = 0.9
GRID_ALPHA         = 0.2

# Loadings arrow aesthetics (for biplot overlays)
LOAD_ARROW_SCALE   = 0.90   # fraction of axis limit for arrow length scaling
LOAD_ARROW_COLOR   = "black"
LOAD_ARROW_ALPHA   = 0.65
HEAD_W_FRAC        = 0.03   # head width as fraction of axis limit
HEAD_L_FRAC        = 0.05   # head length as fraction of axis limit
LABEL_PAD_FRAC     = 0.06   # label offset as fraction of axis limit
LABEL_FONTSIZE     = 9

# Arrow axis-weighting: scale x/y of loadings by PC explained-variance ratio
# options: "none" (no weighting), "evr" (use EVR ratios), "sqrt_evr" (Gabriel-style)
LOADINGS_AXIS_WEIGHTING = "evr"


# ========================
# HELPERS
# ========================
def load_panel_scaled():
    df = pd.read_excel(PANEL_XLSX, sheet_name=PANEL_SHEET, engine="openpyxl")
    need = ["Country Name","Country Code","Year"] + FEATURE_COLS
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {PANEL_SHEET}: {missing}")
    # window
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    if START_YEAR is not None:
        df = df[df["Year"].ge(START_YEAR)]
    if END_YEAR is not None:
        df = df[df["Year"].le(END_YEAR)]
    return df.sort_values(["Country Code","Year"]).reset_index(drop=True)

def load_clusters():
    c = pd.read_excel(CLUST_XLSX, sheet_name=CLUST_SHEET, engine="openpyxl")
    need = ["Country Code","Year","cluster"]
    missing = [x for x in need if x not in c.columns]
    if missing:
        raise KeyError(f"Missing columns in {CLUST_SHEET}: {missing}")
    c["Year"] = pd.to_numeric(c["Year"], errors="coerce").astype("Int64")
    return c[need]

def fit_global_pca(X, n_components=2):
    pca = PCA(n_components=n_components, svd_solver="auto", random_state=None)
    Z = pca.fit_transform(X)
    return pca, Z

def project_centroids(Z_df):
    return Z_df.groupby("cluster")[["PC1","PC2"]].mean().reset_index()

def _cluster_colors(unique_clusters):
    base = [
        "#1f77b4","#ff7f0e","#2ca02c","#d62728",
        "#9467bd","#8c564b","#e377c2","#7f7f7f",
        "#bcbd22","#17becf"
    ]
    colors = {}
    for i, cl in enumerate(sorted(unique_clusters)):
        colors[cl] = base[i % len(base)]
    return colors

def _apply_equal_axes(ax, lim):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal', adjustable='box')
    ax.axhline(0, color=AX_CROSS_COLOR, linewidth=AX_CROSS_WIDTH)
    ax.axvline(0, color=AX_CROSS_COLOR, linewidth=AX_CROSS_WIDTH)

def _draw_loadings_arrows(ax, pca, lim, feature_names, evr, mode=LOADINGS_AXIS_WEIGHTING):
    """
    Draw biplot-style loadings arrows for PC1/PC2.
    Axis-weighting lets the x/y components reflect the PC1/PC2 variance ratio.
      mode="none"      -> no axis weighting (original behavior)
      mode="evr"       -> multiply x by EVR[0]/max(EVR), y by EVR[1]/max(EVR)
      mode="sqrt_evr"  -> same but with sqrt(EVR) (common in biplot literature)
    """
    comps = pca.components_.T  # (n_features, 2)
    if comps.shape[1] < 2:
        return

    # base scale so arrows fit the plot nicely
    scale = LOAD_ARROW_SCALE * lim
    head_w = HEAD_W_FRAC * lim
    head_l = HEAD_L_FRAC * lim
    pad    = LABEL_PAD_FRAC * lim

    # axis weights (PC1, PC2)
    if mode is None or str(mode).lower() == "none":
        w1, w2 = 1.0, 1.0
    else:
        evr2 = np.asarray(evr[:2], dtype=float)
        if str(mode).lower() == "sqrt_evr":
            evr2 = np.sqrt(evr2)
        # normalize so the larger axis keeps weight 1.0 (preserves overall scale)
        evr2 = evr2 / np.max(evr2)
        w1, w2 = evr2[0], evr2[1]

    for k in range(comps.shape[0]):
        # axis-weighted components
        vx = comps[k, 0] * w1 * scale
        vy = comps[k, 1] * w2 * scale
        ax.arrow(0, 0, vx, vy,
                 color=LOAD_ARROW_COLOR, alpha=LOAD_ARROW_ALPHA,
                 length_includes_head=True,
                 head_width=head_w, head_length=head_l, linewidth=1.2)
        # label slightly beyond the arrow tip
        lx = vx + np.sign(vx) * pad
        ly = vy + np.sign(vy) * pad
        ax.text(lx, ly, feature_names[k], fontsize=LABEL_FONTSIZE,
                ha="center", va="center", color=LOAD_ARROW_COLOR)

# ========================
# MAIN
# ========================
def main():
    # Load data
    df  = load_panel_scaled()
    clu = load_clusters()
    # Merge to bring cluster labels
    dfm = df.merge(clu, on=["Country Code","Year"], how="left")
    dfm = dfm.dropna(subset=["cluster"]).copy()
    dfm["cluster"] = dfm["cluster"].astype(int)

    # Prepare matrix and fit PCA on pooled panel (global comparability)
    X = dfm[FEATURE_COLS].astype(float).to_numpy()
    pca, Z = fit_global_pca(X, n_components=2)
    dfm["PC1"] = Z[:,0]
    dfm["PC2"] = Z[:,1]

    # Save scores for any custom plotting
    dfm[["Country Name","Country Code","Year","cluster","PC1","PC2"]].to_csv(
        os.path.join(OUT_DIR, "pca_scores.csv"), index=False
    )

    # Variance explained (for titles/captions)
    evr = pca.explained_variance_ratio_
    evr1, evr2 = float(evr[0]), float(evr[1])
    evr_txt = f"Explained variance: PC1={evr1:.2%}, PC2={evr2:.2%} (sum={evr1+evr2:.2%})"

    # Global axis limits (equal scale, centered at 0)
    max_abs = float(np.nanmax(np.abs(dfm[["PC1","PC2"]].to_numpy())))
    lim = max_abs * 1.05  # 5% padding

    # Color map by cluster
    colors = _cluster_colors(dfm["cluster"].unique())

    # ---------- (A) GLOBAL SCATTER + LOADINGS ARROWS + SLIM, FLUSH KDEs ----------
    
    # ===== Marginal KDE (anchored) tuning =====
    KDE_TOP_AX_FRAC     = 0.16   # top panel thickness (relative to main height)
    KDE_TOP_LEN_FRAC    = 1.00   # top panel horizontal length (0–1 of main width)
    KDE_RIGHT_AX_FRAC   = 0.18   # right panel thickness (relative to main width)
    KDE_RIGHT_LEN_FRAC  = 1.00   # right panel vertical length (0–1 of main height)
    KDE_GAP_FRAC        = 0.02   # gap from main axes (0=flush; negative=slight overlap)
    
    KDE_BW              = 0.40   # gaussian_kde bandwidth multiplier (Scott*BW)
    KDE_RES             = 400    # curve resolution
    KDE_SCALE_TOP       = 1.00   # visual height scale of top KDE
    KDE_SCALE_RIGHT     = 1.00   # visual width scale of right KDE
    KDE_HEADROOM        = 1.05   # headroom so curves don't touch the frame
    
    KDE_LINEWIDTH       = 1.00
    KDE_LINE_ALPHA      = 0.70
    KDE_FILL_ALPHA      = 0.22
    
    TITLE_Y             = 1.10   # suptitle height
    TITLE_FONTSIZE      = 14
    
    # ---------- (A) GLOBAL SCATTER + LOADINGS ARROWS + ANCHORED, PURE KDEs ----------
    try:
        from scipy.stats import gaussian_kde
        _USE_KDE = True
    except Exception:
        _USE_KDE = False
    
    fig = plt.figure(figsize=(8.6, 7.8))
    ax_scatter = fig.add_subplot(111)  # single big axes; marginals anchored to its bbox
    
    # --- Main scatter by cluster ---
    for cl, g in dfm.groupby("cluster"):
        ax_scatter.scatter(g["PC1"], g["PC2"],
                           s=POINT_SIZE_GLOBAL, alpha=POINT_ALPHA_GLOBAL,
                           label=f"Cluster {cl}", c=colors[cl])
    
    # Centroids & loadings
    cent_global = project_centroids(dfm)
    ax_scatter.scatter(cent_global["PC1"], cent_global["PC2"],
                       s=CENTROID_SIZE, edgecolors="k", linewidths=1.0, facecolors="none")
    for _, r in cent_global.iterrows():
        ax_scatter.text(r["PC1"], r["PC2"], f"{int(r['cluster'])}", fontsize=10,
                        ha="center", va="center", color="k", weight="bold")
    
    _apply_equal_axes(ax_scatter, lim)
    _draw_loadings_arrows(ax_scatter, pca, lim, FEATURE_NICE, evr, mode=LOADINGS_AXIS_WEIGHTING)
    
    ax_scatter.set_xlabel("PC1")
    ax_scatter.set_ylabel("PC2")
    ax_scatter.legend(frameon=False, ncol=2, loc="lower left")
    ax_scatter.grid(True, alpha=GRID_ALPHA)
    
    # Draw once so bbox is final
    fig.canvas.draw()
    
    # === Anchor marginal axes to scatter bbox ===
    pos = ax_scatter.get_position()
    main_w, main_h = pos.width, pos.height
    gap_x = KDE_GAP_FRAC * main_w
    gap_y = KDE_GAP_FRAC * main_h
    
    # Top KDE (length & thickness controlled)
    top_w  = main_w * max(0.0, min(1.0, KDE_TOP_LEN_FRAC))
    top_x0 = pos.x0 + (main_w - top_w) / 2.0
    top_h  = KDE_TOP_AX_FRAC * main_h
    top_rect = [top_x0, pos.y1 + gap_y, top_w, top_h]
    ax_kdex = fig.add_axes(top_rect, sharex=ax_scatter)
    
    # Right KDE (length & thickness controlled)
    right_h  = main_h * max(0.0, min(1.0, KDE_RIGHT_LEN_FRAC))
    right_y0 = pos.y0 + (main_h - right_h) / 2.0
    right_w  = KDE_RIGHT_AX_FRAC * main_w
    right_rect = [pos.x1 + gap_x, right_y0, right_w, right_h]
    ax_kdey = fig.add_axes(right_rect, sharey=ax_scatter)
    
    # --- Pure KDE from data; no taper/alteration ---
    xlo, xhi = ax_scatter.get_xlim(); xs = np.linspace(xlo, xhi, KDE_RES)
    ylo, yhi = ax_scatter.get_ylim(); ys = np.linspace(ylo, yhi, KDE_RES)
    
    if _USE_KDE:
        # TOP KDE (PC1)
        top_max = 0.0
        for cl, g in dfm.groupby("cluster"):
            if len(g) < 5: 
                continue
            kde  = gaussian_kde(g["PC1"].astype(float), bw_method=KDE_BW)
            dens = KDE_SCALE_TOP * kde(xs)  # pure KDE
            ax_kdex.plot(xs, dens, color=colors[cl], lw=KDE_LINEWIDTH,
                         alpha=KDE_LINE_ALPHA, solid_capstyle="round")
            ax_kdex.fill_between(xs, 0, dens, color=colors[cl], alpha=KDE_FILL_ALPHA)
            top_max = max(top_max, float(dens.max()))
        ax_kdex.set_xlim(xlo, xhi)
        ax_kdex.set_ylim(0, max(1e-12, top_max) * KDE_HEADROOM)
    
        # RIGHT KDE (PC2)
        right_max = 0.0
        for cl, g in dfm.groupby("cluster"):
            if len(g) < 5: 
                continue
            kde  = gaussian_kde(g["PC2"].astype(float), bw_method=KDE_BW)
            dens = KDE_SCALE_RIGHT * kde(ys)  # pure KDE
            ax_kdey.plot(dens, ys, color=colors[cl], lw=KDE_LINEWIDTH,
                         alpha=KDE_LINE_ALPHA, solid_capstyle="round")
            ax_kdey.fill_betweenx(ys, 0, dens, color=colors[cl], alpha=KDE_FILL_ALPHA)
            right_max = max(right_max, float(dens.max()))
        ax_kdey.set_ylim(ylo, yhi)
        ax_kdey.set_xlim(0, max(1e-12, right_max) * KDE_HEADROOM)
    else:
        # Fallback histograms clipped to same limits
        bins = 30
        for cl, g in dfm.groupby("cluster"):
            ax_kdex.hist(g["PC1"], bins=bins, range=(xlo, xhi), density=True,
                         color=colors[cl], alpha=KDE_FILL_ALPHA)
            ax_kdey.hist(g["PC2"], bins=bins, range=(ylo, yhi), density=True,
                         color=colors[cl], alpha=KDE_FILL_ALPHA, orientation="horizontal")
    
    # --- Tidy marginals: only touching spine, no ticks ---
    ax_kdex.grid(False); ax_kdey.grid(False)
    
    for s in ("top", "left", "right"): ax_kdex.spines[s].set_visible(False)
    ax_kdex.spines["bottom"].set_visible(True); ax_kdex.spines["bottom"].set_linewidth(1.2)
    ax_kdex.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_kdex.tick_params(axis="y", left=False, labelleft=False)
    
    for s in ("top", "bottom", "right"): ax_kdey.spines[s].set_visible(False)
    ax_kdey.spines["left"].set_visible(True); ax_kdey.spines["left"].set_linewidth(1.2)
    ax_kdey.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_kdey.tick_params(axis="y", left=False, labelleft=False)
    
    # Title (not bold)
    fig.suptitle(
        f"PCA scatter (all country-years)\nExplained variance: PC1={evr[0]:.2%}, PC2={evr[1]:.2%} (sum={(evr[0]+evr[1]):.2%})",
        y=TITLE_Y, fontsize=TITLE_FONTSIZE
    )
    
    # No tight_layout (manual axes positioning)
    fig.savefig(os.path.join(OUT_DIR, "pca_global_scatter.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


    # ---------- (B) YEAR-BY-YEAR SCATTERS + LOADINGS ARROWS ----------
    years = sorted(dfm["Year"].dropna().unique())
    for y in years:
        gy = dfm[dfm["Year"] == y]
        if gy.empty:
            continue
        fig, ax = plt.subplots(figsize=(7.2,5.8))
        for cl, g in gy.groupby("cluster"):
            ax.scatter(g["PC1"], g["PC2"], s=POINT_SIZE_YEAR, alpha=0.75,
                       label=f"Cluster {cl}", c=colors[cl])

        # overlay year-specific centroids
        cent_y = gy.groupby("cluster")[["PC1","PC2"]].mean().reset_index()
        ax.scatter(cent_y["PC1"], cent_y["PC2"],
                   s=CENTROID_SIZE, edgecolors="k", linewidths=1.0, facecolors="none")
        for _, r in cent_y.iterrows():
            ax.text(r["PC1"], r["PC2"], f"{int(r['cluster'])}", fontsize=10,
                    ha="center", va="center", color="k", weight="bold")

        _apply_equal_axes(ax, lim)
        _draw_loadings_arrows(ax, pca, lim, FEATURE_NICE, evr, mode=LOADINGS_AXIS_WEIGHTING)

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"PCA scatter — Year {int(y)}\n{evr_txt}")
        ax.legend(frameon=False, ncol=2)
        ax.grid(True, alpha=GRID_ALPHA)
        fig.tight_layout()
        out_path = os.path.join(OUT_BY_YEAR, f"{int(y)}_pca.png")
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

    # ---------- (C) CLUSTER-CENTROID TRAJECTORIES (NO loadings arrows) ----------
    cent = (dfm.groupby(["Year","cluster"])[["PC1","PC2"]]
            .mean().reset_index().sort_values(["cluster","Year"]))

    fig, ax = plt.subplots(figsize=(7.9,6.4))
    for cl, gc in cent.groupby("cluster"):
        ax.plot(gc["PC1"], gc["PC2"], linewidth=LINE_WIDTH, color=colors[cl], label=f"Cluster {cl}")
        # small arrows to indicate direction (every ~5th step)
        for i in range(1, len(gc)):
            if i % 5 == 0 or i == len(gc)-1:
                x0, y0 = gc.iloc[i-1][["PC1","PC2"]]
                x1, y1 = gc.iloc[i][["PC1","PC2"]]
                dx, dy = x1-x0, y1-y0
                ax.arrow(x0, y0, dx, dy, length_includes_head=True,
                         head_width=0.02, head_length=0.03, color=colors[cl], alpha=0.9)
        # annotate start & end
        first = gc.iloc[0]; last = gc.iloc[-1]
        ax.text(first["PC1"], first["PC2"], str(int(first["Year"])), fontsize=8,
                ha="right", va="bottom", color=colors[cl])
        ax.text(last["PC1"],  last["PC2"],  str(int(last["Year"])),  fontsize=8,
                ha="left", va="top", color=colors[cl])

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal', adjustable='box')
    ax.axhline(0, color=AX_CROSS_COLOR, linewidth=AX_CROSS_WIDTH)
    ax.axvline(0, color=AX_CROSS_COLOR, linewidth=AX_CROSS_WIDTH)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"Cluster centroid trajectories in PC space\n{evr_txt}")
    ax.legend(frameon=False, ncol=2)
    ax.grid(True, alpha=GRID_ALPHA)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "pca_cluster_centroid_trajectories.png"), dpi=220)
    plt.close(fig)

    # ---------- (D) Export PCA loadings to Excel ----------
    loadings = pd.DataFrame(
        pca.components_.T,
        index=FEATURE_NICE,
        columns=[f"PC{i+1}" for i in range(pca.components_.shape[0])]
    )
    ev = pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(len(evr))],
        "explained_variance_ratio": evr
    })

    with pd.ExcelWriter(OUT_XLSX, engine="xlsxwriter") as xw:
        loadings.to_excel(xw, sheet_name="pca_loadings", index=True)
        ev.to_excel(xw, sheet_name="explained_variance", index=False)
        pd.DataFrame({"axis_limit_used":[lim],
                      "load_arrow_scale":[LOAD_ARROW_SCALE]}).to_excel(
            xw, sheet_name="plot_params", index=False
        )

    print("[PCA] Saved figures to:", OUT_DIR)
    print("[PCA] Saved loadings to:", OUT_XLSX)
    print(f"[PCA] Global variance explained: PC1={evr1:.2%}, PC2={evr2:.2%}, sum={(evr1+evr2):.2%}")
    print(f"[PCA] Global axis limit used: ±{lim:.3f}")

if __name__ == "__main__":
    main()