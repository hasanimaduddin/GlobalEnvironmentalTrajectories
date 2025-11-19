# -*- coding: utf-8 -*-
"""
Transition matrix C(t) -> C(t+1) from country-year cluster assignments.

Inputs
------
  BASE/cluster_prep/clusters_kmeans.xlsx  (sheet: cluster_assignments)
  Columns required: ["Country Name","Country Code","Year","cluster"]

Outputs (under BASE/temporal_outputs/transitions)
-------------------------------------------------
  - transition_counts.png        (heatmap)
  - transition_probs.png         (row-normalized heatmap)
  - transition_matrices.xlsx     (sheets: counts, probs, summary)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =====================
# CONFIG
# =====================
BASE = r"C:/Users/LEGION/Documents/Paper3 Intergenerational Environmental Efficiency/Analysis102025"
CLUST_XLSX  = os.path.join(BASE, "cluster_prep", "clusters_kmeans.xlsx")
CLUST_SHEET = "cluster_assignments"

OUT_DIR = os.path.join(BASE, "temporal_outputs", "transitions")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_XLSX = os.path.join(OUT_DIR, "transition_matrices.xlsx")
FIG_COUNTS = os.path.join(OUT_DIR, "transition_counts.png")
FIG_PROBS  = os.path.join(OUT_DIR, "transition_probs.png")

# Core columns
ID_COL   = "Country Code"
NAME_COL = "Country Name"
TIME_COL = "Year"
CL_COL   = "cluster"

# Options
YEAR_MIN = None      # e.g., 1995
YEAR_MAX = None      # e.g., 2024
ONLY_CONSECUTIVE = True   # True: use pairs where Year(t+1) == Year(t) + 1
LABEL_PREFIX = "C"        # label prefix for axes: e.g., "C1..Ck" (use "S" for states)

# =====================
# LOAD
# =====================
def _need(df, cols):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise KeyError(f"Missing columns: {miss}")

def load_clusters():
    df = pd.read_excel(CLUST_XLSX, sheet_name=CLUST_SHEET, engine="openpyxl")
    _need(df, [NAME_COL, ID_COL, TIME_COL, CL_COL])
    df = (df[[NAME_COL, ID_COL, TIME_COL, CL_COL]]
          .dropna(subset=[ID_COL, TIME_COL, CL_COL])
          .copy())
    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce").astype(int)
    df[CL_COL]   = pd.to_numeric(df[CL_COL], errors="coerce").astype(int)

    if YEAR_MIN is not None:
        df = df[df[TIME_COL] >= YEAR_MIN]
    if YEAR_MAX is not None:
        df = df[df[TIME_COL] <= YEAR_MAX]

    # if duplicates (same country, same year), keep the last row
    df = (df.sort_values([ID_COL, TIME_COL])
            .drop_duplicates(subset=[ID_COL, TIME_COL], keep="last"))
    return df.sort_values([ID_COL, TIME_COL]).reset_index(drop=True)

# =====================
# TRANSITION MATRIX
# =====================
def build_transitions(cldf: pd.DataFrame):
    """
    Returns (pairs_df, counts_matrix, probs_matrix)

    pairs_df has columns: [ID_COL, TIME_COL, 'c_t', 'c_tp1']
    """
    g = (cldf.sort_values([ID_COL, TIME_COL])
              .groupby(ID_COL, group_keys=False))

    df_pairs = []
    for _, d in g:
        d = d[[ID_COL, TIME_COL, CL_COL]].copy()
        d["c_t"]    = d[CL_COL]
        d["t"]      = d[TIME_COL]
        d["c_tp1"]  = d[CL_COL].shift(-1)
        d["t_next"] = d[TIME_COL].shift(-1)

        if ONLY_CONSECUTIVE:
            d = d[(d["t_next"] == d["t"] + 1)]
        else:
            d = d[~d["c_tp1"].isna()]

        if not d.empty:
            df_pairs.append(d[[ID_COL, "t", "c_t", "c_tp1"]].rename(columns={"t": TIME_COL}))

    pairs = pd.concat(df_pairs, ignore_index=True) if df_pairs else pd.DataFrame(
        columns=[ID_COL, TIME_COL, "c_t", "c_tp1"]
    )

    # Ensure integer types
    for col in ["c_t", "c_tp1"]:
        pairs[col] = pd.to_numeric(pairs[col], errors="coerce").astype("Int64")

    # Count matrix
    levels = sorted(pd.unique(cldf[CL_COL].astype(int)))
    idx = pd.Index(levels, name=f"{LABEL_PREFIX}(t)")
    cols = pd.Index(levels, name=f"{LABEL_PREFIX}(t+1)")

    counts = (
        pd.crosstab(pairs["c_t"], pairs["c_tp1"])
        .reindex(index=idx, columns=cols)
        .fillna(0)
        .astype(int)
    )

    # Row-normalized probabilities (Markov)
    row_sums = counts.sum(axis=1).replace(0, np.nan)
    probs = (counts.div(row_sums, axis=0)).fillna(0.0)

    return pairs, counts, probs

# =====================
# PLOTS
# =====================
def plot_heatmap(mat: pd.DataFrame, out_path: str, title: str | None, cbar_label: str):
    # build pretty tick labels C1..Ck (or S1..Sk)
    states = [int(s) for s in mat.index]
    labels = [f"{LABEL_PREFIX}{s}" for s in states]

    fig, ax = plt.subplots(figsize=(9, 7.2))
    im = ax.imshow(mat.values, cmap="Blues", origin="lower", aspect="equal")

    # ticks & labels
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel(f"{LABEL_PREFIX}(t+1)")
    ax.set_ylabel(f"{LABEL_PREFIX}(t)")

    # title
    if title is None:
        title = f"Transition {cbar_label.lower()}s: {LABEL_PREFIX}(t) → {LABEL_PREFIX}(t+1)"
    ax.set_title(title)

    # annotate cells
    H, W = mat.shape
    for i in range(H):
        for j in range(W):
            val = mat.iat[i, j]
            if cbar_label.lower().startswith("prob"):
                text = f"{val:.2f}"
            else:
                text = f"{val:d}"
            ax.text(j, i, text, ha="center", va="center", color="black", fontsize=10)

    # colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

# =====================
# MAIN
# =====================
def main():
    cl = load_clusters()
    pairs, counts, probs = build_transitions(cl)

    # Save Excel
    with pd.ExcelWriter(OUT_XLSX, engine="xlsxwriter") as xw:
        counts.to_excel(xw, sheet_name="counts")
        probs.to_excel(xw, sheet_name="probs")

        # quick summary
        total_pairs = int(len(pairs))
        diag = int(np.trace(counts.values))
        up   = int((pairs["c_tp1"] > pairs["c_t"]).sum())
        down = int((pairs["c_tp1"] < pairs["c_t"]).sum())
        pd.DataFrame({
            "metric": ["n_pairs", "stay (diag)", "move_up", "move_down",
                       "stay_share", "up_share", "down_share"],
            "value":  [total_pairs, diag, up, down,
                       diag / total_pairs if total_pairs else np.nan,
                       up / total_pairs if total_pairs else np.nan,
                       down / total_pairs if total_pairs else np.nan]
        }).to_excel(xw, sheet_name="summary", index=False)

    # Plots (titles auto-use LABEL_PREFIX)
    plot_heatmap(counts, FIG_COUNTS, title=None, cbar_label="Count")
    plot_heatmap(probs,  FIG_PROBS,  title=None, cbar_label="Probability")

    print("Saved:")
    print(" -", OUT_XLSX)
    print(" -", FIG_COUNTS)
    print(" -", FIG_PROBS)

if __name__ == "__main__":
    main()
