# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 23:35:02 2025

@author: LEGION
"""

# -*- coding: utf-8 -*-
"""
Global K-Means clustering on five-lens panel ([-1,1] scaled features).
Outputs tidy assignments, ordered centroids ([-1,1] and [0,1]), and simple diagnostics.

Requires:
  C:panel_scaled.xlsx
    - sheet 'panel_scaled_pm1' with columns:
      Country Name, Country Code, Year,
      PC_scaled_pm1, DM_scaled_pm1, RE_scaled_pm1, DDR_scaled_pm1, RDP_scaled_pm1
    - (optional) sheet 'panel_scaled_01' for radar-friendly centroids

Notes:
- Single global fit across all years (comparability over time).
- Deterministic via random_state; n_init increased for robustness.
- Cluster labels are re-assigned to 1..k in ascending order of the overall mean composite
  (average of five lenses), so numbering is interpretable and stable.
"""

import os
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ========================
# CONFIG
# ========================
BASE = r"C:\Users\LEGION\Documents\Paper3 Intergenerational Environmental Efficiency\Analysis102025"
IN_XLSX = os.path.join(BASE, "cluster_prep", "panel_scaled.xlsx")
SHEET_PM1 = "panel_scaled_pm1"   # analysisSHEET_01  = "panel_scaled_01"    # for radar centroids ([0,1]) – optional features ([-1,1])


OUT_DIR  = os.path.join(BASE, "cluster_prep")
OUT_XLSX = os.path.join(OUT_DIR, "clusters_kmeans.xlsx")

START_YEAR = 1995     # main window
END_YEAR   = None     # e.g., 2024 to cap

K_CLUSTERS = 4        # <- main choice (set to 3 for robustness run)
RANDOM_STATE = 0
N_INIT = 100
MAX_ITER = 500

FEATURE_COLS_PM1 = [
    "PC_scaled_pm1","DM_scaled_pm1","RE_scaled_pm1","DDR_scaled_pm1","RDP_scaled_pm1"
]

# ========================
# HELPERS
# ========================
def pick_writer_engine():
    try:
        import xlsxwriter  # noqa: F401
        return "xlsxwriter"
    except Exception:
        return "openpyxl"

def load_panel(sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(IN_XLSX, sheet_name=sheet_name, engine="openpyxl")
    # windowing
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    if START_YEAR is not None:
        df = df[df["Year"].ge(START_YEAR)]
    if END_YEAR is not None:
        df = df[df["Year"].le(END_YEAR)]
    df = df.sort_values(["Country Code","Year"]).reset_index(drop=True)
    return df

def relabel_by_composite(labels: np.ndarray, X: np.ndarray, feature_names: list[str]) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """
    Relabel clusters to 1..k ordered by ascending overall mean composite
    (simple average across features; works because all lenses are [-1,1] and "higher=better").
    Returns new_labels, summary_df (mean by cluster), and mapping {old: new}.
    """
    k = len(np.unique(labels))
    df_tmp = pd.DataFrame(X, columns=feature_names)
    df_tmp["label"] = labels
    means = df_tmp.groupby("label")[feature_names].mean()
    means["composite_mean"] = means.mean(axis=1)
    # order clusters by composite
    order = means.sort_values("composite_mean").index.tolist()
    mapping = {old: new for new, old in enumerate(order, start=1)}
    new_labels = np.array([mapping[l] for l in labels], dtype=int)
    means_ordered = means.loc[order].copy()
    means_ordered.index = [mapping[i] for i in means_ordered.index]
    means_ordered.index.name = "cluster"
    return new_labels, means_ordered, mapping

def to01_from_pm1(x: pd.Series | np.ndarray) -> np.ndarray:
    return (np.asarray(x) + 1.0) / 2.0

# ========================
# MAIN
# ========================
def main():
    # --- Load [-1,1] features for clustering ---
    df_pm1 = load_panel(SHEET_PM1)
    need_cols = ["Country Name","Country Code","Year"] + FEATURE_COLS_PM1
    missing = [c for c in need_cols if c not in df_pm1.columns]
    if missing:
        raise KeyError(f"Missing columns in {SHEET_PM1}: {missing}")

    # matrix X and ID frame
    X = df_pm1[FEATURE_COLS_PM1].astype(float).to_numpy()
    ids = df_pm1[["Country Name","Country Code","Year"]].copy()

    # guard against any NaNs (shouldn't happen)
    mask = ~np.isnan(X).any(axis=1)
    X, ids = X[mask], ids.loc[mask].reset_index(drop=True)

    # --- Fit global K-Means ---
    km = KMeans(
        n_clusters=K_CLUSTERS,
        n_init=N_INIT,
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
    )
    labels0 = km.fit_predict(X)

    # --- Relabel clusters by composite mean (ordered 1..k) ---
    labels_ord, means_pm1, mapping = relabel_by_composite(labels0, X, FEATURE_COLS_PM1)

    # --- Compute diagnostics on the final labels ---
    wcss = float(km.inertia_)
    sil  = float(silhouette_score(X, labels0, metric="euclidean"))

    # --- Build assignments table ---
    assign = ids.copy()
    assign["cluster"] = labels_ord  # ordered labels
    # counts
    counts_overall = assign.groupby("cluster").size().reset_index(name="n_rows")
    counts_by_year = (assign.groupby(["Year","cluster"]).size()
                      .reset_index(name="n_rows")
                      .sort_values(["Year","cluster"]))

    # --- Centroids ([-1,1]) in the ordered label space ---
    means_pm1 = means_pm1.reset_index()  # columns: cluster, features..., composite_mean
    # reorder columns nicely
    means_pm1 = means_pm1[["cluster"] + FEATURE_COLS_PM1 + ["composite_mean"]]

    # --- Optional: centroids in [0,1] for radar plots ---
    # If you also want the radar-friendly centroids computed from the same means:
    centroid01 = means_pm1.copy()
    centroid01.rename(columns={
        "PC_scaled_pm1":"PC_scaled_01",
        "DM_scaled_pm1":"DM_scaled_01",
        "RE_scaled_pm1":"RE_scaled_01",
        "DDR_scaled_pm1":"DDR_scaled_01",
        "RDP_scaled_pm1":"RDP_scaled_01",
    }, inplace=True)
    for col in ["PC_scaled_01","DM_scaled_01","RE_scaled_01","DDR_scaled_01","RDP_scaled_01"]:
        src = col.replace("_scaled_01","_scaled_pm1")
        centroid01[col] = to01_from_pm1(means_pm1[src])
    # keep only [0,1] columns + cluster
    centroid01 = centroid01[["cluster","PC_scaled_01","DM_scaled_01","RE_scaled_01","DDR_scaled_01","RDP_scaled_01"]]

    # --- Save Excel ---
    engine = pick_writer_engine()
    with pd.ExcelWriter(OUT_XLSX, engine=engine) as xw:
        assign.to_excel(xw, sheet_name="cluster_assignments", index=False)
        counts_overall.to_excel(xw, sheet_name="counts_overall", index=False)
        counts_by_year.to_excel(xw, sheet_name="counts_by_year", index=False)
        means_pm1.to_excel(xw, sheet_name="centroids_pm1", index=False)
        centroid01.to_excel(xw, sheet_name="centroids_01", index=False)

        # method params & diagnostics
        pd.DataFrame({
            "param": [
                "ALGORITHM","K","RANDOM_STATE","N_INIT","MAX_ITER",
                "FEATURE_SPACE","FEATURES","START_YEAR","END_YEAR",
                "LABEL_ORDERING","WCSS","SILHOUETTE"
            ],
            "value": [
                "KMeans", str(K_CLUSTERS), str(RANDOM_STATE), str(N_INIT), str(MAX_ITER),
                "Global, pooled panel; [-1,1] tanh-z scaled",
                ", ".join([c.replace("_scaled_pm1","") for c in FEATURE_COLS_PM1]),
                str(START_YEAR), str(END_YEAR) if END_YEAR else "None",
                "Ordered by overall mean composite (ascending)",
                f"{wcss:.3f}", f"{sil:.3f}"
            ]
        }).to_excel(xw, sheet_name="method_params", index=False)

    # Console summary
    print("[KMeans] k =", K_CLUSTERS)
    print("[KMeans] WCSS (inertia):", round(wcss, 3))
    print("[KMeans] Silhouette     :", round(sil, 3))
    print("[KMeans] Counts overall:\n", counts_overall.to_string(index=False))
    print("[KMeans] Saved:", OUT_XLSX)

if __name__ == "__main__":
    main()
