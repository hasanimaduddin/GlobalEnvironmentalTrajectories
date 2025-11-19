# -*- coding: utf-8 -*-
"""
Cluster diagnostics (standard-tools only):
1) Elbow (WCSS via KMeans.inertia_)
2) Silhouette score (sklearn.metrics.silhouette_score)
3) Seed stability via ARI (sklearn.metrics.adjusted_rand_score)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

# ========================
# CONFIG
# ========================
BASE = r"C:\Users\LEGION\Documents\Paper3 Intergenerational Environmental Efficiency\Analysis102025"
IN_XLSX = os.path.join(BASE, "cluster_prep", "panel_scaled.xlsx")
IN_SHEET = "panel_scaled_pm1"

OUT_DIR = os.path.join(BASE, "cluster_prep", "diagnostics_standard")
os.makedirs(OUT_DIR, exist_ok=True)

FEATURE_COLS = [
    "PC_scaled_pm1","DM_scaled_pm1","RE_scaled_pm1","DDR_scaled_pm1","RDP_scaled_pm1"
]

K_RANGE = list(range(2, 11))     # evaluate k = 2..10
SEED_LIST = list(range(1, 21))   # stability via 20 different seeds

# KMeans settings (standard sklearn)
KM_N_INIT = 50
KM_MAX_ITER = 500

# ========================
# HELPERS
# ========================
def load_features():
    df = pd.read_excel(IN_XLSX, sheet_name=IN_SHEET, engine="openpyxl")
    need = ["Country Code","Year"] + FEATURE_COLS
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise KeyError(f"Missing columns in input: {miss}")
    X = df[FEATURE_COLS].astype(float).to_numpy()
    ids = df[["Country Code","Year"]].reset_index(drop=True)
    # Drop rows with any NaN just in case
    mask = ~np.isnan(X).any(axis=1)
    return X[mask], ids.loc[mask].reset_index(drop=True)

def fit_kmeans(X, k, random_state):
    km = KMeans(n_clusters=k, n_init=KM_N_INIT, max_iter=KM_MAX_ITER, random_state=random_state)
    labels = km.fit_predict(X)
    return km, labels

# ========================
# MAIN
# ========================
def main():
    X, ids = load_features()

    rows = []
    stability_records = []  # (k, seed, ari)

    for k in K_RANGE:
        # --- Elbow (WCSS) ---
        km0, labels0 = fit_kmeans(X, k, random_state=0)
        wcss = float(km0.inertia_)

        # --- Silhouette (built-in) ---
        # Silhouette needs at least 2 clusters (true for k>=2); labels from km0
        sil = float(silhouette_score(X, labels0, metric="euclidean"))

        # --- Stability across seeds (ARI vs baseline) ---
        aris = []
        for s in SEED_LIST:
            km_s, labels_s = fit_kmeans(X, k, random_state=s)
            ari = float(adjusted_rand_score(labels0, labels_s))
            aris.append(ari)
            stability_records.append({"k": k, "seed": s, "ari": ari})

        rows.append({
            "k": k,
            "wcss": wcss,
            "silhouette": sil,
            "stability_ari_mean": float(np.mean(aris)),
            "stability_ari_min":  float(np.min(aris)),
            "stability_ari_max":  float(np.max(aris)),
        })

    # Save metrics
    metrics = pd.DataFrame(rows)
    metrics_path = os.path.join(OUT_DIR, "metrics_by_k.csv")
    metrics.to_csv(metrics_path, index=False)

    stab = pd.DataFrame(stability_records)
    stab_path = os.path.join(OUT_DIR, "stability_seeds_ari.csv")
    stab.to_csv(stab_path, index=False)

    print("[Diagnostics] Saved:", metrics_path)
    print("[Diagnostics] Saved:", stab_path)

    # --------- Plots (all matplotlib, no seaborn) ---------
    # 1) Elbow
    plt.figure(figsize=(7,5))
    plt.plot(metrics["k"], metrics["wcss"], marker="o")
    plt.xlabel("k")
    plt.ylabel("Within-Cluster Sum of Squares (WCSS)")
    plt.title("Elbow Method (KMeans inertia)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "elbow_wcss.png"), dpi=200)
    plt.close()

    # 2) Silhouette
    plt.figure(figsize=(7,5))
    plt.plot(metrics["k"], metrics["silhouette"], marker="o")
    plt.axhline(0.25, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="rule-of-thumb: 0.25")
    plt.xlabel("k")
    plt.ylabel("Silhouette score")
    plt.title("Silhouette vs k (higher = better)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "silhouette_vs_k.png"), dpi=200)
    plt.close()

    # 3) Stability across seeds (boxplot of ARI per k)
    plt.figure(figsize=(8,5))
    data = [stab.loc[stab["k"] == k, "ari"].to_numpy() for k in K_RANGE]
    plt.boxplot(data, labels=K_RANGE, showmeans=True)
    plt.ylim(0, 1.01)
    plt.xlabel("k")
    plt.ylabel("Adjusted Rand Index vs baseline (seed=0)")
    plt.title("Stability across seeds (higher = more stable)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "stability_seeds_ari.png"), dpi=200)
    plt.close()

    print("[Diagnostics] Figures saved in:", OUT_DIR)
    print(metrics.to_string(index=False))

if __name__ == "__main__":
    main()
