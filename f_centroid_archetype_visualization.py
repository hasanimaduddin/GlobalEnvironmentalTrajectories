# -*- coding: utf-8 -*-
"""
Cluster centroid heatmap visualization (z-score / [-1,1] scaled values)
- Reads centroids directly from Excel (no hard-coded numbers)
- Draws a heatmap of archetypal profiles across the five lenses + composite

Input:
  C:/Users/LEGION/Documents/Paper3 Intergenerational Environmental Efficiency/Analysis102025/cluster_prep/clusters_kmeans.xlsx
    sheet: 'centroids_pm1'
    columns: ['cluster','PC_scaled_pm1','DM_scaled_pm1','RE_scaled_pm1','DDR_scaled_pm1','RDP_scaled_pm1','composite_mean']

Output:
  <BASE>/temporal_outputs/cluster_centroid_heatmap.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------
# CONFIG
# -------------
BASE   = r"C:/Users/LEGION/Documents/Paper3 Intergenerational Environmental Efficiency/Analysis102025"
IN_XLS = os.path.join(BASE, "cluster_prep", "clusters_kmeans.xlsx")
IN_SHEET = "centroids_pm1"

OUT_DIR = os.path.join(BASE, "temporal_outputs")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PNG = os.path.join(OUT_DIR, "cluster_centroid_heatmap.png")

# Column mapping from file -> pretty labels (order preserved)
COLMAP = {
    "PC_scaled_pm1":  "PC",
    "DM_scaled_pm1":  "DM",
    "RE_scaled_pm1":  "RE",
    "DDR_scaled_pm1": "DDR",
    "RDP_scaled_pm1": "RDP",
    "composite_mean": "Composite"
}

# -------------
# LOAD
# -------------
df = pd.read_excel(IN_XLS, sheet_name=IN_SHEET, engine="openpyxl")

# basic checks / cleanup
need = ["cluster"] + list(COLMAP.keys())
missing = [c for c in need if c not in df.columns]
if missing:
    raise KeyError(f"Missing columns in '{IN_SHEET}': {missing}")

df = df[need].copy()
df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce").astype("Int64")
for c in COLMAP.keys():
    df[c] = pd.to_numeric(df[c], errors="coerce")

# rename to pretty labels and set index
df = df.rename(columns=COLMAP).set_index("cluster").sort_index()

# -------------
# HEATMAP
# -------------
plt.figure(figsize=(8, 5))
sns.heatmap(
    df,
    annot=True, fmt=".3f",           # numbers with 3 decimals
    cmap="coolwarm", center=0.0,
    linewidths=0.5, linecolor="gray",
    cbar_kws={"label": "Scaled Value (−1 to 1)"},
    annot_kws={"size": 9}
)

plt.title("Cluster Centroid Profiles (z-score / [-1,1] scaled)", fontsize=13, pad=12)
plt.ylabel("Cluster ID")
plt.xlabel("Environmental performance lenses")

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
plt.close()

print("Loaded from:", f"{IN_XLS} :: {IN_SHEET}")
print("Saved heatmap to:", OUT_PNG)
