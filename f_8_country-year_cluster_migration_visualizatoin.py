# -*- coding: utf-8 -*-
"""
Cluster assignment heatmaps by region (1995–2024)
-------------------------------------------------
- One PNG per region (using the scheme below, with explicit overrides).
- Optional exclusion of countries with no cluster movement.
- Flexible row sorting: alphabetical or by movement intensity.

Inputs
------
BASE/cluster_prep/clusters_kmeans.xlsx  (sheet: "cluster_assignments")
Required columns: ["Country Name", "Country Code", "Year", "cluster"]

Outputs
-------
BASE/temporal_outputs/regions/cluster_heatmap_<region>.png
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# =====================
# CONFIG
# =====================
BASE = r"C:/Users/LEGION/Documents/Paper3 Intergenerational Environmental Efficiency/Analysis102025"
CLUST_XLSX  = os.path.join(BASE, "cluster_prep", "clusters_kmeans.xlsx")
CLUST_SHEET = "cluster_assignments"

OUT_DIR  = os.path.join(BASE, "temporal_outputs", "regions")
os.makedirs(OUT_DIR, exist_ok=True)

ID_COL   = "Country Code"
NAME_COL = "Country Name"
TIME_COL = "Year"
CL_COL   = "cluster"

YEAR_MIN = 1995
YEAR_MAX = 2024

# Row filtering and sorting knobs
EXCLUDE_STATIC = False                  # True → drop countries with no cluster changes across the window
ROW_SORT_MODE  = "movement_then_alpha"  # "alpha" | "movement" | "movement_then_alpha"

# Colors consistent with PCA: C1=blue, C2=orange, C3=green, C4=red
cluster_cmap = mcolors.ListedColormap([
    "#377EB8",  # C1 blue
    "#FF7F00",  # C2 orange
    "#4DAF4A",  # C3 green
    "#E41A1C",  # C4 red
])
cluster_cmap.set_bad("lightgray")  # show NA as gray on the heatmap, but not in the colorbar

# 4 bins only (1..4). NA will be masked (not binned), so the colorbar has no gray segment.
bounds = [0.5, 1.5, 2.5, 3.5, 4.5]
norm = mcolors.BoundaryNorm(bounds, cluster_cmap.N, clip=True)

# =====================
# FONT SIZES
# =====================
TITLE_FONTSIZE          = 18  # title of each heatmap
AXIS_LABEL_FONTSIZE     = 16  # "Year", "Country"
TICK_LABEL_FONTSIZE     = 14  # years on x-axis, country names on y-axis
COLORBAR_TICK_FONTSIZE  = 14  # C1–C4 labels on colorbar

# =====================
# REGION SCHEME + overrides
# =====================
OVERRIDE_REGION = {
    "Turkey":  "Middle East & North Africa",
    "Albania": "Europe East",
    "Armenia": "East & Central Asia",
}

EUROPE_WEST = {
    "Austria","Belgium","Denmark","Finland","France","Germany","Ireland",
    "Italy","Luxembourg","Netherlands","Norway","Portugal","Spain","Sweden",
    "Switzerland","United Kingdom"
}
EUROPE_EAST = {
    "Bulgaria","Croatia","Czech Republic","Estonia","Hungary","Latvia","Lithuania",
    "Poland","Romania","Slovakia","Slovenia","Cyprus","Greece","Albania",
    "Ukraine","Russian Federation"
}
ASEAN = {
    "Brunei Darussalam","Indonesia","Malaysia","Myanmar","Philippines",
    "Singapore","Thailand","Viet Nam","Lao People's Democratic Republic","Cambodia"
}
SOUTH_ASIA = {"Bangladesh","India","Maldives","Nepal","Pakistan","Sri Lanka"}
EAST_CENTRAL_ASIA = {
    "China","Japan","Mongolia","Republic of Korea","Kazakhstan","Armenia"
}
MENA = {
    "Algeria","Bahrain","Egypt","Iran (Islamic Republic of)","Iraq","Israel","Jordan",
    "Kuwait","Morocco","Qatar","Saudi Arabia","Tunisia","United Arab Emirates","Yemen","Turkey"
}
SUB_SAHARAN_AFRICA = {
    "Angola","Benin","Botswana","Burkina Faso","Cameroon","Congo","D.R. of the Congo",
    "Côte d'Ivoire","Ethiopia","Gabon","Gambia","Ghana","Guinea","Kenya","Madagascar",
    "Mauritius","Mozambique","Namibia","Niger","Nigeria","Rwanda","Senegal","South Africa",
    "Eswatini","Togo","Uganda","Zambia","Zimbabwe"
}
NORTH_AMERICA = {"Canada","United States","Mexico"}
LAC = {
    "Argentina","Barbados","Belize","Bolivia (Plurinational State of)","Brazil","Chile",
    "Colombia","Costa Rica","Cuba","Dominican Republic","Ecuador","El Salvador","Guatemala",
    "Haiti","Honduras","Jamaica","Nicaragua","Panama","Paraguay","Peru","Trinidad and Tobago","Uruguay"
}
OCEANIA = {"Australia","New Zealand","Fiji"}

REGION_ORDER = [
    "Europe West",
    "Europe East",
    "Middle East & North Africa",
    "East & Central Asia",
    "South Asia",
    "ASEAN",
    "North America",
    "Latin America & Caribbean",
    "Sub-Saharan Africa",
    "Oceania",
    "Others",
]

REGION_MAP = {
    "Europe West": EUROPE_WEST,
    "Europe East": EUROPE_EAST,
    "Middle East & North Africa": MENA,
    "East & Central Asia": EAST_CENTRAL_ASIA,
    "South Asia": SOUTH_ASIA,
    "ASEAN": ASEAN,
    "North America": NORTH_AMERICA,
    "Latin America & Caribbean": LAC,
    "Sub-Saharan Africa": SUB_SAHARAN_AFRICA,
    "Oceania": OCEANIA,
}

def country_to_region(name: str) -> str:
    if name in OVERRIDE_REGION:
        return OVERRIDE_REGION[name]
    for region, members in REGION_MAP.items():
        if name in members:
            return region
    return "Others"

# =====================
# LOAD & PREP
# =====================
df = pd.read_excel(CLUST_XLSX, sheet_name=CLUST_SHEET, engine="openpyxl")
df = df[[NAME_COL, ID_COL, TIME_COL, CL_COL]].dropna(subset=[ID_COL, TIME_COL, CL_COL]).copy()
df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce").astype(int)
df[CL_COL]   = pd.to_numeric(df[CL_COL], errors="coerce").astype(int)
df = df[(df[TIME_COL] >= YEAR_MIN) & (df[TIME_COL] <= YEAR_MAX)].copy()
df["Region"] = df[NAME_COL].map(country_to_region)

heatmap_data = (
    df.pivot(index=NAME_COL, columns=TIME_COL, values=CL_COL)
      .reindex(columns=list(range(YEAR_MIN, YEAR_MAX + 1)))
      .sort_index()
)

def movement_count(row: pd.Series) -> int:
    x = row.dropna().astype(int).values
    if x.size <= 1:
        return 0
    return int(np.sum(x[1:] != x[:-1]))

movement_series = heatmap_data.apply(movement_count, axis=1)

# =====================
# DRAW
# =====================
def draw_heatmap(subset: pd.DataFrame, title: str, outfile: str, row_labels=None):
    # Mask NaNs so they render with cmap.set_bad(...) and DO NOT appear in the colorbar
    Z = subset.to_numpy(dtype=float)
    Z = np.ma.masked_invalid(Z)

    fig_height = max(3.8, 0.30 * len(subset))
    fig, ax = plt.subplots(figsize=(18, fig_height))
    im = ax.imshow(Z, aspect="auto", cmap=cluster_cmap, norm=norm)

    ax.set_xticks(np.arange(-0.5, Z.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, Z.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Major ticks & labels
    ax.set_yticks(range(len(subset)))
    ax.set_yticklabels(
        row_labels if row_labels is not None else subset.index,
        fontsize=TICK_LABEL_FONTSIZE
    )
    ax.set_xticks(range(len(subset.columns)))
    ax.set_xticklabels(
        subset.columns,
        fontsize=TICK_LABEL_FONTSIZE,
        rotation=90
    )

    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Year", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Country", fontsize=AXIS_LABEL_FONTSIZE)

    # Colorbar without the gray NA segment
    cbar = plt.colorbar(
        im,
        ax=ax,
        orientation="vertical",
        fraction=0.02,
        pad=0.02,
        boundaries=bounds
    )
    cbar.set_ticks([1, 2, 3, 4])
    cbar.set_ticklabels(["C1", "C2", "C3", "C4"])
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)

    plt.tight_layout()
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)

def sanitize_filename(s: str) -> str:
    s = re.sub(r"[^\w\s\-\&]", "", s)
    s = re.sub(r"\s+", "_", s.strip())
    return s

for region in REGION_ORDER:
    members = df.loc[df["Region"] == region, NAME_COL].unique().tolist()
    if not members:
        continue

    sub = heatmap_data.loc[sorted(set(members) & set(heatmap_data.index))]

    if EXCLUDE_STATIC:
        dyn_idx = movement_series.loc[sub.index]
        sub = sub.loc[dyn_idx > 0]

    if sub.empty:
        continue

    if ROW_SORT_MODE == "alpha":
        ordered_countries = sorted(sub.index.tolist())
    elif ROW_SORT_MODE == "movement":
        ordered_countries = (
            movement_series.loc[sub.index]
            .sort_values(ascending=False)
            .index
            .tolist()
        )
    elif ROW_SORT_MODE == "movement_then_alpha":
        tmp = pd.DataFrame({"country": sub.index, "moves": movement_series.loc[sub.index].values})
        tmp = tmp.sort_values(["moves", "country"], ascending=[False, True])
        ordered_countries = tmp["country"].tolist()
    else:
        ordered_countries = sub.index.tolist()

    sub = sub.loc[ordered_countries]

    title = f"Cluster assignment paths — {region} ({YEAR_MIN}–{YEAR_MAX})"
    outfile = os.path.join(OUT_DIR, f"cluster_heatmap_{sanitize_filename(region)}.png")

    row_labels = [f"{c}  (Δ={movement_series[c]})" for c in sub.index]

    draw_heatmap(sub, title, outfile, row_labels=row_labels)

print(f"Saved regional heatmaps to: {OUT_DIR}")
