# -*- coding: utf-8 -*-
"""
Marginal movement & effectiveness (no plots, no regressions) — ROBUST EXCEL VERSION
"""

import os
import numpy as np
import pandas as pd

# ----------------
# CONFIG
# ----------------
BASE = r"C:/Users/LEGION/Documents/Paper3 Intergenerational Environmental Efficiency/Analysis102025"
PANEL_XLSX  = os.path.join(BASE, r"cluster_prep/panel_scaled.xlsx")
PANEL_SHEET = "panel_scaled_pm1"
CLUST_XLSX  = os.path.join(BASE, r"cluster_prep/clusters_kmeans.xlsx")
CLUST_SHEET = "cluster_assignments"

OUT_DIR  = os.path.join(BASE, r"temporal_outputs")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_XLSX = os.path.join(OUT_DIR, "marginal_movement_effectiveness_basic.xlsx")

ID_COL   = "Country Code"
NAME_COL = "Country Name"
TIME_COL = "Year"
CL_COL   = "cluster"
DIM_COLS = ["PC_scaled_pm1", "DM_scaled_pm1", "RE_scaled_pm1", "DDR_scaled_pm1", "RDP_scaled_pm1"]

YEAR_MIN = None
YEAR_MAX = None

IDEAL_POINT = np.ones(len(DIM_COLS), dtype=float)
N_BINS = 8

def pick_writer_engine():
    try:
        import xlsxwriter  # noqa: F401
        return "xlsxwriter"
    except Exception:
        return "openpyxl"

# ----------------
# LOAD
# ----------------
panel = pd.read_excel(PANEL_XLSX, sheet_name=PANEL_SHEET, engine="openpyxl")
panel = panel[[NAME_COL, ID_COL, TIME_COL] + DIM_COLS].dropna(subset=[ID_COL, TIME_COL])
panel[TIME_COL] = panel[TIME_COL].astype(int)
panel = panel.sort_values([ID_COL, TIME_COL]).reset_index(drop=True)

clusters = pd.read_excel(CLUST_XLSX, sheet_name=CLUST_SHEET, engine="openpyxl")
clusters = clusters[[NAME_COL, ID_COL, TIME_COL, CL_COL]].dropna(subset=[ID_COL, TIME_COL, CL_COL])
clusters[TIME_COL] = clusters[TIME_COL].astype(int)
clusters[CL_COL]   = clusters[CL_COL].astype(int)
clusters = clusters.sort_values([ID_COL, TIME_COL]).reset_index(drop=True)

if YEAR_MIN is not None:
    panel    = panel[panel[TIME_COL] >= YEAR_MIN]
    clusters = clusters[clusters[TIME_COL] >= YEAR_MIN]
if YEAR_MAX is not None:
    panel    = panel[panel[TIME_COL] <= YEAR_MAX]
    clusters = clusters[clusters[TIME_COL] <= YEAR_MAX]

# ----------------
# STEP-LEVEL METRICS
# ----------------
def compute_steps_metrics(panel_df: pd.DataFrame, clusters_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for iso, g in panel_df.groupby(ID_COL, sort=False):
        g = g.sort_values(TIME_COL).reset_index(drop=True)
        country = str(g[NAME_COL].iloc[0])
        for k in range(1, len(g)):
            year_prev = int(g.loc[k-1, TIME_COL])
            year_curr = int(g.loc[k,   TIME_COL])

            prev_vec = g.loc[k-1, DIM_COLS].to_numpy(dtype=float)
            curr_vec = g.loc[k,   DIM_COLS].to_numpy(dtype=float)

            step_vec = curr_vec - prev_vec
            step_len = float(np.linalg.norm(step_vec))

            prev_dist = float(np.linalg.norm(IDEAL_POINT - prev_vec))
            curr_dist = float(np.linalg.norm(IDEAL_POINT - curr_vec))

            if prev_dist > 0:
                radial_dir = (IDEAL_POINT - prev_vec) / prev_dist
                CE = float(np.dot(step_vec, radial_dir))
            else:
                CE = 0.0

            lateral_sq  = max(step_len**2 - CE**2, 0.0)
            lateral_len = float(np.sqrt(lateral_sq))

            if step_len > 0:
                radial_share  = max(CE, 0.0) / step_len
                lateral_share = lateral_len / step_len
                effectiveness_step = (prev_dist - curr_dist) / step_len
            else:
                radial_share = 0.0
                lateral_share = 0.0
                effectiveness_step = 0.0

            DDI = prev_dist - curr_dist  # >0 means closer to ideal

            rows.append({
                "iso": iso,
                "country": country,
                "year": year_curr,             # label by arrival year t
                "prev_year": year_prev,
                "step_len": step_len,
                "prev_dist_ideal": prev_dist,
                "next_dist_ideal": curr_dist,
                "DDI": DDI,
                "CE": CE,
                "lateral_len": lateral_len,
                "radial_share": radial_share,
                "lateral_share": lateral_share,
                "effectiveness_step": effectiveness_step
            })

    steps = pd.DataFrame(rows)

    # attach prev_cluster (t-1) and next_cluster (t)
    cl_prev = clusters_df[[ID_COL, TIME_COL, CL_COL]].rename(
        columns={ID_COL: "iso", TIME_COL: "prev_year", CL_COL: "prev_cluster"}
    )
    cl_next = clusters_df[[ID_COL, TIME_COL, CL_COL]].rename(
        columns={ID_COL: "iso", TIME_COL: "year", CL_COL: "next_cluster"}
    )

    steps = steps.merge(cl_prev, on=["iso","prev_year"], how="left") \
                 .merge(cl_next, on=["iso","year"],      how="left")

    # numeric coercions
    for c in ["prev_cluster","next_cluster"]:
        if c in steps.columns:
            steps[c] = pd.to_numeric(steps[c], errors="coerce")
    return steps

steps = compute_steps_metrics(panel, clusters)

# ----------------
# COUNTRY-LEVEL SUMMARY
# ----------------
def compute_country_summary(panel_df: pd.DataFrame, steps_df: pd.DataFrame, clusters_df: pd.DataFrame) -> pd.DataFrame:
    p_sorted = panel_df.sort_values(TIME_COL)
    first    = p_sorted.groupby([ID_COL, NAME_COL]).first()[DIM_COLS]
    last     = p_sorted.groupby([ID_COL, NAME_COL]).last()[DIM_COLS]

    start_dist = pd.Series(
        np.linalg.norm(first.to_numpy(dtype=float) - IDEAL_POINT, axis=1),
        index=first.index, name="start_dist_ideal"
    )
    end_dist   = pd.Series(
        np.linalg.norm(last.to_numpy(dtype=float)  - IDEAL_POINT, axis=1),
        index=last.index,  name="end_dist_ideal"
    )
    net_improve = (start_dist - end_dist).rename("net_improvement_ideal")
    yr_first = p_sorted.groupby([ID_COL, NAME_COL])[TIME_COL].min().rename("first_year")
    yr_last  = p_sorted.groupby([ID_COL, NAME_COL])[TIME_COL].max().rename("last_year")
    years_span = (yr_last - yr_first).rename("years_span")

    base = pd.concat([start_dist, end_dist, net_improve, yr_first, yr_last, years_span], axis=1).reset_index()
    base = base.rename(columns={ID_COL: "iso", NAME_COL: "country"})

    walked = steps_df.groupby(["iso","country"])["step_len"].sum(min_count=1).rename("walked_distance").reset_index()
    steps_count = steps_df.groupby(["iso","country"])["step_len"].size().rename("steps_count").reset_index()
    med_CE  = steps_df.groupby(["iso","country"])["CE"].median().rename("median_CE").reset_index()
    med_DDI = steps_df.groupby(["iso","country"])["DDI"].median().rename("median_DDI").reset_index()
    mean_rad = steps_df.groupby(["iso","country"])["radial_share"].mean().rename("mean_radial_share").reset_index()
    mean_lat = steps_df.groupby(["iso","country"])["lateral_share"].mean().rename("mean_lateral_share").reset_index()

    out = (base.merge(walked, on=["iso","country"], how="left")
                .merge(steps_count, on=["iso","country"], how="left")
                .merge(med_CE, on=["iso","country"], how="left")
                .merge(med_DDI, on=["iso","country"], how="left")
                .merge(mean_rad, on=["iso","country"], how="left")
                .merge(mean_lat, on=["iso","country"], how="left"))

    out["walked_per_year"] = np.where(out["years_span"] > 0,
                                      out["walked_distance"] / out["years_span"], np.nan)
    out["effectiveness_ideal"] = np.where(out["walked_distance"] > 0,
                                          out["net_improvement_ideal"] / out["walked_distance"], np.nan)

    c_sorted = clusters_df.sort_values(TIME_COL)
    cl_first = c_sorted.groupby(ID_COL).first()[CL_COL].rename("start_cluster").reset_index().rename(columns={ID_COL: "iso"})
    cl_last  = c_sorted.groupby(ID_COL).last()[CL_COL].rename("end_cluster").reset_index().rename(columns={ID_COL: "iso"})
    out = out.merge(cl_first, on="iso", how="left").merge(cl_last, on="iso", how="left")

    cols = ["country","iso","first_year","last_year","years_span","steps_count",
            "walked_distance","walked_per_year",
            "start_dist_ideal","end_dist_ideal","net_improvement_ideal","effectiveness_ideal",
            "median_CE","median_DDI","mean_radial_share","mean_lateral_share",
            "start_cluster","end_cluster"]
    return out[cols].sort_values("country").reset_index(drop=True)

country_summary = compute_country_summary(panel, steps, clusters)

# ----------------
# BINNED STATS BY PROXIMITY
# ----------------
def make_proximity_bins(s: pd.Series, n_bins: int):
    lo, hi = float(np.nanmin(s.values)), float(np.nanmax(s.values))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        # fallback to a sensible window
        lo, hi = 0.0, max(1.0, hi + 1.0)
    edges = np.linspace(lo, hi, n_bins + 1)
    cats  = pd.cut(s, bins=edges, include_lowest=True)
    iv    = pd.IntervalIndex(cats.cat.categories)
    centers = iv.mid.values
    lefts   = iv.left.values
    rights  = iv.right.values
    centers_map = {cat: float(mid)  for cat, mid  in zip(cats.cat.categories, centers)}
    left_map    = {cat: float(left) for cat, left in zip(cats.cat.categories, lefts)}
    right_map   = {cat: float(rgt)  for cat, rgt  in zip(cats.cat.categories, rights)}
    return cats, edges, centers_map, left_map, right_map

def summarize_steps_overall(df: pd.DataFrame, centers_map: dict, left_map: dict, right_map: dict) -> pd.DataFrame:
    agg = {
        "step_len":      "mean",
        "DDI":           "mean",
        "CE":            "mean",
        "radial_share":  "mean",
        "lateral_share": "mean",
        "effectiveness_step": "mean",
        "iso":           "nunique",
    }
    grp = (df.groupby("prox_bin")
             .agg(agg)
             .rename(columns={"iso":"n_countries",
                              "effectiveness_step":"mean_effectiveness"})
             .reset_index())
    grp["n_steps"]        = df.groupby("prox_bin").size().values
    grp["bin_left"]       = grp["prox_bin"].map(left_map).astype(float)
    grp["bin_right"]      = grp["prox_bin"].map(right_map).astype(float)
    grp["bin_center"]     = grp["prox_bin"].map(centers_map).astype(float)
    grp["bin_label"]      = grp["prox_bin"].astype(str)
    # drop raw Interval to avoid Excel writer glitches
    grp = grp.drop(columns=["prox_bin"]).sort_values("bin_center").reset_index(drop=True)
    return grp

def summarize_by_prev_cluster(df: pd.DataFrame, centers_map: dict, left_map: dict, right_map: dict) -> pd.DataFrame:
    agg = {
        "step_len":      "mean",
        "DDI":           "mean",
        "CE":            "mean",
        "radial_share":  "mean",
        "lateral_share": "mean",
        "effectiveness_step": "mean",
        "iso":           "nunique",
    }
    grp = (df.groupby(["prev_cluster","prox_bin"])
             .agg(agg)
             .rename(columns={"iso":"n_countries",
                              "effectiveness_step":"mean_effectiveness"})
             .reset_index())
    grp["n_steps"]    = df.groupby(["prev_cluster","prox_bin"]).size().values
    grp["bin_left"]   = grp["prox_bin"].map(left_map).astype(float)
    grp["bin_right"]  = grp["prox_bin"].map(right_map).astype(float)
    grp["bin_center"] = grp["prox_bin"].map(centers_map).astype(float)
    grp["bin_label"]  = grp["prox_bin"].astype(str)
    grp = grp.drop(columns=["prox_bin"]).sort_values(["prev_cluster","bin_center"]).reset_index(drop=True)
    return grp

steps_nonan = steps.dropna(subset=["prev_dist_ideal"]).copy()
cats, edges, centers_map, left_map, right_map = make_proximity_bins(steps_nonan["prev_dist_ideal"], N_BINS)
steps_nonan["prox_bin"] = cats

binned_overall = summarize_steps_overall(steps_nonan, centers_map, left_map, right_map)
binned_by_prev = summarize_by_prev_cluster(steps_nonan, centers_map, left_map, right_map)

# ----------------
# SAVE EXCEL
# ----------------
with pd.ExcelWriter(OUT_XLSX, engine=pick_writer_engine()) as xw:
    steps_to_save = (steps[[
        "country","iso","prev_year","year",
        "prev_cluster","next_cluster",
        "prev_dist_ideal","next_dist_ideal",
        "step_len","DDI","CE","lateral_len","radial_share","lateral_share",
        "effectiveness_step"
    ]]
    .sort_values(["country","year"]))
    steps_to_save.to_excel(xw, sheet_name="steps_metrics", index=False)

    country_summary.to_excel(xw, sheet_name="country_summary", index=False)
    binned_overall.to_excel(xw, sheet_name="binned_stats_overall", index=False)
    binned_by_prev.to_excel(xw, sheet_name="binned_stats_by_prev_cluster", index=False)

    yrs = panel[TIME_COL].agg(["min","max"]).tolist()
    pd.DataFrame({
        "param": ["YEAR_MIN","YEAR_MAX","N_BINS","N_countries","Years_in_data","Bin_edges"],
        "value": [YEAR_MIN if YEAR_MIN is not None else yrs[0],
                  YEAR_MAX if YEAR_MAX is not None else yrs[1],
                  N_BINS,
                  panel[ID_COL].nunique(),
                  f"{yrs[0]}–{yrs[1]}",
                  "; ".join(f"{e:.6g}" for e in edges)]
    }).to_excel(xw, sheet_name="params", index=False)

print("Done. Excel written to:", OUT_XLSX)
