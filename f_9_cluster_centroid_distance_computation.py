# -*- coding: utf-8 -*-
"""
Year-by-year K-Means (k=4) on five-lens [-1,1] features
+ Trend tests on WCD (within-cluster distance) lines:
    • Parametric: OLS WCD ~ Year with Newey–West (HAC) SEs
    • Nonparametric: Mann–Kendall trend test

Outputs:
  - temporal_outputs/wcd_by_year_k4.csv
  - temporal_outputs/wcd_by_year_k4.png
  - temporal_outputs/wcd_trend_tests_k4.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# for tests
from math import sqrt
from scipy.stats import norm
try:
    import statsmodels.api as sm
    HAS_SM = True
except Exception:
    HAS_SM = False

# ----------------
# CONFIG
# ----------------
BASE = r"C:/Users/LEGION/Documents/Paper3 Intergenerational Environmental Efficiency/Analysis102025"
IN_XLSX = os.path.join(BASE, "cluster_prep", "panel_scaled.xlsx")
SHEET   = "panel_scaled_pm1"

OUT_DIR = os.path.join(BASE, "temporal_outputs")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV_WCD   = os.path.join(OUT_DIR, "wcd_by_year_k4.csv")
OUT_FIG_WCD   = os.path.join(OUT_DIR, "wcd_by_year_k4.png")
OUT_CSV_TESTS = os.path.join(OUT_DIR, "wcd_trend_tests_k4.csv")

START_YEAR = 1995
END_YEAR   = None   # e.g., 2024

K_CLUSTERS   = 4
RANDOM_STATE = 0
N_INIT       = 100
MAX_ITER     = 500

ID_COLS   = ["Country Name","Country Code","Year"]
FEATURES  = ["PC_scaled_pm1","DM_scaled_pm1","RE_scaled_pm1","DDR_scaled_pm1","RDP_scaled_pm1"]

# ----------------
# LOAD
# ----------------
df = pd.read_excel(IN_XLSX, sheet_name=SHEET, engine="openpyxl")
need = ID_COLS + FEATURES
missing = [c for c in need if c not in df.columns]
if missing:
    raise KeyError(f"Missing columns in {SHEET}: {missing}")

df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype(int)
if START_YEAR is not None:
    df = df[df["Year"] >= START_YEAR]
if END_YEAR is not None:
    df = df[df["Year"] <= END_YEAR]

df = df.dropna(subset=FEATURES + ["Year"]).copy()
df = df.sort_values(["Country Code","Year"]).reset_index(drop=True)
years = sorted(df["Year"].unique().tolist())

# ----------------
# HELPERS
# ----------------
def order_labels_by_composite(centers: np.ndarray) -> dict:
    comp = centers.mean(axis=1)      # all lenses higher=better
    order_old = np.argsort(comp)     # idx low -> high
    return {int(old): int(new) for new, old in enumerate(order_old, start=1)}  # old->new (1..k)

def mann_kendall(y: np.ndarray):
    """
    Mann–Kendall test for monotonic trend (two-sided).
    Returns: tau, S, varS, z, p_value.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    # S
    S = 0
    for k in range(n-1):
        S += np.sign(y[k+1:] - y[k]).sum()
    # variance with tie correction
    # count ties in y
    _, counts = np.unique(y, return_counts=True)
    tie_term = np.sum(counts*(counts-1)*(2*counts+5))
    varS = (n*(n-1)*(2*n+5) - tie_term) / 18.0
    # z
    if S > 0:
        z = (S - 1) / sqrt(varS)
    elif S < 0:
        z = (S + 1) / sqrt(varS)
    else:
        z = 0.0
    p = 2*(1 - norm.cdf(abs(z)))
    # Kendall's tau
    tau = S / (0.5*n*(n-1))
    return float(tau), float(S), float(varS), float(z), float(p)

def choose_maxlags(n: int) -> int:
    # simple, conservative rule-of-thumb for HAC lag
    return max(1, int(round(n**0.25)))

def ols_newey_west(x_year: np.ndarray, y: np.ndarray):
    """
    OLS y ~ 1 + year with HAC (Newey–West) SEs.
    Returns: slope, se, t, p, n, maxlags
    """
    if not HAS_SM:
        return np.nan, np.nan, np.nan, np.nan, len(y), np.nan
    t = np.asarray(x_year, dtype=float)
    y = np.asarray(y, dtype=float)
    X = sm.add_constant(t)  # [const, year]
    model = sm.OLS(y, X, missing='drop')
    n = X.shape[0]
    maxlags = choose_maxlags(n)
    res = model.fit(cov_type='HAC', cov_kwds={'maxlags': maxlags})
    slope = float(res.params[1])
    se    = float(res.bse[1])
    tval  = float(res.tvalues[1])
    pval  = float(res.pvalues[1])
    return slope, se, tval, pval, n, maxlags

# ----------------
# MAIN: YEAR-BY-YEAR K-MEANS + WCD
# ----------------
rows = []

for y in years:
    dy = df[df["Year"] == y].dropna(subset=FEATURES)
    X = dy[FEATURES].to_numpy(dtype=float)

    if X.shape[0] < K_CLUSTERS:
        continue

    km = KMeans(
        n_clusters=K_CLUSTERS,
        n_init=N_INIT,
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
    )
    labels0 = km.fit_predict(X)
    centers = km.cluster_centers_

    # per-year relabel to 1..k by centroid composite
    map_old2new = order_labels_by_composite(centers)

    # distances to own centroid
    own_centers = centers[labels0]
    dists = np.linalg.norm(X - own_centers, axis=1)

    labels_ord = np.array([map_old2new[int(l)] for l in labels0], dtype=int)

    for c in range(1, K_CLUSTERS+1):
        mask = labels_ord == c
        if not np.any(mask):
            continue
        rows.append({
            "Year": int(y),
            "cluster": int(c),
            "wcd_mean": float(np.mean(dists[mask])),
            "wcd_median": float(np.median(dists[mask])),
            "wcd_std": float(np.std(dists[mask], ddof=0)),
            "n": int(mask.sum())
        })

wcd_df = pd.DataFrame(rows).sort_values(["Year","cluster"]).reset_index(drop=True)
wcd_df.to_csv(OUT_CSV_WCD, index=False)

# ----------------
# TREND TESTS PER CLUSTER
# ----------------
tests = []
for c in range(1, K_CLUSTERS+1):
    gc = wcd_df[wcd_df["cluster"] == c].dropna(subset=["wcd_mean"])
    if gc.empty or gc["Year"].nunique() < 5:
        continue
    years_c = gc["Year"].to_numpy()
    wcd_c   = gc["wcd_mean"].to_numpy()

    # Newey–West OLS
    slope, se, tval, pval, n_nw, maxlags = ols_newey_west(years_c, wcd_c)

    # Mann–Kendall
    tau, S, varS, z, p_mk = mann_kendall(wcd_c)

    tests.append({
        "cluster": c,
        "n_years": int(len(wcd_c)),
        # Newey–West regression
        "slope_per_year": slope,         # change in mean WCD per calendar year
        "se_NW": se,
        "t_NW": tval,
        "p_NW": pval,
        "NW_maxlags": maxlags,
        # Mann–Kendall
        "tau_MK": tau,
        "S_MK": S,
        "z_MK": z,
        "p_MK": p_mk
    })

tests_df = pd.DataFrame(tests).sort_values("cluster")
tests_df.to_csv(OUT_CSV_TESTS, index=False)

# ----------------
# PLOT
# ----------------
plt.figure(figsize=(12, 6))
palette = ["#1f77b4","#ff7f0e","#2ca02c","#d62728"]  # C1..C4

for c in range(1, K_CLUSTERS+1):
    gc = wcd_df[wcd_df["cluster"] == c]
    if gc.empty:
        continue
    plt.plot(gc["Year"], gc["wcd_mean"], linewidth=2, label=f"C{c}", color=palette[(c-1) % len(palette)])

plt.xlabel("Year")
plt.ylabel("Average distance to centroid (Euclidean in 5D)")
plt.title("Within-Cluster Distance (mean) by Year — KMeans refit each year\nClusters ordered per year by centroid composite (low→high)")
plt.grid(True, alpha=0.3)
plt.legend(title="Cluster (ordered per year)", frameon=False, ncol=4)
plt.tight_layout()
plt.savefig(OUT_FIG_WCD, dpi=300, bbox_inches="tight")
plt.close()

print("Saved WCD table to  :", OUT_CSV_WCD)
print("Saved trend tests to:", OUT_CSV_TESTS)
print("Saved figure to     :", OUT_FIG_WCD)
if not HAS_SM:
    print("[WARN] statsmodels not found; Newey–West results are NaN. Install `statsmodels` for HAC regression.")
