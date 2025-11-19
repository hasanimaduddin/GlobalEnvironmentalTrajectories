# -*- coding: utf-8 -*-
"""
Overlay figures only (overall + clusters):
7) overlay_step_vs_dist.png
8) overlay_ddi_vs_dist.png
9) overlay_eff_vs_dist.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE   = r"C:/Users/LEGION/Documents/Paper3 Intergenerational Environmental Efficiency/Analysis102025"
IN_XLS = os.path.join(BASE, "temporal_outputs", "marginal_movement_effectiveness_basic.xlsx")
OUTDIR = os.path.join(BASE, "temporal_outputs", "marginal_viz")
os.makedirs(OUTDIR, exist_ok=True)

# --- Load binned tables
bo = pd.read_excel(IN_XLS, sheet_name="binned_stats_overall")
bc = pd.read_excel(IN_XLS, sheet_name="binned_stats_by_prev_cluster")

# --- Ensure bin centers exist (robust but short)
def _parse_center(s):
    if not isinstance(s, str): return np.nan
    s = s.strip().replace("[","").replace("]","").replace("(","").replace(")","")
    try:
        a, b = map(float, s.split(","))
        return 0.5*(a+b)
    except Exception:
        return np.nan

def ensure_bin_center(df):
    out = df.copy()
    if "bin_center" in out.columns:
        out["bin_center"] = pd.to_numeric(out["bin_center"], errors="coerce")
    elif {"bin_left","bin_right"}.issubset(out.columns):
        out["bin_center"] = (pd.to_numeric(out["bin_left"], errors="coerce") +
                             pd.to_numeric(out["bin_right"], errors="coerce"))/2.0
    elif "bin_label" in out.columns:
        out["bin_center"] = out["bin_label"].apply(_parse_center)
    elif "prox_bin" in out.columns:
        out["bin_center"] = out["prox_bin"].astype(str).apply(_parse_center)
    else:
        raise ValueError("No bin center info found.")
    return out

bo = ensure_bin_center(bo).sort_values("bin_center")
bc = ensure_bin_center(bc)

# --- Small helper to draw one overlay figure
def overlay_plot(ycol, y_label, title, fname):
    fig, ax = plt.subplots(figsize=(10,7))

    # clusters first (thin, translucent)
    sub = bc.dropna(subset=["bin_center", ycol]).copy()
    for cl in sorted(sub["prev_cluster"].dropna().unique(), key=lambda v: int(v)):
        d = sub[sub["prev_cluster"]==cl].sort_values("bin_center")
        ax.plot(d["bin_center"].to_numpy(float),
                d[ycol].to_numpy(float),
                marker="o", linewidth=1.2, alpha=0.35, label=f"C{int(cl)}", zorder=1)

    # bold overall on top
    d0 = bo.dropna(subset=["bin_center", ycol]).copy().sort_values("bin_center")
    ax.plot(d0["bin_center"].to_numpy(float),
            d0[ycol].to_numpy(float),
            marker="o", linewidth=3.0, label="Overall", zorder=3)

    if ycol == "DDI":  # zero line for DDI
        ax.axhline(0.0, linewidth=1)

    ax.set_xlabel("Proximity to ideal at t-1 (bin center; Euclidean distance)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncols=min(5, len(ax.get_legend_handles_labels()[0])))
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, fname), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[Saved]", fname)

# --- Make the 3 overlays
overlay_plot("step_len",
             "Mean step length (||Δz||)",
             "Step size vs distance to ideal — overall (bold) with cluster overlays",
             "overlay_step_vs_dist.png")

overlay_plot("DDI",
             "Mean DDI (prev_dist − next_dist)",
             "Marginal improvement vs distance — overall (bold) with cluster overlays",
             "overlay_ddi_vs_dist.png")

overlay_plot("mean_effectiveness",
             "Mean effectiveness (DDI / step_len)",
             "Effectiveness vs distance — overall (bold) with cluster overlays",
             "overlay_eff_vs_dist.png")

print("Done. Saved to:", OUTDIR)
