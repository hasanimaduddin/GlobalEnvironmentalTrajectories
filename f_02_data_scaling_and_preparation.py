# -*- coding: utf-8 -*-
"""
Five-lens scaling (z-score -> tanh). Separate sheets for raw, [-1,1], [0,1],
plus one summary sheet with Q1/Q2/Q3/Mean/Max/Min/Skew/Kurt/Stdev.S for each dataset.
"""

import os
import numpy as np
import pandas as pd

# ========================
# CONFIG
# ========================
BASE = r"C:\Users\LEGION\Documents\Paper3 Intergenerational Environmental Efficiency\Analysis102025"
IN_FILE  = os.path.join(BASE, "combined_panel_raw_imputed.xlsx")
IN_SHEET = "panel_ready_1995_2024_full"

OUT_DIR  = os.path.join(BASE, "cluster_prep")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_XLSX = os.path.join(OUT_DIR, "panel_scaled.xlsx")

START_YEAR = 1972      # main window
END_YEAR   = None

TAU = 2.0
USE_GLOBAL_SCALING = True  # global across all years

# raw lenses (all higher = better; use DDR_inverted_raw)
LENS_RAW_COLS = ["PC", "DM", "RE", "DDR_inverted_raw", "RDP"]
LENS_ALIAS = {
    "PC": "PC",
    "DM": "DM",
    "RE": "RE",
    "DDR_inverted_raw": "DDR",
    "RDP": "RDP"
}

# ========================
# HELPERS
# ========================
def _safe_num(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return s.replace([np.inf, -np.inf], np.nan)

def _zscore(x: pd.Series) -> pd.Series:
    x = _safe_num(x)
    mu = x.mean()
    sd = x.std(ddof=0)
    if not (pd.notna(sd) and np.isfinite(sd) and sd != 0):
        return pd.Series(np.nan, index=x.index)
    return (x - mu) / sd

def z_tanh_scale(series: pd.Series, tau: float = 2.0) -> pd.Series:
    """Symmetric [-1,1] via tanh(z / tau)."""
    z = _zscore(series)
    return np.tanh(z / tau)

def to_01_from_pm1(x_pm1: pd.Series) -> pd.Series:
    return (x_pm1 + 1.0) / 2.0

def pick_writer_engine():
    try:
        import xlsxwriter  # noqa: F401
        return "xlsxwriter"
    except Exception:
        return "openpyxl"

def summary_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Return stats with rows = metrics (Q1,Q2,Q3,Mean,Max,Min,Skew,Kurt,Stdev.S) and columns = lenses."""
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    stats = pd.DataFrame({
        "Q1":       X.quantile(0.25),
        "Q2":       X.quantile(0.50),
        "Q3":       X.quantile(0.75),
        "Mean":     X.mean(),
        "Max":      X.max(),
        "Min":      X.min(),
        "Skew":     X.skew(),
        "Kurt":     X.kurt(),        # excess kurtosis (normal=0)
        "Stdev.S":  X.std(ddof=1),   # sample std = Excel STDEV.S
    }).T
    # order columns as PC, DM, RE, DDR, RDP (aliases if needed)
    ordered = []
    for raw in LENS_RAW_COLS:
        ordered.append(LENS_ALIAS[raw] if "scaled" in " ".join(X.columns) or raw == "DDR_inverted_raw" else raw)
    # Try to map nicely if columns have alias names
    final_cols = []
    for key in ["PC","DM","RE","DDR","RDP"]:
        # pick the first matching col from available
        candidates = [c for c in stats.columns if c.endswith(key) or c == key]
        if candidates:
            final_cols.append(candidates[0])
    if final_cols:
        stats = stats[final_cols]
    return stats

# ========================
# MAIN
# ========================
def main():
    # Load
    df = pd.read_excel(IN_FILE, sheet_name=IN_SHEET, engine="openpyxl")
    needed = {"Country Name","Country Code","Year"} | set(LENS_RAW_COLS)
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing expected columns in input: {missing}")

    # Windowing
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    if START_YEAR is not None:
        df = df[df["Year"].ge(START_YEAR)]
    if END_YEAR is not None:
        df = df[df["Year"].le(END_YEAR)]
    df = df.sort_values(["Country Code","Year"]).reset_index(drop=True)

    # Base (raw) sheet
    panel_raw = df[["Country Name","Country Code","Year"] + LENS_RAW_COLS].copy()

    # Grouping
    groups = [(None, panel_raw)] if USE_GLOBAL_SCALING else panel_raw.groupby("Year", dropna=False)

    blocks_pm1, blocks_01 = [], []
    for _, g in groups:
        g = g.copy()
        base_cols = ["Country Name","Country Code","Year"]
        # [-1,1]
        g_pm1 = g[base_cols].copy()
        # [0,1]
        g_01  = g[base_cols].copy()

        for raw_col in LENS_RAW_COLS:
            nice = LENS_ALIAS[raw_col]
            s_pm1 = z_tanh_scale(g[raw_col], tau=TAU)
            g_pm1[f"{nice}_scaled_pm1"] = s_pm1
            g_01[f"{nice}_scaled_01"]   = to_01_from_pm1(s_pm1)

        blocks_pm1.append(g_pm1)
        blocks_01.append(g_01)

    panel_pm1 = pd.concat(blocks_pm1, ignore_index=True).sort_values(["Country Code","Year"]).reset_index(drop=True)
    panel_01  = pd.concat(blocks_01,  ignore_index=True).sort_values(["Country Code","Year"]).reset_index(drop=True)

    # ========== Build summary tables (one sheet) ==========
    # Raw lenses use aliases for display
    raw_cols_for_summary = ["PC","DM","RE","DDR_inverted_raw","RDP"]
    raw_display = panel_raw.copy()
    raw_display = raw_display.rename(columns={"DDR_inverted_raw": "DDR"})
    summary_raw  = summary_table(raw_display, ["PC","DM","RE","DDR","RDP"])

    # Scaled [-1,1]
    pm1_cols = [c for c in panel_pm1.columns if c.endswith("_scaled_pm1")]
    pm1_display = panel_pm1.copy()
    pm1_display = pm1_display.rename(columns={
        "PC_scaled_pm1":"PC",
        "DM_scaled_pm1":"DM",
        "RE_scaled_pm1":"RE",
        "DDR_scaled_pm1":"DDR",
        "RDP_scaled_pm1":"RDP"
    })
    summary_pm1  = summary_table(pm1_display, ["PC","DM","RE","DDR","RDP"])

    # Scaled [0,1]
    s01_cols = [c for c in panel_01.columns if c.endswith("_scaled_01")]
    s01_display = panel_01.copy()
    s01_display = s01_display.rename(columns={
        "PC_scaled_01":"PC",
        "DM_scaled_01":"DM",
        "RE_scaled_01":"RE",
        "DDR_scaled_01":"DDR",
        "RDP_scaled_01":"RDP"
    })
    summary_01   = summary_table(s01_display, ["PC","DM","RE","DDR","RDP"])

    # Save (Excel with separate sheets + one combined summary sheet)
    engine = pick_writer_engine()
    with pd.ExcelWriter(OUT_XLSX, engine=engine) as xw:
        panel_raw.to_excel(xw, sheet_name="panel_raw", index=False)
        panel_pm1.to_excel(xw, sheet_name="panel_scaled_pm1", index=False)
        panel_01.to_excel(xw, sheet_name="panel_scaled_01", index=False)

        # method params
        pd.DataFrame({
            "param": ["SCALER","TAU","SPACE","START_YEAR","END_YEAR","LENSES","IN_FILE","IN_SHEET"],
            "value": [f"zscore -> tanh ([-1,1]); radar uses linear remap to [0,1]",
                      str(TAU),
                      "GLOBAL" if USE_GLOBAL_SCALING else "YEAR-WISE",
                      str(START_YEAR), str(END_YEAR) if END_YEAR else "None",
                      ", ".join(["PC","DM","RE","DDR","RDP"]),
                      os.path.basename(IN_FILE), IN_SHEET]
        }).to_excel(xw, sheet_name="method_params", index=False)

        # One sheet with three summary tables stacked
        # Write a small title cell before each block for clarity
        start = 0
        ws_name = "summary_stats"

        # RAW
        pd.DataFrame({"": [f"RAW (original lenses; DDR already inverted)"]}).to_excel(
            xw, sheet_name=ws_name, startrow=start, startcol=0, index=False, header=False)
        summary_raw.to_excel(xw, sheet_name=ws_name, startrow=start+1, startcol=0)
        start += summary_raw.shape[0] + 3

        # SCALED [-1,1]
        pd.DataFrame({"": [f"SCALED [-1,1] (z-score + tanh, tau={TAU})"]}).to_excel(
            xw, sheet_name=ws_name, startrow=start, startcol=0, index=False, header=False)
        summary_pm1.to_excel(xw, sheet_name=ws_name, startrow=start+1, startcol=0)
        start += summary_pm1.shape[0] + 3

        # SCALED [0,1]
        pd.DataFrame({"": [f"SCALED [0,1] (linear remap of [-1,1])"]}).to_excel(
            xw, sheet_name=ws_name, startrow=start, startcol=0, index=False, header=False)
        summary_01.to_excel(xw, sheet_name=ws_name, startrow=start+1, startcol=0)

    print("[Scaling] Rows written:", len(panel_raw))
    print("[Scaling] Years:", int(panel_raw["Year"].min()), "→", int(panel_raw["Year"].max()))
    print("[Scaling] Excel:", OUT_XLSX)

if __name__ == "__main__":
    main()
