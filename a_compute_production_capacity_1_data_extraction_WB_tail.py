# -*- coding: utf-8 -*-
"""
WB Tail Extract — raw panel only (no imputation)

Reads:
  <BASE>/WBRAW.xlsx  (sheet = WDI_SHEET)

Writes:
  <OUT_DIR>/pc_step1_wb_tail_<YEAR_START>_<YEAR_END>.xlsx

Sheets:
  - original_raw_subset  (wide subset of selected indicators for the year window)
  - panel_raw            (Country, Code, Year + selected indicators as columns)
  - panel_data           (same as panel_raw)
"""

import os
import numpy as np
import pandas as pd

# ========================
# CONFIG (edit here)
# ========================
BASE = r"C:\Users\LEGION\Documents\Paper3 Intergenerational Environmental Efficiency"
WDI_EXCEL   = os.path.join(BASE, "WBRAW.xlsx")
WDI_SHEET   = "Data"
OUT_DIR     = os.path.join(BASE, r"Analysis_12082025")

# Year window
YEAR_START  = 2015
YEAR_END    = 2024

# Indicator names in WBRAW.xlsx  -> short column names in panel
INDICATOR_MAP = {
    "Gross fixed capital formation (constant 2015 US$)": "wb_gfcf_const2015",
    "Population, total": "wb_pop",
    "Labor force, total": "wb_l_labor",
    "Employment to population ratio, 15+, total (%) (modeled ILO estimate)": "wb_emp_pop_ratio_pct",  # optional
    "Energy use (kg of oil equivalent per capita)": "E_pc",  # optional but useful
}

# Output
OUT_EXTRACT = os.path.join(OUT_DIR, f"pc_step1_wb_tail_{YEAR_START}_{YEAR_END}.xlsx")
SHEET_WIDE  = "original_raw_subset"
SHEET_PANEL = "panel_raw"
SHEET_DATA  = "panel_data"

# ========================
# Helpers
# ========================
def pick_writer_engine():
    try:
        import xlsxwriter  # noqa
        return "xlsxwriter"
    except Exception:
        return "openpyxl"

def year_columns_in_range(df: pd.DataFrame, y0: int, y1: int):
    years = []
    for c in df.columns:
        try:
            y = int(c)
            if y0 <= y <= y1:
                years.append(str(y))
        except Exception:
            pass
    return sorted(years)

# ========================
# Main
# ========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(WDI_EXCEL):
        raise FileNotFoundError(f"WDI file not found:\n{WDI_EXCEL}")

    # Load WDI (wide)
    raw_wide = pd.read_excel(WDI_EXCEL, sheet_name=WDI_SHEET, dtype={"Country Code": str})
    raw_wide.columns = [c.strip() for c in raw_wide.columns]

    # Keep only indicators we care about
    keep_mask = raw_wide["Indicator Name"].isin(INDICATOR_MAP.keys())
    wide_subset = raw_wide.loc[keep_mask].copy()

    # Limit to year window + id columns
    year_cols = year_columns_in_range(wide_subset, YEAR_START, YEAR_END)
    id_cols = ["Country Name", "Country Code", "Indicator Name", "Indicator Code"]
    wide_subset = wide_subset[id_cols + year_cols]

    # Melt to long (Year, Value)
    long = wide_subset.melt(
        id_vars=id_cols,
        value_vars=year_cols,
        var_name="Year",
        value_name="Value"
    )
    long["Year"] = pd.to_numeric(long["Year"], errors="coerce").astype("Int64")
    long = long.dropna(subset=["Year"])
    long["Year"] = long["Year"].astype(int)

    # Pivot to panel (Country, Code, Year x indicators)
    long["Indicator Key"] = long["Indicator Name"].map(INDICATOR_MAP)
    panel = long.pivot_table(
        index=["Country Name", "Country Code", "Year"],
        columns="Indicator Key",
        values="Value",
        aggfunc="first"
    ).reset_index()

    # Ensure all expected columns exist
    for short_col in INDICATOR_MAP.values():
        if short_col not in panel.columns:
            panel[short_col] = np.nan

    # Sort and output
    panel = panel.sort_values(["Country Code", "Year"]).reset_index(drop=True)

    engine = pick_writer_engine()
    with pd.ExcelWriter(OUT_EXTRACT, engine=engine) as xw:
        wide_subset.to_excel(xw, sheet_name=SHEET_WIDE, index=False)
        panel.to_excel(xw, sheet_name=SHEET_PANEL, index=False)
        panel.to_excel(xw, sheet_name=SHEET_DATA, index=False)

    print(f"[WB Tail • Raw] Saved → {OUT_EXTRACT}")
    print(f"Years: {YEAR_START}-{YEAR_END}")
    print("Panel columns:", list(panel.columns))

if __name__ == "__main__":
    main()
