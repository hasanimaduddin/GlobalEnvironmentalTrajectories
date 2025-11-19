# -*- coding: utf-8 -*-
"""
RDP Step 2 — CLEAN + Imputation (interior-only + forward tail, NO head)

Inputs (from Step 1: rdp_step1_extract.xlsx / panel_raw_RDP):
    CO2_pc, Renewable_share, Water_avail_pc, Forest_area

Outputs:
    rdp_step2_imputed.xlsx with sheets:
      - RDP_imputed
      - universe_RDP
      - cleaning_summary
      - dropped_preview
      - method_params
"""

import os
import pandas as pd
import numpy as np

# ========================
# CONFIG
# ========================
BASE = r"C:\Users\LEGION\Documents\Paper3 Intergenerational Environmental Efficiency\Analysis_12082025"
IN_FILE = os.path.join(BASE, "rdp_step1_extract.xlsx")
IN_SHEET = "panel_raw_RDP"
OUT_FILE = os.path.join(BASE, "rdp_step2_imputed.xlsx")

# Toggle: drop territories (aggregates are always dropped)
EXCLUDE_TERRITORIES = True

# Indicators to impute
INDICATORS = ["CO2_pc", "Renewable_share", "Water_avail_pc", "Forest_area"]

# Aggregates/regions/income groups to ALWAYS exclude (exact strings)
AGGREGATES = {
    "Africa Eastern and Southern", "Africa Western and Central", "Arab World",
    "Caribbean small states", "Central Europe and the Baltics", "Early-demographic dividend",
    "East Asia & Pacific", "East Asia & Pacific (IDA & IBRD countries)", "East Asia & Pacific (excluding high income)",
    "Euro area", "Europe & Central Asia", "Europe & Central Asia (IDA & IBRD countries)",
    "Europe & Central Asia (excluding high income)", "European Union",
    "Fragile and conflict affected situations", "Heavily indebted poor countries (HIPC)",
    "High income", "IBRD only", "IDA & IBRD total", "IDA blend", "IDA only", "IDA total",
    "Late-demographic dividend", "Latin America & Caribbean",
    "Latin America & Caribbean (excluding high income)",
    "Latin America & the Caribbean (IDA & IBRD countries)",
    "Least developed countries: UN classification",
    "Low & middle income", "Low income", "Lower middle income",
    "Middle East, North Africa, Afghanistan & Pakistan",
    "Middle East, North Africa, Afghanistan & Pakistan (IDA & IBRD)",
    "Middle East, North Africa, Afghanistan & Pakistan (excluding high income)",
    "Middle income", "North America", "OECD members", "Other small states",
    "Pacific island small states", "Post-demographic dividend", "Pre-demographic dividend",
    "Small states", "South Asia", "South Asia (IDA & IBRD)",
    "Sub-Saharan Africa", "Sub-Saharan Africa (IDA & IBRD countries)",
    "Sub-Saharan Africa (excluding high income)", "Upper middle income", "World"
}

# Territories/special regions often excluded from clustering
TERRITORIES = {
    "American Samoa", "Anguilla", "Aruba", "Bermuda", "British Virgin Islands", "Cayman Islands",
    "China, Hong Kong SAR", "Hong Kong SAR, China", "China, Macao SAR", "Macao SAR, China",
    "Curacao", "Curaçao", "Faroe Islands", "French Polynesia", "Gibraltar", "Greenland",
    "Guam", "Isle of Man", "Montserrat", "New Caledonia", "Northern Mariana Islands",
    "Puerto Rico (US)", "Sint Maarten (Dutch part)", "St. Martin (French part)",
    "Turks and Caicos Islands", "Virgin Islands (U.S.)", "West Bank and Gaza",
    "U.R. of Tanzania: Mainland", "Channel Islands"
}

def pick_writer_engine():
    try:
        import xlsxwriter  # noqa
        return "xlsxwriter"
    except Exception:
        return "openpyxl"

def impute_series(s: pd.Series) -> pd.Series:
    """
    Interior-only linear interpolation + forward tail fill; NO head fill.
    - interior: interpolate(..., limit_area='inside') prevents head/tail fills
    - tail: ffill() extends last observation forward
    """
    s = pd.to_numeric(s, errors="coerce")
    s_lin = s.interpolate(method="linear", limit_area="inside")
    return s_lin.ffill()  # head remains NaN

def main():
    # Load
    df = pd.read_excel(IN_FILE, sheet_name=IN_SHEET)
    if "Year" not in df.columns or "Country Code" not in df.columns:
        raise RuntimeError("panel_raw_RDP must contain 'Country Code' and 'Year'.")

    # Year type & sort
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df = df.sort_values(["Country Code", "Year"]).reset_index(drop=True)

    # ---------- CLEAN: drop aggregates and (optionally) territories ----------
    n0 = len(df)
    mask_agg = df["Country Name"].isin(AGGREGATES)
    mask_terr = df["Country Name"].isin(TERRITORIES) if EXCLUDE_TERRITORIES else False
    dropped = df[mask_agg | mask_terr].copy()
    panel = df[~(mask_agg | mask_terr)].copy()
    n_after_clean = len(panel)

    # Ensure indicators exist and numeric
    for ind in INDICATORS:
        if ind not in panel.columns:
            raise RuntimeError(f"Indicator '{ind}' missing in Step 1 panel.")
        panel[ind] = pd.to_numeric(panel[ind], errors="coerce")

    # Coverage BEFORE (on cleaned universe)
    cov_before = {ind: float(panel[ind].notna().mean() * 100) for ind in INDICATORS}

    # ---------- IMPUTATION: interior-only + forward tail (no head) ----------
    for ind in INDICATORS:
        panel[f"{ind}_imputed"] = (
            panel.groupby("Country Code", group_keys=False)[ind].apply(impute_series)
        )

    # Coverage AFTER
    cov_after = {f"{ind}_imputed": float(panel[f"{ind}_imputed"].notna().mean() * 100) for ind in INDICATORS}

    # Save
    engine = pick_writer_engine()
    with pd.ExcelWriter(OUT_FILE, engine=engine) as xw:
        panel.to_excel(xw, sheet_name="RDP_imputed", index=False)

        universe = (panel[["Country Name","Country Code"]]
                    .drop_duplicates().sort_values(["Country Code"]))
        universe.to_excel(xw, sheet_name="universe_RDP", index=False)

        summary = pd.DataFrame({
            "metric": ["rows_in", "rows_after_clean", "rows_dropped", "countries_after_clean"],
            "value": [n0, n_after_clean, n0 - n_after_clean, len(universe)]
        })
        summary.to_excel(xw, sheet_name="cleaning_summary", index=False)

        dropped.head(20).to_excel(xw, sheet_name="dropped_preview", index=False)

        meta = pd.DataFrame({
            "param": [
                "STEP", "INTERP_METHOD", "INTERP_DIRECTION",
                "INDICATORS", "EXCLUDE_TERRITORIES", "AGGREGATES_N", "TERRITORIES_N",
                "NOTE"
            ],
            "value": [
                "RDP Step 2 (CLEAN + Imputation)",
                "linear (limit_area='inside') + forward-fill tail",
                "interior-only + forward tail; NO head",
                ",".join(INDICATORS), str(EXCLUDE_TERRITORIES), len(AGGREGATES), len(TERRITORIES),
                "No composite computed here; RDP will be computed in Step 3."
            ]
        })
        meta.to_excel(xw, sheet_name="method_params", index=False)

    # Console summary
    print(f"[RDP Step 2] Saved CLEAN + imputed data to: {OUT_FILE}")
    print("Sheets: RDP_imputed | universe_RDP | cleaning_summary | dropped_preview | method_params")
    print("Coverage BEFORE (%):", {k: round(v,1) for k,v in cov_before.items()})
    print("Coverage AFTER  (%):", {k: round(v,1) for k,v in cov_after.items()})

if __name__ == "__main__":
    main()
