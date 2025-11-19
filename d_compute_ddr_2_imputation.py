# -*- coding: utf-8 -*-
"""
DDR Step 2 — CLEAN + Imputation ONLY (no composite / no final score)

What this does
--------------
1) Drops aggregates/regions (always) and territories/special regions (toggle).
2) For each component, performs interior-only linear interpolation and forward tail fill (NO head imputation).
3) Outputs *_imputed columns; no DDR composite here.

Outputs
-------
ddr_step2_imputed.xlsx with sheets:
  - DDR_imputed         : IDs + raw component columns + *_imputed counterparts
  - universe_DDR        : country list (post-clean)
  - cleaning_summary    : rows in/after/dropped
  - dropped_preview     : sample of dropped rows
  - method_params       : config details
"""

import os
import pandas as pd
import numpy as np

# ========================
# CONFIG
# ========================
BASE = r"C:\Users\LEGION\Documents\Paper3 Intergenerational Environmental Efficiency\Analysis_12082025"
IN_FILE = os.path.join(BASE, "ddr_step1_extract.xlsx")
OUT_FILE = os.path.join(BASE, "ddr_step2_imputed.xlsx")
PANEL_SHEET = "panel_raw_DDR"

# Toggle: drop territories (aggregates are always dropped)
EXCLUDE_TERRITORIES = True

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

# Required primitive components
COMP_KEYS = ["dep_energy_pctGNI", "dep_mineral_pctGNI", "dep_forest_pctGNI"]
# Optional WB total (diagnostics only; not used downstream)
OPT_TOTAL = "dep_natural_total_pctGNI"

def pick_writer_engine():
    try:
        import xlsxwriter  # noqa
        return "xlsxwriter"
    except Exception:
        return "openpyxl"

def impute_series(s: pd.Series) -> pd.Series:
    """
    Interior-only linear interpolation + forward tail fill; NO head fill.
    - interior: pandas interpolate with limit_area='inside' (never touches head/tail)
    - tail: forward-fill (flat extrapolation)
    """
    s = pd.to_numeric(s, errors="coerce")
    s_lin = s.interpolate(method="linear", limit_area="inside")
    return s_lin.ffill()  # head stays NaN; only forward tail filled

def main():
    # Load panel
    df = pd.read_excel(IN_FILE, sheet_name=PANEL_SHEET)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df.sort_values(["Country Code", "Year"], inplace=True)

    # ---------- CLEANING: drop aggregates and (optionally) territories ----------
    n0 = len(df)
    mask_agg = df["Country Name"].isin(AGGREGATES)
    mask_terr = df["Country Name"].isin(TERRITORIES) if EXCLUDE_TERRITORIES else False
    dropped = df[mask_agg | mask_terr].copy()
    panel = df[~(mask_agg | mask_terr)].copy()
    n_after_clean = len(panel)

    # Ensure numeric types for components
    to_numeric_cols = [c for c in COMP_KEYS if c in panel.columns]
    if OPT_TOTAL in panel.columns:
        to_numeric_cols.append(OPT_TOTAL)
    for col in to_numeric_cols:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")

    # Coverage BEFORE (on the cleaned universe)
    cov_before = {
        k: float(panel[k].notna().mean() * 100) if k in panel.columns else np.nan
        for k in COMP_KEYS
    }

    # ---------- IMPUTATION (by country code) ----------
    for k in COMP_KEYS:
        if k not in panel.columns:
            raise RuntimeError(f"Missing required component column '{k}' in {PANEL_SHEET}.")
        panel[f"{k}_imputed"] = panel.groupby("Country Code", group_keys=False)[k].apply(impute_series)

    if OPT_TOTAL in panel.columns:
        panel[f"{OPT_TOTAL}_imputed"] = panel.groupby("Country Code", group_keys=False)[OPT_TOTAL].apply(impute_series)

    # Coverage AFTER
    cov_after = {
        f"{k}_imputed": float(panel[f"{k}_imputed"].notna().mean() * 100)
        for k in COMP_KEYS
    }

    # ---------- SAVE ----------
    id_cols = ["Country Name", "Country Code", "Year"]
    keep_cols = id_cols + COMP_KEYS + [f"{k}_imputed" for k in COMP_KEYS]
    if OPT_TOTAL in panel.columns:
        keep_cols += [OPT_TOTAL, f"{OPT_TOTAL}_imputed"]

    engine = pick_writer_engine()
    with pd.ExcelWriter(OUT_FILE, engine=engine) as xw:
        panel[keep_cols].to_excel(xw, sheet_name="DDR_imputed", index=False)

        # Universe & cleaning summary
        universe = (panel[id_cols[:2]].drop_duplicates().sort_values(["Country Code"]))
        universe.to_excel(xw, sheet_name="universe_DDR", index=False)

        summary = pd.DataFrame({
            "metric": ["rows_in", "rows_after_clean", "rows_dropped",
                       "countries_after_clean"],
            "value": [n0, n_after_clean, n0 - n_after_clean, len(universe)]
        })
        summary.to_excel(xw, sheet_name="cleaning_summary", index=False)

        dropped.head(20).to_excel(xw, sheet_name="dropped_preview", index=False)

        # Method params
        meta = pd.DataFrame({
            "param": [
                "STEP", "INTERP_METHOD", "INTERP_DIRECTION",
                "COMPONENT_COLUMNS", "OUTPUT_COLUMNS",
                "EXCLUDE_TERRITORIES", "AGGREGATES_N", "TERRITORIES_N",
                "NOTE"
            ],
            "value": [
                "DDR Step 2 (CLEAN + Imputation ONLY)",
                "linear (limit_area='inside') + forward-fill tail",
                "interior-only + forward tail; NO head",
                ",".join(COMP_KEYS),
                ",".join([f"{k}_imputed" for k in COMP_KEYS]),
                str(EXCLUDE_TERRITORIES), len(AGGREGATES), len(TERRITORIES),
                "No composite computed here; DDR will be computed in Step 3 from imputed components."
            ]
        })
        meta.to_excel(xw, sheet_name="method_params", index=False)

    # Console
    print(f"[DDR Step 2] Saved to: {OUT_FILE}")
    print("Sheets: DDR_imputed | universe_DDR | cleaning_summary | dropped_preview | method_params")
    print("Coverage BEFORE (%):", {k: round(v, 1) for k, v in cov_before.items()})
    print("Coverage AFTER  (%):", {k: round(v, 1) for k, v in cov_after.items()})
    print(f"Dropped rows (aggregates/territories): {n0 - n_after_clean}")

if __name__ == "__main__":
    main()
