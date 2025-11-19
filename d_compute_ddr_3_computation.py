# -*- coding: utf-8 -*-
"""
DDR Step 3b — Depletion Drag Risk (RAW, NO winsorization, NO scaling)

What this does
--------------
Takes imputed primitive depletion components (% of GNI) from Step 2
and computes DDR_raw_imputed as their SUM (requires all 3 components present).
If any component is missing, DDR is set to NaN.
No fallback to WB total depletion.
No outlier handling, no inversion, no scaling.

Inputs
------
ddr_step2_imputed.xlsx (sheet: "DDR_imputed")
  Expected columns: 
    Country Name, Country Code, Year
    dep_energy_pctGNI_imputed, dep_mineral_pctGNI_imputed, dep_forest_pctGNI_imputed

Outputs
-------
ddr_step3_scores.xlsx with sheets:
  - ddr_burden_raw : Country, Code, Year, DDR_burden_pct_raw
  - ddr_full       : adds components, DDR_raw_imputed, flags, ranks
  - method_params  : SCALING=NONE, WINSOR_RULE=NONE, DIRECTION=HIGHER=WORSE
"""

import os
import numpy as np
import pandas as pd

# ====== PATHS ======
BASE = r"C:\Users\LEGION\Documents\Paper3 Intergenerational Environmental Efficiency"
OUT_DIR = os.path.join(BASE, r"Analysis_12082025")
IN_FILE = os.path.join(OUT_DIR, "ddr_step2_imputed.xlsx")
OUT_FILE = os.path.join(OUT_DIR, "ddr_step3_scores.xlsx")
IN_SHEET = "DDR_imputed"

COMP_KEYS = ["dep_energy_pctGNI_imputed", "dep_mineral_pctGNI_imputed", "dep_forest_pctGNI_imputed"]

def pick_writer_engine():
    try:
        import xlsxwriter  # noqa
        return "xlsxwriter"
    except Exception:
        return "openpyxl"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_excel(IN_FILE, sheet_name=IN_SHEET)

    # Required columns
    req = ["Country Name","Country Code","Year"] + COMP_KEYS
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns in '{IN_SHEET}': {missing}")

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")

    # ---- Compute DDR as SUM, but only if all 3 components are present ----
    df["num_components_used"] = df[COMP_KEYS].notna().sum(axis=1)
    df["DDR_raw_imputed"] = df[COMP_KEYS].sum(axis=1, skipna=False, min_count=3)

    # Raw burden = DDR (higher = worse)
    df["DDR_burden_pct_raw"] = df["DDR_raw_imputed"]

    # Diagnostics
    v = pd.to_numeric(df["DDR_burden_pct_raw"], errors="coerce")
    df["is_negative"] = (v < 0).astype("Int64")
    df["is_missing"]  = v.isna().astype("Int64")

    df["rank_within_year_pct"] = (
        df.groupby("Year")["DDR_burden_pct_raw"]
          .rank(method="average", pct=True)
          .values
    )

    # Slim output
    slim_cols = ["Country Name","Country Code","Year","DDR_burden_pct_raw"]
    ddr_slim = df[slim_cols].sort_values(["Country Code","Year"])

    # Save
    engine = pick_writer_engine()
    with pd.ExcelWriter(OUT_FILE, engine=engine) as xw:
        ddr_slim.to_excel(xw, sheet_name="ddr_burden_raw", index=False)

        keep_full = [
            "Country Name","Country Code","Year",
            *COMP_KEYS,
            "num_components_used",
            "DDR_raw_imputed","DDR_burden_pct_raw",
            "is_negative","is_missing","rank_within_year_pct"
        ]
        df.sort_values(["Country Code","Year"])[keep_full].to_excel(
            xw, sheet_name="ddr_full", index=False
        )

        meta = pd.DataFrame({
            "param": ["SCALING","WINSOR_RULE","DIRECTION","COMP_KEYS","RULE"],
            "value": [
                "NONE",
                "NONE",
                "HIGHER=WORSE (burden % of GNI)",
                ",".join(COMP_KEYS),
                "Valid only if all 3 components present (no partial DDR)"
            ]
        })
        meta.to_excel(xw, sheet_name="method_params", index=False)

    # Console summary
    n = len(df)
    n_na = int(df["is_missing"].sum())
    n_neg = int(df["is_negative"].sum())
    n_all3 = int((df["num_components_used"] == 3).sum())
    print(f"[DDR Step 3b] Saved RAW DDR burden to: {OUT_FILE}")
    print(f"Count: {n} | missing DDR: {n_na} | negatives: {n_neg} | full-components: {n_all3}")
    valid = v.dropna()
    if len(valid):
        print(f"DDR stats (raw % of GNI): min={valid.min():.4f}, p50={valid.median():.4f}, "
              f"p90={valid.quantile(0.9):.4f}, max={valid.max():.4f}")

if __name__ == "__main__":
    main()
