# -*- coding: utf-8 -*-
"""
RDP Step 3b — CO2-only Headroom (clean, NO global scaling)

Composite used downstream:
    RDP_headroom_raw = H_CO2 = clip(1 - CO2_pc / CO2_SAFE, 0, 1)

Inputs
------
rdp_step2_imputed.xlsx (sheet: "RDP_imputed")
  Required columns: Country Name, Country Code, Year, CO2_pc_imputed

Outputs
-------
rdp_step3_scores.xlsx
  - rdp_headroom_raw : Country, Code, Year, RDP_headroom_raw  (only sheet used downstream)
  - method_params    : threshold / rules
"""

import os
import numpy as np
import pandas as pd

# ========================
# CONFIG
# ========================
BASE = r"C:\Users\LEGION\Documents\Paper3 Intergenerational Environmental Efficiency"
OUT_DIR = os.path.join(BASE, r"Analysis_12082025")

IN_FILE  = os.path.join(OUT_DIR, "rdp_step2_imputed.xlsx")
IN_SHEET = "RDP_imputed"
OUT_FILE = os.path.join(OUT_DIR, "rdp_step3_scores.xlsx")

# Guardrail (tCO2 per capita)
CO2_SAFE = 1.9

def pick_writer_engine():
    try:
        import xlsxwriter  # noqa: F401
        return "xlsxwriter"
    except Exception:
        return "openpyxl"

def headroom_co2(co2_pc: pd.Series) -> pd.Series:
    s = pd.to_numeric(co2_pc, errors="coerce")
    return (1.0 - s / CO2_SAFE).clip(lower=0.0, upper=1.0)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load minimal inputs
    df = pd.read_excel(IN_FILE, sheet_name=IN_SHEET, dtype={"Country Code": str})
    need = ["Country Name", "Country Code", "Year", "CO2_pc_imputed"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns in {IN_SHEET}: {missing}")

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")

    # Compute CO2 headroom and set as composite
    df["RDP_headroom_raw"] = headroom_co2(df["CO2_pc_imputed"])

    # Slim sheet ONLY (to avoid downstream confusion)
    slim = df[["Country Name", "Country Code", "Year", "RDP_headroom_raw"]].sort_values(
        ["Country Code", "Year"]
    )

    # Save
    engine = pick_writer_engine()
    with pd.ExcelWriter(OUT_FILE, engine=engine) as xw:
        slim.to_excel(xw, sheet_name="rdp_headroom_raw", index=False)

        pd.DataFrame({
            "param": ["CO2_SAFE", "SCALING", "COMPOSITE_RULE", "OUTPUT_COLUMNS"],
            "value": [CO2_SAFE, "NONE", "CO2_ONLY",
                      "Country Name, Country Code, Year, RDP_headroom_raw"]
        }).to_excel(xw, sheet_name="method_params", index=False)

    # Console summary
    n_total = len(slim)
    n_valid = int(slim["RDP_headroom_raw"].notna().sum())
    print(f"[RDP Step 3b • CO2-only] Saved → {OUT_FILE}")
    print("Sheets: rdp_headroom_raw | method_params")
    print(f"Coverage (rows): valid={n_valid} / total={n_total}")

if __name__ == "__main__":
    main()
