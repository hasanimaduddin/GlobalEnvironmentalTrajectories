# -*- coding: utf-8 -*-
"""
Step 1 — Extract PC inputs (PWT v10.x) and standardize per-capita using WB population

Reads:
  PWT  : C:/Users/LEGION/Documents/Paper3 Intergenerational Environmental Efficiency/Analysis_12082025/pwt1001.xlsx (sheet 'Data')
  WB   : C:/Users/LEGION/Documents/Paper3 Intergenerational Environmental Efficiency/WBRAW.xlsx (sheet 'Data')

Writes:
  C:/Users/LEGION/Documents/Paper3 Intergenerational Environmental Efficiency/Analysis_12082025/pc_step1_extract.xlsx

Sheets:
  - original_raw_subset_pwt   (selected PWT columns as-is)
  - original_raw_subset_wb    (WB population subset as-is)
  - panel_raw                 (Country/Code/Year + per-capita using WB POP)
  - panel_data                (same as panel_raw for downstream compatibility)

Notes:
  • PWT series (rgdpo, cn, rnna, emp) are in *millions*. WB POP is in *persons*.
    Hence we multiply PWT numerators by 1e6 before dividing by WB POP.
  • No fallback: if WB population is missing for a country–year, per-capita = NaN.
"""

import os
import numpy as np
import pandas as pd

# ========================
# CONFIG
# ========================
BASE = r"C:\Users\LEGION\Documents\Paper3 Intergenerational Environmental Efficiency"
OUT_DIR = os.path.join(BASE, r"Analysis_12082025")

# PWT
PWT_EXCEL = os.path.join(OUT_DIR, "pwt1001.xlsx")
PWT_SHEET = "Data"

# WB (for Population, total)
WB_EXCEL = os.path.join(BASE, "WBRAW.xlsx")
WB_SHEET = "Data"
WB_POP_CODES = ["SP.POP.TOTL"]
WB_POP_NAMES = ["Population, total"]  # keep both code and name matching

OUT_EXTRACT = os.path.join(OUT_DIR, "pc_step1_extract.xlsx")

MILLION = 1_000_000.0

def pick_writer_engine():
    try:
        import xlsxwriter  # noqa
        return "xlsxwriter"
    except Exception:
        return "openpyxl"

def year_columns(df: pd.DataFrame):
    years = []
    for c in df.columns:
        try:
            y = int(c)
            if 1900 <= y <= 2100:
                years.append(y)
        except Exception:
            pass
    return sorted(years)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- Load PWT ----
    if not os.path.exists(PWT_EXCEL):
        raise FileNotFoundError(f"PWT file not found:\n{PWT_EXCEL}")

    pwt = pd.read_excel(PWT_EXCEL, sheet_name=PWT_SHEET)
    pwt.columns = [c.strip() for c in pwt.columns]

    # Keep only columns we need (if present)
    want = [
        "countrycode", "country", "currency_unit", "year",
        "rgdpo", "rgdpe", "pop", "emp", "cn", "rnna", "hc",
        "ctfp", "rtfpna"
    ]
    have = [c for c in want if c in pwt.columns]
    pwt_sub = pwt[have].copy()

    # Coerce numeric
    for c in ["year", "rgdpo", "rgdpe", "pop", "emp", "cn", "rnna", "hc", "ctfp", "rtfpna"]:
        if c in pwt_sub.columns:
            pwt_sub[c] = pd.to_numeric(pwt_sub[c], errors="coerce")

    # ---- Load WB Population (wide -> long -> tidy) ----
    wb = pd.read_excel(WB_EXCEL, sheet_name=WB_SHEET, dtype={"Country Code": str})
    wb_cols = [c.strip() for c in wb.columns]
    wb.columns = wb_cols
    years = [str(y) for y in year_columns(wb)]

    mask_pop = False
    if "Indicator Code" in wb.columns:
        mask_pop = wb["Indicator Code"].isin(WB_POP_CODES)
    if "Indicator Name" in wb.columns:
        mask_pop = mask_pop | wb["Indicator Name"].isin(WB_POP_NAMES) if isinstance(mask_pop, pd.Series) else wb["Indicator Name"].isin(WB_POP_NAMES)

    wb_pop = wb.loc[mask_pop].copy()
    if wb_pop.empty:
        raise RuntimeError("WB population ('Population, total') not found in WBRAW.xlsx.")

    id_vars = ["Country Name", "Country Code", "Indicator Name", "Indicator Code"]
    wb_long = wb_pop.melt(id_vars=id_vars, value_vars=years, var_name="Year", value_name="WB_POP").copy()
    wb_long["Year"] = pd.to_numeric(wb_long["Year"], errors="coerce").astype("Int64")
    wb_long["WB_POP"] = pd.to_numeric(wb_long["WB_POP"], errors="coerce")

    wb_pop_tidy = wb_long[["Country Name", "Country Code", "Year", "WB_POP"]].copy()

    # ---- Prepare PWT tidy ----
    pwt_tidy = pwt_sub.rename(columns={
        "country": "Country Name",
        "countrycode": "Country Code",
        "year": "Year"
    }).copy()
    pwt_tidy["Year"] = pwt_tidy["Year"].astype("Int64")

    # ---- Merge PWT with WB POP (outer left on PWT to keep PWT coverage; you may use inner if you prefer strict overlap) ----
    df = pwt_tidy.merge(
        wb_pop_tidy[["Country Code", "Year", "WB_POP"]],
        on=["Country Code", "Year"],
        how="left"
    )

    # ---- Build per-capita using WB population (persons) ----
    # PWT numerators are in millions → multiply by 1e6 before dividing by persons
    def per_capita_million(numer_million, wb_pop_series):
        num = pd.to_numeric(numer_million, errors="coerce") * MILLION
        den = pd.to_numeric(wb_pop_series, errors="coerce")
        return np.where(den > 0, num / den, np.nan)

    gdp_pc_2017ppp = per_capita_million(df.get("rgdpo"), df.get("WB_POP"))
    k_pc_ppp       = per_capita_million(df.get("cn"),   df.get("WB_POP"))
    k_pc_const     = per_capita_million(df.get("rnna"), df.get("WB_POP"))
    # Employment per capita (share): emp (millions of persons) / population (persons)
    l_pc           = per_capita_million(df.get("emp"),  df.get("WB_POP"))  # share in [0,1] if consistent
    h_idx          = pd.to_numeric(df.get("hc"), errors="coerce") if "hc" in df.columns else np.nan

    panel = pd.DataFrame({
        "Country Name": df["Country Name"],
        "Country Code": df["Country Code"],
        "Year": df["Year"],
        # Per-capita/index variables (WB population denominator)
        "gdp_pc_2017ppp": gdp_pc_2017ppp,   # USD 2017 per person
        "k_pc_ppp": k_pc_ppp,               # USD 2017 per person
        "k_pc_const": k_pc_const,           # (constant-price) USD per person
        "l_pc": l_pc,                       # employment share
        "h_idx": h_idx,                     # PWT human capital index
        # Diagnostics / raw levels
        "WB_POP_persons": df["WB_POP"],
        "PWT_rgdpo_mil_2017USD": df.get("rgdpo"),
        "PWT_cn_mil_2017USD": df.get("cn"),
        "PWT_rnna_mil_2017USD": df.get("rnna"),
        "PWT_emp_millions": df.get("emp"),
        "PWT_pop_millions": df.get("pop"),
        "PWT_hc_index": df.get("hc"),
        "PWT_ctfp_USA1": df.get("ctfp"),
        "PWT_rtfpna_2017eq1": df.get("rtfpna"),
        "PWT_currency_unit": df.get("currency_unit")
    }).sort_values(["Country Code", "Year"]).reset_index(drop=True)

    # ---- Save ----
    engine = pick_writer_engine()
    with pd.ExcelWriter(OUT_EXTRACT, engine=engine) as xw:
        pwt_sub.to_excel(xw, sheet_name="original_raw_subset_pwt", index=False)
        wb_pop.to_excel(xw, sheet_name="original_raw_subset_wb", index=False)
        panel.to_excel(xw, sheet_name="panel_raw", index=False)
        panel.to_excel(xw, sheet_name="panel_data", index=False)

    print(f"[Step 1 • PC] Saved extract → {OUT_EXTRACT}")
    print("Sheets: original_raw_subset_pwt | original_raw_subset_wb | panel_raw | panel_data")
    print("Panel columns:", list(panel.columns))

if __name__ == "__main__":
    main()
