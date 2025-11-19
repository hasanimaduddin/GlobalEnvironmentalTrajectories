# -*- coding: utf-8 -*-
"""
DM Step 1 — Extract bridged inputs:
PWT (1968–2018): GDP per cap, Capital per cap, Labor input per cap
WB  (1968–2024): Energy per cap, Population (total & ages 0–14)

Reads
-----
PWT : <BASE>/Analysis_12082025/pwt1001.xlsx (sheet 'Data')
WB  : <BASE>/WBRAW.xlsx (sheet 'Data')

Writes
------
<BASE>/Analysis_12082025/dm_step1_extract.xlsx
  - original_raw_subset_pwt
  - original_raw_subset_wb
  - panel_raw_DM
  - coverage_summary

Notes
-----
• PWT aggregates are in *millions*. WB POP is in persons.
  Per-capita = (PWT series × 1e6) / WB population.
• L_share_pwt = EMP (millions) / POP (persons). Clamped to [0, 1].
"""

import os
import numpy as np
import pandas as pd

# ========================
# CONFIG
# ========================
BASE = r"C:\Users\LEGION\Documents\Paper3 Intergenerational Environmental Efficiency"
OUT_DIR = os.path.join(BASE, r"Analysis_12082025")

# PWT (same file you used for PC step 1)
PWT_EXCEL = os.path.join(OUT_DIR, "pwt1001.xlsx")
PWT_SHEET = "Data"

# WB (energy + population)
WB_EXCEL = os.path.join(BASE, "WBRAW.xlsx")
WB_SHEET = "Data"

# Exact WB indicator names to keep
WB_KEEP_NAMES = [
    "Energy use (kg of oil equivalent per capita)",  # E_pc
    "Population, total",                             # POP
    "Population ages 0-14, total",                   # POP_0_14
]

OUT_EXTRACT = os.path.join(OUT_DIR, "dm_step1_extract.xlsx")
MILLION = 1_000_000.0

# ---------------- helpers ----------------
def pick_writer_engine():
    try:
        import xlsxwriter  # noqa
        return "xlsxwriter"
    except Exception:
        return "openpyxl"

def year_columns(df: pd.DataFrame):
    yrs = []
    for c in df.columns:
        try:
            y = int(c)
            if 1900 <= y <= 2100:
                yrs.append(y)
        except Exception:
            pass
    return sorted(yrs)

def per_capita_from_millions(numer_millions, pop_persons):
    num = pd.to_numeric(numer_millions, errors="coerce") * MILLION
    den = pd.to_numeric(pop_persons,     errors="coerce")
    return np.where(den > 0, num / den, np.nan)

# ---------------- main ----------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---------- Load PWT ----------
    if not os.path.exists(PWT_EXCEL):
        raise FileNotFoundError(f"PWT file not found:\n{PWT_EXCEL}")
    pwt = pd.read_excel(PWT_EXCEL, sheet_name=PWT_SHEET)
    pwt.columns = [c.strip() for c in pwt.columns]

    want_pwt = [
        "country", "countrycode", "year",
        "rgdpo",   # real GDP (output-side), constant 2017 PPP, millions
        "cn",      # capital stock at current PPP, millions
        "emp",     # persons employed, millions
        "pop",     # PWT population, millions (diagnostic only)
    ]
    have = [c for c in want_pwt if c in pwt.columns]
    pwt_sub = pwt[have].copy()

    # tidy PWT
    pwt_sub = pwt_sub.rename(columns={
        "country": "Country Name",
        "countrycode": "Country Code",
        "year": "Year"
    })
    for c in ["Year", "rgdpo", "cn", "emp", "pop"]:
        if c in pwt_sub.columns:
            pwt_sub[c] = pd.to_numeric(pwt_sub[c], errors="coerce")
    pwt_sub["Year"] = pwt_sub["Year"].astype("Int64")

    # ---------- Load WB (Energy + POP) ----------
    wb = pd.read_excel(WB_EXCEL, sheet_name=WB_SHEET, dtype={"Country Code": str})
    wb.columns = [c.strip() for c in wb.columns]
    years = [str(y) for y in year_columns(wb)]

    keep = wb["Indicator Name"].isin(WB_KEEP_NAMES)
    wb_keep = wb.loc[keep].copy()
    if wb_keep.empty:
        raise RuntimeError("Required WB indicators not found. Expected at least: "
                           + ", ".join(WB_KEEP_NAMES))

    id_cols = ["Country Name", "Country Code", "Indicator Name", "Indicator Code"]
    wb_long = wb_keep.melt(id_vars=id_cols, value_vars=years,
                           var_name="Year", value_name="Value").copy()
    wb_long["Year"] = pd.to_numeric(wb_long["Year"], errors="coerce").astype("Int64")
    wb_long["Value"] = pd.to_numeric(wb_long["Value"], errors="coerce")

    wb_wide = (
        wb_long.pivot_table(index=["Country Name", "Country Code", "Year"],
                            columns="Indicator Name", values="Value", aggfunc="first")
               .reset_index()
    )
    wb_wide = wb_wide.rename(columns={
        "Energy use (kg of oil equivalent per capita)": "E_pc",
        "Population, total": "POP",
        "Population ages 0-14, total": "POP_0_14",
    })

    # ---------- Merge & build bridged per-capita inputs ----------
    df = pwt_sub.merge(
        wb_wide[["Country Code","Year","POP","POP_0_14","E_pc"]],
        on=["Country Code","Year"], how="left"
    )

    # per-capita using **WB** population (persons)
    df["gdp_pc_pwt_2017ppp"] = per_capita_from_millions(df.get("rgdpo"), df.get("POP"))
    df["k_pc_pwt_2017ppp"]   = per_capita_from_millions(df.get("cn"),   df.get("POP"))

    # labor share: EMP (millions) / POP (persons) → clamp to [0,1]
    l_share                  = per_capita_from_millions(df.get("emp"),  df.get("POP"))
    df["l_share_pwt"]        = np.clip(l_share, 0.0, 1.0)

    # working-age population (15+)
    df["POP_15p"] = pd.to_numeric(df["POP"], errors="coerce") - \
                    pd.to_numeric(df["POP_0_14"], errors="coerce")
    df.loc[df["POP_15p"] < 0, "POP_15p"] = np.nan  # guard

    # ---------- Final tidy panel ----------
    panel = df[[
        "Country Name","Country Code","Year",
        "gdp_pc_pwt_2017ppp",     # Y (per capita)
        "k_pc_pwt_2017ppp",       # K (per capita)
        "l_share_pwt",            # L (% of population employed)
        "E_pc",                   # Energy per capita (kgoe/person)
        "POP","POP_0_14","POP_15p",
        "rgdpo","cn","emp","pop"  # diagnostics (PWT, millions)
    ]].sort_values(["Country Code","Year"]).reset_index(drop=True)

    # ---------- Coverage summary (quick counts of non-missing by series) ----------
    cov = (panel[["gdp_pc_pwt_2017ppp","k_pc_pwt_2017ppp","l_share_pwt","E_pc"]]
           .notna().sum().rename("NonMissing_rows").to_frame())
    cov["Total_rows"] = len(panel)
    cov["Coverage_%"] = 100 * cov["NonMissing_rows"] / cov["Total_rows"]
    cov = cov.reset_index().rename(columns={"index":"Series"})

    # ---------- Save ----------
    engine = pick_writer_engine()
    with pd.ExcelWriter(OUT_EXTRACT, engine=engine) as xw:
        pwt_sub.to_excel(xw, sheet_name="original_raw_subset_pwt", index=False)
        wb_keep.to_excel(xw, sheet_name="original_raw_subset_wb", index=False)
        panel.to_excel(xw, sheet_name="panel_raw_DM", index=False)
        cov.to_excel(xw, sheet_name="coverage_summary", index=False)

    print(f"[DM Step 1] Saved extract → {OUT_EXTRACT}")
    print("Sheets: original_raw_subset_pwt | original_raw_subset_wb | panel_raw_DM | coverage_summary")
    print("Panel columns:", list(panel.columns))

if __name__ == "__main__":
    main()
