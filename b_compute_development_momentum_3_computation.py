# -*- coding: utf-8 -*-
"""
DM Step 3b — Development Momentum (RAW, no scaling)

Inputs (from Step 2):
  Analysis_12082025/dm_step2_imputed.xlsx  (sheet: 'prepped_panel_DM')

Required columns in that sheet:
  - gdp_pc_pwt_2017ppp   (USD2017 per person, from PWT rgdpo / WB POP)
  - k_pc_pwt_2017ppp     (USD2017 per person, from PWT cn / WB POP)
  - l_share_pwt          (employment share = PWT emp / WB POP)
  - POP                  (persons, WB)
  - E_total_kgoe         (kgoe, built upstream as E_pc * POP)

Outputs:
  Analysis_12082025/dm_step3_scores.xlsx
    - dm_momentum_raw  (Country, Code, Year, DM_smooth)
    - dm_full          (diagnostics: gY,gK,gL,gE,gTFP, plus totals used)
    - method_params
"""

import os
import numpy as np
import pandas as pd

# ====== PATHS ======
BASE = r"C:\Users\LEGION\Documents\Paper3 Intergenerational Environmental Efficiency"
OUT_DIR = os.path.join(BASE, r"Analysis_12082025")
IN_FILE = os.path.join(OUT_DIR, "dm_step2_imputed.xlsx")
OUT_FILE = os.path.join(OUT_DIR, "dm_step3_scores.xlsx")

# ====== PARAMETERS (fixed for comparability) ======
ALPHA_K = 0.35
GAMMA_E = 0.05
BETA_L  = 1.0 - ALPHA_K - GAMMA_E  # = 0.60

LAMBDA      = 0.70   # weight on current gTFP
RHO_DEFAULT = 0.50   # persistence on lagged gTFP

# Column names expected from Step 2
GDP_PC_COL   = "gdp_pc_pwt_2017ppp"
K_PC_COL     = "k_pc_pwt_2017ppp"
L_SHARE_COL  = "l_share_pwt"
POP_COL      = "POP"
E_TOTAL_COL  = "E_total_kgoe"   # already = E_pc * POP upstream

def pick_writer_engine():
    try:
        import xlsxwriter  # noqa
        return "xlsxwriter"
    except Exception:
        return "openpyxl"

def log_growth(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    s = s.replace({0.0: np.nan})
    ln = np.log(s)
    return ln.diff(1)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_excel(IN_FILE, sheet_name="prepped_panel_DM")

    need = ["Country Name","Country Code","Year",
            GDP_PC_COL, K_PC_COL, L_SHARE_COL, POP_COL, E_TOTAL_COL]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns in prepped_panel_DM: {missing}")

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")

    rows = []
    for (cn, cc), g in df.groupby(["Country Name","Country Code"], sort=False):
        g = g.sort_values("Year").copy()

        # ---- Build TOTALS from per-capita / share using WB POP ----
        POP = pd.to_numeric(g[POP_COL], errors="coerce")

        Y_tot = pd.to_numeric(g[GDP_PC_COL], errors="coerce") * POP                  # USD2017
        K_tot = pd.to_numeric(g[K_PC_COL],   errors="coerce") * POP                  # USD2017
        L_tot = pd.to_numeric(g[L_SHARE_COL],errors="coerce") * POP                  # persons
        E_tot = pd.to_numeric(g[E_TOTAL_COL],errors="coerce")                        # kgoe (already total)

        # ---- Log growth rates ----
        gY = log_growth(Y_tot)
        gK = log_growth(K_tot)
        gL = log_growth(L_tot)
        gE = log_growth(E_tot)

        # ---- Solow residual (CRS, fixed elasticities) ----
        gTFP = gY - ALPHA_K*gK - BETA_L*gL - GAMMA_E*gE

        # ---- Momentum smoother ----
        DM_smooth = LAMBDA*gTFP + (1.0 - LAMBDA)*RHO_DEFAULT*gTFP.shift(1)

        out = pd.DataFrame({
            "Country Name": cn,
            "Country Code": cc,
            "Year": g["Year"].values,
            "Y_total_2017ppp": Y_tot.values,
            "K_total_2017ppp": K_tot.values,
            "L_total_persons": L_tot.values,
            "E_total_kgoe":    E_tot.values,
            "gY": gY.values, "gK": gK.values, "gL": gL.values, "gE": gE.values,
            "gTFP": gTFP.values,
            "DM_smooth": DM_smooth.values
        })
        rows.append(out)

    dm = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["Country Name","Country Code","Year","DM_smooth"]
    )

    # Diagnostic: within-year percentile rank (optional)
    if not dm.empty:
        dm["rank_within_year_pct"] = dm.groupby("Year")["DM_smooth"].rank(method="average", pct=True).values

    # ---- Save ----
    engine = pick_writer_engine()
    with pd.ExcelWriter(OUT_FILE, engine=engine) as xw:
        dm[["Country Name","Country Code","Year","DM_smooth"]].sort_values(
            ["Country Code","Year"]
        ).to_excel(xw, sheet_name="dm_momentum_raw", index=False)

        dm.sort_values(["Country Code","Year"]).to_excel(xw, sheet_name="dm_full", index=False)

        pd.DataFrame({
            "param": ["ALPHA_K","BETA_L","GAMMA_E","LAMBDA","RHO_DEFAULT",
                      "Y_def","K_def","L_def","E_def","SCALING"],
            "value": [ALPHA_K, BETA_L, GAMMA_E, LAMBDA, RHO_DEFAULT,
                      "Y_total = gdp_pc_pwt_2017ppp * POP",
                      "K_total = k_pc_pwt_2017ppp * POP",
                      "L_total = l_share_pwt * POP",
                      "E_total = E_total_kgoe",
                      "NONE"]
        }).to_excel(xw, sheet_name="method_params", index=False)

    print(f"[DM Step 3b] Saved raw DM to: {OUT_FILE}")
    print("Sheets: dm_momentum_raw | dm_full | method_params (SCALING=NONE)")
    if not dm.empty:
        print(f"Non-missing counts → gY:{dm['gY'].notna().sum()}  gK:{dm['gK'].notna().sum()}  "
              f"gL:{dm['gL'].notna().sum()}  gE:{dm['gE'].notna().sum()}  gTFP:{dm['gTFP'].notna().sum()}")

if __name__ == "__main__":
    main()
