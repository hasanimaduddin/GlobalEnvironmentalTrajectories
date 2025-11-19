# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 20:02:54 2025

@author: LEGION
"""

# -*- coding: utf-8 -*-
"""
DM Step 3b — Development Momentum (EXTENDED, no scaling)

Adds to the original spec:
  • Multi-lag growth computation (Δ=1 and Δ=3) to reduce noise
  • Clean initialization of the smoother at the first valid year
  • Optional tail carry-forward of DM_smooth (off by default)
  • Coverage diagnostics (by year) for each DM variant

Inputs (from Step 2 • PWT+WB):
  Analysis_12082025/dm_step2_imputed.xlsx  (sheet: 'prepped_panel_DM')
  Required columns:
    - gdp_pc_pwt_2017ppp   (USD2017 per person)
    - k_pc_pwt_2017ppp     (USD2017 per person)
    - l_share_pwt          (employment share = emp/POP)
    - POP                  (persons)
    - E_total_kgoe         (kgoe total)

Outputs:
  Analysis_12082025/dm_step3_scores_extended.xlsx
    - dm_momentum_raw_ext  (Country, Code, Year, DM_smooth)  [primary = Δ=1]
    - dm_full_ext          (diagnostics incl. Δ=1 and Δ=3 variants)
    - coverage_by_year     (non-missing counts by year)
    - method_params
"""

import os
import numpy as np
import pandas as pd

# ====== PATHS ======
BASE = r"C:\Users\LEGION\Documents\Paper3 Intergenerational Environmental Efficiency"
OUT_DIR = os.path.join(BASE, r"Analysis_12082025")
IN_FILE = os.path.join(OUT_DIR, "dm_step2_imputed.xlsx")
OUT_FILE = os.path.join(OUT_DIR, "dm_step3_scores_extended.xlsx")

IN_SHEET = "prepped_panel_DM"

# ====== PARAMETERS (fixed for comparability) ======
ALPHA_K = 0.35
GAMMA_E = 0.05
BETA_L  = 1.0 - ALPHA_K - GAMMA_E  # = 0.60

LAMBDA      = 0.70   # weight on current gTFP
RHO_DEFAULT = 0.50   # persistence on lagged gTFP

# Extended toggles
GROWTH_LAGS             = [1, 3]   # compute Δ=1 (primary) and Δ=3 (alternative)
INIT_DM_WITH_CURRENT    = True     # fill first valid DM_t with current gTFP_t
CARRY_FORWARD_TAIL_DM   = False    # if True, ffill DM at trailing NaNs (rare if Step 2 did tail impute)

# Step-2 column names
GDP_PC_COL   = "gdp_pc_pwt_2017ppp"
K_PC_COL     = "k_pc_pwt_2017ppp"
L_SHARE_COL  = "l_share_pwt"
POP_COL      = "POP"
E_TOTAL_COL  = "E_total_kgoe"

def pick_writer_engine():
    try:
        import xlsxwriter  # noqa
        return "xlsxwriter"
    except Exception:
        return "openpyxl"

def log_growth_lag(series: pd.Series, lag: int) -> pd.Series:
    """g_t = ln(X_t) - ln(X_{t-lag}); requires positive levels."""
    s = pd.to_numeric(series, errors="coerce")
    s = s.replace({0.0: np.nan})
    ln = np.log(s)
    return ln - ln.shift(lag)

def momentum_filter(g_tfp: pd.Series, lam: float, rho: float, init_with_current: bool) -> pd.Series:
    """DM_t = λ*g_tfp_t + (1-λ)*ρ*g_tfp_{t-1}, with optional clean init at first valid t."""
    dm = lam * g_tfp + (1.0 - lam) * rho * g_tfp.shift(1)
    if init_with_current:
        # If the first valid gTFP has NaN DM (no lag), set DM to current gTFP at that t
        first_valid_idx = g_tfp.first_valid_index()
        if first_valid_idx is not None and pd.isna(dm.loc[first_valid_idx]):
            dm.loc[first_valid_idx] = g_tfp.loc[first_valid_idx]
    return dm

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_excel(IN_FILE, sheet_name=IN_SHEET)

    need = ["Country Name","Country Code","Year",
            GDP_PC_COL, K_PC_COL, L_SHARE_COL, POP_COL, E_TOTAL_COL]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns in '{IN_SHEET}': {missing}")

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")

    all_rows = []
    for (cn, cc), g in df.groupby(["Country Name","Country Code"], sort=False):
        g = g.sort_values("Year").copy()

        POP = pd.to_numeric(g[POP_COL], errors="coerce")

        # Build TOT levels
        Y_tot = pd.to_numeric(g[GDP_PC_COL], errors="coerce") * POP
        K_tot = pd.to_numeric(g[K_PC_COL],   errors="coerce") * POP
        L_tot = pd.to_numeric(g[L_SHARE_COL],errors="coerce") * POP
        E_tot = pd.to_numeric(g[E_TOTAL_COL],errors="coerce")

        out = pd.DataFrame({
            "Country Name": cn,
            "Country Code": cc,
            "Year": g["Year"].values,
            "Y_total_2017ppp": Y_tot.values,
            "K_total_2017ppp": K_tot.values,
            "L_total_persons": L_tot.values,
            "E_total_kgoe":    E_tot.values
        })

        # For each lag (Δ=1 primary, Δ=3 alt), compute gY,gK,gL,gE → gTFP → DM
        for LAG in GROWTH_LAGS:
            gY = log_growth_lag(Y_tot, LAG)
            gK = log_growth_lag(K_tot, LAG)
            gL = log_growth_lag(L_tot, LAG)
            gE = log_growth_lag(E_tot, LAG)

            # Require all four present at t to compute gTFP
            valid_mask = gY.notna() & gK.notna() & gL.notna() & gE.notna()
            gTFP = pd.Series(np.nan, index=gY.index, dtype=float)
            gTFP.loc[valid_mask] = gY.loc[valid_mask] - ALPHA_K*gK.loc[valid_mask] - BETA_L*gL.loc[valid_mask] - GAMMA_E*gE.loc[valid_mask]

            DM = momentum_filter(gTFP, LAMBDA, RHO_DEFAULT, INIT_DM_WITH_CURRENT)

            if CARRY_FORWARD_TAIL_DM:
                DM = DM.ffill()

            # attach with suffix by lag
            suf = f"_L{LAG}"
            out[f"gY{suf}"]   = gY.values
            out[f"gK{suf}"]   = gK.values
            out[f"gL{suf}"]   = gL.values
            out[f"gE{suf}"]   = gE.values
            out[f"gTFP{suf}"] = gTFP.values
            out[f"DM_smooth{suf}"] = DM.values

        all_rows.append(out)

    dm = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame(
        columns=["Country Name","Country Code","Year","DM_smooth_L1"]
    )

    # Primary export = Δ=1 variant
    dm["DM_smooth"] = dm.get("DM_smooth_L1", np.nan)

    # Coverage by year
    cov = (dm.groupby("Year")[["DM_smooth_L1","DM_smooth_L3"]]
             .apply(lambda x: x.notna().sum())
             .reset_index()
             .rename(columns={"DM_smooth_L1":"N_DM_L1","DM_smooth_L3":"N_DM_L3"}))

    # Save
    engine = pick_writer_engine()
    with pd.ExcelWriter(OUT_FILE, engine=engine) as xw:
        dm[["Country Name","Country Code","Year","DM_smooth"]].sort_values(
            ["Country Code","Year"]
        ).to_excel(xw, sheet_name="dm_momentum_raw_ext", index=False)

        keep_cols = [
            "Country Name","Country Code","Year",
            "Y_total_2017ppp","K_total_2017ppp","L_total_persons","E_total_kgoe",
            "gY_L1","gK_L1","gL_L1","gE_L1","gTFP_L1","DM_smooth_L1",
            "gY_L3","gK_L3","gL_L3","gE_L3","gTFP_L3","DM_smooth_L3",
            "DM_smooth"
        ]
        dm.sort_values(["Country Code","Year"])[keep_cols].to_excel(
            xw, sheet_name="dm_full_ext", index=False
        )

        cov.sort_values("Year").to_excel(xw, sheet_name="coverage_by_year", index=False)

        pd.DataFrame({
            "param": [
                "ALPHA_K","BETA_L","GAMMA_E",
                "LAMBDA","RHO_DEFAULT",
                "GROWTH_LAGS","INIT_DM_WITH_CURRENT","CARRY_FORWARD_TAIL_DM",
                "Y_def","K_def","L_def","E_def","SCALING"
            ],
            "value": [
                ALPHA_K, BETA_L, GAMMA_E,
                LAMBDA, RHO_DEFAULT,
                str(GROWTH_LAGS), str(INIT_DM_WITH_CURRENT), str(CARRY_FORWARD_TAIL_DM),
                "Y_total = gdp_pc_pwt_2017ppp * POP",
                "K_total = k_pc_pwt_2017ppp * POP",
                "L_total = l_share_pwt * POP",
                "E_total = E_total_kgoe",
                "NONE"
            ]
        }).to_excel(xw, sheet_name="method_params", index=False)

    print(f"[DM Step 3b • EXT] Saved → {OUT_FILE}")
    if not dm.empty:
        n1 = dm["DM_smooth_L1"].notna().sum()
        n3 = dm["DM_smooth_L3"].notna().sum()
        print(f"Coverage (rows): DM_L1={n1} | DM_L3={n3} | total={len(dm)}")

if __name__ == "__main__":
    main()
