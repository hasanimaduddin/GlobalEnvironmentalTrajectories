# -*- coding: utf-8 -*-
"""
PC Step 3 — Guided Bridge (no LFPR fallback):
Fit PC on PWT (<=2019), extend 2020–2024 with WB tail proxies (anchored at 2019)

Changes in this version
-----------------------
- Labor in the tail uses EPR (employment-to-population ratio) **only**.
  If EPR is missing, labor is left NaN (no LFPR fallback).
- Prints coverage for the extension window (2020–2024).

Outputs:
  <OUT_DIR>/pc_step3_pc_scores_guided.xlsx
    - pc_capacity_raw
    - pc_capacity_detail
    - coefficients
    - method_params
"""

import os
import numpy as np
import pandas as pd

# ========================
# CONFIG (edit here)
# ========================
BASE = r"C:\Users\LEGION\Documents\Paper3 Intergenerational Environmental Efficiency"
OUT_DIR = os.path.join(BASE, r"Analysis_12082025")

# Inputs
IN_PWT_PREPPED = os.path.join(OUT_DIR, "pc_step2_imputed.xlsx")             # Step 2 output (PWT-based)
IN_WB_TAIL     = os.path.join(OUT_DIR, "pc_step1_wb_tail_2015_2024.xlsx")   # WB tail panel

# Output
OUT_PC = os.path.join(OUT_DIR, "pc_step3_pc_scores_guided.xlsx")

# Core spec
USE_YEAR_FE        = True
USE_POP_WEIGHTS    = True
RIDGE              = 1e-3
PREFER_K_PPP       = True
INCLUDE_ENERGY_IN_PC = False

# Bridge window
TRAIN_END_YEAR     = 2019
TAIL_START_YEAR    = 2020
TAIL_END_YEAR      = 2024

# Year effect in tail:
#   'carry_2019' -> add gamma(2019) to tail years
#   'none'       -> add 0 (baseline)
TAIL_YEAR_EFFECT   = "carry_2019"

# WB tail column names (as created by your tail extractor)
WB_COL_GFCF   = "wb_gfcf_const2015"
WB_COL_POP    = "wb_pop"
WB_COL_EMPR   = "wb_emp_pop_ratio_pct"   # EPR only (no LFPR fallback)

def pick_writer_engine():
    try:
        import xlsxwriter  # noqa
        return "xlsxwriter"
    except Exception:
        return "openpyxl"

def safe_log(v):
    v = pd.to_numeric(v, errors="coerce").astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(v > 0, np.log(v), np.nan)
    return pd.Series(out, index=v.index)

def fit_ridge_wls(X: np.ndarray, y: np.ndarray, w: np.ndarray | None, ridge=RIDGE):
    if w is not None:
        W = np.sqrt(np.nan_to_num(w, nan=0.0, neginf=0.0, posinf=0.0))[:, None]
        Xw = X * W
        yw = y * W.squeeze()
    else:
        Xw, yw = X, y
    XtX = Xw.T @ Xw
    beta = np.linalg.pinv(XtX + ridge * np.eye(XtX.shape[0])) @ (Xw.T @ yw)
    yhat = X @ beta
    return beta, yhat

def main():
    # ---------------- Load PWT-prepped panel ----------------
    panel = pd.read_excel(IN_PWT_PREPPED, sheet_name="prepped_panel")
    panel["Year"] = pd.to_numeric(panel["Year"], errors="coerce").astype(int)

    # Required cols
    need_cols = ["Country Name","Country Code","Year","gdp_pc_2017ppp","l_pc","h_idx"]
    for c in need_cols:
        if c not in panel.columns:
            raise RuntimeError(f"Missing required column in prepped_panel: {c}")

    # Capital column choice
    if PREFER_K_PPP and "k_pc_ppp" in panel.columns:
        KCOL = "k_pc_ppp"
    elif "k_pc_const" in panel.columns:
        KCOL = "k_pc_const"
    else:
        raise RuntimeError("Capital column missing: need k_pc_ppp or k_pc_const in prepped_panel.")

    # ---------------- Build logs for training ----------------
    df = panel.copy()
    df["ln_y_pc"] = safe_log(df["gdp_pc_2017ppp"])
    df["ln_k_pc"] = safe_log(df[KCOL])
    df["ln_l_pc"] = safe_log(df["l_pc"])
    df["ln_h"]    = safe_log(df["h_idx"])
    if INCLUDE_ENERGY_IN_PC and ("E_pc" in df.columns):
        df["ln_e_pc"] = safe_log(df["E_pc"])
    else:
        df["ln_e_pc"] = np.nan

    base_cols = ["ln_k_pc", "ln_l_pc", "ln_h"]
    use_spec = f"K({KCOL})+L+H"
    if INCLUDE_ENERGY_IN_PC and df["ln_e_pc"].notna().sum() > 100:
        base_cols.append("ln_e_pc"); use_spec += "+E"

    # Training subset (<= TRAIN_END_YEAR)
    fit_keep = ["Country Name","Country Code","Year","ln_y_pc"] + base_cols
    df_fit = df.loc[df["Year"] <= TRAIN_END_YEAR, fit_keep].dropna().copy()
    if df_fit.shape[0] < 30:
        raise RuntimeError("Not enough observations to fit the pooled model (<= TRAIN_END_YEAR).")

    # Design matrix with Year FE
    X_core = df_fit[base_cols].to_numpy()
    if USE_YEAR_FE:
        dummies = pd.get_dummies(df_fit["Year"].astype(int), prefix="Y", drop_first=True)
        X = np.column_stack([np.ones((X_core.shape[0], 1)), X_core, dummies.to_numpy()])
        fe_cols = list(dummies.columns)
    else:
        X = np.column_stack([np.ones((X_core.shape[0], 1)), X_core])
        fe_cols = []
    y = df_fit["ln_y_pc"].to_numpy()

    # Weights
    w = None
    if USE_POP_WEIGHTS:
        if "PWT_pop_millions" in df.columns:
            src = "PWT_pop_millions"
        elif "POP" in df.columns:
            src = "POP"
        else:
            src = None
        if src is not None:
            pop_series = df.set_index(["Country Name","Country Code","Year"])[src]
            idx_fit = df_fit.set_index(["Country Name","Country Code","Year"]).index
            pop_fit = np.array([pop_series.get(i, np.nan) for i in idx_fit], dtype=float)
            if src == "PWT_pop_millions":
                pop_fit = pop_fit * 1e6
            w = pop_fit

    # Fit pooled model
    beta, _ = fit_ridge_wls(X, y, w=w, ridge=RIDGE)

    # Recover gamma_2019 if using Year FE
    if USE_YEAR_FE:
        fe_betas = beta[len(["Intercept"] + base_cols):]
        gamma_2019 = 0.0
        for name, b in zip(fe_cols, fe_betas):
            try:
                yy = int(str(name).split("_")[1])
                if yy == TRAIN_END_YEAR:
                    gamma_2019 = float(b)
                    break
            except Exception:
                pass
    else:
        gamma_2019 = 0.0

    # ---------------- Predict within-sample (<= 2019) ----------------
    pred_keep = ["Country Name","Country Code","Year"] + base_cols
    df_pred = df[pred_keep].copy()
    mask_ok = df_pred[base_cols].notna().all(axis=1) & (df_pred["Year"] <= TRAIN_END_YEAR)

    Xp_core = df_pred.loc[mask_ok, base_cols].to_numpy()
    if USE_YEAR_FE:
        dummies_p = pd.get_dummies(df_pred.loc[mask_ok, "Year"].astype(int), prefix="Y", drop_first=True)
        for col in fe_cols:
            if col not in dummies_p.columns:
                dummies_p[col] = 0
        dummies_p = dummies_p[fe_cols]
        Xp = np.column_stack([np.ones((Xp_core.shape[0], 1)), Xp_core, dummies_p.to_numpy()])
    else:
        Xp = np.column_stack([np.ones((Xp_core.shape[0], 1)), Xp_core])

    ln_y_hat = np.full(len(df_pred), np.nan, dtype=float)
    ln_y_hat[mask_ok.values] = (Xp @ beta)
    pc_hat = np.exp(ln_y_hat)

    detail_pwt = df.copy()
    detail_pwt["k_source"] = KCOL
    detail_pwt["use_spec"] = use_spec
    detail_pwt["ln_y_pc_hat"] = ln_y_hat
    detail_pwt["PC_capacity_pc_raw"] = pc_hat
    detail_pwt["resid_ln"] = detail_pwt["ln_y_pc"] - detail_pwt["ln_y_pc_hat"]
    detail_pwt["source_used"] = np.where(detail_pwt["Year"] <= TRAIN_END_YEAR, "PWT", np.nan)

    # ---------------- Load WB tail panel (raw, 2015–2024) ----------------
    wb_tail = pd.read_excel(IN_WB_TAIL, sheet_name="panel_data")
    wb_tail["Year"] = pd.to_numeric(wb_tail["Year"], errors="coerce").astype(int)

    # Build tail predictors for 2020–2024
    tail = wb_tail[(wb_tail["Year"] >= TAIL_START_YEAR) & (wb_tail["Year"] <= TAIL_END_YEAR)].copy()

    # Per-capita GFCF (as proxy for K movement)
    tail["wb_gfcf_pc_const2015"] = (
        pd.to_numeric(tail.get(WB_COL_GFCF), errors="coerce")
        / pd.to_numeric(tail.get(WB_COL_POP), errors="coerce")
    )

    # Labor ratio: EPR only (no fallback)
    tail["wb_l_pc"] = (
        pd.to_numeric(tail.get(WB_COL_EMPR), errors="coerce") / 100.0
        if WB_COL_EMPR in tail.columns else np.nan
    )

    # Overlap 2015–2019 for scaling (no LFPR anywhere)
    wb_overlap = wb_tail[(wb_tail["Year"] >= 2015) & (wb_tail["Year"] <= TRAIN_END_YEAR)].copy()
    wb_overlap["wb_gfcf_pc_const2015"] = (
        pd.to_numeric(wb_overlap.get(WB_COL_GFCF), errors="coerce")
        / pd.to_numeric(wb_overlap.get(WB_COL_POP), errors="coerce")
    )
    wb_overlap["wb_l_pc"] = (
        pd.to_numeric(wb_overlap.get(WB_COL_EMPR), errors="coerce") / 100.0
        if WB_COL_EMPR in wb_overlap.columns else np.nan
    )

    # Pull PWT levels for scaling (2015–2019)
    pwt_overlap = df[["Country Name","Country Code","Year", KCOL, "l_pc", "h_idx"]]
    pwt_overlap = pwt_overlap[(pwt_overlap["Year"] >= 2015) & (pwt_overlap["Year"] <= TRAIN_END_YEAR)].copy()

    # Merge to compute scale factors per country
    ol = pwt_overlap.merge(
        wb_overlap[["Country Name","Country Code","Year","wb_gfcf_pc_const2015","wb_l_pc"]],
        on=["Country Name","Country Code","Year"], how="left"
    )

    def compute_scale(group, num, den):
        r = (group[num] / group[den]).replace([np.inf, -np.inf], np.nan)
        val = np.nanmedian(r.values)
        if np.isnan(val):
            g2019 = group[group["Year"] == TRAIN_END_YEAR]
            if not g2019.empty:
                denv = g2019[den].values[0]
                val = (g2019[num].values[0] / denv) if pd.notna(denv) and denv != 0 else np.nan
        return val

    scales = []
    for (cn, cc), g in ol.groupby(["Country Name","Country Code"], sort=False):
        sK = compute_scale(g, KCOL, "wb_gfcf_pc_const2015")
        sL = compute_scale(g, "l_pc", "wb_l_pc")
        h_hist = df[(df["Country Name"]==cn) & (df["Country Code"]==cc) & (df["Year"]<=TRAIN_END_YEAR)]["h_idx"]
        h0 = pd.to_numeric(h_hist, errors="coerce").dropna()
        h0 = h0.iloc[-1] if len(h0)>0 else np.nan
        scales.append({"Country Name": cn, "Country Code": cc, "scaleK": sK, "scaleL": sL, "h_tail_level": h0})
    scales = pd.DataFrame(scales)

    # Apply scales to tail predictors
    tail = tail.merge(scales, on=["Country Name","Country Code"], how="left")
    tail["k_tail_adj"] = tail["scaleK"] * tail["wb_gfcf_pc_const2015"]
    tail["l_tail_adj"] = tail["scaleL"] * tail["wb_l_pc"]
    tail["h_tail_adj"] = tail["h_tail_level"]

    # Build logs for tail
    tail["ln_k_pc"] = safe_log(tail["k_tail_adj"])
    tail["ln_l_pc"] = safe_log(tail["l_tail_adj"])
    tail["ln_h"]    = safe_log(tail["h_tail_adj"])
    tail["ln_e_pc"] = np.nan  # energy excluded by construction

    # Predict tail using fixed coefficients + chosen year effect
    intercept = float(beta[0])
    betas = {c: float(b) for c, b in zip(base_cols, beta[1:1+len(base_cols)])}
    add_gamma = gamma_2019 if TAIL_YEAR_EFFECT == "carry_2019" else 0.0

    ln_y_hat_tail = []
    for _, row in tail.iterrows():
        val = intercept + sum(betas[c] * row[c] for c in base_cols)
        val += add_gamma
        ln_y_hat_tail.append(val)
    tail["ln_y_pc_hat"] = ln_y_hat_tail
    tail["PC_capacity_pc_raw"] = np.exp(tail["ln_y_pc_hat"])
    tail["source_used"] = "WB-bridged"
    tail["use_spec"] = use_spec
    tail["k_source"] = f"{KCOL}~WB_bridge"
    tail["ln_y_pc"] = np.nan
    tail["resid_ln"] = np.nan

    # Harmonize columns with detail_pwt
    keep_out = [
        "Country Name","Country Code","Year",
        "ln_y_pc","ln_k_pc","ln_l_pc","ln_h","ln_y_pc_hat",
        "PC_capacity_pc_raw","resid_ln","use_spec","k_source","source_used",
        # tail drivers for transparency
        "wb_gfcf_pc_const2015","wb_l_pc","scaleK","scaleL","h_tail_level","k_tail_adj","l_tail_adj","h_tail_adj"
    ]
    for col in ["wb_gfcf_pc_const2015","wb_l_pc","scaleK","scaleL","h_tail_level","k_tail_adj","l_tail_adj","h_tail_adj"]:
        if col not in tail.columns:
            tail[col] = np.nan
    detail_tail = tail[keep_out].copy()

    detail_pwt["source_used"] = detail_pwt["source_used"].fillna("PWT")
    keep_out_pwt = [
        "Country Name","Country Code","Year",
        "ln_y_pc","ln_k_pc","ln_l_pc","ln_h","ln_y_pc_hat",
        "PC_capacity_pc_raw","resid_ln","use_spec","k_source","source_used"
    ]
    detail_pwt2 = detail_pwt[keep_out_pwt].copy()
    for c in ["wb_gfcf_pc_const2015","wb_l_pc","scaleK","scaleL","h_tail_level","k_tail_adj","l_tail_adj","h_tail_adj"]:
        detail_pwt2[c] = np.nan
    detail_pwt2 = detail_pwt2[keep_out]

    # Combine & sort
    detail_all = pd.concat([detail_pwt2, detail_tail], ignore_index=True).sort_values(["Country Code","Year"])
    slim = detail_all[["Country Name","Country Code","Year","PC_capacity_pc_raw","use_spec","source_used"]].copy()

    # Coefficients table
    names = ["Intercept"] + base_cols + (fe_cols if USE_YEAR_FE else [])
    coef_vals = list(beta[:1+len(base_cols)]) + (list(beta[1+len(base_cols):]) if USE_YEAR_FE else [])
    coef_df = pd.DataFrame({"param": names, "value": [float(b) for b in coef_vals]})

    # Method params
    method = pd.DataFrame({
        "param": [
            "train_end_year","tail_start_year","tail_end_year",
            "use_year_fe","tail_year_effect","ridge","use_pop_weights",
            "capital_var_used","include_energy",
            "bridge_k_method","bridge_l_method"
        ],
        "value": [
            TRAIN_END_YEAR, TAIL_START_YEAR, TAIL_END_YEAR,
            str(USE_YEAR_FE), TAIL_YEAR_EFFECT, RIDGE, str(USE_POP_WEIGHTS),
            KCOL, str(INCLUDE_ENERGY_IN_PC),
            "scaled WB GFCF per capita anchored on 2019",
            "WB employment-to-population ratio (EPR) anchored on 2019 — no fallback"
        ]
    })

    # ---------------- Coverage for extension window (2020–2024) ----------------
    ext = detail_all[(detail_all["Year"] >= TAIL_START_YEAR) & (detail_all["Year"] <= TAIL_END_YEAR)]
    def cov(col): 
        return float(ext[col].notna().mean() * 100.0) if len(ext) else 0.0

    print("[Coverage • Extension 2020–2024] "
          f"k_tail_adj={cov('k_tail_adj'):.1f}% | "
          f"l_tail_adj={cov('l_tail_adj'):.1f}% | "
          f"h_tail_adj={cov('h_tail_adj'):.1f}% | "
          f"PC_capacity_pc_raw={cov('PC_capacity_pc_raw'):.1f}%")

    # Save
    os.makedirs(OUT_DIR, exist_ok=True)
    engine = pick_writer_engine()
    with pd.ExcelWriter(OUT_PC, engine=engine) as xw:
        slim.to_excel(xw, sheet_name="pc_capacity_raw", index=False)
        detail_all.to_excel(xw, sheet_name="pc_capacity_detail", index=False)
        coef_df.to_excel(xw, sheet_name="coefficients", index=False)
        method.to_excel(xw, sheet_name="method_params", index=False)

    print(f"[PC Step 3 • Guided Bridge] Saved → {OUT_PC}")
    print("Sheets: pc_capacity_raw | pc_capacity_detail | coefficients | method_params")
    print(f"Spec: {use_spec} | Capital: {KCOL} | Tail year effect: {TAIL_YEAR_EFFECT} | Train<= {TRAIN_END_YEAR}")

if __name__ == "__main__":
    main()
