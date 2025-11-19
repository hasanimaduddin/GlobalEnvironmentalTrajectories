# -*- coding: utf-8 -*-
"""
PC Step 3b — Production Capacity from pooled Cobb–Douglas (PWT inputs, no winsorization/normalization)

Changes vs previous:
- Coefficients estimated ONLY on 1970–2018 (PWT window)
- Uses WB population weights ONLY (WB_POP_persons); if missing -> unweighted
- Capital input fixed to k_pc_ppp (no fallback)
- Energy excluded by construction (no accidental inclusion)
- No fallbacks anywhere in computation
"""

import os
import numpy as np
import pandas as pd

BASE = r"C:\Users\LEGION\Documents\Paper3 Intergenerational Environmental Efficiency"
OUT_DIR = os.path.join(BASE, r"Analysis_12082025")
IN_IMPUTED = os.path.join(OUT_DIR, "pc_step2_imputed.xlsx")
OUT_PC = os.path.join(OUT_DIR, "pc_step3_pc_scores.xlsx")

USE_YEAR_FE = True          # year fixed effects in the pooled regression
USE_POP_WEIGHTS = True      # WLS using WB population
RIDGE = 1e-3                # small ridge for numerical stability

EST_START, EST_END = 1970, 2018  # estimation window per design
CAPITAL_VAR = "k_pc_ppp"         # fixed choice (no fallback)
POP_VAR = "WB_POP_persons"       # fixed choice (no fallback)

def pick_writer_engine():
    try:
        import xlsxwriter  # noqa
        return "xlsxwriter"
    except Exception:
        return "openpyxl"

def safe_log(v):
    v = pd.to_numeric(v, errors="coerce").astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return pd.Series(np.where(v > 0, np.log(v), np.nan), index=v.index)

def fit_ridge_wls(X, y, w, ridge=RIDGE):
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
    df = pd.read_excel(IN_IMPUTED, sheet_name="prepped_panel")

    need = ["Country Name","Country Code","Year","gdp_pc_2017ppp","l_pc","h_idx", CAPITAL_VAR, POP_VAR]
    for c in need:
        if c not in df.columns:
            raise RuntimeError(f"Missing required column in prepped_panel: {c}")

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["Year"]).copy()
    df["Year"] = df["Year"].astype(int)

    # Logs
    df["ln_y_pc"] = safe_log(df["gdp_pc_2017ppp"])
    df["ln_k_pc"] = safe_log(df[CAPITAL_VAR])
    df["ln_l_pc"] = safe_log(df["l_pc"])
    df["ln_h"]    = safe_log(df["h_idx"])

    # ----- ESTIMATION SAMPLE: 1970–2018 only -----
    est = df[(df["Year"] >= EST_START) & (df["Year"] <= EST_END)].copy()
    base_cols = ["ln_k_pc","ln_l_pc","ln_h"]
    keep_cols = ["Country Name","Country Code","Year","ln_y_pc"] + base_cols + [POP_VAR]
    est = est[keep_cols].dropna()

    if est.shape[0] < 30:
        raise RuntimeError("Not enough observations in 1970–2018 to fit the pooled model.")

    # Design matrix (estimation)
    X_core = est[base_cols].to_numpy()
    if USE_YEAR_FE:
        dummies = pd.get_dummies(est["Year"].astype(int), prefix="Y", drop_first=True)
        X = np.column_stack([np.ones((X_core.shape[0], 1)), X_core, dummies.to_numpy()])
        fe_cols = list(dummies.columns)
    else:
        X = np.column_stack([np.ones((X_core.shape[0], 1)), X_core])
        fe_cols = []
    y = est["ln_y_pc"].to_numpy()

    # Weights: WB population only (no fallback). If all-NaN -> OLS.
    w = est[POP_VAR].to_numpy(dtype=float)
    if not USE_POP_WEIGHTS or np.all(np.isnan(w)):
        w = None

    # Fit
    beta, _ = fit_ridge_wls(X, y, w, ridge=RIDGE)

    # ----- PREDICTION for all years where inputs exist (1970–2024) -----
    pred = df[["Country Name","Country Code","Year"] + base_cols].copy()
    mask_ok = pred[base_cols].notna().all(axis=1)

    Xp_core = pred.loc[mask_ok, base_cols].to_numpy()
    if USE_YEAR_FE:
        dummies_p = pd.get_dummies(pred.loc[mask_ok, "Year"].astype(int), prefix="Y", drop_first=True)
        # align to estimation FE columns; unseen years get zeros
        for col in fe_cols:
            if col not in dummies_p.columns:
                dummies_p[col] = 0
        dummies_p = dummies_p[fe_cols]
        Xp = np.column_stack([np.ones((Xp_core.shape[0], 1)), Xp_core, dummies_p.to_numpy()])
    else:
        Xp = np.column_stack([np.ones((Xp_core.shape[0], 1)), Xp_core])

    ln_y_hat = np.full(len(pred), np.nan, dtype=float)
    ln_y_hat[mask_ok.values] = Xp @ beta
    y_pc_hat = np.exp(ln_y_hat)

    # Outputs
    detail = df.copy()
    detail["k_source"] = CAPITAL_VAR
    detail["use_spec"] = "K({})+L+H".format(CAPITAL_VAR)
    detail["ln_y_pc_hat"] = ln_y_hat
    detail["PC_capacity_pc_raw"] = y_pc_hat
    detail["resid_ln"] = detail["ln_y_pc"] - detail["ln_y_pc_hat"]

    keep_out = [
        "Country Name","Country Code","Year",
        "gdp_pc_2017ppp", CAPITAL_VAR, "l_pc", "h_idx",
        "ln_y_pc","ln_k_pc","ln_l_pc","ln_h",
        "ln_y_pc_hat","PC_capacity_pc_raw","resid_ln","use_spec","k_source"
    ]
    detail = detail[keep_out].sort_values(["Country Code","Year"])

    slim = detail[["Country Name","Country Code","Year","PC_capacity_pc_raw","use_spec"]].copy()

    # Coefficients
    names = ["Intercept"] + base_cols + fe_cols
    coef_df = pd.DataFrame({"param": names, "value": [float(b) for b in beta]})

    # Method params
    method = pd.DataFrame({
        "param": [
            "estimation_years","use_year_fe","use_spec","ridge",
            "use_pop_weights","weight_var","capital_var_used",
            "winsorization","capping","normalization","energy_included"
        ],
        "value": [
            f"{EST_START}-{EST_END}", str(USE_YEAR_FE), "K+L+H", RIDGE,
            str(USE_POP_WEIGHTS and w is not None), POP_VAR, CAPITAL_VAR,
            "False","False","False","False"
        ]
    })

    os.makedirs(OUT_DIR, exist_ok=True)
    engine = pick_writer_engine()
    with pd.ExcelWriter(OUT_PC, engine=engine) as xw:
        slim.to_excel(xw, sheet_name="pc_capacity_raw", index=False)
        detail.to_excel(xw, sheet_name="pc_capacity_detail", index=False)
        coef_df.to_excel(xw, sheet_name="coefficients", index=False)
        method.to_excel(xw, sheet_name="method_params", index=False)

    print(f"[PC Step 3b] Saved capacity to: {OUT_PC}")
    print("Spec: K(like)=k_pc_ppp + L + H | Year FE:", USE_YEAR_FE,
          "| Weights: WB_POP_persons" if USE_POP_WEIGHTS else "| Weights: OLS")
    print("Estimation window:", f"{EST_START}-{EST_END}")

if __name__ == "__main__":
    main()
