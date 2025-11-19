# -*- coding: utf-8 -*-
"""
Resource Efficiency (RE) — Step 3b: Year-wise normalization (log + z), STRICT composite
- Requires both components present for RE; otherwise RE=NaN (comparability > coverage)
- No winsorization, no renewables augmenter, equal weights (0.5, 0.5)
"""

import os
import numpy as np
import pandas as pd

# ========================
# CONFIG
# ========================
BASE = r"C:\Users\LEGION\Documents\Paper3 Intergenerational Environmental Efficiency"
OUT_DIR = os.path.join(BASE, r"Analysis_12082025")

IN_IMPUTED = os.path.join(OUT_DIR, "re_step2_imputed.xlsx")
OUT_SCORES = os.path.join(OUT_DIR, "re_step3_scores.xlsx")

# Equal weights (used only if both z-scores are available)
W_E   = 0.5
W_GHG = 0.5

# Optional gentle tail compression (kept False to preserve distances)
COMPRESS_TAILS = False   # True uses tanh(z/2); False keeps plain z

def pick_writer_engine():
    try:
        import xlsxwriter  # noqa
        return "xlsxwriter"
    except Exception:
        return "openpyxl"

def yearwise_z_from_raw(series: pd.Series, year: pd.Series) -> pd.Series:
    """
    Compute within-year z-scores on log of series:
      z = (ln(series) - mean_y ln(series)) / sd_y ln(series)
    """
    x = np.log(pd.to_numeric(series, errors="coerce"))
    mu = x.groupby(year).transform("mean")
    sd = x.groupby(year).transform("std")
    z = (x - mu) / (sd.replace({0.0: np.nan}))
    return z

def maybe_compress(z: pd.Series, compress: bool) -> pd.Series:
    return np.tanh(z/2.0) if compress else z

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_excel(IN_IMPUTED, sheet_name="prepped_panel_RE")

    required = ["Country Name","Country Code","Year","GDP_per_energy_final","Y_per_GHG"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns in prepped_panel_RE: {missing}")

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")

    # Raw components (higher is better)
    df["C_E"]   = pd.to_numeric(df["GDP_per_energy_final"], errors="coerce")   # Y/E
    df["C_GHG"] = pd.to_numeric(df["Y_per_GHG"],            errors="coerce")   # Y/GHG

    # Year-wise normalization (log + z), component by component
    df["z_E"]   = yearwise_z_from_raw(df["C_E"],   df["Year"])
    df["z_GHG"] = yearwise_z_from_raw(df["C_GHG"], df["Year"])

    # (Optional) mild tail compression (off by default)
    df["r_E"]   = maybe_compress(df["z_E"],   COMPRESS_TAILS)
    df["r_GHG"] = maybe_compress(df["z_GHG"], COMPRESS_TAILS)

    # STRICT composite: require both components present; else RE=NaN
    mask_both = df["r_E"].notna() & df["r_GHG"].notna()
    df["RE_composite_raw"] = np.where(
        mask_both,
        W_E * df["r_E"] + W_GHG * df["r_GHG"],
        np.nan
    )

    # Diagnostics / coverage
    df["has_E"]   = df["C_E"].notna().astype(int)
    df["has_GHG"] = df["C_GHG"].notna().astype(int)
    df["both_available"] = (df["has_E"] & df["has_GHG"]).astype(int)

    # Save
    engine = pick_writer_engine()
    with pd.ExcelWriter(OUT_SCORES, engine=engine) as xw:
        # Slim output
        df[["Country Name","Country Code","Year","RE_composite_raw"]].sort_values(
            ["Country Code","Year"]
        ).to_excel(xw, sheet_name="re_composite_raw", index=False)

        # Full diagnostics
        keep_full = [
            "Country Name","Country Code","Year",
            "C_E","C_GHG","z_E","z_GHG","r_E","r_GHG",
            "RE_composite_raw","has_E","has_GHG","both_available"
        ]
        df.sort_values(["Country Code","Year"])[keep_full].to_excel(
            xw, sheet_name="re_full", index=False
        )

        # Normalization params by year (auditable benchmark)
        # computed on the available set for each component (broader is fine; composite still requires both)
        norm_params = (
            pd.DataFrame({
                "Year": df["Year"],
                "ln_C_E": np.log(df["C_E"]),
                "ln_C_GHG": np.log(df["C_GHG"])
            })
            .groupby("Year")
            .agg(
                mu_E   = ("ln_C_E","mean"),
                sd_E   = ("ln_C_E","std"),
                N_E    = ("ln_C_E", lambda s: s.notna().sum()),
                mu_GHG = ("ln_C_GHG","mean"),
                sd_GHG = ("ln_C_GHG","std"),
                N_GHG  = ("ln_C_GHG", lambda s: s.notna().sum())
            )
            .reset_index()
        )
        norm_params.to_excel(xw, sheet_name="norm_params_by_year", index=False)

        # Params
        meta = pd.DataFrame({
            "param": [
                "WEIGHT_E","WEIGHT_GHG","SCALING","YEARWISE_TRANSFORM",
                "COMPRESS_TAILS","STRICT_COMPLETENESS"
            ],
            "value": [W_E, W_GHG, "YEAR_Z", "log + z (mean/SD per year)",
                      str(COMPRESS_TAILS), "True"]
        })
        meta.to_excel(xw, sheet_name="method_params", index=False)

    # Console summary
    n = len(df)
    both = int(df["both_available"].sum())
    only_e = int(((df["has_E"]==1) & (df["has_GHG"]==0)).sum())
    only_g = int(((df["has_E"]==0) & (df["has_GHG"]==1)).sum())
    none = n - both - only_e - only_g
    print(f"[RE Step 3b] Saved STRICT composite to: {OUT_SCORES}")
    print("Sheets: re_composite_raw | re_full | norm_params_by_year | method_params")
    print(f"Coverage (rows): both={both} | only E={only_e} | only GHG={only_g} | none={none} | total={n}")
    print(f"Config → weights: ({W_E},{W_GHG}), compress_tails={COMPRESS_TAILS}, STRICT_COMPLETENESS=True")
    
if __name__ == "__main__":
    main()
