# -*- coding: utf-8 -*-
"""
Step 2 — Impute panel (linear interior, forward-tail) for PWT-based PC inputs.

Reads:  Analysis_12082025/pc_step1_extract.xlsx  (sheet: panel_raw)
Writes: Analysis_12082025/pc_step2_imputed.xlsx  (sheets: prepped_panel, imputation_flags)

Notes:
- Works on new variable names from Step 1 (PWT core):
    gdp_pc_2017ppp, k_pc_ppp, k_pc_const, l_pc, h_idx
  (+ optionally E_pc, PWT_rtfpna_2017eq1 if present — included automatically if found)
- Does NOT fill leading (head) NaNs; only interior + forward tail.
- Clips to non-negative values (useful for econ quantities).
"""

import os
import numpy as np
import pandas as pd

# ========================
# CONFIG
# ========================
BASE = r"C:\Users\LEGION\Documents\Paper3 Intergenerational Environmental Efficiency"
OUT_DIR = os.path.join(BASE, r"Analysis_12082025")
IN_EXTRACT = os.path.join(OUT_DIR, "pc_step1_extract.xlsx")   # from Step 1
OUT_IMPUTED = os.path.join(OUT_DIR, "pc_step2_imputed.xlsx")  # this step's output

# Core series for PC (new names). Others can be auto-added if present.
CORE_SERIES = ["gdp_pc_2017ppp", "k_pc_ppp", "k_pc_const", "l_pc", "h_idx"]

# Optional extras to include if present (won't error if missing)
OPTIONAL_SERIES = ["E_pc", "PWT_rtfpna_2017eq1"]

def pick_writer_engine():
    try:
        import xlsxwriter  # noqa
        return "xlsxwriter"
    except Exception:
        return "openpyxl"

def linear_interpolate_and_tail_extrapolate(years_idx: pd.Index, values: pd.Series):
    """
    Interpolate interior NaNs on a YEAR-AWARE axis; forward-extrapolate tail.
    - Do NOT fill leading/head NaNs (before first valid).
    - Use slope from last two known points for tails; if only one known point, flat-fill.
    Inputs:
        years_idx: Index of integer years (sorted, unique, ideally continuous)
        values   : Series aligned to years_idx (float)
    Returns:
        filled (Series[float]), imputed_flag (Series[bool])
    """
    s = values.astype(float).copy()
    imputed = pd.Series(False, index=years_idx, dtype=bool)

    # Keep original for interior-flagging
    orig = s.copy()

    # Year-aware interpolation (index = years) for interior gaps
    s_interp = s.interpolate(method="index", limit_direction="forward")

    # Identify first/last valid from the ORIGINAL data
    first_valid = s.first_valid_index()
    last_valid  = s.last_valid_index()

    # Do not fill the head (before first original valid)
    if first_valid is not None:
        s_interp.loc[years_idx[years_idx < first_valid]] = np.nan

    # Mark interior interpolations: only between original first/last valid
    if (first_valid is not None) and (last_valid is not None):
        interior_mask = (years_idx > first_valid) & (years_idx < last_valid)
        imputed |= interior_mask & orig.isna() & s_interp.notna()

    s = s_interp

    # Tail extrapolation (forward only)
    if last_valid is not None and last_valid != years_idx.max():
        valid_years = years_idx[s.notna()]
        if len(valid_years) >= 2:
            y1, y2 = valid_years[-2], valid_years[-1]
            v1, v2 = s.loc[y1], s.loc[y2]
            dy = (y2 - y1)
            slope = (v2 - v1) / dy if dy != 0 else 0.0
            for y in years_idx[years_idx > y2]:
                if pd.isna(s.loc[y]):
                    s.loc[y] = v2 + slope * (y - y2)
                    imputed.loc[y] = True
        elif len(valid_years) == 1:
            y2 = valid_years[-1]
            v2 = s.loc[y2]
            for y in years_idx[years_idx > y2]:
                if pd.isna(s.loc[y]):
                    s.loc[y] = v2
                    imputed.loc[y] = True

    # Econ safety: clip at zero (l_pc may exceed 1 in some historical artifacts; we don't cap upper bound here)
    s = s.clip(lower=0)

    return s, imputed

def main():
    # Read panel from Step 1
    panel = pd.read_excel(IN_EXTRACT, sheet_name="panel_raw")

    # Ensure Year is integer-like and index properly
    panel["Year"] = pd.to_numeric(panel["Year"], errors="coerce").astype("Int64")
    panel = panel.dropna(subset=["Year"]).copy()
    panel["Year"] = panel["Year"].astype(int)
    panel = panel.set_index(["Country Name", "Country Code", "Year"]).sort_index()

    # Determine which series to process (present in file)
    cols = [c for c in CORE_SERIES + OPTIONAL_SERIES if c in panel.columns]
    if not cols:
        raise RuntimeError(
            "No expected PWT-based columns found in 'panel_raw'. "
            f"Looked for: {CORE_SERIES + OPTIONAL_SERIES}"
        )

    # Prepare flags container
    flags = pd.DataFrame(index=panel.index)

    # Diagnostics: before completeness
    print("=== BEFORE completeness (non-NA share) ===")
    for c in cols:
        non_na = panel[c].notna().mean() if len(panel) else 0.0
        print(f"{c:>18}: {non_na:.2%}")

    # Process per indicator, per country
    for col in cols:
        series = panel[col].astype(float)

        filled_parts = []
        flag_parts = []

        # Group by country (Name, Code)
        for (cn, cc), g in series.groupby(level=[0, 1], sort=False):
            g_year = g.droplevel([0, 1])

            if g_year.empty:
                continue

            original_years = g_year.index.unique().astype(int)
            yr_min = int(original_years.min())
            yr_max = int(original_years.max())

            # Build continuous year range for this country
            full_years = pd.Index(range(yr_min, yr_max + 1), name="Year")

            # Align to continuous range
            g_full = g_year.reindex(full_years)

            filled, flag = linear_interpolate_and_tail_extrapolate(full_years, g_full)

            # Any year introduced by reindexing is flagged as imputed
            new_years_mask = ~full_years.isin(original_years)
            if new_years_mask.any():
                flag.loc[full_years[new_years_mask]] = True

            # Reattach MultiIndex
            mi = pd.MultiIndex.from_product([[cn], [cc], full_years],
                                            names=["Country Name", "Country Code", "Year"])
            filled_parts.append(pd.Series(filled.values, index=mi, name=col))
            flag_parts.append(pd.Series(flag.values, index=mi, name=col + "_imputed"))

        # Concatenate all countries back
        if filled_parts:
            filled_col = pd.concat(filled_parts).sort_index()
            flag_col = pd.concat(flag_parts).sort_index()
            panel[col] = filled_col
            flags[flag_col.name] = flag_col

            # Downcast float to save space
            if panel[col].dtype == "float64":
                panel[col] = pd.to_numeric(panel[col], downcast="float")

    # Diagnostics: after completeness
    print("\n=== AFTER completeness (non-NA share) ===")
    for c in cols:
        non_na = panel[c].notna().mean() if len(panel) else 0.0
        print(f"{c:>18}: {non_na:.2%}")

    # Save output
    os.makedirs(OUT_DIR, exist_ok=True)
    engine = pick_writer_engine()
    with pd.ExcelWriter(OUT_IMPUTED, engine=engine) as xw:
        panel.reset_index().to_excel(xw, sheet_name="prepped_panel", index=False)
        flags.reset_index().to_excel(xw, sheet_name="imputation_flags", index=False)

    print(f"\n[Step 2] Saved imputed panel to: {OUT_IMPUTED}")
    print(f"Columns imputed: {cols}")

if __name__ == "__main__":
    main()
