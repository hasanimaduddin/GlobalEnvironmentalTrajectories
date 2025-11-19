# -*- coding: utf-8 -*-
"""
DM Step 2 — Imputation (interior-only + forward-tail; NO head fill)
Inputs : Analysis_12082025/dm_step1_extract.xlsx (sheet: panel_raw_DM)
Outputs: Analysis_12082025/dm_step2_imputed.xlsx
"""

import os
import numpy as np
import pandas as pd

# ========================
# CONFIG
# ========================
BASE = r"C:\Users\LEGION\Documents\Paper3 Intergenerational Environmental Efficiency"
OUT_DIR = os.path.join(BASE, r"Analysis_12082025")

IN_EXTRACT = os.path.join(OUT_DIR, "dm_step1_extract.xlsx")
OUT_IMPUTED = os.path.join(OUT_DIR, "dm_step2_imputed.xlsx")

# Keep exactly the Step-1 column names we’ll impute
SERIES_COLS = [
    "gdp_pc_pwt_2017ppp",   # Y (per cap, PWT rgdpo / WB POP)
    "k_pc_pwt_2017ppp",     # K (per cap, PWT cn / WB POP)
    "l_share_pwt",          # L share (EMP / POP)
    "E_pc",                 # Energy per cap (kgoe/person)
    "POP",                  # Population, total (persons)
    "POP_0_14",             # 0–14 population (persons)
    "POP_15p",              # built in step 1, re-imputed for continuity
]

# Aggregates & territories to exclude
AGGREGATES = {
    "Africa Eastern and Southern", "Africa Western and Central", "Arab World",
    "Caribbean small states", "Central Europe and the Baltics", "Early-demographic dividend",
    "East Asia & Pacific", "East Asia & Pacific (IDA & IBRD countries)", "East Asia & Pacific (excluding high income)",
    "Euro area", "Europe & Central Asia", "Europe & Central Asia (IDA & IBRD countries)",
    "Europe & Central Asia (excluding high income)", "European Union",
    "Fragile and conflict affected situations", "Heavily indebted poor countries (HIPC)",
    "High income", "IBRD only", "IDA & IBRD total", "IDA blend", "IDA only", "IDA total",
    "Late-demographic dividend", "Latin America & Caribbean",
    "Latin America & Caribbean (excluding high income)",
    "Latin America & the Caribbean (IDA & IBRD countries)",
    "Least developed countries: UN classification", "Low & middle income", "Low income",
    "Lower middle income", "Middle East, North Africa, Afghanistan & Pakistan",
    "Middle East, North Africa, Afghanistan & Pakistan (IDA & IBRD)",
    "Middle East, North Africa, Afghanistan & Pakistan (excluding high income)",
    "Middle income", "North America", "OECD members", "Other small states",
    "Pacific island small states", "Post-demographic dividend", "Pre-demographic dividend",
    "Small states", "South Asia", "South Asia (IDA & IBRD)",
    "Sub-Saharan Africa", "Sub-Saharan Africa (IDA & IBRD countries)",
    "Sub-Saharan Africa (excluding high income)", "Upper middle income", "World"
}
TERRITORIES = {
    "American Samoa", "Anguilla", "Aruba", "Bermuda", "British Virgin Islands", "Cayman Islands",
    "China, Hong Kong SAR", "Hong Kong SAR, China", "China, Macao SAR", "Macao SAR, China",
    "Curacao", "Curaçao", "Faroe Islands", "French Polynesia", "Gibraltar", "Greenland",
    "Guam", "Isle of Man", "Montserrat", "New Caledonia", "Northern Mariana Islands",
    "Puerto Rico (US)", "Sint Maarten (Dutch part)", "St. Martin (French part)",
    "Turks and Caicos Islands", "Virgin Islands (U.S.)", "West Bank and Gaza",
    "U.R. of Tanzania: Mainland", "Channel Islands"
}

def pick_writer_engine():
    try:
        import xlsxwriter  # noqa
        return "xlsxwriter"
    except Exception:
        return "openpyxl"

def linear_interpolate_and_tail_extrapolate(years_idx: pd.Index, values: pd.Series):
    """
    Fill ONLY interior gaps (linear) and forward-tail (using last two valid points).
    DO NOT fill the head (leading years before first valid).
    """
    s = values.astype(float).copy()
    flags = pd.Series(False, index=years_idx, dtype=bool)

    orig = s.copy()
    s_interp = s.interpolate(method="linear", limit_direction="both")

    # keep head as NaN
    first_valid = s.first_valid_index()
    if first_valid is not None:
        s_interp.loc[years_idx[years_idx < first_valid]] = np.nan

    # mark interior fills
    last_valid = s.last_valid_index()
    if first_valid is not None and last_valid is not None:
        interior = (years_idx > first_valid) & (years_idx < last_valid)
        flags |= interior & orig.isna() & s_interp.notna()

    s = s_interp

    # forward-only tail based on last two valid points
    if last_valid is not None and last_valid != years_idx.max():
        valid_years = years_idx[s.notna()]
        if len(valid_years) >= 2:
            y1, y2 = valid_years[-2], valid_years[-1]
            v1, v2 = float(s.loc[y1]), float(s.loc[y2])
            dy = (y2 - y1) if (y2 - y1) != 0 else 1
            slope = (v2 - v1) / dy
            for y in years_idx[years_idx > y2]:
                if pd.isna(s.loc[y]):
                    s.loc[y] = v2 + slope * (y - y2)
                    flags.loc[y] = True

    return s, flags

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load Step-1 panel
    df = pd.read_excel(IN_EXTRACT, sheet_name="panel_raw_DM")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")

    # Drop aggregates/territories
    mask_excl = df["Country Name"].isin(AGGREGATES | TERRITORIES)
    before = len(df)
    df = df.loc[~mask_excl].copy()
    after = len(df)
    print(f"[DM Step 2] Dropped aggregates/territories: {before - after} rows")

    # Set index for per-country processing
    df = df.set_index(["Country Name", "Country Code", "Year"]).sort_index()

    cols = [c for c in SERIES_COLS if c in df.columns]
    if not cols:
        raise RuntimeError("None of the expected series were found. Check Step-1 output/column names.")

    flags = pd.DataFrame(index=df.index)

    # Impute each series country-wise
    for col in cols:
        ser = df[col].astype(float)
        filled_chunks, flag_chunks = [], []

        for (cn, cc), g in ser.groupby(level=[0, 1]):
            gy = g.droplevel([0, 1])
            years = gy.index.unique().sort_values()
            filled, flg = linear_interpolate_and_tail_extrapolate(years, gy.reindex(years))

            mi = pd.MultiIndex.from_product([[cn], [cc], years],
                                            names=["Country Name", "Country Code", "Year"])
            filled_chunks.append(pd.Series(filled.values, index=mi, name=col))
            flag_chunks.append(pd.Series(flg.values,    index=mi, name=f"{col}_imputed"))

        df[col] = pd.concat(filled_chunks).sort_index()
        flags[flag_chunks[0].name] = pd.concat(flag_chunks).sort_index()

        if df[col].dtype == "float64":
            df[col] = pd.to_numeric(df[col], downcast="float")

    # Reconstruct POP_15p (in case POP/POP_0_14 moved after imputation)
    if {"POP", "POP_0_14"}.issubset(df.columns):
        df["POP_15p"] = (pd.to_numeric(df["POP"], errors="coerce") -
                         pd.to_numeric(df["POP_0_14"], errors="coerce")).astype("float32")

    # Convenience: total energy (for diagnostics or optional use)
    if {"E_pc", "POP"}.issubset(df.columns):
        df["E_total_kgoe"] = (pd.to_numeric(df["E_pc"], errors="coerce") *
                              pd.to_numeric(df["POP"], errors="coerce")).astype("float32")

    # Coverage quicklook
    cov_before = {c: int(df[c].notna().sum()) for c in cols}
    print("[DM Step 2] Non-missing counts after imputation:", cov_before)

    # Save
    engine = pick_writer_engine()
    with pd.ExcelWriter(OUT_IMPUTED, engine=engine) as xw:
        df.reset_index().to_excel(xw, sheet_name="prepped_panel_DM", index=False)
        flags.reset_index().to_excel(xw, sheet_name="imputation_flags_DM", index=False)

    print(f"[DM Step 2] Saved → {OUT_IMPUTED}")
    print("Sheets: prepped_panel_DM | imputation_flags_DM")

if __name__ == "__main__":
    main()
