# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd

# ========================
# CONFIG
# ========================
BASE = r"C:\Users\LEGION\Documents\Paper3 Intergenerational Environmental Efficiency"
OUT_DIR = os.path.join(BASE, r"Analysis_12082025")

IN_EXTRACT = os.path.join(OUT_DIR, "re_step1_extract.xlsx")      # from Step 1
OUT_IMPUTED = os.path.join(OUT_DIR, "re_step2_imputed.xlsx")     # this step's output

# Cleaning switches
EXCLUDE_TERRITORIES = True   # set False if you want to keep territories
STRICT_POSITIVITY = True     # enforce positive denominators for ratios

# Aggregates/regions/income groups to ALWAYS exclude (exact strings)
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
    "Least developed countries: UN classification",
    "Low & middle income", "Low income", "Lower middle income",
    "Middle East, North Africa, Afghanistan & Pakistan",
    "Middle East, North Africa, Afghanistan & Pakistan (IDA & IBRD)",
    "Middle East, North Africa, Afghanistan & Pakistan (excluding high income)",
    "Middle income", "North America", "OECD members", "Other small states",
    "Pacific island small states", "Post-demographic dividend", "Pre-demographic dividend",
    "Small states", "South Asia", "South Asia (IDA & IBRD)",
    "Sub-Saharan Africa", "Sub-Saharan Africa (IDA & IBRD countries)",
    "Sub-Saharan Africa (excluding high income)", "Upper middle income", "World"
}

# Territories/special regions often excluded from clustering
TERRITORIES = {
    "American Samoa", "Anguilla", "Aruba", "Bermuda", "British Virgin Islands", "Cayman Islands",
    "China, Hong Kong SAR", "Hong Kong SAR, China", "China, Macao SAR", "Macao SAR, China",
    "Curacao", "Curaçao", "Faroe Islands", "French Polynesia", "Gibraltar", "Greenland",
    "Guam", "Isle of Man", "Montserrat", "New Caledonia", "Northern Mariana Islands",
    "Puerto Rico (US)", "Sint Maarten (Dutch part)", "St. Martin (French part)",
    "Turks and Caicos Islands", "Virgin Islands (U.S.)", "West Bank and Gaza",
    "U.R. of Tanzania: Mainland", "Channel Islands"
}

# Only the raw series we actually need for MANUAL computation + context columns
SERIES_COLS = [
    "GHG_total",               # total GHG
    "GDP_const",               # GDP constant USD
    "E_pc",                    # energy use per capita (kgoe)
    "POP",                     # population
    "Renewable_share",         # optional augmenter (kept for reference)
    # direct columns are optional and kept ONLY for reference if present:
    "Energy_intensity_direct",
    "GDP_per_energy",
]

def pick_writer_engine():
    try:
        import xlsxwriter  # noqa
        return "xlsxwriter"
    except Exception:
        return "openpyxl"

def linear_interpolate_and_tail_extrapolate(years_idx: pd.Index, values: pd.Series):
    """
    Interpolate interior NaNs; forward-extrapolate tail using last two known points; do NOT fill head.
    """
    s = values.astype(float).copy()
    imputed = pd.Series(False, index=years_idx, dtype=bool)

    orig = s.copy()
    # Only fill interior gaps (never head/tail)
    s_interp = s.interpolate(method="linear", limit_area="inside")

    # mark interior fills
    first_valid = s.first_valid_index()
    last_valid  = s.last_valid_index()
    if first_valid is not None and last_valid is not None:
        interior = (years_idx > first_valid) & (years_idx < last_valid)
        imputed |= interior & orig.isna() & s_interp.notna()

    s = s_interp

    # forward-only tail extrapolation using the last two valid points
    if last_valid is not None and last_valid != years_idx.max():
        valid_years = years_idx[s.notna()]
        if len(valid_years) >= 2:
            y1, y2 = valid_years[-2], valid_years[-1]
            v1, v2 = s.loc[y1], s.loc[y2]
            dy = y2 - y1
            slope = (v2 - v1) / dy if dy != 0 else 0.0
            for y in years_idx[years_idx > y2]:
                if pd.isna(s.loc[y]):
                    s.loc[y] = v2 + slope * (y - y2)
                    imputed.loc[y] = True

    return s, imputed

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load raw panel
    raw = pd.read_excel(IN_EXTRACT, sheet_name="panel_raw_RE")
    raw["Year"] = pd.to_numeric(raw["Year"], errors="coerce").astype("Int64")

    # ---------- Cleaning: drop aggregates (+ territories if selected) ----------
    n0 = len(raw)
    countries0 = raw["Country Code"].nunique()

    drop_names = set(AGGREGATES)
    if EXCLUDE_TERRITORIES:
        drop_names |= set(TERRITORIES)

    clean = raw[~raw["Country Name"].isin(drop_names)].copy()

    n1 = len(clean)
    countries1 = clean["Country Code"].nunique()

    # ---------- Keep only needed columns ----------
    keep_cols = ["Country Name", "Country Code", "Year"] + [c for c in SERIES_COLS if c in clean.columns]
    panel = clean[keep_cols].dropna(subset=["Year"]).copy()

    # Set index for processing
    panel = panel.set_index(["Country Name", "Country Code", "Year"]).sort_index()

    # Imputation flags aligned to the same MultiIndex
    flags = pd.DataFrame(index=panel.index)

    # Impute each series per country (interior + forward tail; no head)
    for col in [c for c in SERIES_COLS if c in panel.columns]:
        series = pd.to_numeric(panel[col], errors="coerce")

        filled_parts, flag_parts = [], []
        for (cn, cc), g in series.groupby(level=[0, 1], sort=False):
            g_year = g.droplevel([0, 1])
            yrs = g_year.index.unique().sort_values()
            filled, flag = linear_interpolate_and_tail_extrapolate(yrs, g_year.reindex(yrs))

            mi = pd.MultiIndex.from_product([[cn], [cc], yrs],
                                            names=["Country Name", "Country Code", "Year"])
            filled_parts.append(pd.Series(filled.values, index=mi, name=col))
            flag_parts.append(pd.Series(flag.values, index=mi, name=col + "_imputed"))

        panel[col] = pd.concat(filled_parts).sort_index()
        flags[col + "_imputed"] = pd.concat(flag_parts).sort_index().reindex(panel.index)

        # compact dtype
        if panel[col].dtype == "float64":
            panel[col] = pd.to_numeric(panel[col], downcast="float")

    # ===== Derived metrics (MANUAL ONLY for consistency) =====
    # Total energy use (kgoe)
    if {"E_pc", "POP"}.issubset(panel.columns):
        panel["E_total_kgoe"] = (panel["E_pc"].astype(float) * panel["POP"].astype(float)).astype("float32")
    else:
        panel["E_total_kgoe"] = np.nan

    # Enforce positivity for denominators if requested
    if STRICT_POSITIVITY:
        for c in ["GDP_const", "E_total_kgoe", "GHG_total"]:
            if c in panel.columns:
                panel.loc[panel[c] <= 0, c] = np.nan

    # Energy intensity (kgoe per $ of GDP_const)
    if {"E_total_kgoe", "GDP_const"}.issubset(panel.columns):
        denom = panel["GDP_const"].replace({0: np.nan})
        panel["Energy_intensity_manual"] = (panel["E_total_kgoe"] / denom).astype("float32")
    else:
        panel["Energy_intensity_manual"] = np.nan

    # GDP per energy ($ per kgoe)
    if {"GDP_const", "E_total_kgoe"}.issubset(panel.columns):
        denom = panel["E_total_kgoe"].replace({0: np.nan})
        panel["GDP_per_energy_manual"] = (panel["GDP_const"] / denom).astype("float32")
    else:
        panel["GDP_per_energy_manual"] = np.nan

    # Final columns (manual definitions)
    panel["Energy_intensity_final"] = panel["Energy_intensity_manual"].astype("float32")
    panel["GDP_per_energy_final"]   = panel["GDP_per_energy_manual"].astype("float32")

    # GDP per GHG
    if {"GDP_const", "GHG_total"}.issubset(panel.columns):
        panel["Y_per_GHG"] = (panel["GDP_const"] / panel["GHG_total"].replace({0: np.nan})).astype("float32")
    else:
        panel["Y_per_GHG"] = np.nan

    # Save
    engine = pick_writer_engine()
    with pd.ExcelWriter(OUT_IMPUTED, engine=engine) as xw:
        panel.reset_index().to_excel(xw, sheet_name="prepped_panel_RE", index=False)
        flags.reset_index().to_excel(xw, sheet_name="imputation_flags_RE", index=False)

        # Record cleaning summary in a small metadata sheet
        meta = pd.DataFrame({
            "metric": ["rows_in","countries_in","rows_after_clean","countries_after_clean",
                       "exclude_territories","strict_positivity"],
            "value":  [n0, countries0, n1, countries1, str(EXCLUDE_TERRITORIES), str(STRICT_POSITIVITY)]
        })
        meta.to_excel(xw, sheet_name="meta_cleaning", index=False)

    print(f"[RE Step 2] Saved manual-only panel to: {OUT_IMPUTED}")
    print(f"Cleaning: rows {n0} -> {n1}; countries {countries0} -> {countries1}; "
          f"exclude_territories={EXCLUDE_TERRITORIES}, strict_positivity={STRICT_POSITIVITY}")
    print("Using Energy_intensity_final = E_total_kgoe/GDP_const and GDP_per_energy_final = GDP_const/E_total_kgoe")

if __name__ == "__main__":
    main()
