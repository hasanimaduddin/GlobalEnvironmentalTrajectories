# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 07:46:19 2025

@author: LEGION
"""

# -*- coding: utf-8 -*-
import os
import pandas as pd

# ========================
# CONFIG
# ========================
BASE = r"C:\Users\LEGION\Documents\Paper3 Intergenerational Environmental Efficiency"
INPUT_EXCEL = os.path.join(BASE, "WBRAW.xlsx")
INPUT_SHEET = "Data"

OUT_DIR = os.path.join(BASE, r"Analysis_12082025")
OUT_EXTRACT = os.path.join(OUT_DIR, "re_step1_extract.xlsx")

# ---- Candidate sets (try codes first; names are fallback) ----
# We standardize to the keys on the left so later steps are stable.
CANDIDATES = {
    # GDP per unit energy (PPP const $ per kgoe). WB sometimes updates the base year in the name.
    "GDP_per_energy": {
        "codes": ["EG.GDP.PUSE.KO.PP.KD"],
        "names": [
            "GDP per unit of energy use (constant 2021 PPP $ per kg of oil equivalent)",
            "GDP per unit of energy use (constant 2017 PPP $ per kg of oil equivalent)",
            "GDP per unit of energy use (constant PPP $ per kg of oil equivalent)",
        ],
    },

    # Total GHG emissions (excluding LULUCF), usually in kt CO2e. We'll handle units later.
    "GHG_total": {
        "codes": ["EN.ATM.GHGT.KT.CE"],  # kt CO2e
        "names": [
            "Total greenhouse gas emissions excluding LULUCF (kt of CO2 equivalent)",
            "Total greenhouse gas emissions excluding LULUCF (Mt CO2e)",
            "Total greenhouse gas emissions excluding LULUCF (kt CO2e)",
        ],
    },

    # If you prefer CO2-only (not total GHG), keep as optional for robustness later.
    "CO2_kt": {
        "codes": ["EN.ATM.CO2E.KT"],
        "names": ["CO2 emissions (kt)"],
    },

    # GDP constant USD (for intensities)
    "GDP_const": {
        "codes": ["NY.GDP.MKTP.KD"],
        "names": ["GDP (constant 2015 US$)", "GDP (constant 2017 US$)", "GDP (constant US$)"],
    },

    # Energy use per capita (kgoe) + Population (to build total energy use)
    "E_pc": {
        "codes": ["EG.USE.PCAP.KG.OE"],
        "names": ["Energy use (kg of oil equivalent per capita)"],
    },
    "POP": {
        "codes": ["SP.POP.TOTL"],
        "names": ["Population, total"],
    },

    # Optional: direct energy intensity (MJ per $2017 PPP GDP). Often sparse; we still extract if present.
    "Energy_intensity_direct": {
        "codes": ["EG.EGY.PRIM.PP.KD"],
        "names": ["Energy intensity level of primary energy (MJ/$2017 PPP GDP)"],
    },

    # Optional: renewables share (augmenter)
    "Renewable_share": {
        "codes": ["EG.FEC.RNEW.ZS"],
        "names": ["Renewable energy consumption (% of total final energy consumption)"],
    },
}

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

def pick_writer_engine():
    try:
        import xlsxwriter  # noqa
        return "xlsxwriter"
    except Exception:
        return "openpyxl"

def compute_coverage_table(wide_subset: pd.DataFrame) -> pd.DataFrame:
    years = [str(y) for y in year_columns(wide_subset)]
    if not years:
        return pd.DataFrame(columns=["Indicator Name", "Coverage_%"])
    grp = wide_subset.groupby("Indicator Name")[years]
    cov = grp.apply(lambda g: g.notna().sum().sum())
    total = grp.size() * len(years)
    cov_pct = (cov / total) * 100
    out = cov_pct.reset_index().rename(columns={0: "Coverage_%", "index": "Indicator Name"})
    return out.sort_values("Coverage_%", ascending=False)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    raw = pd.read_excel(INPUT_EXCEL, sheet_name=INPUT_SHEET, dtype={"Country Code": str})
    years = [str(y) for y in year_columns(raw)]

    found = []
    for std_key, spec in CANDIDATES.items():
        codes = spec.get("codes", [])
        names = spec.get("names", [])

        mask_code = raw["Indicator Code"].isin(codes) if "Indicator Code" in raw.columns else False
        mask_name = raw["Indicator Name"].isin(names) if "Indicator Name" in raw.columns else False
        sel = raw[mask_code | mask_name].copy()
        if not sel.empty:
            sel["__std_key__"] = std_key
            found.append(sel)

    if not found:
        raise RuntimeError("No matching RE indicators found. Check that WBRAW.xlsx has 'Indicator Code/Name' columns and codes match.")

    subset = pd.concat(found, ignore_index=True)

    # Console summary of what we found
    print("== Found RE indicators ==")
    for std_key in CANDIDATES.keys():
        hit = subset.loc[subset["__std_key__"] == std_key]
        codes_preview = ", ".join(hit["Indicator Code"].dropna().astype(str).unique()[:3]) if not hit.empty else "-"
        print(f"  {std_key:24s}: {'OK' if not hit.empty else 'MISSING'}  (codes: {codes_preview})")

    # Melt → long
    id_vars = ["Country Name", "Country Code", "Indicator Name", "Indicator Code", "__std_key__"]
    long = subset.melt(id_vars=id_vars, value_vars=years, var_name="Year", value_name="Value")
    long["Year"] = long["Year"].astype(int)

    # Pivot using standardized keys
    panel = long.pivot_table(index=["Country Name", "Country Code", "Year"],
                             columns="__std_key__", values="Value", aggfunc="first").reset_index()
    panel = panel.sort_values(["Country Code", "Year"]).reset_index(drop=True)

    # Coverage summary
    coverage = compute_coverage_table(subset.drop(columns="__std_key__"))

    # Save
    engine = pick_writer_engine()
    with pd.ExcelWriter(OUT_EXTRACT, engine=engine) as xw:
        subset.to_excel(xw, sheet_name="original_raw_subset_RE", index=False)
        panel.to_excel(xw, sheet_name="panel_raw_RE", index=False)
        coverage.to_excel(xw, sheet_name="coverage_summary", index=False)

    print(f"[RE Step 1] Saved extract to: {OUT_EXTRACT}")
    print("Sheets: original_raw_subset_RE | panel_raw_RE | coverage_summary")
    print("Next: Step 2 will impute and (if needed) compute manual energy intensity = (E_pc × POP) / GDP_const.")

if __name__ == "__main__":
    main()
