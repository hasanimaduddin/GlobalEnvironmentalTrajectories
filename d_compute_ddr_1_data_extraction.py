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
OUT_EXTRACT = os.path.join(OUT_DIR, "ddr_step1_extract.xlsx")

# ---- Candidate sets (prefer CODE; names are fallback) ----
CANDIDATES = {
    # % of GNI series only (comparable across countries)
    "dep_mineral_pctGNI": {
        "codes": ["NY.ADJ.DMIN.GN.ZS"],
        "names": ["Adjusted savings: mineral depletion (% of GNI)"],
    },
    "dep_energy_pctGNI": {
        "codes": ["NY.ADJ.DNGY.GN.ZS"],
        "names": ["Adjusted savings: energy depletion (% of GNI)"],
    },
    "dep_forest_pctGNI": {
        "codes": ["NY.ADJ.DFOR.GN.ZS"],
        "names": ["Adjusted savings: net forest depletion (% of GNI)"],
    },
    # Aggregate fallback (often ≈ energy + mineral + forest)
    "dep_natural_total_pctGNI": {
        "codes": ["NY.ADJ.DRES.GN.ZS"],
        "names": ["Adjusted savings: natural resources depletion (% of GNI)"],
    },
    # We intentionally DO NOT include the "current US$" variants here.
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
        raise RuntimeError("No DDR indicators found. Check 'Indicator Code/Name' in WBRAW.xlsx.")

    subset = pd.concat(found, ignore_index=True)

    # Console summary
    print("== Found DDR indicators ==")
    for std_key in CANDIDATES.keys():
        hit = subset.loc[subset["__std_key__"] == std_key]
        codes_preview = ", ".join(hit["Indicator Code"].dropna().astype(str).unique()[:3]) if not hit.empty else "-"
        print(f"  {std_key:26s}: {'OK' if not hit.empty else 'MISSING'}  (codes: {codes_preview})")

    # Melt → long
    id_vars = ["Country Name", "Country Code", "Indicator Name", "Indicator Code", "__std_key__"]
    long = subset.melt(id_vars=id_vars, value_vars=years, var_name="Year", value_name="Value")
    long["Year"] = long["Year"].astype(int)

    # Pivot to standardized keys
    panel = long.pivot_table(index=["Country Name", "Country Code", "Year"],
                             columns="__std_key__", values="Value", aggfunc="first").reset_index()
    panel = panel.sort_values(["Country Code", "Year"]).reset_index(drop=True)

    # Coverage summary
    coverage = compute_coverage_table(subset.drop(columns="__std_key__"))

    # Save
    engine = pick_writer_engine()
    with pd.ExcelWriter(OUT_EXTRACT, engine=engine) as xw:
        subset.to_excel(xw, sheet_name="original_raw_subset_DDR", index=False)
        panel.to_excel(xw, sheet_name="panel_raw_DDR", index=False)
        coverage.to_excel(xw, sheet_name="coverage_summary", index=False)

    print(f"[DDR Step 1] Saved extract to: {OUT_EXTRACT}")
    print("Sheets: original_raw_subset_DDR | panel_raw_DDR | coverage_summary")

if __name__ == "__main__":
    main()
