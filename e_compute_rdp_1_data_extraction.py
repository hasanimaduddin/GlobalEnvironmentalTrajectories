# -*- coding: utf-8 -*-
"""
RDP Step 1 — Extract by Indicator *Name* (PROXY water version)
Targets:
  - CO2_pc           -> "Carbon dioxide (CO2) emissions excluding LULUCF per capita (t CO2e/capita)"
  - Renewable_share  -> "Renewable energy consumption (% of total final energy consumption)"
  - Water_avail_pc   -> "Renewable internal freshwater resources per capita (cubic meters)"
  - Forest_area      -> "Forest area (% of land area)"

Outputs:
  rdp_step1_extract.xlsx with:
    • original_raw_subset_RDP
    • panel_raw_RDP
    • coverage_summary
"""

import os
import pandas as pd

# ========================
# CONFIG
# ========================
BASE = r"C:\Users\LEGION\Documents\Paper3 Intergenerational Environmental Efficiency"
INPUT_EXCEL = os.path.join(BASE, "WBRAW.xlsx")
INPUT_SHEET = "Data"

OUT_DIR = os.path.join(BASE, r"Analysis_12082025")
OUT_EXTRACT = os.path.join(OUT_DIR, "rdp_step1_extract.xlsx")

# ---- Indicator Name candidates (NAMES only) ----
NAME_CANDIDATES = {
    "CO2_pc": [
        "Carbon dioxide (CO2) emissions excluding LULUCF per capita (t CO2e/capita)",
        # common minor variants
        "Carbon dioxide (CO2) emissions excluding LULUCF per capita (t CO2e / capita)",
        "CO2 emissions excluding LULUCF per capita (t CO2e/capita)",
    ],
    "Renewable_share": [
        "Renewable energy consumption (% of total final energy consumption)",
        "Renewable energy consumption (% of total final energy consumption) ",
    ],
    # PROXY water availability (higher = more headroom)
    "Water_avail_pc": [
        "Renewable internal freshwater resources per capita (cubic meters)",
        "Renewable internal freshwater resources per capita (cubic meters) ",
        "Renewable internal freshwater resources per capita",
    ],
    "Forest_area": [
        "Forest area (% of land area)",
    ],
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

    # Filter rows by Indicator Name candidates
    mask_any = False
    for _, name_list in NAME_CANDIDATES.items():
        m = raw["Indicator Name"].isin(name_list)
        mask_any = m if mask_any is False else (mask_any | m)
    subset = raw.loc[mask_any].copy()

    if subset.empty:
        raise RuntimeError("No RDP indicators found by Indicator Name. Check names in WBRAW.xlsx > 'Data'.")

    # Attach standardized keys
    def std_key_from_name(name: str) -> str:
        for std_key, name_list in NAME_CANDIDATES.items():
            if name in name_list:
                return std_key
        return "UNKNOWN"

    subset["__std_key__"] = subset["Indicator Name"].apply(std_key_from_name)

    # Melt → long
    id_vars = ["Country Name", "Country Code", "Indicator Name", "Indicator Code", "__std_key__"]
    long = subset.melt(id_vars=id_vars, value_vars=years, var_name="Year", value_name="Value")
    long["Year"] = long["Year"].astype(int)

    # Pivot → tidy panel
    panel = long.pivot_table(index=["Country Name", "Country Code", "Year"],
                             columns="__std_key__", values="Value", aggfunc="first").reset_index()
    panel = panel.sort_values(["Country Code", "Year"]).reset_index(drop=True)

    # Coverage summary
    coverage = compute_coverage_table(subset.drop(columns="__std_key__"))

    # Save
    engine = pick_writer_engine()
    with pd.ExcelWriter(OUT_EXTRACT, engine=engine) as xw:
        subset.to_excel(xw, sheet_name="original_raw_subset_RDP", index=False)
        panel.to_excel(xw, sheet_name="panel_raw_RDP", index=False)
        coverage.to_excel(xw, sheet_name="coverage_summary", index=False)

    # Console summary
    print("== Found RDP indicators (by name) ==")
    for k in NAME_CANDIDATES.keys():
        hit = subset.loc[subset["__std_key__"] == k]
        print(f"  {k:14s}: {'OK' if not hit.empty else 'MISSING'}")
    print(f"[RDP Step 1] Saved extract to: {OUT_EXTRACT}")
    print("Sheets: original_raw_subset_RDP | panel_raw_RDP | coverage_summary")

if __name__ == "__main__":
    main()
