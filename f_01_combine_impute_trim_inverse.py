# -*- coding: utf-8 -*-
"""
Five-lens panel assembly (PC, DM, RE, DDR, RDP)
- merges by Country Code + Year
- interior interpolation + forward-only tail extrapolation (no head fill)
- drops aggregates/territories
- strict completeness (all five present)
- adds DDR_inverted_raw (no scaling yet)
- POST: cut to 1995–2024 and KEEP ONLY countries with full coverage in that window
"""

import os
import numpy as np
import pandas as pd
from functools import reduce

BASE = r"C:\Users\LEGION\Documents\Paper3 Intergenerational Environmental Efficiency\Analysis102025"

FILES = {
    # lens : path, sheet, column, final_name
    "PC":  dict(path=os.path.join(BASE, "pc_step3_pc_scores_guided.xlsx"),
                sheet="pc_capacity_raw",
                col="PC_capacity_pc_raw",
                out="PC"),
    "DM":  dict(path=os.path.join(BASE, "dm_step3_scores_extended.xlsx"),
                sheet="dm_momentum_raw_ext",
                col="DM_smooth",
                out="DM"),
    "RE":  dict(path=os.path.join(BASE, "re_step3_scores.xlsx"),
                sheet="re_composite_raw",
                col="RE_composite_raw",
                out="RE"),
    "DDR": dict(path=os.path.join(BASE, "ddr_step3_scores.xlsx"),
                sheet="ddr_burden_raw",
                col="DDR_burden_pct_raw",
                out="DDR"),
    "RDP": dict(path=os.path.join(BASE, "rdp_step3_scores.xlsx"),
                sheet="rdp_headroom_raw",
                col="RDP_headroom_raw",
                out="RDP"),
}

OUT_FILE = os.path.join(BASE, "combined_panel_raw_imputed.xlsx")

# ---- POST-PROCESSING WINDOW (editable) ----
YEAR_MIN, YEAR_MAX = 1995, 2024

EXCLUDE_AGGREGATES  = True
EXCLUDE_TERRITORIES = True

AGGREGATES = {
    "Africa Eastern and Southern","Africa Western and Central","Arab World",
    "Caribbean small states","Central Europe and the Baltics","Early-demographic dividend",
    "East Asia & Pacific","East Asia & Pacific (IDA & IBRD countries)","East Asia & Pacific (excluding high income)",
    "Euro area","Europe & Central Asia","Europe & Central Asia (IDA & IBRD countries)",
    "Europe & Central Asia (excluding high income)","European Union",
    "Fragile and conflict affected situations","Heavily indebted poor countries (HIPC)",
    "High income","IBRD only","IDA & IBRD total","IDA blend","IDA only","IDA total",
    "Late-demographic dividend","Latin America & Caribbean",
    "Latin America & Caribbean (excluding high income)",
    "Latin America & the Caribbean (IDA & IBRD countries)",
    "Least developed countries: UN classification","Low & middle income","Low income","Lower middle income",
    "Middle East, North Africa, Afghanistan & Pakistan",
    "Middle East, North Africa, Afghanistan & Pakistan (IDA & IBRD)",
    "Middle East, North Africa, Afghanistan & Pakistan (excluding high income)",
    "Middle income","North America","OECD members","Other small states",
    "Pacific island small states","Post-demographic dividend","Pre-demographic dividend",
    "Small states","South Asia","South Asia (IDA & IBRD)",
    "Sub-Saharan Africa","Sub-Saharan Africa (IDA & IBRD countries)",
    "Sub-Saharan Africa (excluding high income)","Upper middle income","World"
}
TERRITORIES = {
    "American Samoa","Anguilla","Aruba","Bermuda","British Virgin Islands","Cayman Islands",
    "China, Hong Kong SAR","Hong Kong SAR, China","China, Macao SAR","Macao SAR, China",
    "Curacao","Curaçao","Faroe Islands","French Polynesia","Gibraltar","Greenland",
    "Guam","Isle of Man","Montserrat","New Caledonia","Northern Mariana Islands",
    "Puerto Rico (US)","Sint Maarten (Dutch part)","St. Martin (French part)",
    "Turks and Caicos Islands","Virgin Islands (U.S.)","West Bank and Gaza",
    "U.R. of Tanzania: Mainland","Channel Islands"
}

def pick_writer_engine():
    try:
        import xlsxwriter  # noqa
        return "xlsxwriter"
    except Exception:
        return "openpyxl"

def linear_interpolate_and_tail_extrapolate(years_idx: pd.Index, values: pd.Series):
    """Interior linear interpolation + forward-only tail; NO head imputation."""
    s = pd.to_numeric(values, errors="coerce").astype(float).copy()
    s_interp = s.interpolate(method="linear", limit_direction="both")

    # remove any head fill
    first_valid = s.first_valid_index()
    if first_valid is not None:
        s_interp.loc[years_idx[years_idx < first_valid]] = np.nan

    s = s_interp
    last_valid = s.last_valid_index()
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
    return s

def load_lens(spec):
    df = pd.read_excel(spec["path"], sheet_name=spec["sheet"], dtype={"Country Code": str})
    keep = [c for c in ["Country Name","Country Code","Year", spec["col"]] if c in df.columns]
    df = df[keep].copy()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df[spec["out"]] = pd.to_numeric(df[spec["col"]], errors="coerce")
    df = df.drop_duplicates(subset=["Country Code","Year"])
    return df[["Country Name","Country Code","Year", spec["out"]]]

def main():
    # Load each lens
    frames = []
    name_ref = None
    for key, spec in FILES.items():
        d = load_lens(spec)
        if name_ref is None:
            name_ref = d[["Country Code","Country Name"]].dropna().drop_duplicates("Country Code")
        frames.append(d.drop(columns=["Country Name"]))

    # Merge on Country Code + Year
    panel = reduce(lambda L,R: pd.merge(L, R, on=["Country Code","Year"], how="outer"), frames)
    # Attach one Country Name mapping
    panel = panel.merge(name_ref, on="Country Code", how="left")
    panel = panel[["Country Name","Country Code","Year","PC","DM","RE","DDR","RDP"]]

    # Clean aggregates/territories
    if EXCLUDE_AGGREGATES or EXCLUDE_TERRITORIES:
        mask = pd.Series(True, index=panel.index)
        if EXCLUDE_AGGREGATES:  mask &= ~panel["Country Name"].isin(AGGREGATES)
        if EXCLUDE_TERRITORIES: mask &= ~panel["Country Name"].isin(TERRITORIES)
        panel = panel.loc[mask].copy()

    # Time-series imputation (interior + tail; no head) per country per lens
    for lens in ["PC","DM","RE","DDR","RDP"]:
        parts = []
        for cc, g in panel.groupby("Country Code", sort=False):
            g = g.sort_values("Year")
            yrs = pd.Index(sorted(g["Year"].dropna().astype("Int64").unique()))
            filled = linear_interpolate_and_tail_extrapolate(
                yrs, g.set_index("Year")[lens].reindex(yrs)
            )
            parts.append(pd.DataFrame({"Country Code": cc, "Year": yrs, lens: filled.values}))
        filled_df = pd.concat(parts, ignore_index=True)
        panel = panel.drop(columns=[lens]).merge(filled_df, on=["Country Code","Year"], how="left")

    # Strict completeness
    before = len(panel)
    complete = panel.dropna(subset=["PC","DM","RE","DDR","RDP"]).copy()
    after = len(complete)

    # DDR inversion
    complete["DDR_inverted_raw"] = -complete["DDR"]

    # Coverage prints
    print("[Combine] rows before strict completeness:", before)
    print("[Combine] rows after strict completeness :", after)
    print("Per-lens non-missing counts (before strict drop):")
    for col in ["PC","DM","RE","DDR","RDP"]:
        print(f"  {col}: {int(panel[col].notna().sum())}")

    # --- POST-PROCESS: cut to 1995–2024 and keep only countries with FULL coverage in that window ---
    YEAR_RANGE = pd.Index(range(YEAR_MIN, YEAR_MAX + 1), dtype="Int64")
    N_YEARS_REQUIRED = len(YEAR_RANGE)

    complete_cut = complete.loc[complete["Year"].between(YEAR_MIN, YEAR_MAX)].copy()
    year_counts = complete_cut.groupby("Country Code")["Year"].nunique()
    full_countries = year_counts[year_counts == N_YEARS_REQUIRED].index

    final = (
        complete_cut[complete_cut["Country Code"].isin(full_countries)]
        .sort_values(["Country Code", "Year"])
        .copy()
    )

    print(f"[Post] Year window: {YEAR_MIN}-{YEAR_MAX} (N_years_required={N_YEARS_REQUIRED})")
    print(f"[Post] Countries with FULL coverage in window: {len(full_countries)}")
    print(f"[Post] Final rows to save: {len(final)}")

    # Save (write BOTH: original 'complete' and filtered 'final')
    engine = pick_writer_engine()
    with pd.ExcelWriter(OUT_FILE, engine=engine) as xw:
        # Original strict-complete panel (all available years) — for reference
        complete.sort_values(["Country Code","Year"]).to_excel(
            xw, sheet_name="panel_raw_imputed", index=False
        )

        # Ready-to-process panel: 1995–2024, full-country coverage only
        final.to_excel(
            xw, sheet_name="panel_ready_1995_2024_full", index=False
        )

        # Coverage by year for the final panel (should be flat = #full_countries)
        (
            final.groupby("Year").size().reset_index(name="N_countries_final")
        ).to_excel(xw, sheet_name="coverage_by_year_final", index=False)

        # Per-lens availability before strict drop (for transparency)
        (
            panel.groupby("Year")[["PC","DM","RE","DDR","RDP"]]
            .apply(lambda d: d.notna().sum()).reset_index()
        ).to_excel(xw, sheet_name="coverage_by_lens", index=False)

        # Parameter log
        pd.DataFrame({
            "param": [
                "PC_source_sheet","PC_source_file","IMPUTATION","HEAD_POLICY","TAIL_POLICY",
                "EXCLUDE_AGGREGATES","EXCLUDE_TERRITORIES","DDR_INVERTED_COL","SCALING",
                "YEAR_MIN","YEAR_MAX","KEEP_ONLY_FULL_COUNTRIES","N_YEARS_REQUIRED",
                "N_COUNTRIES_FINAL"
            ],
            "value": [
                "pc_capacity_raw", os.path.basename(FILES["PC"]["path"]),
                "linear (interior) + forward-tail (slope of last two points)",
                "NO HEAD IMPUTATION","Forward-only extrapolation",
                str(EXCLUDE_AGGREGATES), str(EXCLUDE_TERRITORIES),
                "DDR_inverted_raw","NOT APPLIED",
                str(YEAR_MIN), str(YEAR_MAX), "True", str(N_YEARS_REQUIRED),
                str(len(full_countries))
            ]
        }).to_excel(xw, sheet_name="method_params", index=False)

    print("Saved:", OUT_FILE)

if __name__ == "__main__":
    main()
