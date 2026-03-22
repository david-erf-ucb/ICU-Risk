#!/usr/bin/env python3
"""
Compare prediction features used by METRE vs Partner pipelines.

Creates a variable-level dataset with in_metre and in_partner flags.
Rows = each variable; columns = variable, in_metre, in_partner.

Usage:
    python compare_prediction_features.py --input_dir ../METRE/output
    python compare_prediction_features.py --input_dir /path/to/parquet/dir -o feature_comparison.csv
"""
import argparse
import ast
import os
import re
import sys

import pandas as pd

# METRE intervention columns (from compile_meep_to_npy.py)
INV_COLS = [
    "vent", "antibiotic", "dopamine", "epinephrine", "norepinephrine",
    "phenylephrine", "vasopressin", "dobutamine", "milrinone", "heparin",
    "crrt", "rbc_trans", "platelets_trans", "ffp_trans", "colloid_bolus", "crystalloid_bolus",
]

# Partner exclusions (from copy_of_overall_mimic_eda_3_1.py)
LEAKAGE_COLS = [
    "mort_icu", "mort_hosp", "hospital_expire_flag",
    "deathtime", "dischtime", "readmission_30",
    "los_icu", "los_hosp", "length_of_stay",
    "max_hours",
]
SUSPICIOUS_KEYWORDS = [
    "death", "expire", "discharge", "outtime", "los",
    "readmit", "post", "after", "total_stay", "final_",
]
ID_COLS = ["subject_id", "hadm_id", "stay_id", "tbin"]
LABEL_COL = "mort_icu"


def _col_to_var(c):
    """Convert column to normalized variable name for comparison."""
    if isinstance(c, tuple):
        return "_".join(str(x) for x in c if x)
    s = str(c)
    # Parse "('so2', 'mean')" -> "so2_mean"
    if s.startswith("(") and "'" in s:
        try:
            t = ast.literal_eval(s)
            if isinstance(t, tuple):
                return "_".join(str(x) for x in t if x)
        except (ValueError, SyntaxError):
            pass
    return s


def get_metre_variables(vital: pd.DataFrame) -> set:
    """METRE uses all vital columns + INV_COLS (200 total)."""
    vars_ = set()
    for c in vital.columns:
        vars_.add(_col_to_var(c))
    for c in INV_COLS:
        vars_.add(c)
    return vars_


def get_partner_variables(vital: pd.DataFrame, static: pd.DataFrame, inv: pd.DataFrame) -> set:
    """
    Simulate partner pipeline: merge vital+static+inv, collapse to 4h bins,
    flatten, exclude leakage/suspicious, drop count cols.
    Uses a small sample for speed (we only need column names).
    """
    # Sample to avoid loading full merge (faster)
    vital_reset = vital.reset_index()
    sample_stays = vital_reset["stay_id"].unique()[:500]
    vital_s = vital_reset[vital_reset["stay_id"].isin(sample_stays)]
    static_reset = static.reset_index()
    inv_reset = inv.reset_index()

    # Merge (same as partner)
    merged = pd.merge(
        vital_s, static_reset,
        on=["subject_id", "hadm_id", "stay_id"],
        how="left",
    )
    merged = pd.merge(
        merged, inv_reset,
        on=["subject_id", "hadm_id", "stay_id", "hours_in"],
        how="left",
    )

    # Collapse to 4h bins (hours 0-23)
    df = merged[merged["hours_in"].between(0, 23)].copy()
    df["tbin"] = (df["hours_in"] // 4).astype("int16")

    exclude = set(ID_COLS + [LABEL_COL, "hours_in"])
    feat_cols = [c for c in df.columns if c not in exclude]

    # Flatten column names
    def flatten_col(c):
        if isinstance(c, tuple):
            return "_".join(str(x) for x in c if x)
        s = str(c)
        if s.startswith("(") and "'" in s:
            try:
                t = ast.literal_eval(s)
                if isinstance(t, tuple):
                    return "_".join(str(x) for x in t if x)
            except (ValueError, SyntaxError):
                pass
        return s

    # Build binned (small sample for speed - we only need column names)
    numeric_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(df[c])]
    # Handle both tuple and string-repr columns like "('so2', 'mean')"
    def _is_mean_col(c):
        if isinstance(c, tuple) and len(c) == 2:
            return c[1] == "mean"
        s = str(c)
        return "mean" in s.lower() and "count" not in s.lower()
    def _is_count_col(c):
        if isinstance(c, tuple) and len(c) == 2:
            return c[1] == "count"
        return "count" in str(c).lower()
    mean_cols = [c for c in numeric_cols if _is_mean_col(c) and not _is_count_col(c)]
    count_cols = [c for c in numeric_cols if _is_count_col(c)]
    other_cols = [c for c in numeric_cols if c not in mean_cols and c not in count_cols]

    agg = {}
    for c in mean_cols:
        agg[c] = "mean"
    for c in count_cols:
        agg[c] = "sum"
    for c in other_cols:
        agg[c] = "mean"

    id_cols = ["subject_id", "hadm_id", "stay_id", "tbin"]
    binned = df.groupby(id_cols, sort=False).agg(agg).reset_index()

    # Flatten column names
    binned.columns = [flatten_col(c) for c in binned.columns]

    # Drop count columns
    count_cols_flat = [c for c in binned.columns if "_count" in str(c) or str(c).endswith("_count")]
    binned = binned.drop(columns=count_cols_flat, errors="ignore")

    # Apply partner's get_numeric_feature_cols_leakage_safe logic
    exclude = set(ID_COLS + [LABEL_COL] + LEAKAGE_COLS)
    candidates = [c for c in binned.columns if c not in exclude]
    numeric = [c for c in candidates if pd.api.types.is_numeric_dtype(binned[c])]

    def is_suspicious(name):
        n = str(name).lower()
        return any(k in n for k in SUSPICIOUS_KEYWORDS)

    safe = [c for c in numeric if not is_suspicious(c)]
    return set(safe)


def main():
    parser = argparse.ArgumentParser(
        description="Compare prediction features: METRE vs Partner"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "METRE", "output"),
        help="Directory containing MEEP_MIMIC_*.parquet files",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path to save variable-level dataset (CSV)",
    )
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    print(f"Loading MEEP parquet from: {input_dir}\n")

    vital_path = os.path.join(input_dir, "MEEP_MIMIC_vital.parquet")
    static_path = os.path.join(input_dir, "MEEP_MIMIC_static.parquet")
    inv_path = os.path.join(input_dir, "MEEP_MIMIC_inv.parquet")

    for p in [vital_path, static_path, inv_path]:
        if not os.path.exists(p):
            print(f"Error: {p} not found.")
            sys.exit(1)

    vital = pd.read_parquet(vital_path)
    static = pd.read_parquet(static_path)
    inv = pd.read_parquet(inv_path)

    metre_vars = get_metre_variables(vital)
    partner_vars = get_partner_variables(vital, static, inv)

    all_vars = sorted(metre_vars | partner_vars)

    rows = []
    for v in all_vars:
        rows.append({
            "variable": v,
            "in_metre": v in metre_vars,
            "in_partner": v in partner_vars,
        })

    df = pd.DataFrame(rows)

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Saved feature comparison to {args.output} ({len(df)} rows)\n")

    # Summary
    print("=" * 60)
    print("FEATURE COMPARISON: METRE vs Partner")
    print("=" * 60)
    print(f"\nTotal unique variables: {len(all_vars)}")
    print(f"METRE uses:   {len(metre_vars)}")
    print(f"Partner uses: {len(partner_vars)}")
    print(f"In both:     {len(metre_vars & partner_vars)}")
    print(f"METRE only:  {len(metre_vars - partner_vars)}")
    print(f"Partner only: {len(partner_vars - metre_vars)}")

    print("\n--- METRE-only variables (Partner excludes) ---")
    metre_only = sorted(metre_vars - partner_vars)
    for v in metre_only[:30]:
        print(f"  {v}")
    if len(metre_only) > 30:
        print(f"  ... and {len(metre_only) - 30} more")

    print("\n--- Partner-only variables (METRE does not use) ---")
    partner_only = sorted(partner_vars - metre_vars)
    for v in partner_only[:30]:
        print(f"  {v}")
    if len(partner_only) > 30:
        print(f"  ... and {len(partner_only) - 30} more")


if __name__ == "__main__":
    main()
