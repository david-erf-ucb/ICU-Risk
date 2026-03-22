#!/usr/bin/env python3
"""
Compare METRE hosp_mort vs partner mort_icu populations (prior to train/val/test split).

Creates a stay-level dataset with flags (in_metre, in_partner) and key columns (LOS, mort_hosp,
mort_icu) so you can inspect and recreate the comparison numbers.

METRE hosp_mort: LOS >= 28h, mort_hosp not NaN
Partner mort_icu: All stays with hours 0-23 (no LOS filter beyond MEEP 24-240h)

Usage:
    python compare_mortality_populations.py --input_dir ../METRE/output
    python compare_mortality_populations.py --input_dir /path/to/parquet/dir --output stays.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

# METRE hosp_mort filter params
THRESH = 24
GAP = 4
MIN_LOS_METRE = THRESH + GAP  # 28


def load_meep_parquet(input_dir: str):
    """Load MEEP MIMIC parquet files."""
    vital_path = os.path.join(input_dir, "MEEP_MIMIC_vital.parquet")
    static_path = os.path.join(input_dir, "MEEP_MIMIC_static.parquet")
    inv_path = os.path.join(input_dir, "MEEP_MIMIC_inv.parquet")

    for p in [vital_path, static_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Parquet not found: {p}\n"
                f"Run extract_database.py first to generate MEEP parquet files."
            )

    vital = pd.read_parquet(vital_path)
    static = pd.read_parquet(static_path)
    inv = pd.read_parquet(inv_path) if os.path.exists(inv_path) else None

    return vital, static, inv


def get_stay_level(vital: pd.DataFrame) -> str:
    """Return the stay-level index name (stay_id for MIMIC)."""
    if isinstance(vital.index, pd.MultiIndex):
        names = vital.index.names
        if "stay_id" in names:
            return "stay_id"
        # eICU uses patientunitstayid
        for n in names:
            if "stay" in str(n).lower() or "patient" in str(n).lower():
                return n
    return "stay_id"


def build_stay_level_dataset(vital: pd.DataFrame, static: pd.DataFrame) -> pd.DataFrame:
    """
    Build stay-level dataset with LOS, mort_hosp, mort_icu, and flags for each source.

    METRE: LOS >= 28h AND mort_hosp not NaN (replicates filter_los)
    Partner: All stays in vital (MEEP has LOS 24-240h; partner uses hours 0-23, no extra filter)
    """
    stay_level = get_stay_level(vital)
    vital_reset = vital.reset_index()

    # LOS = number of hours per stay
    los_per_stay = vital_reset.groupby(stay_level).size().rename("los")

    # Get subject_id, hadm_id from vital (guaranteed)
    id_cols = [c for c in ["subject_id", "hadm_id", stay_level] if c in vital_reset.columns]
    id_df = vital_reset[id_cols].drop_duplicates(subset=[stay_level])

    # Get mort_icu and mort_hosp from static
    static_reset = static.reset_index()
    mort_cols = [c for c in ["mort_icu", "mort_hosp"] if c in static_reset.columns]
    static_stays = static_reset[[stay_level] + mort_cols].drop_duplicates(subset=[stay_level])

    df = los_per_stay.reset_index().merge(id_df, on=stay_level, how="left")
    df = df.merge(static_stays, on=stay_level, how="left")

    # Flags: METRE = LOS>=28 and mort_hosp not NaN; Partner = all stays
    df["in_metre"] = (df["los"] >= MIN_LOS_METRE) & df["mort_hosp"].notna()
    df["in_partner"] = True  # Partner includes all MEEP stays (no filter beyond vital)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Compare METRE hosp_mort vs partner mort_icu populations"
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
        help="Path to save stay-level dataset (CSV). Columns: stay_id, LOS, mort_hosp, mort_icu, in_metre, in_partner",
    )
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    print(f"Loading MEEP parquet from: {input_dir}\n")

    try:
        vital, static, inv = load_meep_parquet(input_dir)
    except FileNotFoundError as e:
        print(str(e))
        sys.exit(1)

    stay_level = get_stay_level(vital)
    print(f"Vital shape: {vital.shape}, index: {vital.index.names}")
    print(f"Static shape: {static.shape}")
    print()

    # Build stay-level dataset with flags (prior to any train/val/test split)
    df = build_stay_level_dataset(vital, static)

    metre_df = df[df["in_metre"]]
    partner_df = df[df["in_partner"]]

    metre_stays = set(metre_df[stay_level].unique())
    partner_stays = set(partner_df[stay_level].unique())

    overlap = metre_stays & partner_stays
    metre_only = metre_stays - partner_stays
    partner_only = partner_stays - metre_stays

    # --- Save dataset ---
    if args.output:
        out_cols = [c for c in ["subject_id", "hadm_id", stay_level, "los", "mort_hosp", "mort_icu", "in_metre", "in_partner"] if c in df.columns]
        df_out = df[out_cols].copy()
        df_out.to_csv(args.output, index=False)
        print(f"Saved stay-level dataset to {args.output} ({len(df_out):,} rows)\n")

    # --- Print results (recreated from dataset) ---
    print("=" * 60)
    print("POPULATION COMPARISON (prior to train/val/test split)")
    print("=" * 60)

    print("\n--- Counts ---")
    print(f"METRE hosp_mort:     {len(metre_stays):,} stays")
    print(f"Partner mort_icu:    {len(partner_stays):,} stays")
    print(f"Overlap (both):     {len(overlap):,} stays")
    print(f"METRE only:         {len(metre_only):,} stays")
    print(f"Partner only:       {len(partner_only):,} stays")

    print("\n--- Mortality prevalence ---")
    m_pos = metre_df["mort_hosp"].sum()
    m_prev = 100 * m_pos / len(metre_df) if len(metre_df) > 0 else 0
    print(f"METRE:   mort_hosp positive = {m_pos:.0f} / {len(metre_df)} = {m_prev:.2f}%")

    if "mort_icu" in partner_df.columns:
        p_pos = partner_df["mort_icu"].sum()
        p_prev = 100 * p_pos / len(partner_df) if len(partner_df) > 0 else 0
        print(f"Partner: mort_icu positive = {p_pos:.0f} / {len(partner_df)} = {p_prev:.2f}%")

    # In overlap, compare mort_hosp vs mort_icu
    overlap_df = df[df[stay_level].isin(overlap)]
    if len(overlap_df) > 0 and "mort_icu" in overlap_df.columns:
        o_icu = overlap_df["mort_icu"].sum()
        o_hosp = overlap_df["mort_hosp"].sum()
        print(f"Overlap: mort_icu = {o_icu:.0f}, mort_hosp = {o_hosp:.0f}")

    print("\n--- LOS distribution ---")
    print("METRE (in_metre=True):")
    print(metre_df["los"].describe())
    print("\nPartner (in_partner=True):")
    print(partner_df["los"].describe())

    # Partner-only stays (excluded by METRE)
    if len(partner_only) > 0:
        partner_only_df = df[df[stay_level].isin(partner_only)]
        print("\nPartner-only stays (in_partner & ~in_metre):")
        print(partner_only_df["los"].describe())
        los_short = partner_only_df[partner_only_df["los"] < MIN_LOS_METRE]
        print(f"  Stays with LOS < 28h: {len(los_short)}")
        if "mort_icu" in partner_only_df.columns:
            p_only_prev = 100 * partner_only_df["mort_icu"].sum() / len(partner_only_df)
            print(f"  mort_icu prevalence in Partner-only: {p_only_prev:.2f}%")
        # Check for other exclusion reasons (NaN mort_hosp)
        nan_hosp = partner_only_df["mort_hosp"].isna().sum()
        if nan_hosp > 0:
            print(f"  Stays with NaN mort_hosp: {nan_hosp}")

    print("\n--- Attribution summary ---")
    print("Why populations differ:")
    print("  1. LOS filter: METRE requires LOS >= 28h; Partner has no filter (MEEP has 24-240h).")
    print(f"  2. Partner includes {len(partner_only)} stays that METRE excludes.")
    los_reason = (df[~df["in_metre"]]["los"] < MIN_LOS_METRE).sum()
    nan_reason = df[~df["in_metre"]]["mort_hosp"].isna().sum()
    print(f"  3. Exclusion reasons: LOS < 28h: {los_reason}, NaN mort_hosp: {nan_reason}")
    print("  4. Outcome: METRE uses mort_hosp (in-hospital); Partner uses mort_icu (in-ICU).")
    print("\nPotential leakage drivers for Partner overperformance:")
    print("  - LOS 24-27h stays may have different (easier?) mortality predictability.")
    print("  - mort_icu is narrower than mort_hosp; different prevalence/signal.")
    print("  - Short-stay deaths may be more predictable from early vitals.")


if __name__ == "__main__":
    main()
