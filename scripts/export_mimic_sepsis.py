#!/usr/bin/env python3
"""
Export MIMIC-IV sepsis data from BigQuery.

Queries physionet-data.mimiciv_3_1_derived.sepsis3 and produces:
  1. mimic_sepsis_stays.csv — stay_id, sepsis (minimal)
  2. mimic_sepsis_full.csv — full sepsis3 table (all columns)
  3. mimic_sepsis_all_stays.csv — all ICU stays with sepsis indicator (optional)

Requires: gcloud auth application-default login
Uses PROJECT_ID from constants.py for BigQuery billing.

Usage:
    python export_mimic_sepsis.py
    python export_mimic_sepsis.py --output_dir METRE/output
    python export_mimic_sepsis.py --all_stays  # include non-sepsis stays with sepsis=0
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from constants import PROJECT_ID

import pandas as pd
from google.cloud import bigquery


def gcp2df(client, sql):
    return client.query(sql).result().to_dataframe()


def main():
    parser = argparse.ArgumentParser(description="Export MIMIC sepsis data from BigQuery")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "METRE", "output"),
    )
    parser.add_argument(
        "--project_id",
        type=str,
        default=PROJECT_ID,
    )
    parser.add_argument(
        "--all_stays",
        action="store_true",
        help="Also produce file with ALL ICU stays and sepsis indicator (0/1)",
    )
    parser.add_argument(
        "--sofa",
        action="store_true",
        help="Also pull SOFA distribution by stay_id (from sofa table)",
    )
    args = parser.parse_args()

    os.environ["GOOGLE_CLOUD_PROJECT"] = args.project_id
    client = bigquery.Client(project=args.project_id)

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Query sepsis3 table (use SELECT * to handle schema variations)
    print("Querying sepsis3 table...")
    query_full = """
    SELECT *
    FROM physionet-data.mimiciv_3_1_derived.sepsis3
    ORDER BY stay_id
    """
    sepsis_full = gcp2df(client, query_full)
    sepsis_full["sepsis"] = 1  # All rows in sepsis3 are sepsis cases

    # 2. Minimal file: stay_id, sepsis
    sepsis_minimal = sepsis_full[["stay_id", "sepsis"]].drop_duplicates()
    minimal_path = os.path.join(args.output_dir, "mimic_sepsis_stays.csv")
    sepsis_minimal.to_csv(minimal_path, index=False)
    print(f"Saved {minimal_path} ({len(sepsis_minimal):,} sepsis stays)")

    # 3. Full table
    full_path = os.path.join(args.output_dir, "mimic_sepsis_full.csv")
    sepsis_full.to_csv(full_path, index=False)
    print(f"Saved {full_path} ({len(sepsis_full):,} rows)")

    # 4. Optional: all ICU stays with sepsis indicator
    if args.all_stays:
        print("Querying all ICU stays for sepsis indicator...")
        query_all = """
        WITH sepsis_stays AS (
            SELECT stay_id, 1 AS sepsis
            FROM physionet-data.mimiciv_3_1_derived.sepsis3
        )
        SELECT
            i.stay_id,
            COALESCE(s.sepsis, 0) AS sepsis
        FROM physionet-data.mimiciv_3_1_derived.icustay_detail i
        LEFT JOIN sepsis_stays s ON i.stay_id = s.stay_id
        WHERE i.stay_id IS NOT NULL
        ORDER BY i.stay_id
        """
        all_stays = gcp2df(client, query_all)
        all_path = os.path.join(args.output_dir, "mimic_sepsis_all_stays.csv")
        all_stays.to_csv(all_path, index=False)
        n_sepsis = all_stays["sepsis"].sum()
        print(f"Saved {all_path} ({len(all_stays):,} stays, {n_sepsis:,} sepsis)")

    # 5. Optional: SOFA distribution by stay
    if args.sofa:
        print("Querying SOFA table (may take a few minutes)...")
        # SOFA table has one row per 24h window per stay - multiple rows per stay
        query_sofa = """
        SELECT *
        FROM physionet-data.mimiciv_3_1_derived.sofa
        ORDER BY stay_id, starttime
        """
        sofa_raw = gcp2df(client, query_sofa)
        # Find sofa score column (may be sofa_24hours or similar)
        sofa_col = next((c for c in sofa_raw.columns if "sofa" in c.lower() and "score" not in c.lower()), None)
        if sofa_col is None:
            sofa_col = [c for c in sofa_raw.columns if "sofa" in c.lower()][0] if any("sofa" in c.lower() for c in sofa_raw.columns) else sofa_raw.columns[-1]
        sofa_raw = sofa_raw.rename(columns={sofa_col: "sofa_score"})
        sofa_path = os.path.join(args.output_dir, "mimic_sofa_by_window.csv")
        sofa_raw.to_csv(sofa_path, index=False)
        print(f"Saved {sofa_path} ({len(sofa_raw):,} rows)")

        # Per-stay summary: count, min, max, mean SOFA
        sofa_summary = sofa_raw.groupby("stay_id").agg(
            n_windows=("sofa_score", "count"),
            sofa_min=("sofa_score", "min"),
            sofa_max=("sofa_score", "max"),
            sofa_mean=("sofa_score", "mean"),
        ).reset_index()
        summary_path = os.path.join(args.output_dir, "mimic_sofa_by_stay.csv")
        sofa_summary.to_csv(summary_path, index=False)
        print(f"Saved {summary_path} ({len(sofa_summary):,} stays)")

        # Distribution of SOFA scores (across all windows)
        sofa_dist = sofa_raw["sofa_score"].value_counts().sort_index().reset_index()
        sofa_dist.columns = ["sofa_score", "count"]
        dist_path = os.path.join(args.output_dir, "mimic_sofa_score_distribution.csv")
        sofa_dist.to_csv(dist_path, index=False)
        print(f"Saved {dist_path} (SOFA score distribution)")

        # Rows per stay distribution
        stays_per_row = sofa_summary["n_windows"].value_counts().sort_index().reset_index()
        stays_per_row.columns = ["n_windows", "n_stays"]
        rows_dist_path = os.path.join(args.output_dir, "mimic_sofa_windows_per_stay.csv")
        stays_per_row.to_csv(rows_dist_path, index=False)
        print(f"Saved {rows_dist_path} (windows per stay distribution)")

    print(f"\nSepsis data saved to {args.output_dir}")


if __name__ == "__main__":
    main()
