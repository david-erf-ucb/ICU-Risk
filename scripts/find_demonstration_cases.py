#!/usr/bin/env python3
"""
Find patients who look similar and normal at entry, but some die and some survive.
Goal: demonstrate model value by showing it predicted deaths among "looks fine" patients.

Uses METRE hosp_mort predictions (first 24h of data). Defines "normal at entry" as
baseline vitals (first 8h mean) within physiologic ranges.

Usage:
    python find_demonstration_cases.py
    python find_demonstration_cases.py -o METRE/output/demonstration_cases.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

# Physiologic "normal" ranges for first 8h mean (tighter = more clearly stable at entry)
NORMAL_RANGES = {
    "heart_rate": (60, 100),
    "sbp": (90, 140),
    "resp_rate": (12, 22),
    "so2": (95, 100),
    "temperature": (36.0, 37.5),
}

VITAL_KEYS = ["heart_rate", "sbp", "resp_rate", "so2", "temperature"]


def _find_vital_col(df, key):
    """Find column matching key."""
    for c in df.columns:
        s = str(c)
        if isinstance(key, tuple):
            if key[0] in s and (len(key) == 1 or key[1] in s):
                return c
        elif key in s and "mean" in s and "count" not in s:
            return c
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--meep_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "METRE", "output"),
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "METRE", "output", "benchmarks", "test_predictions.csv"),
    )
    parser.add_argument("-o", "--output", type=str, default=None)
    parser.add_argument("-n", "--top_n", type=int, default=10)
    args = parser.parse_args()

    if not os.path.exists(args.predictions):
        print(f"Error: {args.predictions} not found. Run export_predictions.py first.")
        sys.exit(1)

    pred = pd.read_csv(args.predictions)
    # hosp_mort task only
    pred = pred[pred["hosp_mort_24h_gap4h_gt"].notna()].copy()
    pred = pred.rename(columns={
        "hosp_mort_24h_gap4h_gt": "mort",
        "hosp_mort_24h_gap4h_LR_prob": "prob_lr",
        "hosp_mort_24h_gap4h_RF_prob": "prob_rf",
    })

    vital_path = os.path.join(args.meep_dir, "MEEP_MIMIC_vital.parquet")
    if not os.path.exists(vital_path):
        print(f"Error: {vital_path} not found.")
        sys.exit(1)

    vital = pd.read_parquet(vital_path)
    vital_reset = vital.reset_index()

    # Get baseline (first 8h mean) for key vitals
    baseline = vital_reset[vital_reset["hours_in"] < 8].copy()
    stay_ids = set(pred["stay_id"].astype(int))
    baseline = baseline[baseline["stay_id"].isin(stay_ids)]

    vital_col_map = {}
    for k in VITAL_KEYS:
        c = _find_vital_col(vital, k)
        if c is not None:
            vital_col_map[k] = c

    if len(vital_col_map) < 3:
        print("Warning: Could not find enough vital columns. Using available ones.")

    agg_dict = {vital_col_map[k]: "mean" for k in vital_col_map}
    baseline_agg = baseline.groupby("stay_id").agg(agg_dict).reset_index()
    baseline_agg.columns = ["stay_id"] + list(vital_col_map.keys())

    # Merge with predictions
    pred = pred.merge(baseline_agg, on="stay_id", how="inner")

    # Filter to "normal at entry" - all key vitals in range
    mask = pd.Series(True, index=pred.index)
    for k, (lo, hi) in NORMAL_RANGES.items():
        if k in pred.columns:
            mask &= pred[k].between(lo, hi, inclusive="both")

    normal = pred[mask].copy()
    if len(normal) == 0:
        print("No stays with all vitals in normal range. Relaxing criteria...")
        # Relax: at least 3 vitals in range
        for k, (lo, hi) in NORMAL_RANGES.items():
            if k in pred.columns:
                pred[f"{k}_ok"] = pred[k].between(lo, hi, inclusive="both")
        n_ok = pred[[f"{k}_ok" for k in VITAL_KEYS if f"{k}_ok" in pred.columns]].sum(axis=1)
        normal = pred[n_ok >= 3].copy()

    # Need both deaths and survivors in this group
    n_died = normal["mort"].sum()
    n_survived = len(normal) - n_died
    if n_died == 0 or n_survived == 0:
        print("No mixed outcomes in normal-at-entry group. Try relaxing NORMAL_RANGES.")
        sys.exit(1)

    # Sort by predicted prob (RF) - deaths should rank higher if model works
    normal = normal.sort_values("prob_rf", ascending=False).reset_index(drop=True)

    # Select top demonstration cases: mix of deaths (predicted high) and survivors (predicted low)
    died = normal[normal["mort"] == 1]
    survived = normal[normal["mort"] == 0]
    n_show = min(args.top_n, len(died), len(survived))
    # Top predicted deaths that actually died
    n_deaths = (n_show + 1) // 2
    n_survivors = n_show - n_deaths
    top_deaths = died.head(n_deaths)
    # Survivors with lowest predicted risk (model correctly said low risk)
    low_risk_survivors = survived.tail(n_survivors)
    demo = pd.concat([top_deaths, low_risk_survivors]).sort_values("prob_rf", ascending=False)

    print("=" * 60)
    print("DEMONSTRATION CASES: Normal at entry, mixed outcomes")
    print("=" * 60)
    print(f"\nStays with 'normal' baseline vitals (first 8h): {len(normal)}")
    print(f"  Died: {n_died}, Survived: {n_survived}")
    print(f"\nSelected {len(demo)} cases (deaths model predicted high, survivors predicted low):")
    print(demo[["stay_id", "age", "mort", "prob_lr", "prob_rf"] + [c for c in VITAL_KEYS if c in demo.columns]].to_string(index=False))

    # Check: did model discriminate?
    auc_deaths = died["prob_rf"].mean()
    auc_surv = survived["prob_rf"].mean()
    print(f"\nAmong normal-at-entry: mean RF prob for deaths={auc_deaths:.3f}, survivors={auc_surv:.3f}")

    if args.output:
        demo.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
