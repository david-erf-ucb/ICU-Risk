#!/usr/bin/env python3
"""
Query BigQuery for raw (charttime-level) vitals, interventions, and labs
for the 40 dashboard patients. Saves CSVs to dashboard/data/ for offline use.

The point of querying BigQuery rather than using MEEP parquets is to get
sub-hourly granularity -- individual measurements at the exact minute they
were charted -- so the dashboard time slider can simulate real-time arrival.

Requires: gcloud auth application-default login
Uses PROJECT_ID from constants.py for BigQuery billing.

Usage:
    python scripts/export_dashboard_events.py
    python scripts/export_dashboard_events.py --force   # re-pull even if cached
    python scripts/export_dashboard_events.py --n 20     # fewer patients
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from constants import PROJECT_ID

PREDICTIONS_CSV = os.path.join(
    os.path.dirname(__file__), "..", "METRE", "output", "benchmarks", "test_predictions.csv"
)
OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "METRE", "output", "benchmarks", "dashboard", "data"
)

# ── itemid mappings ──────────────────────────────────────────────────────────

VITAL_ITEMIDS = {
    220045: "heart_rate",
    220179: "sbp", 220050: "sbp",
    220180: "dbp", 220051: "dbp",
    220052: "mbp", 220181: "mbp", 225312: "mbp",
    220210: "resp_rate", 224690: "resp_rate",
    223761: "temperature_f", 223762: "temperature_c",
    220277: "spo2",
    225664: "glucose", 220621: "glucose", 226537: "glucose",
}
VITAL_ITEMID_STR = ",".join(str(i) for i in VITAL_ITEMIDS)

LAB_ITEMIDS = {
    50813: "lactate",
    50912: "creatinine",
    51301: "wbc",
    51265: "platelets",
    51222: "hemoglobin",
    50882: "bicarbonate",
    50931: "glucose_lab",
}
LAB_ITEMID_STR = ",".join(str(i) for i in LAB_ITEMIDS)

VASOPRESSOR_ITEMIDS = {
    221906: "norepinephrine",
    221289: "epinephrine",
    221662: "dopamine",
    222315: "vasopressin",
    221749: "phenylephrine",
    221653: "dobutamine",
}
VASOPRESSOR_ITEMID_STR = ",".join(str(i) for i in VASOPRESSOR_ITEMIDS)


def get_dashboard_stay_ids(predictions_csv, n=40):
    """Reproduce the same N-patient sample used by export_dashboard_data.py."""
    df = pd.read_csv(predictions_csv)
    n_sample = min(n, len(df))
    idx = np.random.RandomState(42).choice(len(df), size=n_sample, replace=False)
    sample = df.iloc[idx]
    return sample["stay_id"].astype(int).tolist()


def gcp2df(client, sql):
    return client.query(sql).result().to_dataframe()


# ── queries ──────────────────────────────────────────────────────────────────

def query_patient_info(client, stay_ids):
    ids = ",".join(str(s) for s in stay_ids)
    return gcp2df(client, f"""
        SELECT i.stay_id, i.subject_id, i.hadm_id,
               i.icu_intime, i.icu_outtime, i.los_icu
        FROM `physionet-data.mimiciv_3_1_derived.icustay_detail` i
        WHERE i.stay_id IN ({ids})
    """)


def query_vitals(client, stay_ids):
    """Raw chartevents -- one row per measurement at exact charttime."""
    ids = ",".join(str(s) for s in stay_ids)
    return gcp2df(client, f"""
        SELECT ce.stay_id, ce.charttime, ce.itemid, ce.valuenum,
               i.icu_intime,
               TIMESTAMP_DIFF(ce.charttime, i.icu_intime, MINUTE) AS minutes_since_admit
        FROM `physionet-data.mimiciv_3_1_icu.chartevents` ce
        INNER JOIN `physionet-data.mimiciv_3_1_derived.icustay_detail` i
            ON ce.stay_id = i.stay_id
        WHERE ce.stay_id IN ({ids})
          AND ce.itemid IN ({VITAL_ITEMID_STR})
          AND ce.charttime BETWEEN i.icu_intime AND i.icu_outtime
          AND ce.valuenum IS NOT NULL
        ORDER BY ce.stay_id, ce.charttime
    """)


def query_labs(client, stay_ids):
    """Raw labevents -- one row per lab result at exact charttime."""
    ids = ",".join(str(s) for s in stay_ids)
    return gcp2df(client, f"""
        SELECT le.subject_id, le.charttime, le.itemid, le.valuenum,
               i.stay_id, i.icu_intime,
               TIMESTAMP_DIFF(le.charttime, i.icu_intime, MINUTE) AS minutes_since_admit
        FROM `physionet-data.mimiciv_3_1_hosp.labevents` le
        INNER JOIN `physionet-data.mimiciv_3_1_derived.icustay_detail` i
            ON le.subject_id = i.subject_id
           AND le.charttime BETWEEN i.icu_intime AND i.icu_outtime
        WHERE i.stay_id IN ({ids})
          AND le.itemid IN ({LAB_ITEMID_STR})
          AND le.valuenum IS NOT NULL
        ORDER BY i.stay_id, le.charttime
    """)


def query_vasopressors(client, stay_ids):
    """Vasopressor infusions from inputevents -- start/end times + rate."""
    ids = ",".join(str(s) for s in stay_ids)
    return gcp2df(client, f"""
        SELECT ie.stay_id, ie.starttime, ie.endtime,
               ie.itemid, ie.amount, ie.rate, ie.rateuom,
               i.icu_intime,
               TIMESTAMP_DIFF(ie.starttime, i.icu_intime, MINUTE) AS start_minutes,
               TIMESTAMP_DIFF(ie.endtime, i.icu_intime, MINUTE) AS end_minutes
        FROM `physionet-data.mimiciv_3_1_icu.inputevents` ie
        INNER JOIN `physionet-data.mimiciv_3_1_derived.icustay_detail` i
            ON ie.stay_id = i.stay_id
        WHERE ie.stay_id IN ({ids})
          AND ie.itemid IN ({VASOPRESSOR_ITEMID_STR})
          AND ie.starttime BETWEEN i.icu_intime AND i.icu_outtime
        ORDER BY ie.stay_id, ie.starttime
    """)


def query_ventilation(client, stay_ids):
    """Ventilation periods from the derived ventilation table."""
    ids = ",".join(str(s) for s in stay_ids)
    return gcp2df(client, f"""
        SELECT v.stay_id, v.starttime, v.endtime, v.ventilation_status,
               i.icu_intime,
               TIMESTAMP_DIFF(v.starttime, i.icu_intime, MINUTE) AS start_minutes,
               TIMESTAMP_DIFF(v.endtime, i.icu_intime, MINUTE) AS end_minutes
        FROM `physionet-data.mimiciv_3_1_derived.ventilation` v
        INNER JOIN `physionet-data.mimiciv_3_1_derived.icustay_detail` i
            ON v.stay_id = i.stay_id
        WHERE v.stay_id IN ({ids})
          AND v.starttime < i.icu_outtime
          AND v.endtime > i.icu_intime
        ORDER BY v.stay_id, v.starttime
    """)


def query_antibiotics(client, stay_ids):
    """Antibiotic administrations from the derived antibiotic table."""
    ids = ",".join(str(s) for s in stay_ids)
    return gcp2df(client, f"""
        SELECT ab.subject_id, ab.antibiotic, ab.starttime, ab.stoptime,
               ab.route,
               i.stay_id, i.icu_intime,
               TIMESTAMP_DIFF(ab.starttime, i.icu_intime, MINUTE) AS start_minutes,
               TIMESTAMP_DIFF(ab.stoptime, i.icu_intime, MINUTE) AS end_minutes
        FROM `physionet-data.mimiciv_3_1_derived.antibiotic` ab
        INNER JOIN `physionet-data.mimiciv_3_1_derived.icustay_detail` i
            ON ab.hadm_id = i.hadm_id
        WHERE i.stay_id IN ({ids})
          AND ab.starttime < i.icu_outtime
          AND ab.stoptime > i.icu_intime
        ORDER BY i.stay_id, ab.starttime
    """)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Export raw BigQuery events for dashboard patients")
    parser.add_argument("--predictions", type=str, default=PREDICTIONS_CSV)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--project_id", type=str, default=PROJECT_ID)
    parser.add_argument("--n", type=int, default=40, help="Number of patients (must match dashboard sample)")
    parser.add_argument("--force", action="store_true", help="Re-query even if cached CSVs exist")
    args = parser.parse_args()

    if not os.path.exists(args.predictions):
        print(f"Error: {args.predictions} not found. Run training pipeline first.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    stay_ids = get_dashboard_stay_ids(args.predictions, n=args.n)
    print(f"Dashboard sample: {len(stay_ids)} stays")
    print(f"  stay_ids: {stay_ids[:5]}...") if len(stay_ids) > 5 else None

    files = {
        "patient_info.csv": query_patient_info,
        "vitals.csv": query_vitals,
        "labs.csv": query_labs,
        "vasopressors.csv": query_vasopressors,
        "ventilation.csv": query_ventilation,
        "antibiotics.csv": query_antibiotics,
    }

    cached = {f for f in files if os.path.exists(os.path.join(args.output_dir, f))}
    if cached and not args.force:
        print(f"\nAlready cached ({len(cached)}/{len(files)}): {', '.join(sorted(cached))}")
        files = {f: q for f, q in files.items() if f not in cached}
        if not files:
            print("All files cached. Use --force to re-pull.")
            return 0

    from google.cloud import bigquery
    os.environ["GOOGLE_CLOUD_PROJECT"] = args.project_id
    client = bigquery.Client(project=args.project_id)

    for filename, query_fn in files.items():
        label = filename.replace(".csv", "")
        print(f"\n  Querying {label}...")
        try:
            df = query_fn(client, stay_ids)
            path = os.path.join(args.output_dir, filename)
            df.to_csv(path, index=False)
            print(f"    Saved {filename} ({len(df):,} rows)")
        except Exception as e:
            print(f"    ERROR querying {label}: {e}")

    print(f"\nDone. Data cached in {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
