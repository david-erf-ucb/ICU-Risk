#!/usr/bin/env python3
"""
Query BigQuery for raw data of demonstration case stays and save to case_study folder.

Requires: gcloud auth application-default login
Uses PROJECT_ID from constants.py for BigQuery billing.

Usage:
    python export_case_study_to_bigquery.py
    python export_case_study_to_bigquery.py --cases METRE/output/demonstration_cases.csv
"""
import argparse
import os
import sys

# Add project root for constants
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from constants import PROJECT_ID

import pandas as pd
from google.cloud import bigquery


def gcp2df(client, sql):
    return client.query(sql).result().to_dataframe()


def query_patient_info(client, stay_ids):
    """Get icustay_detail + admissions for case study stays."""
    ids = ",".join(str(int(s)) for s in stay_ids)
    query = f"""
    SELECT
        i.subject_id, i.hadm_id, i.stay_id,
        i.gender, i.admission_age as age, i.race,
        i.hospital_expire_flag, i.los_icu,
        i.admittime, i.dischtime, i.icu_intime, i.icu_outtime,
        a.admission_type, a.insurance, a.deathtime, a.discharge_location,
        CASE WHEN a.deathtime BETWEEN i.icu_intime AND i.icu_outtime THEN 1 ELSE 0 END AS mort_icu,
        CASE WHEN a.deathtime BETWEEN i.admittime AND i.dischtime THEN 1 ELSE 0 END AS mort_hosp
    FROM physionet-data.mimiciv_3_1_derived.icustay_detail i
    INNER JOIN physionet-data.mimiciv_3_1_hosp.admissions a ON i.hadm_id = a.hadm_id
    WHERE i.stay_id IN ({ids})
    ORDER BY i.stay_id
    """
    return gcp2df(client, query)


def query_vitals(client, stay_ids):
    """Get aggregated vitals (heart_rate, sbp, dbp, resp_rate, temperature, spo2, glucose) per charttime."""
    ids = ",".join(str(int(s)) for s in stay_ids)
    query = f"""
    WITH vitalsign AS (
        SELECT
            ce.subject_id, ce.stay_id, ce.charttime,
            AVG(CASE WHEN itemid IN (220045) AND valuenum > 0 AND valuenum < 9999 THEN valuenum END) AS heart_rate,
            AVG(CASE WHEN itemid IN (220179,220050) AND valuenum > 0 AND valuenum < 9999 THEN valuenum END) AS sbp,
            AVG(CASE WHEN itemid IN (220180,220051) AND valuenum > 0 AND valuenum < 9999 THEN valuenum END) AS dbp,
            AVG(CASE WHEN itemid IN (220052,220181,225312) AND valuenum > 0 AND valuenum < 9999 THEN valuenum END) AS mbp,
            AVG(CASE WHEN itemid IN (220210,224690) AND valuenum > 0 AND valuenum < 9999 THEN valuenum END) AS resp_rate,
            ROUND(AVG(CASE WHEN itemid IN (223761) AND valuenum > 70 AND valuenum < 120 THEN (valuenum-32)/1.8
                          WHEN itemid IN (223762) AND valuenum > 10 AND valuenum < 50 THEN valuenum END), 2) AS temperature,
            AVG(CASE WHEN itemid IN (220277) AND valuenum > 0 AND valuenum <= 100 THEN valuenum END) AS spo2,
            AVG(CASE WHEN itemid IN (225664,220621,226537) AND valuenum > 0 THEN valuenum END) AS glucose
        FROM physionet-data.mimiciv_3_1_icu.chartevents ce
        WHERE ce.stay_id IN ({ids})
        AND ce.itemid IN (220045,220179,220180,220181,220050,220051,220052,225312,220210,224690,220277,225664,220621,226537,223761,223762)
        GROUP BY ce.subject_id, ce.stay_id, ce.charttime
    )
    SELECT v.*, i.icu_intime,
           TIMESTAMP_DIFF(v.charttime, i.icu_intime, HOUR) AS hours_since_icu_admit
    FROM vitalsign v
    INNER JOIN physionet-data.mimiciv_3_1_derived.icustay_detail i ON v.stay_id = i.stay_id
    WHERE v.charttime BETWEEN i.icu_intime AND i.icu_outtime
    ORDER BY v.stay_id, v.charttime
    """
    return gcp2df(client, query)


def query_raw_chartevents(client, stay_ids, itemids=None):
    """Get raw chartevents for key vitals (optional - for full granularity)."""
    ids = ",".join(str(int(s)) for s in stay_ids)
    if itemids is None:
        itemids = "220045,220179,220180,220181,220050,220051,220052,225312,220210,224690,220277,223761,223762"
    query = f"""
    SELECT ce.subject_id, ce.stay_id, ce.charttime, ce.itemid, ce.value, ce.valuenum, ce.valueuom
    FROM physionet-data.mimiciv_3_1_icu.chartevents ce
    WHERE ce.stay_id IN ({ids})
    AND ce.itemid IN ({itemids})
    ORDER BY ce.stay_id, ce.charttime
    """
    return gcp2df(client, query)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "METRE", "output", "demonstration_cases.csv"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "METRE", "output", "case_study"),
    )
    parser.add_argument("--project_id", type=str, default=PROJECT_ID)
    args = parser.parse_args()

    if not os.path.exists(args.cases):
        print(f"Error: {args.cases} not found. Run find_demonstration_cases.py first.")
        sys.exit(1)

    cases = pd.read_csv(args.cases)
    stay_ids = cases["stay_id"].astype(int).unique().tolist()
    print(f"Querying BigQuery for {len(stay_ids)} stays...")

    os.environ["GOOGLE_CLOUD_PROJECT"] = args.project_id
    client = bigquery.Client(project=args.project_id)

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Patient/ICU stay info
    print("  Querying patient info...")
    patient = query_patient_info(client, stay_ids)
    patient.to_csv(os.path.join(args.output_dir, "patient_info.csv"), index=False)
    print(f"    Saved patient_info.csv ({len(patient)} rows)")

    # 2. Aggregated vitals
    print("  Querying vitals...")
    vitals = query_vitals(client, stay_ids)
    vitals.to_csv(os.path.join(args.output_dir, "vitals.csv"), index=False)
    print(f"    Saved vitals.csv ({len(vitals)} rows)")

    # 3. Copy demonstration_cases summary into case_study folder
    cases.to_csv(os.path.join(args.output_dir, "demonstration_cases_summary.csv"), index=False)
    print(f"    Saved demonstration_cases_summary.csv")

    print(f"\nCase study data saved to {args.output_dir}")


if __name__ == "__main__":
    main()
