"""
Export aggregate case-load stats from test_predictions.csv for the ICU dashboard.

Outputs dashboard_summary.json with:
  - total_stays
  - per-task: n_total, n_high_risk, n_elevated, pct_high_risk, median_risk, distribution bins

With --patient-detail: also exports patient_detail.json with vitals and interventions
per patient (requires MEEP parquet files). No stay_id or PHI in output.

With --events: exports patient_events.json with raw-granularity (per-minute) events
from cached BigQuery CSVs (run scripts/export_dashboard_events.py first).

No patient IDs or PHI. Safe for sharing.
"""
import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

# Key vitals to export (parquet column may be tuple ("heart_rate","mean") or str)
VITAL_KEYS = [
    ("heart_rate", "mean"), ("sbp", "mean"), ("dbp", "mean"), ("mbp", "mean"),
    ("resp_rate", "mean"), ("temperature", "mean"), ("glucose", "mean"),
    ("so2", "mean"),
]
INV_COLS = [
    "vent", "antibiotic", "dopamine", "epinephrine", "norepinephrine",
    "phenylephrine", "vasopressin", "dobutamine", "milrinone", "heparin",
    "crrt", "rbc_trans", "platelets_trans", "ffp_trans", "colloid_bolus", "crystalloid_bolus",
]

TASKS = [
    "hosp_mort_24h_gap4h",
    "ARF_2h_gap4h",
    "ARF_6h_gap4h",
    "shock_2h_gap4h",
    "shock_6h_gap4h",
]
TASK_LABELS = {
    "hosp_mort_24h_gap4h": "In-hospital mortality",
    "ARF_2h_gap4h": "ARF (2h)",
    "ARF_6h_gap4h": "ARF (6h)",
    "shock_2h_gap4h": "Shock (2h)",
    "shock_6h_gap4h": "Shock (6h)",
}
BINS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
LOS_BINS = [0, 24, 48, 72, 96, 120, 168, 9999]
LOS_LABELS = ["0–24h", "24–48h", "48–72h", "72–96h", "96–120h", "120–168h", "168h+"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "output", "benchmarks", "test_predictions.csv"),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "output", "benchmarks", "dashboard", "dashboard_summary.json"),
    )
    parser.add_argument("--model", choices=["LR", "RF"], default="LR")
    parser.add_argument("--threshold-high", type=float, default=0.5)
    parser.add_argument("--threshold-elevated", type=float, default=0.2)
    parser.add_argument("--patient-list", type=int, default=40, metavar="N",
                        help="Export N patient rows to patient_list.json (0=skip)")
    parser.add_argument("--patient-detail", action="store_true",
                        help="Export vitals and interventions per patient (requires MEEP parquet)")
    parser.add_argument("--meep-dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "output"),
                        help="Directory with MEEP_MIMIC_vital.parquet etc.")
    parser.add_argument("--events", action="store_true",
                        help="Export patient_events.json from cached BigQuery CSVs "
                             "(run scripts/export_dashboard_events.py first)")
    parser.add_argument("--events-dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "output",
                                             "benchmarks", "dashboard", "data"),
                        help="Directory with cached BigQuery CSVs (vitals.csv, labs.csv, etc.)")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.patient_list > 0:
        n_sample = min(args.patient_list, len(df))
        idx = np.random.RandomState(42).choice(len(df), size=n_sample, replace=False)
        sample = df.iloc[idx]
    else:
        sample = df
        n_sample = len(df)

    tasks_out = {}
    for task in TASKS:
        prob_col = f"{task}_{args.model}_prob"
        if prob_col not in df.columns:
            continue
        sub = sample[[prob_col]].dropna()
        probs = sub[prob_col].values
        n_total = len(probs)

        n_high = int((probs >= args.threshold_high).sum())
        n_elevated = int(((probs >= args.threshold_elevated) & (probs < args.threshold_high)).sum())
        pct_high = round(100 * n_high / n_total, 1) if n_total > 0 else 0
        median_risk = round(float(np.median(probs)), 3) if n_total > 0 else 0

        hist, _ = np.histogram(probs, bins=BINS)
        distribution = [int(x) for x in hist]

        tasks_out[task] = {
            "label": TASK_LABELS[task],
            "n_total": n_total,
            "n_high_risk": n_high,
            "n_elevated": n_elevated,
            "pct_high_risk": pct_high,
            "median_risk": median_risk,
            "distribution": distribution,
            "bin_edges": BINS,
        }

    los_dist = None
    if "los_icu" in df.columns:
        los_vals = sample["los_icu"].dropna()
        los_vals = los_vals.apply(lambda v: float(v) * 24 if v < 50 else float(v))
        hist, _ = np.histogram(los_vals, bins=LOS_BINS)
        los_dist = {
            "distribution": [int(x) for x in hist],
            "labels": LOS_LABELS,
        }

    out = {
        "total_stays": n_sample,
        "updated": datetime.now().isoformat(),
        "model": args.model,
        "threshold_high": args.threshold_high,
        "threshold_elevated": args.threshold_elevated,
        "los_distribution": los_dist,
        "tasks": tasks_out,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Saved {args.output} (n={n_sample})")

    if args.patient_list > 0:
        patients = []
        for j, (_, row) in enumerate(sample.iterrows()):
            p = {"id": f"P{j + 1:03d}"}
            if "age" in df.columns:
                v = row.get("age")
                p["age"] = int(v) if pd.notna(v) else None
            if "gender" in df.columns:
                v = row.get("gender")
                p["gender"] = str(v) if pd.notna(v) and v else None
            if "admission_type" in df.columns:
                v = row.get("admission_type")
                p["admission_type"] = str(v) if pd.notna(v) and v else None
            if "los_icu" in df.columns:
                v = row.get("los_icu")
                if pd.notna(v):
                    val = float(v)
                    p["los_icu"] = round(val * 24, 1) if val < 50 else round(val, 1)
                else:
                    p["los_icu"] = None
            for task in TASKS:
                col = f"{task}_{args.model}_prob"
                if col in df.columns:
                    v = row[col]
                    p[task] = round(float(v), 3) if pd.notna(v) else None
                else:
                    p[task] = None
            patients.append(p)
        pl_path = os.path.join(os.path.dirname(args.output), "patient_list.json")
        with open(pl_path, "w") as f:
            json.dump({"model": args.model, "patients": patients}, f, indent=2)
        print(f"Saved {pl_path} ({n_sample} patients)")

        if args.patient_detail:
            detail_path = os.path.join(os.path.dirname(args.output), "patient_detail.json")
            details = _export_patient_details(sample, args.meep_dir)
            if details:
                with open(detail_path, "w") as f:
                    json.dump(details, f, indent=2)
                print(f"Saved {detail_path} ({len(details)} patients)")
            else:
                print("Skipped patient_detail.json (parquet not found or no matching stays)")

    if args.events:
        events_path = os.path.join(os.path.dirname(args.output), "patient_events.json")
        events = _export_patient_events(sample, args.events_dir, args.model)
        if events:
            with open(events_path, "w") as f:
                json.dump(events, f)
            print(f"Saved {events_path} ({len(events)} patients)")
        else:
            print("Skipped patient_events.json (cached CSVs not found in --events-dir)")

    return 0


def _find_vital_col(df, key):
    """Find column matching key (tuple or str)."""
    for c in df.columns:
        if isinstance(c, tuple) and c == key:
            return c
        if isinstance(c, tuple) and len(c) >= 1 and c[0] == (key[0] if isinstance(key, tuple) else key):
            return c
        if isinstance(c, str) and c == str(key):
            return c
    return None


def _export_patient_details(sample, meep_dir):
    """Export vitals and interventions per patient. Returns dict {P001: {...}} or {}."""
    vital_path = os.path.join(meep_dir, "MEEP_MIMIC_vital.parquet")
    inv_path = os.path.join(meep_dir, "MEEP_MIMIC_inv.parquet")
    if not os.path.isfile(vital_path) or not os.path.isfile(inv_path):
        return {}

    vital = pd.read_parquet(vital_path)
    inv = pd.read_parquet(inv_path)
    stay_level = "stay_id"

    inv_cols = [c for c in INV_COLS if c in inv.columns]
    if not inv_cols:
        return {}

    vital_cols = {}
    for k in VITAL_KEYS:
        c = _find_vital_col(vital, k)
        if c is not None:
            vital_cols[k[0] if isinstance(k, tuple) else k] = c

    merged = vital.merge(inv[inv_cols], left_index=True, right_index=True, how="outer")
    stay_to_idx = {}
    for j, (_, row) in enumerate(sample.iterrows()):
        sid = row.get("stay_id")
        if pd.notna(sid):
            stay_to_idx[int(sid)] = j

    details = {}
    for stay_id, group in merged.groupby(level=stay_level):
        sid = int(stay_id)
        if sid not in stay_to_idx:
            continue
        pid = f"P{stay_to_idx[sid] + 1:03d}"

        group = group.reset_index()
        if "hours_in" in group.columns:
            group = group.sort_values("hours_in").drop_duplicates(subset=["hours_in"], keep="first")
        hrs = group["hours_in"].values if "hours_in" in group.columns else list(range(len(group)))

        vitals_by_hour = []
        for i, h in enumerate(hrs):
            rec = {"hour": int(h) if pd.notna(h) else i}
            for label, col in vital_cols.items():
                v = group[col].iloc[i] if i < len(group) else None
                if pd.notna(v):
                    rec[label] = round(float(v), 2)
            vitals_by_hour.append(rec)

        inv_by_hour = []
        for i, h in enumerate(hrs):
            rec = {"hour": int(h) if pd.notna(h) else i}
            for col in inv_cols:
                v = group[col].iloc[i] if i < len(group) else 0
                if pd.notna(v) and float(v) >= 0.5:
                    rec[col] = 1
            inv_by_hour.append(rec)

        details[pid] = {"vitals": vitals_by_hour, "interventions": inv_by_hour}

    return details


VITAL_ITEMID_NAMES = {
    220045: "heart_rate",
    220179: "sbp", 220050: "sbp",
    220180: "dbp", 220051: "dbp",
    220052: "mbp", 220181: "mbp", 225312: "mbp",
    220210: "resp_rate", 224690: "resp_rate",
    223761: "temperature_f", 223762: "temperature_c",
    220277: "spo2",
    225664: "glucose", 220621: "glucose", 226537: "glucose",
}
LAB_ITEMID_NAMES = {
    50813: "lactate",
    50912: "creatinine",
    51301: "wbc",
    51265: "platelets",
    51222: "hemoglobin",
    50882: "bicarbonate",
    50931: "glucose_lab",
}
VASOPRESSOR_ITEMID_NAMES = {
    221906: "norepinephrine",
    221289: "epinephrine",
    221662: "dopamine",
    222315: "vasopressin",
    221749: "phenylephrine",
    221653: "dobutamine",
}


def _export_patient_events(sample, events_dir, model):
    """Build patient_events.json from cached BigQuery CSVs.

    Returns dict {P001: {max_minutes, events: [...], predictions: {...}}} or {}.
    """
    vitals_path = os.path.join(events_dir, "vitals.csv")
    if not os.path.isfile(vitals_path):
        return {}

    stay_to_pid = {}
    stay_to_preds = {}
    for j, (_, row) in enumerate(sample.iterrows()):
        sid = int(row["stay_id"])
        pid = f"P{j + 1:03d}"
        stay_to_pid[sid] = pid
        preds = {}
        for task in TASKS:
            col = f"{task}_{model}_prob"
            v = row.get(col)
            preds[task] = round(float(v), 3) if pd.notna(v) else None
        stay_to_preds[sid] = preds

    patients = {pid: {"max_minutes": 0, "events": [], "predictions": stay_to_preds[sid]}
                for sid, pid in stay_to_pid.items()}

    # Vitals: each row is one measurement at exact charttime
    vitals = pd.read_csv(vitals_path)
    for _, row in vitals.iterrows():
        sid = int(row["stay_id"])
        if sid not in stay_to_pid:
            continue
        pid = stay_to_pid[sid]
        t = int(round(float(row["minutes_since_admit"])))
        itemid = int(row["itemid"])
        name = VITAL_ITEMID_NAMES.get(itemid)
        if name is None:
            continue
        val = float(row["valuenum"])
        if name == "temperature_f":
            val = round((val - 32) / 1.8, 2)
            name = "temperature"
        elif name == "temperature_c":
            val = round(val, 2)
            name = "temperature"
        else:
            val = round(val, 1)
        if t < 0:
            continue
        patients[pid]["events"].append({"t": t, "type": "vital", "name": name, "value": val})
        patients[pid]["max_minutes"] = max(patients[pid]["max_minutes"], t)

    # Labs
    labs_path = os.path.join(events_dir, "labs.csv")
    if os.path.isfile(labs_path):
        labs = pd.read_csv(labs_path)
        for _, row in labs.iterrows():
            sid = int(row["stay_id"])
            if sid not in stay_to_pid:
                continue
            pid = stay_to_pid[sid]
            t = int(round(float(row["minutes_since_admit"])))
            itemid = int(row["itemid"])
            name = LAB_ITEMID_NAMES.get(itemid)
            if name is None or t < 0:
                continue
            val = round(float(row["valuenum"]), 2)
            patients[pid]["events"].append({"t": t, "type": "lab", "name": name, "value": val})
            patients[pid]["max_minutes"] = max(patients[pid]["max_minutes"], t)

    # Vasopressors: start/end intervals -> start and stop events
    vaso_path = os.path.join(events_dir, "vasopressors.csv")
    if os.path.isfile(vaso_path):
        vaso = pd.read_csv(vaso_path)
        for _, row in vaso.iterrows():
            sid = int(row["stay_id"])
            if sid not in stay_to_pid:
                continue
            pid = stay_to_pid[sid]
            itemid = int(row["itemid"])
            name = VASOPRESSOR_ITEMID_NAMES.get(itemid, f"vaso_{itemid}")
            t_start = int(round(float(row["start_minutes"])))
            t_end = int(round(float(row["end_minutes"])))
            if t_start < 0:
                t_start = 0
            patients[pid]["events"].append(
                {"t": t_start, "type": "intervention", "name": name, "value": 1})
            patients[pid]["events"].append(
                {"t": t_end, "type": "intervention", "name": name, "value": 0})
            patients[pid]["max_minutes"] = max(patients[pid]["max_minutes"], t_end)

    # Ventilation: start/end intervals
    vent_path = os.path.join(events_dir, "ventilation.csv")
    if os.path.isfile(vent_path):
        vent = pd.read_csv(vent_path)
        for _, row in vent.iterrows():
            sid = int(row["stay_id"])
            if sid not in stay_to_pid:
                continue
            pid = stay_to_pid[sid]
            status = str(row.get("ventilation_status", ""))
            t_start = int(round(float(row["start_minutes"])))
            t_end = int(round(float(row["end_minutes"])))
            if t_start < 0:
                t_start = 0
            patients[pid]["events"].append(
                {"t": t_start, "type": "intervention", "name": "vent",
                 "value": 1, "detail": status})
            patients[pid]["events"].append(
                {"t": t_end, "type": "intervention", "name": "vent", "value": 0})
            patients[pid]["max_minutes"] = max(patients[pid]["max_minutes"], t_end)

    # Antibiotics: start/end intervals
    abx_path = os.path.join(events_dir, "antibiotics.csv")
    if os.path.isfile(abx_path):
        abx = pd.read_csv(abx_path)
        for _, row in abx.iterrows():
            sid = int(row["stay_id"])
            if sid not in stay_to_pid:
                continue
            pid = stay_to_pid[sid]
            drug = str(row.get("antibiotic", "antibiotic"))
            t_start = int(round(float(row["start_minutes"])))
            t_end = int(round(float(row["end_minutes"])))
            if t_start < 0:
                t_start = 0
            patients[pid]["events"].append(
                {"t": t_start, "type": "intervention", "name": "antibiotic",
                 "value": 1, "detail": drug})
            patients[pid]["events"].append(
                {"t": t_end, "type": "intervention", "name": "antibiotic", "value": 0})
            patients[pid]["max_minutes"] = max(patients[pid]["max_minutes"], t_end)

    # Sort events by time within each patient
    for pid in patients:
        patients[pid]["events"].sort(key=lambda e: (e["t"], e["type"]))

    return patients


if __name__ == "__main__":
    exit(main())
