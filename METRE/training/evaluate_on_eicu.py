"""
Evaluate MIMIC-trained LR/RF models on eICU test set (cross-database validation).

Loads eICU_compile.npy, applies same task filters, runs MIMIC models on eICU test set,
and reports AUC, AP, precision, recall, confusion matrix.

Usage:
    python evaluate_on_eicu.py --eicu_path ../output/eICU_compile.npy --models_dir ../output/benchmarks/models
"""
import argparse
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    log_loss,
)

import sys
sys.path.insert(0, os.path.dirname(__file__))
from run_benchmarks_lr_rf import filter_los, filter_arf, filter_shock, flatten_for_sklearn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eicu_path", type=str, default=os.path.join(os.path.dirname(__file__), "..", "output", "eICU_compile.npy"))
    parser.add_argument("--models_dir", type=str, default=os.path.join(os.path.dirname(__file__), "..", "output", "benchmarks", "models"))
    parser.add_argument("--output_path", type=str, default=os.path.join(os.path.dirname(__file__), "..", "output", "benchmarks", "eICU_test_metrics.csv"))
    args = parser.parse_args()

    eicu_path = os.path.abspath(args.eicu_path)
    models_dir = os.path.abspath(args.models_dir)
    output_path = os.path.abspath(args.output_path)

    if not os.path.exists(eicu_path):
        print(f"Error: {eicu_path} not found. Run compile_meep_to_npy.py --database eICU first.")
        return 1
    if not os.path.exists(models_dir):
        print(f"Error: {models_dir} not found. Run run_benchmarks_lr_rf.py first.")
        return 1

    data = np.load(eicu_path, allow_pickle=True).item()
    test_head = data["test_head"]
    s_test = np.stack(data["static_test_filter"], axis=0)

    task_map = {0: "hosp_mort", 1: "ARF", 2: "shock"}
    tasks = [(0, 24, 4), (1, 2, 4), (1, 6, 4), (2, 2, 4), (2, 6, 4)]

    rows = []
    for target_idx, thresh, gap in tasks:
        task_name = f"{task_map[target_idx]}_{thresh}h_gap{gap}h"
        if target_idx == 0:
            static_test, test_data = filter_los(s_test, test_head, thresh, gap)
            test_label = static_test[:, 0]
        elif target_idx == 1:
            test_data, test_label = filter_arf(test_head, thresh, gap)
        else:
            test_data, test_label = filter_shock(test_head, thresh, gap)

        X_test = flatten_for_sklearn(test_data)
        lr_model = joblib.load(os.path.join(models_dir, f"LR_{task_name}.joblib"))
        rf_model = joblib.load(os.path.join(models_dir, f"RF_{task_name}.joblib"))
        lr_prob = lr_model.predict_proba(X_test)[:, 1]
        rf_prob = rf_model.predict_proba(X_test)[:, 1]

        def _row(y_true, y_prob, model_name):
            y_pred = (y_prob >= 0.5).astype(int)
            tp = int(((y_true == 1) & (y_pred == 1)).sum())
            fp = int(((y_true == 0) & (y_pred == 1)).sum())
            tn = int(((y_true == 0) & (y_pred == 0)).sum())
            fn = int(((y_true == 1) & (y_pred == 0)).sum())
            eps = 1e-15
            p = np.clip(y_prob, eps, 1 - eps)
            bce = log_loss(y_true, np.column_stack([1 - p, p]), labels=[0, 1])
            return {
                "task": task_name, "model": model_name, "database": "eICU",
                "auc": roc_auc_score(y_true, y_prob),
                "ap": average_precision_score(y_true, y_prob),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "log_loss": bce, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                "n_total": len(y_true), "n_positive": int(y_true.sum()),
            }

        rows.append(_row(test_label, lr_prob, "LR"))
        rows.append(_row(test_label, rf_prob, "RF"))

        print(f"{task_name} (eICU test): LR AUC {roc_auc_score(test_label, lr_prob):.4f}  RF AUC {roc_auc_score(test_label, rf_prob):.4f}  n={len(test_label)}")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved {output_path}")
    return 0


if __name__ == "__main__":
    exit(main())
