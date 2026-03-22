#!/usr/bin/env python3
"""Generate patient_panel.html from demonstration_cases.csv."""
import argparse
import os

import pandas as pd


def round_vital(val):
    if pd.isna(val):
        return ""
    if val >= 100:
        return f"{val:.0f}"
    if val >= 10:
        return f"{val:.1f}"
    return f"{val:.2f}"


def score_class(prob):
    if prob >= 0.4:
        return "score-high"
    if prob >= 0.15:
        return "score-mid"
    return "score-low"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "METRE", "output", "demonstration_cases.csv"),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "METRE", "output", "case_study", "patient_panel.html"),
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible shuffle")
    args = parser.parse_args()

    df = pd.read_csv(args.cases)
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # Build noisy table rows
    noisy_rows = []
    for _, row in df.iterrows():
        hr = round_vital(row["heart_rate"])
        sbp = round_vital(row["sbp"])
        rr = round_vital(row["resp_rate"])
        so2 = round_vital(row["so2"])
        temp = round_vital(row["temperature"])
        so2_str = f"{so2}%" if so2 else ""
        temp_str = f"{temp}°C" if temp else ""
        noisy_rows.append(
            f"""        <tr>
          <td>{int(row['age'])}</td>
          <td>{row['gender']}</td>
          <td>{row['admission_type']}</td>
          <td><span class="vital-val">{hr}</span></td>
          <td><span class="vital-val">{sbp}</span></td>
          <td><span class="vital-val">{rr}</span></td>
          <td><span class="vital-val">{so2_str}</span></td>
          <td><span class="vital-val">{temp_str}</span></td>
        </tr>"""
        )

    # Build reveal table rows
    reveal_rows = []
    for _, row in df.iterrows():
        vitals_str = f"HR {round_vital(row['heart_rate'])}, SBP {round_vital(row['sbp'])}, RR {round_vital(row['resp_rate'])}, SpO2 {round_vital(row['so2'])}%, Temp {round_vital(row['temperature'])}°C"
        prob_lr = row["prob_lr"]
        prob_rf = row["prob_rf"]
        lr_class = score_class(prob_lr)
        rf_class = score_class(prob_rf)
        outcome = "Died" if row["mort"] == 1 else "Survived"
        outcome_class = "outcome-died" if row["mort"] == 1 else "outcome-survived"
        reveal_rows.append(
            f"""          <tr>
            <td>{int(row['age'])}</td>
            <td>{row['gender']}</td>
            <td>{row['admission_type']}</td>
            <td>{vitals_str}</td>
            <td class="col-score {lr_class}">{prob_lr:.2f}</td>
            <td class="col-score {rf_class}">{prob_rf:.2f}</td>
            <td class="col-outcome {outcome_class}">{outcome}</td>
          </tr>"""
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ICU Case Study — Patient Panel</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #0f1419;
      --surface: #1a2332;
      --surface-alt: #243044;
      --text: #e6edf3;
      --text-muted: #8b949e;
      --accent: #58a6ff;
      --success: #3fb950;
      --danger: #f85149;
      --warning: #d29922;
    }}

    * {{ box-sizing: border-box; }}

    body {{
      font-family: 'IBM Plex Sans', system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      margin: 0;
      padding: 2rem;
      min-height: 100vh;
      line-height: 1.6;
    }}

    .container {{
      max-width: 1200px;
      margin: 0 auto;
    }}

    h1 {{
      font-size: 1.25rem;
      font-weight: 600;
      color: var(--text-muted);
      margin-bottom: 0.5rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}

    h2 {{
      font-size: 1rem;
      font-weight: 500;
      color: var(--text);
      margin: 2rem 0 0.75rem;
    }}

    .subtitle {{
      font-size: 0.875rem;
      color: var(--text-muted);
      margin-bottom: 1.5rem;
    }}

    .table-noisy {{
      width: 100%;
      border-collapse: collapse;
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.75rem;
      background: var(--surface);
      border: 1px solid var(--surface-alt);
      border-radius: 8px;
      overflow: hidden;
    }}

    .table-noisy th,
    .table-noisy td {{
      padding: 0.5rem 0.6rem;
      text-align: left;
      border-bottom: 1px solid var(--surface-alt);
    }}

    .table-noisy th {{
      background: var(--surface-alt);
      color: var(--text-muted);
      font-weight: 500;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }}

    .table-noisy tr:last-child td {{
      border-bottom: none;
    }}

    .table-noisy th,
    .table-noisy td {{
      border-right: 1px solid var(--surface-alt);
    }}

    .table-noisy th:last-child,
    .table-noisy td:last-child {{
      border-right: none;
    }}

    .vital-val {{ color: var(--text); }}

    .reveal-section {{
      margin-top: 2.5rem;
      padding-top: 2rem;
      border-top: 1px solid var(--surface-alt);
    }}

    .reveal-badge {{
      display: inline-block;
      background: var(--accent);
      color: var(--bg);
      font-size: 0.7rem;
      font-weight: 600;
      padding: 0.25rem 0.5rem;
      border-radius: 4px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 0.75rem;
    }}

    .reveal-note {{
      font-size: 0.8rem;
      color: var(--text-muted);
      margin-bottom: 1rem;
      font-style: italic;
    }}

    .table-reveal {{
      width: 100%;
      border-collapse: collapse;
      font-family: 'IBM Plex Sans', system-ui, sans-serif;
      font-size: 0.8rem;
      background: var(--surface);
      border: 1px solid var(--surface-alt);
      border-radius: 8px;
      overflow: hidden;
    }}

    .table-reveal th,
    .table-reveal td {{
      padding: 0.5rem 0.6rem;
      text-align: left;
      border-bottom: 1px solid var(--surface-alt);
    }}

    .table-reveal th {{
      background: var(--surface-alt);
      color: var(--text-muted);
      font-weight: 500;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }}

    .table-reveal tr:last-child td {{
      border-bottom: none;
    }}

    .table-reveal .col-score {{
      font-family: 'IBM Plex Mono', monospace;
      font-weight: 600;
    }}

    .table-reveal .col-outcome {{ font-weight: 600; }}

    .outcome-survived {{ color: var(--success); }}
    .outcome-died {{ color: var(--danger); }}

    .score-low {{ color: var(--success); }}
    .score-mid {{ color: var(--warning); }}
    .score-high {{ color: var(--danger); }}
  </style>
</head>
<body>
  <div class="container">
    <h1>ICU Case Study</h1>
    <p class="subtitle">Twenty-five patients with normal first 8h vitals (HR 60–100, SBP 90–140, RR 12–22, SpO2 95–100%, Temp 36–37.5°C)</p>

    <h2>At the bedside — what clinicians see</h2>
    <p class="subtitle">Dense, noisy data. No obvious way to distinguish who will deteriorate.</p>

    <table class="table-noisy">
      <thead>
        <tr>
          <th>Age</th>
          <th>Gender</th>
          <th>Admission</th>
          <th>HR</th>
          <th>SBP</th>
          <th>RR</th>
          <th>SpO2</th>
          <th>Temp</th>
        </tr>
      </thead>
      <tbody>
{chr(10).join(noisy_rows)}
      </tbody>
    </table>

    <div class="reveal-section">
      <span class="reveal-badge">The reveal</span>
      <h2>Model prediction after 24h observation</h2>
      <p class="reveal-note">Model score = predicted in-hospital mortality (0–1 scale; LR = logistic regression, RF = random forest)</p>

      <table class="table-reveal">
        <thead>
          <tr>
            <th>Age</th>
            <th>Gender</th>
            <th>Admission</th>
            <th>First 8h vitals</th>
            <th>LR</th>
            <th>RF</th>
            <th>Outcome</th>
          </tr>
        </thead>
        <tbody>
{chr(10).join(reveal_rows)}
        </tbody>
      </table>
    </div>
  </div>
</body>
</html>
"""

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(html)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
