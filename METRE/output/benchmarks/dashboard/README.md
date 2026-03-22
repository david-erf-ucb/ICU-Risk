# ICU Case Load Dashboard

Summary dashboard for hospital admins and clinicians. Shows total stays, LOS distribution, high-risk counts for mortality, ARF, and shock (METRE predictions).

## Technology

- **HTML** — Static markup
- **CSS** — Custom styles (no framework)
- **JavaScript** — Vanilla JS (no React/Vue/etc.)
- **Chart.js** — Bar charts (loaded from CDN)
- **Data** — JSON from `fetch()` or embedded fallback

## Pages

- **index.html** — Summary: total stays, LOS distribution, high-risk counts, risk distribution charts. Time slider (if event data available).
- **patient_list.html** — Individual patient rows with risk scores, live vitals, and active interventions. Time slider scrubs "present time" — events arrive at real charttime granularity.
- **patient_detail.html** — Per-patient vitals and interventions (requires `--patient-detail` export or event data).
- **settings.html** — Threshold tuning: adjust high/elevated risk cutoffs, with histogram preview. Persists to localStorage.

## Usage

1. **Generate summary + patient list** (from project root):
   ```bash
   python training/export_dashboard_data.py
   ```
   Reads `test_predictions.csv`, outputs `dashboard_summary.json` and `patient_list.json` (40 patients by default).

2. **(Optional) Pull raw event data from BigQuery** for real-time simulation:
   ```bash
   python scripts/export_dashboard_events.py
   ```
   Queries vitals, labs, interventions at charttime granularity for the 40 dashboard patients. Saves CSVs to `dashboard/data/` (cached — won't re-query if files exist).

3. **Build event JSON** from cached BigQuery data:
   ```bash
   python training/export_dashboard_data.py --events
   ```
   Produces `patient_events.json` — per-patient events at per-minute granularity.

4. **View dashboard**: Open `index.html` in a browser.
   - Works with `file://` (embedded demo fallback).
   - Or run a local server: `cd output/benchmarks/dashboard && python -m http.server 8080` → http://localhost:8080

## Data

- **dashboard_summary.json** — Aggregate stats (no PHI).
- **patient_list.json** — Sample of anonymized patient rows with risk scores (no PHI).
- **patient_events.json** — Per-minute event data (vitals, labs, interventions) for dashboard patients.
- **dashboard_summary_demo.json** — Fallback demo data if export hasn't been run.
- **data/** — Cached BigQuery CSVs (gitignored, not committed).

## Settings

Risk thresholds (high/elevated) are stored in `localStorage` under the key `metre_settings`. All pages read from localStorage on load, so changing thresholds on the Settings page immediately affects Summary and Patient List color coding.
