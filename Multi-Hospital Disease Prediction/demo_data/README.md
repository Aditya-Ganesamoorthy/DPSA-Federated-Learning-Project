# Synthetic demo hospitals

Four CSV files for demos and the viva. **All rows are invented** (no row comes from the original Kaggle file and no row is shared between hospitals). They use exactly the original layout, so the dashboard's **Train & Contribute** page and the command-line client accept them as they are.

| File | Rows | Cancer rate | Mean age | Notes |
|---|---|---|---|---|
| `hospital_A_city_general.csv` | 300 | 46.7% | 53.2 | youngest; symptoms reported slightly more often (52% yes) |
| `hospital_B_district_medical.csv` | 310 | 56.1% | 58.4 | highest cancer rate; fewer male patients (40%) |
| `hospital_C_regional_clinic.csv` | 290 | 54.5% | 56.1 | symptoms reported slightly more often (52% yes); 51% male |
| `hospital_D_community_care.csv` | 300 | 53.7% | 59.4 | oldest; symptoms reported less often (47% yes) |

The hospitals differ slightly in size, cancer rate, age and symptom rates, so they are mildly non-IID like real sites.

**Columns:** `GENDER` (M/F), `AGE` (30 to 80), 13 symptom/habit columns coded `1` = no and `2` = yes, and `LUNG_CANCER` (YES/NO).

**Behaviour:** like the original dataset, the diagnosis is independent of the other columns (the strongest correlation across all four files is 0.05). Models trained on these files therefore score near chance (ROC-AUC about 0.5), which is expected and matches the report.

## Use in a demo
1. Open the app, **Register hospital** (use a different browser window per hospital), then **Train and Contribute**.
2. Drop one file per hospital, for example A and B. Try **Differential privacy** on one of them.
3. With `MIN_CLIENTS=2` the round closes after the second hospital. Use C and D for a second round.

Command line: `python -m client.cli run --data demo_data/hospital_A_city_general.csv --epochs 2`

## Regenerate
`python scripts/make_demo_data.py` rewrites the files identically (fixed seed 2026).
