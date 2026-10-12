# Data decisions (Step 0 output)

Inspected file: `Full Lung Cancer dataset.csv` (Kaggle `akashnath29/lung-cancer-dataset`).

## What the file really contains
- **3,000 rows x 16 columns**, no missing values, **2 exact duplicate rows** (removed before any split -> 2,998 rows).
- Columns: `GENDER` (M/F), `AGE` (30-80, integer), 13 survey fields coded 1/2 (`SMOKING`, `YELLOW_FINGERS`, `ANXIETY`, `PEER_PRESSURE`, `CHRONIC_DISEASE`, `FATIGUE`, `ALLERGY`, `WHEEZING`, `ALCOHOL_CONSUMING`, `COUGHING`, `SHORTNESS_OF_BREATH`, `SWALLOWING_DIFFICULTY`, `CHEST_PAIN`) and the target `LUNG_CANCER` (YES/NO).
- The Kaggle page mentions vital signs, treatment and outcomes; **none of those columns exist**. The report describes only the 15 real feature columns.
- Direct identifiers: none. Leakage columns (post-diagnosis fields): none identified.

## Decisions
| # | Decision | Detail |
|---|---|---|
| 1 | Target | `lung_cancer`, `NO=0`, `YES=1`. Prevalence 50.6% (1,517 positives of 2,998), so classes are balanced and no `pos_weight` is needed. |
| 2 | Features (15) | `gender` (F=0, M=1), `age` (divided by the fixed constant 100), 13 binary survey fields (1->0 "no", 2->1 "yes", the standard convention of this survey dataset). Order is fixed in `ml/schema.py`. |
| 3 | Scaling | Age is divided by a **fixed constant**. No scaler is fitted on any data, so nothing leaks across hospitals or into the test set. |
| 4 | Quasi-identifiers | `age` (generalised to 10-year bands, `age_band`) and `gender`. `age_band` is used only for the k-anonymity report, never as a model input. |
| 5 | Splits | Stratified 60% train / 10% validation / 20% test / 10% **attack hold-out** (seed 42). The hold-out is a deviation from the plan's 70/10/20: it supplies membership-inference non-members that were never used in training, validation or testing. |
| 6 | Hospitals | 3 clients. IID control: stratified thirds. Non-IID (main): quantity skew 50/30/20 and label skew with prevalence drawn from Dirichlet(alpha=1), resampled until prevalence is in [0.15, 0.85]. |

## Honest data findings (these shape the whole report)
1. **The label is almost unpredictable from these features.** A 5-fold cross-validated logistic regression on the pooled data gives ROC-AUC 0.501 and a random forest 0.523, and the per-class feature means are nearly identical. The dataset behaves like (near-)random survey data. We therefore **expect every model, centralized or federated, to score close to chance (about 0.5 AUC / 50% accuracy)**. We do not tune until a gap appears and we do not claim clinical value.
2. Consequence for the project goals: the accuracy comparison is expected to show *no meaningful difference* between centralized, federated and federated+DP. The privacy, communication and security analyses (which do not depend on a strong signal) are the substance of the work. Pipeline correctness is verified separately with unit tests, including a synthetic planted-signal check, so a near-chance result cannot be blamed on a pipeline bug.
3. Membership inference on a model with no real signal is also expected to sit near 0.5; a positive-control overfit model shows the attack code works.
4. The data is survey-style/synthetic-looking, not real clinical records. Stated as a limitation.
