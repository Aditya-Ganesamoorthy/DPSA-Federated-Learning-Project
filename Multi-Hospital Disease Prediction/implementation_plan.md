# Privacy-Preserving Lung Cancer Prediction using Federated Learning
### Complete Implementation Plan (Data Privacy and Security Analytics Lab)

**Project title (from brief):** *Decentralized Machine Learning: Implementing Federated Learning for Privacy Preservation*
**Dataset (locked):** https://www.kaggle.com/datasets/akashnath29/lung-cancer-dataset
**Stack (locked):** Python 3.10/3.11, pandas, scikit-learn, NumPy, PyTorch, Flower (`flwr`), Opacus, matplotlib, FastAPI + Uvicorn, React (Vite) + Recharts. No database (CSV/JSON files only).

> **Status flags used in this document**
> - **[VERIFY]** = depends on the real CSV, which has not been inspected yet. Step 0 resolves all of these.
> - **[OPTIONAL]** = only if time remains. Never at the cost of a required item.

---

## 0. Step 0: Inspect the real data first (blocks everything else)

Facts known so far: the dataset listing shows **3,000 rows and 16 columns**, and the first three column names are `gender`, `age`, `smoking`. The other 13 names, the target column, and its values are **not yet confirmed**.

Run once and keep the output in the report appendix:

```python
import pandas as pd
df = pd.read_csv("data/raw/lung_cancer.csv")
print(df.shape)
print(df.columns.tolist())
print(df.dtypes)
print(df.head())
print(df.isna().sum())
print("duplicates:", df.duplicated().sum())
for c in df.columns:
    print(c, df[c].nunique(), df[c].unique()[:10])
print(df.iloc[:, -1].value_counts(normalize=True))   # target guess: last column
```

Decisions that come out of Step 0 (write them in `docs/data_decisions.md`):
1. Exact target column and its encoding.
2. Exact list of feature columns and their types (binary, categorical, numeric).
3. Whether any column is a direct identifier (expected: none) or a leakage column.
4. Class balance, duplicate count, missing-value count.
5. Whether the description on the Kaggle page (it mentions vital signs, treatment, outcomes) matches the real columns. Write only what the file really contains.

---

## 1. Goals and requirement traceability

### 1.1 Goals
1. Simulate several hospitals, each holding a private slice of lung cancer data.
2. Train a lung cancer prediction model with **federated learning** (Flower, FedAvg) without pooling raw data.
3. Add **Differential Privacy** and **Secure Aggregation** (we implement both) for extra protection.
4. **Compare** federated vs centralized learning on accuracy and privacy.
5. **Analyze** communication efficiency, security risks and performance under heterogeneous data.
6. Deliver code, results, a report answering the five guiding questions, slides, and a demo dashboard.

### 1.2 Requirement to deliverable matrix

| # | Requirement in the brief | Implemented by | Evidence produced |
|---|---|---|---|
| R1 | Dataset simulating multiple users/devices (healthcare) | `ml/partition.py` builds hospital_1..3 CSVs | Client table and class-distribution plot |
| R2 | FL using TFF / PySyft / Flower | `ml/federated.py` (Flower FedAvg) | Per-round training logs and plots |
| R3 | Compare FL vs centralized (accuracy) | `ml/centralized.py`, shared test set, 5 seeds | Comparison table (mean ± std) |
| R4 | Compare FL vs centralized (privacy) | `ml/attack.py` (membership inference) | Attack AUC per model |
| R5 | Communication efficiency analysis | `ml/comm_analysis.py` | Bytes/round, total MB, rounds-to-target, vs raw data size |
| R6 | Security risk analysis | Threat model plus attack results | Threat model table in report and dashboard |
| R7 | Model performance in FL setting | IID vs non-IID experiments, per-client metrics | Heterogeneity plots |
| R8 | Secure Aggregation or Differential Privacy | `ml/dp.py` (Opacus) and `ml/secagg.py` | ε vs utility curve; secagg exactness check |
| R9 | Guiding questions 1 to 5 | Report section per question, backed by measured results | Report |
| R10 | Documentation, presentation, demonstration | Report, slides, FastAPI + React dashboard | Files and a 2-minute demo |

### 1.3 Marking scheme alignment (40 marks)

| Marks | Criterion | What the examiner must be able to see |
|---|---|---|
| 8 | Dataset, client setup, problem definition | Data audit, privacy preprocessing, non-IID hospital clients, clear problem statement |
| 10 | Implementation of FL / DP / aggregation | Flower FedAvg running, DP-SGD with real ε, secure aggregation simulation that provably cancels |
| 8 | Comparison | Centralized vs FL vs FL+DP on one fixed test set, multiple seeds |
| 10 | Privacy, security, communication, results analysis | ε curve, attack results, communication numbers, threat model, honest interpretation |
| 4 | Documentation, presentation, demo | Report, slides, working dashboard and live run |

---

## 2. Scope

**In scope:** tabular lung cancer prediction; 3 simulated hospitals (IID and non-IID); centralized baseline; Flower FedAvg; DP-SGD; secure aggregation simulation; membership inference attack; communication analysis; FastAPI backend; React dashboard; report; slides.

**Out of scope (state this in the report):** real hospital deployment; real cryptographic secure aggregation with key exchange and dropout recovery; medical-grade validation; CT image models; running federated training live inside the web app.

**[OPTIONAL] extras, in priority order:** (1) FedProx to reduce non-IID drift, (2) 5-client scaling run, (3) 8-bit update quantization for communication, (4) a data-poisoning experiment.

---

## 3. System architecture

```
                    OFFLINE PIPELINE (Python)                          ONLINE (demo)
┌──────────────┐   ┌───────────────┐   ┌──────────────────────┐   ┌──────────────┐
│ Raw CSV      │──▶│ Preprocess +  │──▶│ Partition into       │   │ React (Vite) │
│ (Kaggle)     │   │ privacy step  │   │ hospital_1..3 + test │   │ Dashboard    │
└──────────────┘   └───────────────┘   └──────────┬───────────┘   └──────▲───────┘
                                                  │                      │ HTTP/JSON
        ┌─────────────────────────────────────────┼──────────────┐       │
        ▼                    ▼                    ▼              ▼       │
 ┌─────────────┐   ┌───────────────┐   ┌──────────────┐  ┌────────────┐  │
 │ Centralized │   │ FL: Flower    │   │ FL + DP      │  │ FL + SecAgg│  │
 │ baseline    │   │ FedAvg        │   │ (Opacus)     │  │ simulation │  │
 └──────┬──────┘   └───────┬───────┘   └──────┬───────┘  └─────┬──────┘  │
        └──────────────────┴──────────┬───────┴────────────────┘         │
                                      ▼                                  │
                 ┌────────────────────────────────────────┐              │
                 │ Evaluation: metrics, attack, comm cost │              │
                 └───────────────────┬────────────────────┘              │
                                     ▼                                   │
                        results/*.json, *.csv, *.png, models/*.pt        │
                                     │                                   │
                                     └───────────▶ FastAPI backend ──────┘
```

Key principle: **hospital CSVs are only opened by client code.** The server code (aggregation, evaluation on the shared test set) never reads a hospital file. State this in the viva.

### 3.1 Folder structure

```
fl-lung-cancer/
├── README.md
├── requirements.txt                 # pinned versions
├── .gitignore                       # data/raw, node_modules, venv
├── data/
│   ├── raw/lung_cancer.csv
│   ├── processed/clean.csv
│   └── clients/ hospital_1.csv hospital_2.csv hospital_3.csv test.csv val.csv
├── docs/
│   ├── data_decisions.md
│   ├── threat_model.md
│   └── report/ (report, slides)
├── ml/
│   ├── config.py                    # seeds, paths, hyperparameters in ONE place
│   ├── schema.py                    # feature list, order, encodings (single source of truth)
│   ├── preprocess.py                # cleaning + privacy transforms
│   ├── partition.py                 # IID / non-IID splits, k-anonymity report
│   ├── model.py                     # LogReg / MLP definitions
│   ├── metrics.py                   # accuracy, recall, F1, AUC, confusion matrix
│   ├── centralized.py
│   ├── federated.py                 # Flower server strategy + client
│   ├── fedavg_reference.py          # hand-written FedAvg to cross-check Flower
│   ├── dp.py                        # Opacus wrapper and persistent accountant
│   ├── secagg.py                    # pairwise-mask simulation
│   ├── attack.py                    # membership inference
│   ├── comm_analysis.py
│   ├── run_all.py                   # runs every experiment, writes results/
│   └── tests/
├── results/                         # json, csv, png, run_meta.json
├── models/                          # state_dict .pt + model_config.json
├── backend/
│   ├── main.py  routers/  schemas.py  services/  tests/
└── frontend/
    └── (Vite React app)
```

---

## 4. Data layer

### 4.1 Cleaning
- Load with explicit encoding. Strip whitespace from column names and lower-case them; map names in ONE place (`schema.py`).
- Check missing values; if the count is zero, say so in the report. If not zero, impute with the training-set mode/median **after** the split and **per client** (so no cross-hospital statistics leak).
- **Remove exact duplicate rows BEFORE any splitting.** Duplicates split across train and test inflate test accuracy, and across clients they blur the "separate hospital" story. Report how many were removed.
- Encode target to `{0,1}` with a documented mapping (for example `YES=1`, `NO=0`). Encode survey-style `1/2` fields to `0/1`. Document the mapping in `schema.py`; the API reuses it.

### 4.2 Privacy-oriented preprocessing (the "data minimization" layer)

| Column type | Action | Notes |
|---|---|---|
| Direct identifiers (name, ID, phone, address) | Drop | Expected to be absent in this dataset; the script still checks and asserts none remain |
| Quasi-identifiers (`age`, `gender`) | Generalize | Age into 10-year bands for the privacy report |
| Clinical features (smoking, symptoms, etc.) | Keep | Protected by FL, DP and SecAgg, not by deletion |
| Target | Keep, local to each hospital | Never sent to the server in FL |
| Leakage columns (anything only known after diagnosis) | Drop | **[VERIFY]** in Step 0 |

- **k-anonymity check:** group by (`age_band`, `gender`, plus any other quasi-identifier) and report the minimum group size `k`, for the pooled data and for each hospital. Report honestly if `k` is small for some group.
- State clearly in the report: *dropping identifiers does not make data anonymous.* The layers are minimization, then FL, then DP/SecAgg.

### 4.3 Feature scaling without leakage
- Most features are expected to be binary; they need no scaling.
- Scale `age` with a **fixed constant** (for example `age / 100`) rather than fitting a scaler on data. This avoids fitting statistics on the test set and avoids pooling statistics across hospitals, which would contradict the federated setting.
- If another numeric feature exists **[VERIFY]**, use fixed domain-based bounds, or fit a scaler on training data only and document how the federated case handles it.

### 4.4 Splits
1. Global stratified split: **70% train / 10% validation / 20% test**, fixed seed.
2. The **test set** is held by the evaluator only and used for final numbers; every method (centralized, FL, FL+DP) is evaluated on the exact same file.
3. The **validation set** is used for choosing learning rate, rounds and thresholds. Never tune on the test set.
4. The train portion is partitioned into 3 hospital clients.

### 4.5 Hospital partitioning (R1, R7)
- **IID split:** random stratified into 3 near-equal parts (control).
- **Non-IID split (main scenario):**
  - *Quantity skew:* for example 50% / 30% / 20% of training rows.
  - *Label skew:* Dirichlet(α) over class proportions per client (α around 0.5), producing different cancer prevalence per hospital.
- Guarantees to enforce with `assert`:
  - Clients are disjoint and their union equals the training set.
  - Every client has **both classes** and at least a minimum number of rows (for example 100).
  - No client row appears in val or test.
- Save a `client_summary.json`: rows, positives, prevalence, age-band counts, and `k` per client. The dashboard reads it.

---

## 5. Model layer

### 5.1 Architectures
- **Primary:** small MLP `input → 32 → 16 → 1` with ReLU. About 1k parameters, so communication numbers are easy to explain.
- **Secondary:** logistic regression (`input → 1`) as a simple, interpretable reference.
- **Opacus compatibility:** no BatchNorm (use none, or GroupNorm). Validate with Opacus `ModuleValidator`.

### 5.2 Training setup
- Loss: `BCEWithLogitsLoss`, with `pos_weight` if the classes are imbalanced.
- Optimizer: SGD with momentum or Adam, one setting shared across all methods for fairness.
- **Fair budget:** centralized training gets the same *total number of gradient steps per sample* as the federated runs (rounds × local epochs), or report both "best" and "matched-budget" results. Do not compare 100 centralized epochs against 10 federated ones without saying so.

### 5.3 Metrics (`metrics.py`)
Accuracy, precision, **recall (sensitivity)**, specificity, F1, **ROC-AUC**, confusion matrix, plus ROC curve points saved for plotting. Recall and AUC lead the discussion because missing a cancer case is the costly error.
- Guard AUC against a single-class evaluation set (return `NaN` and log it instead of crashing).

### 5.4 Reproducibility
- Seeds `0..4` for the 5 repeated runs; set Python, NumPy and PyTorch seeds, and `torch.use_deterministic_algorithms(True)` where supported.
- Save `results/run_meta.json` with package versions, seeds, hyperparameters and timestamp.

---

## 6. Federated learning layer (R2)

### 6.1 Flower setup
- **Strategy:** FedAvg, aggregation weighted by each client's number of training examples.
- **Clients:** `NumPyClient` subclasses. Each loads **only its own hospital CSV**, trains for `E` local epochs, and returns parameters and `num_examples`.
- **Server:** with 3 clients, `fraction_fit=1.0`, `min_available_clients=3`, `min_fit_clients=3`.
- **Evaluation:** centralized evaluation function on the shared test set after each round (gives the global model's curve), plus per-client evaluation on each hospital's validation share.
- **Rounds:** 20 by default; also run early-stopping-free fixed rounds so curves are comparable.
- **Hyperparameters to sweep (small):** local epochs `E ∈ {1, 2, 5}`, learning rate, batch size.
- Log per round: global accuracy, recall, AUC, loss, per-client metrics, bytes sent/received, wall time. Save to `results/fl_<scenario>_seed<k>.json`.
- Save the final global model as `state_dict` plus `model_config.json`. **Do not pickle whole model objects.**

### 6.2 Reference implementation (verification, and the viva explanation)
Write `fedavg_reference.py`: a short hand-written FedAvg loop (about 40 lines). Use it to:
1. Cross-check that one round of Flower matches the manual weighted average within a small tolerance.
2. Explain aggregation line by line in the viva.
3. Act as a fallback if Flower fails to install on a machine.

### 6.3 Aggregation (guiding question 3)
`w_global = Σ_k (n_k / N) · w_k`, where `n_k` is client k's sample count and `N = Σ n_k`. Explain why weighting by `n_k` matters when hospitals have different sizes.

---

## 7. Differential Privacy layer (R8)

### 7.1 Method
Record-level **DP-SGD** applied to each client's local training with Opacus: per-sample gradient clipping with norm `C`, plus Gaussian noise `N(0, σ²C²)` added to the summed gradient. The server only ever sees noisy updates.

### 7.2 Parameters
- Clip norm `C`: 1.0 (tune briefly on validation).
- Noise multiplier sweep: `σ ∈ {0 (no DP), 0.3, 0.6, 1.0, 1.5, 2.0}` (adjust after seeing the results so the ε range is meaningful).
- δ: fixed below `1 / n_client`, for example `1e-5`. State it with every ε.
- Report **ε per client** and the maximum across clients, for each σ.

### 7.3 Accounting pitfalls (real bugs to avoid)
- **Accountant reset:** in Flower simulation, clients are re-created every round. If a fresh `PrivacyEngine`/accountant is created each round, ε is **under-reported**, because composition across rounds is lost. Fix: keep a per-client accountant history that persists across rounds (an external dict or file), and compute ε from the full history.
- **Poisson sampling:** Opacus uses variable batch sizes, so an empty batch is possible with small data. Handle it (skip the step) rather than crash.
- **Tiny client datasets:** large σ may collapse accuracy. That is an expected, reportable result, not a bug.
- Use consistent batch size and `sample_rate = batch_size / n_client` in all accounting.

### 7.4 Outputs
Table and plot: σ, ε (max over clients), accuracy, recall, AUC (mean ± std over 5 seeds). A plot of ε vs recall is the key privacy-utility figure.

State the interpretation precisely: this is *record-level DP against the server and anyone who sees the shared model updates*, not protection against a compromised hospital.

---

## 8. Secure Aggregation simulation (R8)

### 8.1 Protocol simulated (pairwise masking)
1. Each pair of clients `(i, j)` shares a seed `s_ij`.
2. Client `i` generates mask `m_ij = PRG(s_ij)` and adds `+m_ij` if `i < j`, or `−m_ij` if `i > j`, to its (pre-weighted) update.
3. The server receives only masked updates and sums them. Pairwise masks cancel, leaving the true aggregate.

### 8.2 Implementation details
- Use **integer arithmetic modulo 2^32** after fixed-point quantization of updates (scale, round). This makes cancellation **exact** and avoids float rounding noise. A float64 variant may be shown as a comparison with a reported max-abs-error.
- Clients pre-multiply their update by `n_k / N` so the plain sum equals FedAvg.
- Check to show in the report and demo:
  - `max |masked_sum − plain_sum|` is 0 (integer) or below `1e-6` (float).
  - A single masked update has near-zero correlation with the true update.
- Integrate as a variant strategy ("FedAvg + SecAgg") in the full run, and show the final model matches plain FedAvg.

### 8.3 Honest limitations (say these before the examiner asks)
- This **simulates** the masking idea; seeds are shared in the simulation, not derived by a real key-agreement protocol.
- If a client drops out mid-round, masks no longer cancel. Real protocols (Bonawitz et al.) use secret sharing to recover from dropouts. Demonstrate the failure case on purpose and explain it.
- Secure aggregation hides individual updates from the server; it does **not** protect the final aggregated model from inference. That is why DP is combined with it.

---

## 9. Privacy attack evaluation (R4, R6)

### 9.1 Membership inference (loss-threshold attack)
- Idea: a model tends to have lower loss on records it trained on. The attacker guesses "member" when the loss is below a threshold.
- Members: a sample of records from client training sets. Non-members: records never used in training (use records held out for this purpose, **not** the validation or test sets used elsewhere, or document the overlap).
- **Balance** member and non-member counts, so that 0.5 AUC means chance.
- Targets: centralized model, FL global model, FL+DP global model at several σ.
- Report attack AUC and accuracy.

### 9.2 Validity safeguards
- **Positive control:** deliberately overfit a larger MLP on a small subset to show that the attack *can* succeed. Without this, a result near 0.5 could just mean the attack code is broken.
- Honest reporting: if a tiny, well-regularized model gives attack AUC near 0.5 in every setting, say so and discuss why (little overfitting, small model). Do not tune the attack until a story appears.

### 9.3 Threat model (documented in `docs/threat_model.md` and the dashboard)

| Threat | Who | Mitigation in this project | Residual risk |
|---|---|---|---|
| Raw data exposure | Central server / breach | FL (data never leaves clients) | Updates still leak information |
| Membership inference | Server or model user | DP-SGD | Needs ε small; utility cost |
| Gradient / update inversion | Honest-but-curious server | SecAgg and DP | Simulated SecAgg is not cryptographic |
| Model poisoning / malicious client | Compromised hospital | Discussed; **[OPTIONAL]** experiment | Not mitigated by default |
| Client dropout / availability | Network | Discussed; failure demo for SecAgg | Needs real protocol |

---

## 10. Communication efficiency analysis (R5)

- **Parameter count** `P` from the model; bytes per client per round = `P × 4` (float32) uploaded plus the same downloaded.
- Per round total = `P × 4 × 2 × K` (K clients). Total over `R` rounds.
- **Centralized comparison:** raw data size on disk (bytes) as the "upload once" cost. Report honestly: with a tiny dataset and many rounds, FL may transfer **more** bytes than centralizing the data. FL's benefit here is privacy, not bandwidth.
- Measure: rounds to reach a target (for example 95% of the centralized AUC), wall-clock per round, effect of local epochs `E` (more local work means fewer rounds).
- **[OPTIONAL]** 8-bit quantization of updates and its accuracy cost.
- Outputs: bytes-per-round table, rounds-to-target plot, total-MB comparison vs raw data.

---

## 11. Experiment matrix

| ID | Experiment | Config | Seeds | Output |
|---|---|---|---|---|
| E1 | Centralized baseline | MLP and LogReg, pooled train | 5 | metrics |
| E2 | FL, IID | 3 clients, FedAvg, R rounds | 5 | metrics and per-round curve |
| E3 | FL, non-IID | quantity and label skew | 5 | metrics and per-client metrics |
| E4 | FL + DP sweep | σ grid, non-IID | 5 | ε table, utility curve |
| E5 | FL + SecAgg | non-IID | 1 to 5 | equality check with E3 |
| E6 | Communication | vary E, R | 3 | bytes, rounds-to-target |
| E7 | Membership inference | E1, E3, E4 models, plus positive control | 5 | attack AUC |
| E8 | Local-only (no collaboration) | each hospital trains alone | 5 | shows why FL helps |

E8 is an important addition: it shows that each hospital alone does worse than the federated model, which is the motivation for FL.

**Reporting rules:** mean ± std over seeds; all methods on the same test file; recall and AUC highlighted; same preprocessing and architecture across methods.

### 11.1 Results file layout
```
results/
├── run_meta.json
├── client_summary.json
├── comparison.json / comparison.csv
├── rounds_<scenario>.json
├── privacy_curve.json
├── attack.json
├── communication.json
├── secagg_check.json
├── roc_<model>.json
└── plots/*.png
```
All backend endpoints read from these files only.

---

## 12. Backend (FastAPI)

### 12.1 General
- Python module `backend/main.py`, app factory, routers split by topic, Pydantic schemas.
- CORS: allow the Vite dev origin (`http://localhost:5173`) via an env variable.
- Load results JSON and the model once at startup (cached), with a clear error if files are missing.
- Consistent error format: `{"error": {"code": "...", "message": "..."}}`.
- Version prefix `/api/v1`.

### 12.2 Endpoints

| Method | Path | Purpose | Notes |
|---|---|---|---|
| GET | `/api/v1/health` | Liveness, and whether results and model are loaded | Used by the frontend to show a "backend offline" banner |
| GET | `/api/v1/dataset/summary` | Rows, columns, class balance, duplicates removed, k-anonymity | From `client_summary.json` |
| GET | `/api/v1/clients` | Per-hospital rows, prevalence, age bands | |
| GET | `/api/v1/results/comparison` | Metrics table: centralized, FL IID, FL non-IID, FL+DP, local-only | mean ± std |
| GET | `/api/v1/results/rounds?scenario=` | Per-round curves | `scenario` validated against an allowlist |
| GET | `/api/v1/results/privacy-curve` | σ, ε, accuracy, recall, AUC | `epsilon: null` means "no DP" |
| GET | `/api/v1/results/attack` | Membership inference AUC per model, positive control | |
| GET | `/api/v1/results/communication` | Bytes per round, total MB, rounds-to-target, raw-data baseline | |
| GET | `/api/v1/results/roc?model=` | ROC points | optional |
| GET | `/api/v1/secagg/demo?seed=` | Live toy demo: 3 small vectors, masked versions, sums | Cheap and deterministic; no training |
| GET | `/api/v1/meta` | Package versions, seeds, hyperparameters | |
| POST | `/api/v1/predict` | Risk prediction from patient features | See below |

### 12.3 `POST /api/v1/predict`
- **Request:** JSON of feature fields in a canonical API format (booleans for yes/no, integer age), field names from `schema.py`, plus `model` ∈ `centralized | federated | federated_dp`.
- **Validation (Pydantic):** age within a plausible range (for example 1 to 120), required fields present, unknown fields rejected, enum for `model`. Return HTTP 422 with field-level messages.
- **Response:** `{probability, label, threshold, model_used, disclaimer}`.
- **Processing:** the API applies the **same** preprocessing function and **same feature order** as training, imported from the shared `schema.py`. Never re-implement encoding in the backend separately; a mismatch silently produces wrong predictions.
- **Safety:** always return a disclaimer ("educational demo, not medical advice"). **Do not log request bodies.**
- Models are loaded from `state_dict` + `model_config.json`; CPU only.

### 12.4 Backend edge cases
- A results file is missing (pipeline not yet run): return 503 with the message "run `python ml/run_all.py` first", not a stack trace.
- `epsilon = infinity` is not valid JSON: serialize as `null` and let the frontend label it "No DP".
- `NaN` values (single-class AUC) are not valid JSON: convert to `null`.
- Concurrent requests: the model is read-only after load, so this is safe.
- Reject extra fields to avoid silently ignoring typos.
- Large requests: Pydantic size limits; no file uploads.

---

## 13. Frontend (React + Vite + Recharts)

### 13.1 Pages and what each shows

| Page | Contents | Data source |
|---|---|---|
| **Overview** | Problem statement, why FL, architecture diagram, pipeline steps, rubric-style summary | static, `/meta` |
| **Data & Hospitals** | Dataset summary cards; table of 3 hospitals (rows, prevalence); class-distribution bar chart; before/after privacy preprocessing; k-anonymity value | `/dataset/summary`, `/clients` |
| **Training** | Round-by-round global accuracy/recall/AUC lines for IID vs non-IID vs centralized reference line; per-client curves; scenario selector | `/results/rounds` |
| **Comparison** | Metrics table (mean ± std) for all methods; metric toggle; optional ROC chart; confusion matrices | `/results/comparison`, `/results/roc` |
| **Privacy & Security** | ε vs recall/AUC curve; attack AUC bars with the 0.5 chance line; threat model table; short explainer of ε | `/results/privacy-curve`, `/results/attack` |
| **Communication** | Bytes per round, total MB, rounds-to-target, comparison against raw-data size | `/results/communication` |
| **Secure Aggregation** | Interactive toy demo: three updates, masked versions, sum equality, "drop a client" button showing masks failing to cancel | `/secagg/demo` |
| **Predict** | Form with all features, model selector, probability result, disclaimer | `POST /predict` |
| **Q&A / Report** | Answers to the 5 guiding questions and limitations | static |

### 13.2 Components
`Layout` (nav, offline banner), `StatCard`, `DataTable`, `MetricToggle`, `LineChartCard`, `BarChartCard`, `RiskGauge`, `PredictForm`, `SecAggDemo`, `Loading`, `ErrorBox`, `EmptyState`, `Tooltip` (explains ε, δ, FedAvg).

### 13.3 UI states and quality rules
- Every data component has **loading, error and empty** states.
- Backend offline: global banner; pages show a retry button.
- API base URL from `VITE_API_BASE_URL`; no hard-coded host.
- ε = `null` rendered as "No DP"; `NaN`/`null` metrics rendered as "n/a".
- Format numbers consistently (3 decimals for metrics, MB with 2 decimals).
- Do not rely on color alone: use markers/patterns and labelled legends; keep sufficient contrast; form inputs have labels.
- Responsive layout down to phone width; charts resize.
- Predict form: client-side validation mirrors server rules, but the server remains the authority; show server 422 messages next to the fields.
- No browser storage required; keep state in React state.
- Always show the "educational demo, not medical advice" notice on the Predict page.
- The form fields must be generated from the real feature list **[VERIFY]**; do not hard-code fields before Step 0.

---

## 14. Testing and validation

### 14.1 Automated tests (`pytest`)
- **Preprocess:** no identifier columns remain; no NaN; feature order equals `schema.py`; target is in `{0,1}`; duplicates removed.
- **Partition:** clients disjoint; union equals train; each client has both classes and minimum rows; no overlap with val/test; stratification within tolerance.
- **Flower vs reference:** one round of Flower FedAvg equals manual weighted average within tolerance (for example `1e-5`).
- **Weighted average:** unequal client sizes produce the expected weights.
- **SecAgg:** integer masked sum equals plain sum exactly; a dropped client breaks it (test that the failure is detected).
- **DP:** ε increases as σ decreases; with `σ = 0` and large `C` the model is close to the non-DP model; accountant history persists across rounds (ε after R rounds is greater than after 1).
- **Attack:** positive-control model has attack AUC clearly above 0.5; a randomly initialized model is near 0.5.
- **API:** FastAPI `TestClient` for each endpoint, including 422 on bad input, 503 on missing results, and predict output range `[0,1]`.
- **Serving parity:** predictions from the API equal predictions from the training-time pipeline on the same rows.

### 14.2 Manual checks before submission
- Re-run `run_all.py` from a clean clone and confirm the same numbers (given the same seeds).
- Open every dashboard page with the backend running and with it stopped.
- Time a full demo run.

---

## 15. Catalogue of edge cases, bugs and risks

| # | Area | Problem | Symptom | Prevention / fix |
|---|---|---|---|---|
| 1 | Data | Duplicate rows across train/test | Inflated test accuracy | Dedupe before any split; report count |
| 2 | Data | Target encoding mistake (`1/2` vs `0/1`) | Model predicts the inverted class; metrics look terrible or perfect | One mapping in `schema.py`; assert on unique values; test |
| 3 | Data | Column names with spaces/case/BOM | `KeyError` | Normalize names on load |
| 4 | Data | Scaler fit on all data | Test leakage; breaks FL purity | Fixed constants for age; no data-dependent fit |
| 5 | Data | Class imbalance | High accuracy but poor recall | `pos_weight`, report recall/AUC, stratified splits |
| 6 | Data | Dataset too easy (near 100% accuracy) | No visible centralized-vs-FL gap | Say so in the report; emphasize privacy, communication and attack results; do not manufacture differences |
| 7 | Data | Description mismatch (vital signs, treatment mentioned on page) | Report claims features that do not exist | Describe only real columns |
| 8 | Data | Synthetic/survey data | Examiner questions realism | State the limitation openly in the report |
| 9 | Partition | A client ends up with one class | AUC undefined; training degenerate | Asserts on both classes and minimum rows; resample Dirichlet if violated |
| 10 | Partition | Non-IID too extreme | FL diverges | Choose moderate α; optionally show FedProx |
| 11 | Partition | Hospital file read by server code | Breaks the FL story | Server code only touches the shared test file |
| 12 | Model | BatchNorm with Opacus | Opacus raises an error | No BatchNorm; run `ModuleValidator` |
| 13 | Model | Parameter ordering mismatch between client and server | Garbage global model | Use a single helper for get/set parameters; test round-trip |
| 14 | Model | dtype mismatch (float64 vs float32) | Size and numerical issues | Cast to float32 consistently |
| 15 | Fairness | Centralized trained with more epochs than FL | Unfair comparison | Matched-budget comparison and disclose |
| 16 | Eval | Tuning on the test set | Optimistic numbers | Use validation set; touch test only at the end |
| 17 | Eval | Single seed | Results look like luck | 5 seeds, mean ± std |
| 18 | Eval | AUC on single-class set | Exception or NaN | Guard and log, return null |
| 19 | Flower | API/version differences | Code from the web does not run | Pin versions; consult the installed version's docs; keep the reference FedAvg as fallback |
| 20 | Flower | `ray` install/simulation issues on Windows | Simulation will not start | Use Colab/Linux or WSL; fallback to the reference loop |
| 21 | Flower | Clients re-created each round | State (accountant, optimizer) lost | Persist state externally where needed (see DP) |
| 22 | DP | Accountant reset every round | ε under-reported (a serious correctness error) | Persist accountant history across rounds; test it |
| 23 | DP | Empty Poisson batch | Crash or NaN loss | Skip empty batches |
| 24 | DP | δ larger than `1/n` | ε claim not meaningful | Fix δ below `1/n_client`; report with every ε |
| 25 | DP | Large σ collapses accuracy on tiny clients | Utility drop looks like a bug | Expected; report as the privacy-utility trade-off |
| 26 | DP | Misinterpreting ε | Overclaiming privacy in viva | Be able to explain ε, δ, record-level scope |
| 27 | SecAgg | Float rounding does not cancel exactly | Tiny nonzero error | Integer modular arithmetic with fixed-point quantization |
| 28 | SecAgg | Client dropout | Masks do not cancel | Demonstrate it; explain the real protocol needs secret sharing |
| 29 | SecAgg | Overclaiming security | Examiner challenges it | Call it a simulation; list limitations |
| 30 | Attack | Members/non-members unbalanced | Misleading AUC | Balance the sets |
| 31 | Attack | Non-members actually used in training | Attack looks weaker than reality | Use a dedicated untouched set |
| 32 | Attack | Result near 0.5 everywhere | Looks like a broken attack | Positive-control overfit model |
| 33 | Comm | Counting only uploads | Understated cost | Count both directions; per client and total |
| 34 | Comm | FL transfers more than raw data | Surprising to examiners | State honestly; FL's benefit here is privacy |
| 35 | Backend | Model/preprocessing mismatch at serving | Wrong predictions silently | Share `schema.py` and a preprocessing function; parity test |
| 36 | Backend | `inf`/`NaN` in JSON | Invalid JSON or 500 | Serialize as `null` |
| 37 | Backend | CORS not configured | Frontend gets blocked requests | Allow the dev origin through env |
| 38 | Backend | Results missing | 500 error | 503 with clear message |
| 39 | Backend | Logging patient inputs | Privacy contradiction | Do not log request bodies |
| 40 | Frontend | Backend offline during the demo | Blank pages | Banner, retry button; run the demo from saved results |
| 41 | Frontend | Hard-coded API URL | Breaks on another machine | `VITE_API_BASE_URL` |
| 42 | Frontend | Form fields not matching real features | Predict errors | Generate fields from the real feature list |
| 43 | Ops | Colab session reset | Lost files | Save results and models to Drive or commit to git often |
| 44 | Ops | Different package versions across machines | Non-reproducible results | Pinned `requirements.txt`; record versions in `run_meta.json` |
| 45 | Ops | Kaggle download needs a login | Delays | Download once and share the same CSV |
| 46 | Ops | Windows path issues | Scripts fail | Use `pathlib` everywhere |
| 47 | Ops | Live federated training during the demo | Timeouts and failures | Demo precomputed results; show a small, safe live element only (secagg demo) |

---

## 16. Implementation order, with acceptance criteria and cut lines

| Phase | Work | Done when | Approx. time |
|---|---|---|---|
| 0 | Environment, pinned `requirements.txt`, download CSV, **Step 0 inspection**, `schema.py` | Columns, target and encodings written down; imports work | 1 h |
| 1 | Preprocess, privacy step, k-anonymity, partition IID/non-IID, tests | Tests pass; `client_summary.json` produced | 1 h |
| 2 | Model, metrics, centralized baseline, local-only baseline | E1 and E8 numbers with 5 seeds | 1 h |
| 3 | Reference FedAvg, then Flower FedAvg, cross-check test | E2 and E3 run; Flower equals reference within tolerance | 2 h |
| 4 | DP with persistent accountant, σ sweep | E4 table; ε monotonic in σ | 1.5 h |
| 5 | SecAgg module, integrate variant, dropout demo | E5 exact; tests pass | 1 h |
| 6 | Membership inference with positive control; communication analysis | E6 and E7 outputs | 1.5 h |
| 7 | Backend endpoints and tests | All endpoints return real results | 1 h |
| 8 | Frontend pages | All pages render real data; offline state works | 2 h |
| 9 | Report, slides, demo script, viva practice | Report answers Q1 to Q5 with real numbers | 2.5 h |

**Cut lines if time runs short (cut in this order):** 1) frontend polish and extra charts, 2) optional extras, 3) ROC page, 4) Predict page (keep Comparison, Privacy and Communication pages), 5) never cut: Flower FedAvg, comparison, DP with correct ε, SecAgg exactness check, attack with positive control, report.

---

## 17. Report outline (maps to the marks)

1. Introduction and problem statement (centralized ML privacy risk, healthcare setting)
2. Dataset: source, real columns, audit results, privacy preprocessing, k-anonymity, limitations (survey/synthetic)
3. System design: architecture diagram, hospital clients, IID vs non-IID
4. Methods: FedAvg and weighting, DP-SGD (C, σ, δ, accounting), secure aggregation, membership inference
5. Experiments and results: E1 to E8 tables and plots
6. Analysis: privacy-utility trade-off, communication efficiency, heterogeneity, security risks and threat model
7. Answers to the five guiding questions
   1. FL vs centralized ML: where data lives, what is shared, evidence from our results
   2. Challenges: communication overhead, heterogeneity (our non-IID results), security risks
   3. Model aggregation: the FedAvg formula and our weighted example
   4. Privacy and security techniques: DP, secure aggregation, our measurements
   5. Real-world applications in healthcare and finance, with cited examples
8. Limitations and future work (real cryptographic SecAgg, larger datasets, real clinical data, poisoning defenses)
9. Conclusion; references; appendix (Step 0 output, hyperparameters, versions)

### Slides (about 12)
Title; problem; why FL; dataset and privacy preprocessing; architecture; methods (FedAvg, DP, SecAgg); experiment design; accuracy comparison; privacy-utility curve; attack and communication results; limitations; conclusion and demo.

### Demo script (2 minutes)
1) Show the hospital split; 2) show comparison table; 3) show the ε curve; 4) run the secagg demo and drop a client; 5) make one prediction; 6) state limitations.

---

## 18. Viva preparation: questions that must be answerable

1. Why does FL improve privacy if updates still leak information?
2. What exactly does the server see in each of your three setups?
3. Walk through one FedAvg round with numbers.
4. Why weight by number of samples?
5. What is ε? What is δ? What is the privacy unit in your DP?
6. Why did you choose σ and C? What happens as σ grows?
7. How did you account for privacy across rounds? (accountant persistence)
8. Is your secure aggregation really secure? What does it not protect?
9. What happens when a client drops out in secure aggregation?
10. How did you make the hospitals non-IID, and why?
11. Why is the federated model not better than the centralized one (or why is it close)?
12. How did you verify Flower's aggregation was correct? (reference loop test)
13. How do you know your membership inference attack works? (positive control)
14. Why report recall and AUC rather than accuracy only?
15. How do you know there was no train/test leakage? (dedupe, fixed-constant scaling, validation set)
16. Does FL reduce communication here? (honest answer with numbers)
17. What are the limitations of your dataset?
18. What would you change for real hospitals?
19. Where do the hospital CSVs get read, and why never by the server?
20. How would you defend against a malicious client?

**Rehearsal rule:** every person must be able to explain every module. Spend the last hour asking each other these questions.

---

## 19. Final deliverables checklist

- [ ] Code repository with `README.md` (how to install, run pipeline, run backend, run frontend)
- [ ] `requirements.txt` with pinned versions and `package.json`
- [ ] `results/` with all JSON/CSV/PNG and `run_meta.json`
- [ ] Trained models and `model_config.json`
- [ ] Passing tests
- [ ] Report (PDF/Word) with Step 0 output in the appendix
- [ ] Slides
- [ ] Working dashboard and a rehearsed demo
- [ ] Viva Q&A rehearsed

---

## 20. Open items

1. Paste the Step 0 output so schema, target and preprocessing can be finalized.
2. Confirm the compute environment (Colab or local machines, and the OS), which decides how Flower simulation is run.
3. Install the stack and record the versions of `flwr`, `torch` and `opacus`; the Flower and Opacus code is written against those exact versions.
