# Decentralized Machine Learning: Implementing Federated Learning for Privacy Preservation

Data Privacy and Security Analytics Lab, CAT-2 project. Code, results and dashboard: this repository. Every number below is read from `results/*.json` (mean ± std over 5 seeds unless stated).

## 1. Introduction and problem statement
Training a disease-prediction model normally means copying every hospital's patient records to one place, which concentrates privacy risk (breach, misuse, regulation). Federated learning (FL) keeps each hospital's rows local and shares only model updates. Updates can still leak information, so we add differential privacy (DP) and secure aggregation, and we measure what each layer costs and protects.

We simulate three hospitals, train a lung-cancer classifier five ways (centralized, local-only, FedAvg, FedAvg + DP, FedAvg + secure aggregation), compare accuracy and privacy leakage on one shared test set, and analyse communication cost, heterogeneity and security risks.

## 2. Dataset
**Source:** Kaggle `akashnath29/lung-cancer-dataset`. **Audit (Step 0, see Appendix A):** 3,000 rows × 16 columns, no missing values, 2 exact duplicates removed (2,998 rows used). Features: gender, age (30–80) and 13 yes/no survey fields (smoking, yellow fingers, anxiety, peer pressure, chronic disease, fatigue, allergy, wheezing, alcohol consuming, coughing, shortness of breath, swallowing difficulty, chest pain). Target `LUNG_CANCER` (50.6% positive). The Kaggle page mentions vital signs and treatment; those columns do not exist and are not claimed.

**Privacy preprocessing:** no direct identifiers exist (the code checks and asserts this). Age is generalised to 10-year bands for the privacy report; the model uses `age / 100` (a fixed constant, so no statistic is fitted on any data). Pooled k-anonymity over (age band, gender) is 25; per hospital it is 3–8 in the IID split and **1**–8 in the non-IID split (hospital 3 has a group of one). Dropping identifiers is not anonymisation; the protection comes from the layers below.

**Key limitation: the label is essentially unlearnable from these features.** A 5-fold cross-validated logistic regression on the pooled data scores ROC-AUC 0.501 and a random forest 0.523, and per-class feature means are nearly identical. The data behaves like random survey data. All accuracy comparisons are therefore near chance (about 0.5), and we say so instead of tuning until a gap appears. Pipeline correctness is established separately on synthetic planted-signal labels (§6.6).

**Splits** (stratified, seed 42): 60% train (1,798), 10% validation (300), 20% test (600), 10% attack hold-out (300; used only as membership-inference non-members). This deviates from the plan's 70/10/20 to obtain never-trained non-members.

## 3. System design
Three simulated hospitals; only the `HospitalClient` code opens a hospital's CSV. The server code aggregates and evaluates on the shared test file only.

| Scenario | H1 rows / prevalence | H2 | H3 |
|---|---|---|---|
| IID control | 600 / 50.7% | 599 / 50.6% | 599 / 50.6% |
| Non-IID (main) | 900 / 43.0% | 539 / 58.1% | 359 / 58.5% |

Non-IID construction: quantity skew 50/30/20, plus label skew where each hospital's prevalence comes from a Dirichlet(1) draw, resampled until it lies in [0.15, 0.85], with asserted guarantees (disjoint clients, union = train, both classes, ≥100 rows, no overlap with val/test/attack).

Model: MLP 15 → 32 → 16 → 1 (1,057 parameters, no BatchNorm so it is Opacus-compatible). SGD (lr 0.05, momentum 0.9, batch 32). FL: 20 rounds × 2 local epochs. Centralized training uses a matched budget of 40 epochs.

## 4. Methods
**FedAvg.** `w_global = Σ_k (n_k / N) · w_k`. Example: two hospitals with n = 1 and 3 and weights 1 and 4 give (1·1 + 3·4)/4 = 3.25. Weighting by n_k stops a small hospital from counting as much as a large one. We use Flower 1.39's `FedAvg` strategy and `NumPyClient` clients, run in-process (ray/gRPC simulation is unreliable on Windows). A hand-written 40-line FedAvg (`fedavg_reference.py`) reproduces a full Flower round to within 6e-8.

**DP-SGD (Opacus).** Per-sample gradient clipping at C = 1 plus Gaussian noise σ·C, Poisson sampling. δ = 1e-5, below 1/n_k for every hospital. Flower re-creates clients every round, so a naïve accountant would reset and under-report ε; we keep a persistent per-client RDP accountant history across all rounds (tested: ε after 3 rounds exceeds ε after 1). ε is reported per hospital and as the maximum. Scope: record-level DP against the server and anyone who sees updates; not against a compromised hospital.

**Secure aggregation (simulation).** Each pair of hospitals shares a seed; the lower-indexed one adds a pseudorandom mask and the higher-indexed one subtracts it. Updates are pre-weighted by n_k/N, fixed-point quantised (2^-20) and masked in integer arithmetic mod 2^32, so masks cancel exactly. This is a simulation: seeds are shared directly (no key agreement) and there is no dropout recovery.

**Membership inference.** Loss-threshold attack: guess "member" when per-sample loss is low. Members are training rows, non-members come from the untouched hold-out, sets are balanced (AUC 0.5 = chance). A positive control (a deliberately overfit 256-256 MLP on 150 rows) must succeed, and an untrained model must sit at chance.

## 5. Experiments
E1 centralized (MLP, LogReg); E2 FL IID; E3 FL non-IID; E4 FL + DP sweep σ ∈ {0.5, 0.8, 1.0, 1.5, 2.0, 3.0}; E5 FL + SecAgg; E6 communication (local epochs 1/2/5); E7 membership inference; E8 local-only hospitals. Seeds 0–4 (E6: 0–2). Hyperparameters were fixed up front and not tuned on the test set.

## 6. Results and analysis

### 6.1 Accuracy comparison (test set, 600 rows)
| Method | ROC-AUC | Recall | Accuracy |
|---|---|---|---|
| Centralized MLP | 0.529 ± 0.019 | 0.515 ± 0.066 | 0.520 ± 0.018 |
| Centralized LogReg | 0.505 ± 0.009 | 0.304 ± 0.292 | 0.506 ± 0.015 |
| Local-only hospital 1 / 2 / 3 | 0.515 / 0.510 / 0.502 | 0.473 / 0.702 / 0.685 | 0.511 / 0.503 / 0.499 |
| FedAvg IID | 0.524 ± 0.010 | 0.574 ± 0.150 | 0.517 ± 0.009 |
| FedAvg non-IID | 0.515 ± 0.015 | 0.569 ± 0.096 | 0.513 ± 0.010 |
| FedAvg + SecAgg (non-IID) | 0.516 ± 0.015 | 0.545 ± 0.107 | 0.511 ± 0.009 |

The 95% chance band for AUC at n = 600 is 0.5 ± 0.046. **Every method lies inside it.** There is no measurable accuracy difference between centralized, local-only and federated learning on this dataset. The honest conclusion is "no detectable accuracy cost from federating", not "FL is better". High local-only recall (0.70) with AUC 0.5 just means those models predict the majority class (hospitals 2 and 3 have 58% prevalence). Heterogeneity (IID 0.524 vs non-IID 0.515) is likewise within noise.

### 6.2 Differential privacy: ε vs utility
| σ | ε (max over hospitals) | AUC | Recall |
|---|---|---|---|
| none | ∞ | 0.515 ± 0.015 | 0.569 ± 0.096 |
| 0.5 | 69.6 | 0.510 ± 0.013 | 0.582 ± 0.059 |
| 0.8 | 22.5 | 0.508 ± 0.012 | 0.577 ± 0.090 |
| 1.0 | 14.3 | 0.504 ± 0.018 | 0.568 ± 0.105 |
| 1.5 | 7.2 | 0.502 ± 0.028 | 0.495 ± 0.121 |
| 2.0 | 4.8 | 0.491 ± 0.033 | 0.510 ± 0.163 |
| 3.0 | 2.9 | 0.499 ± 0.026 | 0.517 ± 0.180 |

(δ = 1e-5, 20 rounds × 2 epochs.) Smaller hospitals get a larger ε at the same σ (σ = 1: 8.5, 11.7, 14.3 for hospitals 1, 2, 3) because their sampling rate is higher. Utility does not visibly fall as ε shrinks because every setting is already at chance; there is no signal to lose. The real utility cost of DP is visible on planted-signal labels (§6.6).

### 6.3 Privacy leakage (membership inference)
| Target model | Attack AUC |
|---|---|
| Centralized MLP | 0.624 ± 0.013 |
| FedAvg (non-IID) | 0.558 ± 0.020 |
| FedAvg + DP σ = 0.5 / 1.0 / 2.0 / 3.0 | 0.529 / 0.526 / 0.513 / 0.510 |
| Positive control (overfit, 150 rows) | 0.858 ± 0.021 |
| Negative control (untrained) | 0.491 ± 0.024 |

The attack works (positive control) and is calibrated (untrained model at chance). **Leakage falls from centralized (0.624) to FedAvg (0.558) to FedAvg + DP, approaching chance as σ grows.** The models leak at all because, with random labels, a 1,057-parameter MLP trained for 40 epochs memorises rows, which is precisely what membership inference exploits. This is the answer to "compare federated and centralized in terms of privacy": under the same budget the federated model leaks less, and DP reduces it further. Caveat: this is one simple attack; stronger attacks (shadow models) may do better, and a low attack AUC is not a proof of privacy.

### 6.4 Secure aggregation
Integer masked sum vs plain integer sum: max difference **0** (exact). Error vs float FedAvg in one round: 1.3e-6 (fixed-point step). A single masked update has correlation with the true update of −0.03, −0.05 and +0.01 (no usable information). If one hospital drops out, the masks do not cancel and the aggregate error is about 2,048, so the failure is detected (the strategy aborts the round). Over 20 rounds the SecAgg model drifts from plain FedAvg by 0.06 in max parameter difference while test AUC differs by ≤ 0.01. Control: perturbing plain FedAvg's initial weights by only 1e-6 produces a 0.033 drift, so this drift is SGD sensitivity on noise-only labels, not a secure-aggregation error (`results/secagg_check.json`, `ml/secagg_control.py`).

### 6.5 Communication
The model has 1,057 float32 parameters: 8.5 KB per hospital per round (upload + download), 25 KB per round for three hospitals, **0.51 MB over 20 rounds**, independent of local epochs. Uploading the raw data once costs 0.084 MB (CSV). **FL moves about 6× more bytes than centralizing here**; its benefit on this dataset is privacy, not bandwidth. With larger data or fewer rounds the balance reverses. More local epochs cost more wall time per round (0.25 s for E = 1, 0.51 s for E = 5). An AUC-based rounds-to-target is meaningless at AUC ≈ 0.5 (the untrained model already meets it), and the training-loss convergence proxy is trivial for E ≤ 2 (the loss is flat near 0.69) and 16 rounds for E = 5.

### 6.6 Pipeline validation on synthetic labels (not the real data)
To show the pipeline can learn, we keep the real feature rows and hospital partitions but plant a signal in the labels (oracle AUC 0.83). Test AUC: centralized 0.778 ± 0.011; FedAvg IID 0.805 ± 0.004; FedAvg non-IID 0.794 ± 0.010; FedAvg + DP (σ = 1, ε = 14.3) 0.804 ± 0.007; FedAvg + SecAgg 0.794 ± 0.010 (3 seeds). Federated and federated + DP models are close to, and here slightly above, centralized (averaging and noise act as regularisation). So the near-chance real-data results come from the data, not from a pipeline bug.

### 6.7 Threat model
See `docs/threat_model.md`: raw-data exposure (mitigated by FL), membership inference (DP; measured), update inversion (SecAgg + DP; simulated), poisoning (not mitigated), dropout (SecAgg failure demonstrated), re-identification (k-anonymity measured; hospital 3 has k = 1).

## 7. Answers to the five guiding questions
1. **FL vs centralized ML.** Centralized ML copies all records to one server; FL keeps rows in each hospital and shares only weights. Measured: no accuracy difference (all within the chance band), but lower membership-inference leakage for FL (0.558 vs 0.624).
2. **Main challenges.** *Communication:* 0.51 MB vs 0.084 MB here. *Heterogeneity:* hospitals differ in size and prevalence; local-only models collapse to the majority class, and the global model's fit to each hospital's own training data differs by hospital (visible on the Training page). *Security:* updates leak (membership inference above chance), clients can poison, dropouts break masking.
3. **Aggregation.** FedAvg, `w = Σ (n_k/N) w_k`; verified against a hand-written loop (6e-8).
4. **Privacy and security techniques.** DP-SGD with persistent accounting (ε 2.9–69.6 across σ 3.0–0.5, δ = 1e-5), secure aggregation (exact cancellation, dropout detected), data minimisation and k-anonymity; evaluated with an attack that has a positive control.
5. **Applications.** Healthcare: multi-hospital imaging/EHR models, for example the EXAM study predicting COVID-19 oxygen needs across 20 hospitals and the FeTS brain-tumour segmentation initiative. Finance: cross-bank fraud and anti-money-laundering detection without sharing customer transactions, credit scoring across institutions (US-UK PETs prize challenges targeted federated fraud detection). Consumer: keyboard prediction (Gboard).

## 8. Limitations and future work
Near-random synthetic-looking dataset (so accuracy claims are void); simulated, not cryptographic, secure aggregation; no poisoning defence (robust aggregation would be next); one simple attack; in-process Flower instead of a networked deployment; three hospitals and 1,798 training rows; Opacus noise is not from a secure RNG; not medically validated. Future: real clinical data, real key-agreement SecAgg with dropout recovery, shadow-model attacks, FedProx under stronger skew.

## 9. Conclusion
We built and verified an end-to-end federated pipeline (Flower, DP-SGD with correct cross-round accounting, simulated secure aggregation, membership inference with controls, communication analysis, API and dashboard). On this dataset the label carries no learnable signal, so accuracy cannot discriminate between methods; the informative findings are that federating and adding DP progressively reduce membership-inference leakage (0.624 → 0.558 → ~0.51), that secure aggregation is exact and fails detectably on dropout, and that FL costs more bytes than uploading this small dataset once.

## Appendix A: Step 0 output
`shape (3000, 16)`; columns `GENDER, AGE, SMOKING, YELLOW_FINGERS, ANXIETY, PEER_PRESSURE, CHRONIC_DISEASE, FATIGUE, ALLERGY, WHEEZING, ALCOHOL_CONSUMING, COUGHING, SHORTNESS_OF_BREATH, SWALLOWING_DIFFICULTY, CHEST_PAIN, LUNG_CANCER`; missing 0; duplicates 2; every column except AGE (51 values) has 2 unique values; target YES 50.6% / NO 49.4%. See `docs/data_decisions.md` and `results/preprocess_report.json`.

## Appendix B: reproducibility
`results/run_meta.json` records versions (Python 3.14, torch 2.14.1, flwr 1.39.0, opacus 1.6.0), seeds and hyperparameters. Re-run with `python -m ml.run_all`; tests: `python -m pytest ml/tests backend/tests -q` (25 ML + 30 backend).
