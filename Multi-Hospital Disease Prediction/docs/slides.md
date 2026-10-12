---
marp: true
theme: default
paginate: true
---

# Decentralized ML: Federated Learning for Privacy Preservation
Lung-cancer prediction across three simulated hospitals
Flower · Opacus (DP-SGD) · Secure aggregation · Membership inference

---

# Problem
- Centralizing patient records concentrates privacy risk (breach, misuse, regulation)
- Federated learning: train together, keep rows local, share only model updates
- Updates still leak, so we add DP and secure aggregation and **measure** the cost and the protection

---

# Why federated?
```
Hospital 1 ─┐ local training
Hospital 2 ─┼─ weights only ─▶ Server: w = Σ (n_k / N) · w_k ─▶ global model back to all
Hospital 3 ─┘ raw rows never leave
```
- Server code never opens a hospital file; it only sees the shared test set

---

# Dataset and privacy preprocessing
- Kaggle lung-cancer: 3,000 rows, 15 features (gender, age, 13 yes/no), 2 duplicates removed → 2,998
- No identifiers; age → 10-year bands for the report, model uses age/100 (fixed constant)
- k-anonymity: pooled 25; per hospital 1–8 (non-IID hospital 3 has k = 1)
- **Finding:** label is ~unlearnable (CV AUC: logistic 0.501, forest 0.523) → everything is near chance. Reported honestly.

---

# Architecture and hospitals
- 60 / 10 / 20 / 10 split: train / validation / test / attack hold-out
- IID control vs **non-IID**: sizes 900 / 539 / 359, prevalence 43% / 58% / 58%
- MLP 15→32→16→1 (1,057 params), 20 rounds × 2 local epochs; centralized gets a matched 40 epochs
- Flower `FedAvg` + `NumPyClient`, 5 seeds, one shared test file

---

# Methods
- **FedAvg:** weight by sample count; verified against a 40-line reference (diff 6e-8)
- **DP-SGD:** clip C = 1, noise σ·C, δ = 1e-5, **persistent accountant** across rounds (a per-round reset under-reports ε)
- **Secure aggregation:** pairwise masks, integer mod 2³² → exact cancellation (simulation)
- **Membership inference:** loss threshold, balanced sets, positive and negative controls

---

# Accuracy: no difference (data has no signal)
| Method | AUC |
|---|---|
| Centralized MLP | 0.529 ± 0.019 |
| FedAvg IID / non-IID | 0.524 / 0.515 |
| FedAvg + SecAgg | 0.516 |
| Local-only (3 hospitals) | 0.515 / 0.510 / 0.502 |

All inside the 95% chance band 0.5 ± 0.046. Pipeline validated on planted-signal labels: centralized 0.78, FedAvg 0.79–0.81, FedAvg + DP 0.80.

---

# Privacy-utility curve
| σ | ε | AUC |
|---|---|---|
| none | ∞ | 0.515 |
| 0.5 | 69.6 | 0.510 |
| 1.0 | 14.3 | 0.504 |
| 2.0 | 4.8 | 0.491 |
| 3.0 | 2.9 | 0.499 |

No visible utility drop because there is no signal to lose. Smaller hospitals get larger ε at equal σ.

---

# Attack results: leakage falls
| Target | Attack AUC |
|---|---|
| Centralized MLP | **0.624** |
| FedAvg | **0.558** |
| FedAvg + DP σ = 0.5 → 3.0 | 0.529 → 0.510 |
| Positive control | 0.858 (attack works) |
| Untrained model | 0.491 (chance) |

---

# Secure aggregation and communication
- Integer masked sum = plain sum (diff **0**); one masked update ≈ uncorrelated with the truth
- Dropout: masks do not cancel (error ≈ 2,048) → round aborted; real protocols need secret sharing
- Communication: 8.5 KB / hospital / round, **0.51 MB** for 20 rounds vs **0.084 MB** to upload the raw data once → here FL costs ~6× more bytes; its benefit is privacy

---

# Limitations
- Near-random survey data; no clinical claim
- Simulated (not cryptographic) secure aggregation; no poisoning defence
- One simple attack; in-process Flower; 3 hospitals, ~1.8k rows
- Next: real data, real key agreement, shadow-model attacks, robust aggregation

---

# Conclusion and demo
- Verified end-to-end FL pipeline with correct DP accounting and exact secure aggregation
- Federating and adding DP progressively reduce membership-inference leakage (0.624 → 0.558 → ~0.51)
- Accuracy cannot separate methods on this dataset
- **Demo:** dashboard → hospitals → comparison → ε curve → secure-aggregation dropout → one prediction
