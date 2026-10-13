# Threat model

Assets: patient records held by each hospital; the trained model; individual model updates.
Trust assumptions: the server is *honest-but-curious* (follows the protocol, tries to learn from what it sees). Hospitals are honest unless stated.

| Threat | Adversary | Mitigation in this project | Evidence | Residual risk |
|---|---|---|---|---|
| Raw data exposure | Central server / data breach | FL: rows never leave hospital; server code reads only the shared test/val files | Architecture; `HospitalClient` is the only reader of hospital CSVs | Updates still carry information about training data |
| Membership inference | Server or model user | DP-SGD (per-sample clipping C=1, noise σ) | `results/attack.json`: attack AUC per model vs positive control | Needs small ε; stronger attacks (shadow models, label-only) were not run |
| Update / gradient inversion | Honest-but-curious server | Secure aggregation (server sees only masked sums) + DP | `results/secagg_check.json`: masked updates uncorrelated with true updates; integer sum exact | Simulated, not cryptographic; final aggregate is still visible |
| Model poisoning / malicious client | Compromised hospital | Not mitigated (discussed). Robust aggregation (median, trimmed mean, norm clipping) would be the first step | none | A single bad client can shift FedAvg |
| Dropout / availability | Network failure or malicious abort | SecAgg failure demo: a missing client leaves masks uncancelled and the round is aborted | `ml/tests/test_fl.py::test_secagg_dropout_breaks_cancellation` | A real protocol (Bonawitz et al.) recovers with secret sharing |
| Re-identification from quasi-identifiers | Anyone with side data | Age generalised to 10-year bands for the report; k-anonymity measured | `results/client_summary.json` (pooled k=25; per-hospital k from 1 to 8) | Hospital 3 (non-IID) has k=1; dropping identifiers is not anonymisation |
| Serving-time leakage | API operator | Request bodies are never logged; no storage | Backend code review | Predictions themselves can leak about inputs |

## Live system (Render + Hugging Face)
| Threat | Adversary | Mitigation | Residual risk |
|---|---|---|---|
| Unauthorised upload/download | Internet attacker | HMAC-signed, expiring bearer tokens; hospital-only and admin-only routes | Stolen token valid until expiry (8 h default) |
| Credential guessing | Internet attacker | 20-character derived passwords; 5 failures lock a username for 60 s; constant-time comparison | Throttle is per process and in memory |
| Stale / replayed update | Honest client or replayer | Updates carry the base version; anything not built on the current version gets 409 | A replay of a still-current update just replaces that hospital's pending update |
| Poisoned or diverged update | Compromised hospital | L2 norm cap; optional median aggregator (measured: 0.598 to 0.799 AUC under label flipping) | Needs 3+ hospitals; an attacker staying under the norm cap with a small bias is not detected |
| Malformed payload / parser abuse | Attacker | `np.load(allow_pickle=False)`, shape and finiteness checks, 200 KB request cap | none known |
| Server sees individual updates | Honest-but-curious operator | Optional client-side DP (`--sigma`); secure aggregation only simulated | Updates are visible to the server |
| Data in transit | Network attacker | Render HTTPS | Terminates at Render's edge |
| Secret leakage | Anyone with repo access | `SECRET_KEY`, `ADMIN_PASSWORD`, `HF_TOKEN` only in Render's environment, never in git | Rotating `SECRET_KEY` invalidates all hospital passwords |
| Hospital identity in public history | Public | Use pseudonymous hospital ids (documented) | Ids and row counts are public in `history.json` |

## DP guarantee, stated precisely
Record-level (ε, δ)-DP per hospital over the full training run, composed across all rounds with a persistent RDP accountant, at δ = 1e-5 (below 1 / n_k for every hospital). It protects a single patient's record against anyone who sees the shared updates or final model. It does **not** protect against a hospital that is itself malicious, and does not hide *that* a hospital participated.
