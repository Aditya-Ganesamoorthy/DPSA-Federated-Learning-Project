# Viva preparation: 20 questions with answers from our own results

Rehearsal rule: every team member must be able to explain every module. Numbers come from `results/*.json`.

1. **Why does FL improve privacy if updates still leak?** It removes the biggest exposure: raw rows are never copied to a server, so a server breach exposes only weights. Updates do leak (our loss-threshold attack gets AUC 0.558 on FedAvg), which is exactly why we add DP and secure aggregation. FL is a first layer, not a guarantee. It also leaked less than the centralized model under the same budget (0.558 vs 0.624).

2. **What does the server see in each setup?** Centralized: all rows and labels. Plain FedAvg: each hospital's full weight vector every round. FedAvg + DP: clipped, noised weight updates. FedAvg + SecAgg: only masked uint32 vectors that look random (correlation with the truth about -0.03 to +0.01); it learns only the weighted sum.

3. **Walk through one FedAvg round with numbers.** Server sends w to 3 hospitals (900, 539, 359 rows). Each trains 2 local epochs and returns its weights and its row count. Server computes w = (900·w1 + 539·w2 + 359·w3)/1798. Toy check: n = 1, 3 and weights 1, 4 give (1·1 + 3·4)/4 = 3.25.

4. **Why weight by number of samples?** The averaged weights approximate the gradient of the pooled loss, in which every sample counts once. Unweighted averaging would let the 359-row hospital count as much as the 900-row one.

5. **What is ε? δ? What is the privacy unit?** ε bounds how much any single output can change when one record is added or removed (smaller = stronger). δ is the small probability that the bound fails; it must be below 1/n_k (we use 1e-5 vs n_k ≥ 359). The unit is a single patient record (record-level DP), per hospital.

6. **Why those σ and C? What happens as σ grows?** C = 1 is a standard clip norm for a small MLP; σ was swept over {0.5 ... 3.0}. Larger σ gives smaller ε (σ = 0.5: 69.6; σ = 3.0: 2.9) and more noise, which would cost utility on real signal. On planted-signal labels σ = 1 (ε 14.3) kept AUC at 0.804 versus 0.794 without DP; on the real data utility cannot fall because it is already at chance.

7. **How did you account for privacy across rounds?** Flower re-creates clients every round, so a fresh Opacus accountant would reset and under-report ε. We keep a persistent per-client history of (σ, sample rate, steps) outside the clients and compute ε from the whole history. Test: ε after 3 rounds exceeds ε after 1.

8. **Is your secure aggregation really secure? What does it not protect?** No: it is a simulation. Pair seeds are shared directly, with no key agreement, no authentication, and no dropout recovery. It hides individual updates from the server but not the final aggregate, which can still leak, hence DP on top.

9. **What happens when a client drops out in SecAgg?** Its partners' masks are no longer cancelled; the aggregate is garbage (error about 2,048). Our server detects the missing client and aborts the round. Real protocols (Bonawitz et al.) secret-share the seeds so survivors can reconstruct and remove the dropped client's masks.

10. **How did you make the hospitals non-IID, and why?** Quantity skew 50/30/20 and label skew via a Dirichlet(1) draw of each hospital's prevalence (43% / 58% / 58%), resampled until valid. Real hospitals differ in size and case mix; IID is the optimistic control.

11. **Why is the federated model not better than centralized?** Neither can be better: the label carries no learnable signal (CV AUC 0.50), so all methods sit inside the chance band (0.5 ± 0.046). FL's value here is privacy, not accuracy. We demonstrated that the pipeline learns on planted-signal labels (0.78 to 0.81 AUC).

12. **How did you verify Flower's aggregation?** A hand-written 40-line FedAvg (`fedavg_reference.py`) run with the same seeds reproduces a full Flower round to within 6e-8; there is also a unit test of Flower's `aggregate_fit` against the weighted average.

13. **How do you know the membership-inference attack works?** Positive control: a deliberately overfit 256-256 MLP on 150 rows gives AUC 0.858. Negative control: an untrained model gives 0.491. Members and non-members are balanced, and non-members come from a hold-out never used elsewhere.

14. **Why recall and AUC rather than accuracy only?** A missed cancer is the costly error, and accuracy hides majority-class prediction (local-only hospitals 2 and 3 get recall 0.70 with AUC 0.50 simply by predicting "cancer" for most patients).

15. **How do you know there was no train/test leakage?** Duplicates removed before any split; age scaled by a fixed constant (no fitted scaler); hyperparameters fixed before looking at test results; separate validation and attack hold-out; asserts that client rows never overlap val/test/attack.

16. **Does FL reduce communication here?** No. 20 rounds cost 0.51 MB versus 0.084 MB to upload the raw data once (about 6×). FL's benefit here is privacy. For large datasets and few rounds the balance reverses.

17. **What are the dataset's limitations?** Survey-style/synthetic-looking, 3,000 rows, no learnable signal, no vital signs or treatment fields despite the Kaggle description. No clinical claim is made.

18. **What would you change for real hospitals?** Real clinical data and a governance agreement; a networked Flower server with TLS and client authentication; real SecAgg with key agreement and dropout recovery; secure-RNG DP; shadow-model attack evaluation; robust aggregation; per-hospital validation sets.

19. **Where are the hospital CSVs read, and why never by the server?** Only inside `HospitalClient` (`ml/federated.py`). The server code aggregates weights and evaluates on the shared test file, so no server path ever touches a hospital file; that is the federated trust boundary.

20. **How would you defend against a malicious client?** We measured it: a hospital that flips all its labels hurts FedAvg, and coordinate-wise median aggregation recovers part of the loss (see `results/extras.json` and the report, section 6.8). Further options: update-norm clipping, trimmed mean/Krum, anomaly detection on updates, and secure aggregation combined with per-client clipping.
