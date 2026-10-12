# Progress log

Plan: [implementation_plan.md](implementation_plan.md). Environment: Python 3.14, Windows 11, CPU. Versions pinned in `requirements.txt`.

| Phase | Work | Status |
|---|---|---|
| 0 | Environment, pinned requirements, Step 0 data inspection, `schema.py` | Done |
| 1 | Preprocess, privacy step, k-anonymity, IID/non-IID partition, tests | Done |
| 2 | Model, metrics, centralized + local-only baselines (E1, E8) | Done, 5 seeds |
| 3 | Reference FedAvg, Flower FedAvg, cross-check (E2, E3) | Done: Flower round == hand-written FedAvg (6e-8) |
| 4 | DP with persistent accountant, sigma sweep (E4) | Done: 6 sigmas, eps 2.9 to 69.6 |
| 5 | Secure aggregation + dropout demo (E5) | Done: integer sum exact, dropout detected, sensitivity control |
| 6 | Membership inference + communication analysis (E6, E7) | Done, with positive/negative controls |
| 7 | FastAPI backend + tests | Done: 30 tests |
| 8 | React dashboard (9 pages) | Done: built and checked page by page against real results |
| 9 | Report, slides, demo script | Done: `docs/report.md`, `docs/slides.md`, `docs/demo_script.md` |

Tests: 25 ML + 30 backend, all passing (`python -m pytest ml/tests backend/tests -q`).

## Key findings
- **Dataset caveat:** the label is essentially unlearnable from the features (CV logistic AUC 0.501, random forest 0.523). All accuracy comparisons are within the 95% chance band (0.5 +/- 0.046). Reported honestly; the pipeline is validated on synthetic planted-signal labels (centralized 0.78, FedAvg 0.79-0.81, FedAvg + DP 0.80 AUC).
- **Membership inference leakage falls:** centralized 0.624, FedAvg 0.558, FedAvg + DP 0.529 (sigma 0.5) to 0.510 (sigma 3.0); positive control 0.858, untrained 0.491.
- **DP:** eps (max over hospitals, delta 1e-5, 20 rounds) = 69.6, 22.5, 14.3, 7.2, 4.8, 2.9 for sigma 0.5 to 3.0; smaller hospitals get a larger eps.
- **SecAgg:** masked integer sum equals plain sum exactly; dropout breaks cancellation. Over 20 rounds the model drifts 0.06 from plain FedAvg, but a 1e-6 perturbation of plain FedAvg alone drifts 0.033, so that is SGD sensitivity, not an error.
- **Communication:** 0.51 MB for 20 rounds vs 0.084 MB to upload the raw data once (FL costs about 6x more bytes here).

## Deviations from the plan (all documented)
- Python 3.14 (only interpreter available) instead of 3.10/3.11; torch, flwr, opacus, fastapi all work.
- Flower driven in-process (real `NumPyClient` + real `FedAvg`), no ray/gRPC simulation.
- 60/10/20/10 split (extra 10% attack hold-out for membership-inference non-members) instead of 70/10/20.
- Sigma grid {0.5, 0.8, 1.0, 1.5, 2.0, 3.0} instead of {0.3 ... 2.0}: at 20 rounds sigma 0.3 gives a meaningless eps.
- No per-hospital validation shares; per-hospital curves evaluate the global model on each hospital's own training data (it rises because the model memorises it).
- Optional extras (FedProx, quantisation, poisoning) not implemented.
- Not done: a re-run from a fresh clone (the full run takes about 1 hour on CPU); results are reproducible with the same seeds via `python -m ml.run_all`.
