# Progress log

Plan: [implementation_plan.md](implementation_plan.md). Environment: Python 3.14, Windows 11, CPU. Versions pinned in `requirements.txt`.

| Phase | Work | Status |
|---|---|---|
| 0 | Environment, pinned requirements, Step 0 data inspection, `schema.py` | Done |
| 1 | Preprocess, privacy step, k-anonymity, IID/non-IID partition, tests | Done (9 tests pass) |
| 2 | Model, metrics, centralized + local-only baselines | Pending |
| 3 | Reference FedAvg, Flower FedAvg, cross-check | Pending |
| 4 | DP with persistent accountant, sigma sweep | Pending |
| 5 | Secure aggregation + dropout demo | Pending |
| 6 | Membership inference + communication analysis | Pending |
| 7 | FastAPI backend + tests | Pending |
| 8 | React dashboard | Pending |
| 9 | Report, slides, demo script | Pending |

## Phase 0-1 notes
- 3,000 rows x 16 cols, 0 missing, 2 duplicates removed -> 2,998 rows; target balanced (50.6% positive).
- **Dataset caveat:** label is essentially unpredictable from the features (CV logistic regression AUC 0.501, random forest 0.523). Expect near-chance metrics everywhere; reported honestly. See `docs/data_decisions.md`.
- Pooled k-anonymity (age band x gender) = 25. Per hospital k is 1-8; non-IID hospital 3 has k = 1 (a small group), reported as-is.
- Python 3.14 (not 3.10/3.11) was the only interpreter on the machine; torch, flwr, opacus and fastapi all install and run on it.
- Ray is not installed, so Flower is driven in-process (real `NumPyClient` + real `FedAvg` strategy, no gRPC/ray simulation).
