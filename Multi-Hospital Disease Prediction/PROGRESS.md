# Progress log

Plan: [implementation_plan.md](implementation_plan.md). Environment: Python 3.14, Windows 11, CPU. Versions pinned in `requirements.txt` (research) and `requirements-server.txt` (deployed server).

## Original plan (offline simulation)
| Phase | Work | Status |
|---|---|---|
| 0 | Environment, pinned requirements, Step 0 data inspection, `schema.py` | Done |
| 1 | Preprocess, privacy step, k-anonymity, IID/non-IID partition, tests | Done |
| 2 | Model, metrics, centralized + local-only baselines (E1, E8) | Done, 5 seeds |
| 3 | Reference FedAvg, Flower FedAvg, cross-check (E2, E3) | Done: Flower round == hand-written FedAvg (6e-8) |
| 4 | DP with persistent accountant, sigma sweep (E4) | Done: 6 sigmas, eps 2.9 to 69.6 |
| 5 | Secure aggregation + dropout demo (E5) | Done: integer sum exact, dropout detected, sensitivity control |
| 6 | Membership inference + communication analysis (E6, E7) | Done, with positive/negative controls |
| 7 | FastAPI backend + tests | Done |
| 8 | React dashboard | Done |
| 9 | Report, slides (`docs/slides.pptx`), demo script, viva Q&A | Done |
| Extras | FedProx, 8/4-bit quantisation, label-flip poisoning + median defence | Done (`results/extras.json`, report 6.8) |

## Deployment plan (added later): primary model on Hugging Face, live federation, Render
| Item | Status |
|---|---|
| Primary global model v0 (numpy `.npz` exchange format, `models/global_v0.npz`) | Done |
| Live federation coordinator: login, `GET /fl/model`, `POST /fl/update`, FedAvg, versioning, history, test metrics | Done, tested |
| Safeguards: stateless HMAC auth + throttle, stale 409, shape/NaN checks, update-norm cap, optional median aggregation | Done, tested |
| Hospital client (`python -m client.cli`): download, local training on own CSV, optional local DP with persisted accountant, upload | Done; real 3-process run against a local server (v0 to v1) |
| Dashboard: login, live federation page (status, history, download, admin tools), Predict on the live global model | Done |
| Torch-free server (numpy forward pass, verified vs torch 6e-8), Dockerfile, `requirements-server.txt`, `render.yaml` | Done |
| Render deployment: https://fl-lung-cancer.onrender.com (free Docker service, Singapore) | **Live**; smoke test passes; real 2-hospital round v0 to v1 verified on it |
| Hugging Face: primary model v0 + model card published to https://huggingface.co/Adhi-gg029/lung-cancer-federated-learning | **Done and verified** (public download returns v0, identical to local) |
| Server auto-publish + restore (`HF_TOKEN` + `HF_REPO_ID` set on Render) | **Verified live**: a real 2-hospital round published v1 to the Hub by itself; after a redeploy (wiped disk) the server restored v1 and its history from the Hub. Then reset to clean v0. |

Tests: 25 ML + 50 backend/federation = 75, all passing (`python -m pytest ml/tests backend/tests -q`).

## What the owner still has to do
1. Create real hospital credentials in the dashboard (Live Federation, admin) and share them with each hospital.
2. Revoke the Hugging Face token that was pasted into the chat, create a fresh one, and update `HF_TOKEN` in Render (Environment, then Manual Deploy). The token file `Hugging face access token.txt` is git-ignored and was never committed.
3. Rotate `ADMIN_PASSWORD` in Render if the initial one (shared in the chat) should not be kept.
4. After pushing code, press Manual Deploy in Render (this service has no GitHub webhook).

## Key findings
- **Dataset caveat:** the label is essentially unlearnable from the features (CV logistic AUC 0.501, random forest 0.523). All accuracy comparisons are within the 95% chance band (0.5 +/- 0.046). Reported honestly; the pipeline is validated on synthetic planted-signal labels (centralized 0.78, FedAvg 0.79-0.81, FedAvg + DP 0.80 AUC).
- **Membership inference leakage falls:** centralized 0.624, FedAvg 0.558, FedAvg + DP 0.529 (sigma 0.5) to 0.510 (sigma 3.0); positive control 0.858, untrained 0.491.
- **DP:** eps (max over hospitals, delta 1e-5, 20 rounds) = 69.6, 22.5, 14.3, 7.2, 4.8, 2.9 for sigma 0.5 to 3.0; smaller hospitals get a larger eps.
- **SecAgg:** masked integer sum equals plain sum exactly; dropout breaks cancellation. Multi-round drift vs plain FedAvg is SGD sensitivity (a 1e-6 perturbation of plain FedAvg drifts as much), not an error.
- **Communication:** 0.51 MB for 20 rounds vs 0.084 MB to upload the raw data once; 8-bit updates cut uploads 74% with no accuracy loss.
- **Poisoning:** one label-flipping hospital drops FedAvg AUC 0.794 to 0.598 (planted-signal data); median aggregation restores 0.799.

## Deviations from the original plan (all documented)
- Python 3.14 (only interpreter available); Flower is run in-process in experiments (no ray/gRPC). The live server uses numpy FedAvg (tested equal to Flower's) to fit Render's 512 MB free tier.
- 60/10/20/10 split (extra attack hold-out); sigma grid {0.5 ... 3.0}; no per-hospital validation shares.
- Secure aggregation stays a simulation; the live protocol shows updates to the server (mitigations: local DP option, HTTPS, norm cap, median).
- The Render free tier sleeps when idle and has an ephemeral disk; the global model persists across restarts only once `HF_TOKEN` is configured.
- Not done: re-running the whole 1-hour experiment pipeline from a fresh clone (the 55 tests do pass in a fresh clone and a seed-0 rerun reproduces stored results bit-for-bit).
