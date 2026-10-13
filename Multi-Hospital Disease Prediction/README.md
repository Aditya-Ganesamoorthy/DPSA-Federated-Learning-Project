# Decentralized ML: Federated Learning for Privacy Preservation

Privacy-preserving lung-cancer prediction with **real client-server federated learning**:

```
        ┌────────────────────────── Hugging Face model repo ──────────────────────────┐
        │  Adhi-gg029/lung-cancer-federated-learning   (public download point,         │
        │  versioned global models, model card)                                        │
        └───────────▲──────────────────────────────────────────────┬───────────────────┘
                    │ publish v(k+1) after each round              │ download (optional)
        ┌───────────┴───────────── Render web service ──────────────▼───────────────────┐
        │  FastAPI: login · /fl/model · /fl/update · FedAvg · versions · dashboard     │
        └───────▲───────────────────────▲───────────────────────▲──────────────────────┘
                │ weights + n_k only    │                       │
          Hospital A client       Hospital B client       Hospital C client
          (own CSV, never sent)   (own CSV, never sent)   (own CSV, never sent)
```

1. A **primary global model (v0)** was trained with FedAvg across three simulated hospitals and is published on Hugging Face.
2. The **Render server** is the centralized coordinator. Hospitals log in, **download** the current global model, **train on their own data** with the hospital client, and **upload only weights**. When enough hospitals have reported, the server averages them (FedAvg) into the next version and publishes it.
3. The same Render service serves the **dashboard** (results, live federation status, prediction).

> **Read this first.** The Kaggle label is almost unpredictable from the features (cross-validated logistic regression AUC 0.50), so every model scores near chance. The privacy, communication and security analyses and the live federation machinery are the substance of the work; no clinical claims are made. Details: [docs/data_decisions.md](docs/data_decisions.md).

Design: [implementation_plan.md](implementation_plan.md) · Status: [PROGRESS.md](PROGRESS.md) · Report: [docs/report.md](docs/report.md) · Slides: [docs/slides.pptx](docs/slides.pptx) · Demo: [docs/demo_script.md](docs/demo_script.md) · Viva: [docs/viva_qa.md](docs/viva_qa.md) · Deployment: [docs/deployment.md](docs/deployment.md)

## Setup (Windows, Python 3.14 tested)
```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python -m pip install -r "Multi-Hospital Disease Prediction/requirements.txt"
```
All commands below run from `Multi-Hospital Disease Prediction/`.

## 1. Experiments (offline simulation: E1-E8, extras, plots)
```powershell
python -m ml.run_all            # about 1 hour on CPU; resumes from results/runs
python -m ml.extras             # FedProx, 8/4-bit quantisation, poisoning + median defence
python scripts/export_npz.py    # torch .pt -> numpy .npz (the exchange format; also writes global_v0.npz)
```

## 2. Run the live server locally
```powershell
$env:SECRET_KEY="<long random string>"; $env:ADMIN_PASSWORD="<admin password>"; $env:MIN_CLIENTS="2"
python -m uvicorn backend.main:app --port 8000      # API + dashboard (after: cd frontend; npm run build)
python -m backend.manage create-hospital "City General Hospital"   # prints that hospital's credentials
```
Frontend dev mode: `cd frontend; npm install; npm run dev` (http://localhost:5173, talks to :8000).

## 3. Act as a hospital (client)
```powershell
pip install -r client/requirements.txt
$env:FL_SERVER="https://fl-lung-cancer.onrender.com"; $env:FL_USERNAME="<hospital-id>"; $env:FL_PASSWORD="<password>"
python -m client.cli status
python -m client.cli run --data my_hospital.csv --epochs 2           # add --sigma 1.0 for local DP, --source hf to download from Hugging Face
```
`my_hospital.csv` may use the raw Kaggle layout or this project's encoded layout. It is read on your machine only.

## 4. Tests and checks
```powershell
python -m pytest ml/tests backend/tests -q                                   # 75 tests
python scripts/smoke_server.py --url https://fl-lung-cancer.onrender.com --admin-password <pw>   # black-box check of a running server
```

## Server configuration (environment variables)
| Variable | Purpose | Default |
|---|---|---|
| `SECRET_KEY` | signs tokens and derives hospital passwords (keep stable) | random per process (logins reset on restart) |
| `ADMIN_USERNAME` / `ADMIN_PASSWORD` | administrator account (disabled if no password) | `admin` / none |
| `MIN_CLIENTS` | hospitals needed before a new global version is built | `2` |
| `AGGREGATOR` | `fedavg` or `median` (median needs ≥ 3 hospitals; robust to poisoned updates) | `fedavg` |
| `MAX_UPDATE_NORM` | reject updates farther than this (L2) from the global model | `20` |
| `HF_REPO_ID`, `HF_TOKEN` | publish each version to Hugging Face and restore it after a restart | off |
| `REVOKED_HOSPITALS` | comma-separated hospital ids that may no longer log in | none |
| `STATE_DIR`, `CORS_ORIGINS`, `TOKEN_TTL_SECONDS` | state folder, allowed browser origins, session length | see `backend/settings.py` |

## Layout
```
ml/        offline pipeline: config, schema, preprocess, partition, model, metrics, centralized, fedavg_reference,
           federated (Flower), dp (Opacus + persistent accountant), secagg, attack, comm_analysis, sanity, extras,
           model_io (numpy exchange format), run_all, plots, tests/
backend/   FastAPI: auth (stateless HMAC), federation (live FedAvg coordinator), hf_sync, predictor (numpy), routers, tests/
client/    hospital client: core (local training, optional DP), api (HTTP), cli
frontend/  Vite + React + Recharts dashboard (login, live federation, results)
models/    trained models (.pt for research, .npz exchange format; global_v0.npz is the primary model)
results/   experiment outputs read by the dashboard
huggingface/  model card published to the Hub        scripts/  export_npz, publish_to_hf, smoke_server
Dockerfile, requirements-server.txt, ../render.yaml   deployment
```

## Honest implementation notes
- Experiments run Flower **in-process** (real `NumPyClient` + real `FedAvg`, no ray/gRPC). The **live server** uses the numpy FedAvg that the tests verify equal to Flower's, so it fits Render's 512 MB free instance without torch.
- Secure aggregation is a **simulation** (demo page and experiments). The live protocol protects data by keeping it on-site, authenticated TLS transport, optional client-side DP, update-norm checks and an optional median aggregator; updates are visible to the server.
- Render's free tier sleeps when idle (first request can take about a minute) and has an ephemeral disk. Hospital credentials are derived from `SECRET_KEY` (nothing to lose), and the global model is restored from Hugging Face after a restart when `HF_TOKEN` is set. Without it, a restart returns to v0.
- DP is record-level against the server; Opacus is used without its secure-RNG mode.
- Hospital CSVs are opened only inside the client code; server code reads only the shared test file.
