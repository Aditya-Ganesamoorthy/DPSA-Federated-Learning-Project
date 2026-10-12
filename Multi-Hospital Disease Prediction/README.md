# Decentralized ML: Federated Learning for Privacy Preservation

Privacy-preserving lung-cancer prediction across three simulated hospitals, using Flower (FedAvg), Opacus (DP-SGD),
a pairwise-masking secure-aggregation simulation, a membership-inference attack, a FastAPI backend and a React dashboard.

> **Read this first.** The Kaggle label is almost unpredictable from the features (cross-validated logistic regression AUC 0.50),
> so every model scores near chance. The privacy, communication and security analyses are the substance of the work; no clinical
> claims are made. Details: [docs/data_decisions.md](docs/data_decisions.md). The pipeline is validated separately on synthetic planted-signal labels.

Design: [implementation_plan.md](implementation_plan.md) · Status: [PROGRESS.md](PROGRESS.md) · Report: [docs/report.md](docs/report.md) · Slides: [docs/slides.md](docs/slides.md) · Demo: [docs/demo_script.md](docs/demo_script.md)

## Setup (Windows, Python 3.14 tested; 3.10/3.11 should also work)
```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python -m pip install -r "Multi-Hospital Disease Prediction/requirements.txt"
```
All commands below run from `Multi-Hospital Disease Prediction/`.

## 1. Run the pipeline (preprocess, partition, E1-E8, sanity check, plots)
```powershell
python -m ml.run_all            # about 40 min on CPU; resumes from results/runs
python -m ml.run_all --quick    # 1 seed / 3 rounds smoke test into a temp folder
```
Outputs: `results/*.json|csv`, `results/plots/*.png`, `models/*.pt` + `model_config.json`, `results/run_meta.json`.

## 2. Tests
```powershell
python -m pytest ml/tests backend/tests -q
```

## 3. Backend
```powershell
python -m uvicorn backend.main:app --port 8000
```
Docs at http://localhost:8000/docs. Endpoints are under `/api/v1` (health, features, meta, dataset/summary, clients, results/*, secagg/demo, predict).
Env: `CORS_ORIGINS` (default Vite dev origin). Request bodies are never logged.

## 4. Frontend
```powershell
cd frontend
npm install
copy .env.example .env      # VITE_API_BASE_URL=http://localhost:8000
npm run dev                 # http://localhost:5173
```
(The npm scripts call vite through `node` so they also work in folders whose path contains `&`.)

## Layout
```
ml/        config, schema (single source of truth), preprocess, partition, model, metrics, centralized,
           fedavg_reference, federated (Flower), dp (Opacus + persistent accountant), secagg, attack,
           comm_analysis, sanity (synthetic planted signal), plots, run_all, tests/
backend/   FastAPI app (routers, services, schemas, tests)
frontend/  Vite + React + Recharts dashboard
data/      processed/clean.csv and clients/*.csv (derived, reproducible)
results/   experiment outputs read by the backend
docs/      data decisions, threat model, report, slides, demo script
```

## Honest implementation notes
- Flower runs **in-process**: real `NumPyClient` clients and the real `flwr.server.strategy.FedAvg` aggregation, but no ray/gRPC simulation (not reliable on Windows).
- Secure aggregation is a **simulation** (shared pair seeds, integer arithmetic mod 2^32, no key agreement, no dropout recovery).
- DP is record-level against the server; Opacus is used without its secure-RNG mode.
- Hospital CSVs are opened only inside `HospitalClient`; server code reads only the shared validation/test files.
