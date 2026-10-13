# Deployment guide: Hugging Face (model hub) + Render (server + dashboard)

## Architecture
* **Hugging Face model repo** `Adhi-gg029/lung-cancer-federated-learning`: public download point for the global model (`global_model.npz`, `versions/global_v<k>.npz`, `history.json`, model card). It also lets the free Render instance recover the latest model after it restarts.
* **Render web service** (Docker, free plan): FastAPI = authentication + live federated-learning coordinator + dashboard, one origin.
* **Hospital clients** run on each hospital's own machine (`python -m client.cli`). Raw data never leaves them.

## One-time: publish the primary model (v0) to Hugging Face
1. Create a **write** token: https://huggingface.co/settings/tokens (type "Write").
2. From `Multi-Hospital Disease Prediction/`:
```powershell
$env:HF_TOKEN="hf_xxx"
python scripts/publish_to_hf.py            # uploads README (model card), global_model.npz, versions/global_v0.npz, config, history
```
3. In the Render dashboard, open the service, **Environment**, and add `HF_TOKEN` with the same token. After the next restart the server publishes every new global version automatically (and restores the latest one when it wakes up).

Keep the token secret: never commit it, and revoke it at huggingface.co if it leaks.

## Render service
| Setting | Value |
|---|---|
| Runtime / plan | Docker / Free |
| Repo / branch | `https://github.com/Aditya-Ganesamoorthy/DPSA-Federated-Learning-Project` / `main` |
| Docker context / file | `Multi-Hospital Disease Prediction` / `Multi-Hospital Disease Prediction/Dockerfile` |
| Auto-deploy | Enabled, but **it did not fire** for this service: it was created from the public repo URL, so Render has no GitHub webhook. After pushing code, press **Manual Deploy → Deploy latest commit** in the Render dashboard (or connect the repo through Render's GitHub integration to get automatic deploys). A redeploy also resets the free instance's ephemeral state to v0 unless `HF_TOKEN` is configured. |
| Health check | `/api/v1/health` |

Environment variables are listed in the README (`SECRET_KEY`, `ADMIN_PASSWORD`, `MIN_CLIENTS`, `AGGREGATOR`, `HF_REPO_ID`, `HF_TOKEN`, ...). **`SECRET_KEY` must stay constant**: it signs sessions and derives every hospital's password. Changing it logs everyone out and invalidates all hospital passwords.

## Administering hospitals
1. Log in to the dashboard as `admin`.
2. **Live Federation → Administrator tools → New hospital name → Create credentials.** Send the shown hospital id and password to that hospital over a secure channel. (Nothing is stored: the password is derived from `SECRET_KEY`, so you can re-issue it by creating the same name again.)
3. To remove a hospital, add its id to `REVOKED_HOSPITALS` and redeploy.
4. **Aggregate pending updates now** builds a new version from whoever has reported, without waiting for `MIN_CLIENTS`.
5. Use pseudonymous hospital ids: they appear in the public `history.json`.

## A hospital joining
```powershell
git clone https://github.com/Aditya-Ganesamoorthy/DPSA-Federated-Learning-Project ; cd "DPSA-Federated-Learning-Project/Multi-Hospital Disease Prediction"
pip install torch --index-url https://download.pytorch.org/whl/cpu ; pip install -r client/requirements.txt ; pip install -r requirements.txt
$env:FL_SERVER="https://fl-lung-cancer.onrender.com"; $env:FL_USERNAME="<id>"; $env:FL_PASSWORD="<password>"
python -m client.cli run --data my_hospital.csv --epochs 2 --sigma 1.0
```
The first request can take up to a minute while a sleeping free instance wakes up. Use `--source hf` to download the model from Hugging Face instead of the server.

## Verifying a deployment
```powershell
python scripts/smoke_server.py --url https://fl-lung-cancer.onrender.com --admin-password <pw>
```
Read-only by default (health, dashboard, login, hospital creation, download, prediction, rejection of stale and oversized updates). `--mutating` additionally fakes a round and must only be used on a throw-away server.

## Known limits of the free setup
* The instance sleeps after about 15 minutes idle and its disk is ephemeral. Pending (not yet aggregated) updates are lost on restart; hospitals simply resubmit. The published global model survives only if `HF_TOKEN` is set.
* 512 MB RAM: the server is numpy-only (no torch). Do not add heavy dependencies.
* Transport security comes from Render's HTTPS. Updates are visible to the server (secure aggregation is only simulated here).
