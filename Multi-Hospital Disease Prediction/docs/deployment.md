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
3. In the Render dashboard, open the service, **Environment**, and add `HF_TOKEN` with the same token (saving triggers a deploy). The server then publishes every new global version automatically and restores the latest one (model + history) when it restarts. This was verified live: after a real round the Hub showed `Global model v1 (FedAvg round)`, and a redeploy with a wiped disk came back at v1.

Keep the token secret: never commit it, and revoke it at huggingface.co if it leaks.

## Render service
| Setting | Value |
|---|---|
| Runtime / plan | Docker / Free |
| Repo / branch | `https://github.com/Aditya-Ganesamoorthy/DPSA-Federated-Learning-Project` / `main` |
| Docker context / file | `Multi-Hospital Disease Prediction` / `Multi-Hospital Disease Prediction/Dockerfile` |
| Auto-deploy | Enabled, but **it did not fire** for this service: it was created from the public repo URL, so Render has no GitHub webhook. After pushing code, press **Manual Deploy → Deploy latest commit** in the Render dashboard (or connect the repo through Render's GitHub integration to get automatic deploys). A redeploy also resets the free instance's ephemeral state to v0 unless `HF_TOKEN` is configured. |
| Health check | `/api/v1/health` |

Environment variables are listed in the README (`SECRET_KEY`, `ADMIN_PASSWORD`, `MIN_CLIENTS`, `AGGREGATOR`, `HF_REPO_ID`, `HF_TOKEN`, ...). **`SECRET_KEY` must stay constant**: it signs sessions and peppers the stored password hashes. Changing it logs everyone out and invalidates every hospital password (hospitals must register again).

## Hospitals join themselves
Hospitals and sub-hospitals **register on their own** (dashboard: *Register hospital*; or `python -m client.cli register`). The administrator does not create accounts.
* To limit who can join, set `INVITE_CODE` in Render and share the code only with partner hospitals. Registration is rate-limited (`REGISTRATIONS_PER_HOUR`) and capped (`MAX_ACCOUNTS`).
* The administrator (`ADMIN_USERNAME` / `ADMIN_PASSWORD`) can **suspend**, **restore** or **remove** any hospital on the Live Federation page, and force an aggregation.
* Passwords are never stored. Each account keeps a random salt and `HMAC(SECRET_KEY, PBKDF2(password, salt))`. With `HF_TOKEN` set, `accounts.json` (these peppered verifiers plus hospital names and usernames) is backed up to the **public** Hugging Face repo so accounts survive restarts. Keep `SECRET_KEY` secret: without it the verifiers cannot be attacked offline. Use pseudonymous usernames; they also appear in the round history.

## A hospital joining
```powershell
git clone https://github.com/Aditya-Ganesamoorthy/DPSA-Federated-Learning-Project ; cd "DPSA-Federated-Learning-Project/Multi-Hospital Disease Prediction"
pip install torch --index-url https://download.pytorch.org/whl/cpu ; pip install -r client/requirements.txt ; pip install -r requirements.txt
$env:FL_SERVER="https://fl-lung-cancer.onrender.com"; $env:FL_USERNAME="<id>"; $env:FL_PASSWORD="<password>"
python -m client.cli register --hospital-name "My Hospital"      # once
python -m client.cli run --data my_hospital.csv --epochs 2 --sigma 1.0
```
Or, with no installation, open the dashboard, register, and use **Train & Contribute**: the CSV is trained on inside the browser.
The first request can take up to a minute while a sleeping free instance wakes up. Use `--source hf` to download the model from Hugging Face instead of the server.

## Verifying a deployment
```powershell
python scripts/smoke_server.py --url https://fl-lung-cancer.onrender.com --admin-password <pw>
```
Safe by default for a live server (40 checks: dashboards, authentication, self-registration of two throw-away hospitals, download, validation, rejection of stale/oversized/malformed updates, admin suspend/restore). The throw-away hospitals are deleted at the end. `--mutating` additionally completes a fake round and must only be used on a disposable server.

## Known limits of the free setup
* The instance sleeps after about 15 minutes idle and its disk is ephemeral. Pending (not yet aggregated) updates are lost on restart; hospitals simply resubmit. The published global model survives only if `HF_TOKEN` is set.
* 512 MB RAM: the server is numpy-only (no torch). Do not add heavy dependencies.
* Transport security comes from Render's HTTPS. Updates are visible to the server (secure aggregation is only simulated here).
