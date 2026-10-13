# Demo script (about 2 minutes)

**Before the demo:** open the deployed app https://fl-lung-cancer.onrender.com (wake it up a minute early; the free instance sleeps), or run locally (`python -m uvicorn backend.main:app --port 8000` after `npm run build`). Have admin credentials and two hospital credentials ready. Everything shown is precomputed from `results/`; only the secure-aggregation page and the prediction compute live (cheap, deterministic, no training).

1. **Data & Hospitals (20 s).** "Three hospitals, non-IID: 900 / 539 / 359 rows, prevalence 43% / 58% / 58%. Raw rows never leave a hospital; only client code opens these files." Point at k-anonymity: pooled 25, hospital 3 has k = 1, so removing identifiers is not anonymisation.
2. **Comparison (25 s).** "All methods, one test file, 5 seeds. Every AUC sits inside the grey chance band: this dataset's label carries no learnable signal, so I do not claim FL beats centralized. The green table below shows the same pipeline learns when a signal exists (0.78 to 0.81 AUC)."
3. **Privacy & Security (35 s).** "ε falls from 69.6 to 2.9 as noise grows, with δ = 1e-5 and accounting across all 20 rounds. The attack plot: centralized leaks 0.62, FedAvg 0.56, with DP about 0.51. The orange positive control (0.86) proves the attack works; the untrained model sits at chance."
4. **Secure Aggregation (25 s).** Click "New random updates": masked numbers look random, yet server aggregate = plain FedAvg, error about 1e-7. Click **Drop hospital 2**: masks no longer cancel and the error explodes, which is why real protocols use secret sharing. "This is a simulation."
5. **Live Federation (30 s).** Log in as admin: show global model version, the history table and the Hugging Face repo link. Create a hospital (credentials appear). In a terminal as that hospital run `python -m client.cli run --data data/clients/noniid_hospital_1.csv`: it downloads the model, trains locally, uploads weights only; repeat as a second hospital: "Round complete: new global model v1". Refresh the page to see v1 in the history with participants and test AUC.
6. **Predict (10 s).** Tick a few symptoms, keep the "global (live)" model (log in first), press Predict. Point to the disclaimer: every model is near chance here, so the output has no clinical meaning.
7. **Limitations (5 s).** Near-random data, simulated secure aggregation (live server sees updates), optional median aggregation as the only poisoning defence, one simple attack, free Render tier sleeps.

**If the server is unreachable or asleep:** a red "Server unreachable" banner with a Retry button appears; wait a minute for Render to wake (or restart uvicorn locally) and click Retry.
**Fallback without the frontend:** show `results/plots/*.png` and http://localhost:8000/docs.
