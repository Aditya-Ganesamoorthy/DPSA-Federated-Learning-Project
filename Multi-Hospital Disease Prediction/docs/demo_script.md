# Demo script (about 2 minutes)

**Before the demo:** backend running (`python -m uvicorn backend.main:app --port 8000`), frontend running (`npm run dev` in `frontend/`), open http://localhost:5173. Everything shown is precomputed from `results/`; only the secure-aggregation page and the prediction compute live (cheap, deterministic, no training).

1. **Data & Hospitals (20 s).** "Three hospitals, non-IID: 900 / 539 / 359 rows, prevalence 43% / 58% / 58%. Raw rows never leave a hospital; only client code opens these files." Point at k-anonymity: pooled 25, hospital 3 has k = 1, so removing identifiers is not anonymisation.
2. **Comparison (25 s).** "All methods, one test file, 5 seeds. Every AUC sits inside the grey chance band: this dataset's label carries no learnable signal, so I do not claim FL beats centralized. The green table below shows the same pipeline learns when a signal exists (0.78 to 0.81 AUC)."
3. **Privacy & Security (35 s).** "ε falls from 69.6 to 2.9 as noise grows, with δ = 1e-5 and accounting across all 20 rounds. The attack plot: centralized leaks 0.62, FedAvg 0.56, with DP about 0.51. The orange positive control (0.86) proves the attack works; the untrained model sits at chance."
4. **Secure Aggregation (25 s).** Click "New random updates": masked numbers look random, yet server aggregate = plain FedAvg, error about 1e-7. Click **Drop hospital 2**: masks no longer cancel and the error explodes, which is why real protocols use secret sharing. "This is a simulation."
5. **Predict (10 s).** Tick a few symptoms, choose the federated model, press Predict. Point to the disclaimer: every model is near chance here, so the output has no clinical meaning.
6. **Limitations (5 s).** Near-random data, simulated secure aggregation, no poisoning defence, one simple attack.

**If the backend is stopped:** a red "Backend offline" banner with a Retry button appears; restart uvicorn and click Retry.
**Fallback without the frontend:** show `results/plots/*.png` and http://localhost:8000/docs.
