---
license: mit
library_name: numpy
tags:
  - federated-learning
  - differential-privacy
  - healthcare
  - tabular-classification
  - flower
  - education
---

# Lung-cancer federated learning: primary global model

The **centralized server model** of a federated-learning project (course project: *Decentralized Machine Learning:
Implementing Federated Learning for Privacy Preservation*). Hospitals download this model, train it further on their own
data **without sharing any raw rows**, and send back only updated weights. The coordinator averages the updates (FedAvg) into
the next version and publishes it here.

> **Educational demo, not medical advice. Do not use for diagnosis.**
> On the public Kaggle dataset used here the label is essentially unpredictable from the 15 features (cross-validated
> logistic regression ROC-AUC 0.50, random forest 0.52). This model scores about 0.51 AUC on a 600-row test set, which is
> **chance level**. It exists to demonstrate the federated, privacy-preserving *process*, not to predict cancer.

## Files
| File | What it is |
|---|---|
| `global_model.npz` | **Latest** global model (arrays `p0..p5`, plus `version`). |
| `versions/global_v<k>.npz` | Every published version (`v0` = initial primary model). |
| `history.json` | One entry per version: time, participating hospital ids, training-row counts, test metrics. |
| `model_config.json` | Architecture and the exact feature order. |

## Model
MLP `15 → 32 → 16 → 1` (ReLU, sigmoid output, 1,057 parameters). Feature order:
`gender (F=0, M=1), age/100, smoking, yellow_fingers, anxiety, peer_pressure, chronic_disease, fatigue, allergy, wheezing,
alcohol_consuming, coughing, shortness_of_breath, swallowing_difficulty, chest_pain` (yes = 1, no = 0).

Version 0 was trained with FedAvg (Flower) across three simulated, non-IID hospitals (900 / 539 / 359 rows), 20 rounds × 2 local epochs.

## Use it (numpy only, no torch needed)
```python
import numpy as np
from huggingface_hub import hf_hub_download

path = hf_hub_download("Adhi-gg029/lung-cancer-federated-learning", "global_model.npz")
z = np.load(path)
W = [z[f"p{i}"] for i in range(6)]
print("version", int(z["version"]))

def predict_proba(x):                      # x: (n, 15) float32 in the feature order above
    h = np.maximum(x @ W[0].T + W[1], 0)
    h = np.maximum(h @ W[2].T + W[3], 0)
    return 1 / (1 + np.exp(-(h @ W[4].T + W[5])[:, 0]))
```

## Take part as a hospital (federated update)
1. Get credentials from the project administrator.
2. Clone the project repository and run, on your own machine (your CSV never leaves it):
```
set FL_SERVER=<server url>  &  set FL_USERNAME=<hospital-id>  &  set FL_PASSWORD=<password>
python -m client.cli run --data my_hospital.csv --epochs 2 --source hf
```
`--source hf` downloads this model; the update is uploaded to the coordinator server, which publishes the next version here.
Add `--sigma 1.0` for local differential privacy (DP-SGD with a persistent privacy accountant).

## Privacy and security notes
* Raw patient data is never uploaded; only weights and the number of training rows.
* Optional record-level differential privacy on the client. Reference results (6 noise levels): ε 2.9 to 69.6 at δ = 1e-5.
* The coordinator authenticates hospitals, rejects stale or malformed updates, caps update norm, and can use median aggregation
  against poisoned updates. Secure aggregation is only *simulated* in this project (see the project report).
* `history.json` lists pseudonymous hospital ids: use ids that do not reveal the institution.

## Limitations
Near-random, survey-style dataset; tiny model; 600-row evaluation set (95% chance band for AUC is ±0.046); not clinically validated.
