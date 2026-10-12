"""All paths, seeds and hyperparameters in ONE place."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "Full Lung Cancer dataset.csv"
DATA_DIR = ROOT / "data"
PROCESSED_CSV = DATA_DIR / "processed" / "clean.csv"
CLIENT_DIR = DATA_DIR / "clients"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
MODELS_DIR = ROOT / "models"
DOCS_DIR = ROOT / "docs"

N_CLIENTS = 3
SEEDS = [0, 1, 2, 3, 4]
SPLIT_SEED = 42            # the global split and partitions are fixed; SEEDS vary training
SPLIT_FRACTIONS = {"train": 0.60, "val": 0.10, "test": 0.20, "attack": 0.10}
QUANTITY_SKEW = [0.5, 0.3, 0.2]
DIRICHLET_ALPHA = 1.0
MIN_CLIENT_ROWS = 100
PREVALENCE_BOUNDS = (0.15, 0.85)

# model / optimisation (shared by every method for fairness)
HIDDEN = (32, 16)
LR = 0.05
MOMENTUM = 0.9
BATCH_SIZE = 32
FL_ROUNDS = 20
LOCAL_EPOCHS = 2
THRESHOLD = 0.5

# differential privacy
CLIP_NORM = 1.0
SIGMAS = [0.3, 0.6, 1.0, 1.5, 2.0]
DELTA = 1e-5

# secure aggregation
SECAGG_SCALE_BITS = 20
