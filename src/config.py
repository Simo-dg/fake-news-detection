import os
import torch
from pathlib import Path

# --- 1. PATHS ---
SRC_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SRC_DIR.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
PLOTS_DIR = ROOT_DIR / "plots"

# Ensure key directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Specific Model Paths
TFIDF_PATH = MODELS_DIR / "tfidf_logreg_robust.joblib"
BERT_OLD_PATH = MODELS_DIR / "bert_finetuned"
BERT_FINAL_PATH = MODELS_DIR / "bert_final"

# --- 2. DEVICE & SYSTEM ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 3. HYPERPARAMETERS ---
# Model Settings
BERT_MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 512
CHUNK_OVERLAP = 128

# Training Settings
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
EPOCHS = 2

# Data Generation Settings
TARGET_SIZE = 200_000
TARGET_FAKE = TARGET_SIZE // 2
TARGET_REAL = TARGET_SIZE // 2

# Hugging Face Hub IDs
HF_MODEL_ID = "Simingasa/fake-news-bert-finetuned"
HF_MODEL_ID_V2 = "Simingasa/fake-news-bert-v2"

