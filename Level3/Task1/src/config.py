from pathlib import Path

# =========================
# Project Paths
# =========================

BASE_PATH = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_PATH / "data"
MODEL_DIR = BASE_PATH / "models"
VECTOR_DIR = BASE_PATH / "models"
IMAGES_DIR = BASE_PATH / "images"
# Dataset
DATASET_NAME = "sentiment.csv"
DATASET_PATH = DATA_DIR / DATASET_NAME

# Model
MODEL_NAME = "nlp_svm.joblib"
MODEL_PATH = MODEL_DIR / MODEL_NAME

# Vectorizer
VECTOR_NAME = "tfidf_vectorizer.joblib"
VECTOR_PATH = VECTOR_DIR / VECTOR_NAME
# =========================
# Randomness
# =========================

RANDOM_STATE = 42

# =========================
# Train/Test Split
# =========================

TEST_SIZE = 0.2

# =========================
# Scaling
# =========================

SCALER = "standard"

# =========================
# SVM Hyperparameters
# =========================

LINEAR_KERNEL = "linear"
RBF_KERNEL = "rbf"

DEFAULT_C = 1.0
DEFAULT_GAMMA = "scale"
