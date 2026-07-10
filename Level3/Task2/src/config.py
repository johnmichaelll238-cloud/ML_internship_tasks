from pathlib import Path

BASE_PATH = Path(__file__).parent

MODEL_DIR = BASE_PATH / "production_models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "mnist_ffnn.keras"

RANDOM_SEED = 42

BATCH_SIZE = 64

EPOCHS = 20

HIDDEN_UNITS = 128

INPUT_DIM = 784

NUM_CLASSES = 10

VALIDATION_SPLIT = 0.2