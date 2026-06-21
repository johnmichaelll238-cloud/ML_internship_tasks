from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RANDOM_STATE = 42
TEST_SIZE = 0.2
CRITERION = "gini"

DATA_PATH = BASE_DIR / "iris.csv"
MODEL_FILENAME = BASE_DIR / "decision_tree2_model.joblib"
TREE_IMAGE_FILENAME = BASE_DIR / "pruned_tree2.png"
