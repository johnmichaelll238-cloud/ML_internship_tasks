from pathlib import Path

BASE_PATH = Path(__file__).resolve().parent.parent

RANDOM_STATE = 42
TEST_SIZE = 0.2
CRITERION = "gini"

DATA_PATH = BASE_PATH / "data" / "iris.csv"
MODEL_FILENAME = BASE_PATH / "models" / "decision_tree_model.joblib"
TREE_IMAGE_FILENAME = BASE_PATH / "images" / "pruned_tree.png"