import pandas as pd
from sklearn.model_selection import train_test_split
try:
    from .config import RANDOM_STATE, TEST_SIZE
except ImportError:
    from config import RANDOM_STATE, TEST_SIZE
def load_data(csv_path):
    """
    Loads and prepares the Iris dataset.
    Parameters:
     csv_path (str)
     Returns:
      X_train,
      X_test,
      y_train,
      y_test
    """
    df = pd.read_csv(csv_path)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    