"""
Simple Linear Regression pipeline using the housing prices dataset.

Workflow:
1. Load dataset
2. Split into train/test
3. Train Linear Regression
4. Evaluate using MSE and R²
"""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ==========================
# Configuration
# ==========================

DATA_PATH = Path("Level1") / ("Task2") / "housing_prices.csv"

ROOMS_COL = 5
PRICE_COL = 13


def run_regression_pipeline() -> tuple[LinearRegression, float, float]:
    """
    Executes the end-to-end Simple Linear Regression pipeline.

    Returns:
        tuple:
            model (LinearRegression): Trained regression model.
            mse (float): Mean Squared Error on the test set.
            r2 (float): R² score on the test set.
    """

    print("=== [1/4] Loading Space-Separated Dataset ===")

    try:
        df = pd.read_csv("housing_prices.csv", header=None, sep=r"\s+")
    except FileNotFoundError:
        print(f"[ERROR] Dataset not found: {DATA_PATH}")
        raise

    if df.shape[1] <= max(ROOMS_COL, PRICE_COL):
        raise ValueError(
            f"Dataset must contain at least {max(ROOMS_COL, PRICE_COL) + 1} columns."
        )

    X = df[ROOMS_COL].to_numpy().reshape(-1, 1)
    y = df[PRICE_COL].to_numpy()

    print(f"[✓] Dataset successfully loaded.")
    print(f"    -> Total samples: {len(df)}")
    print(f"    -> Features: {X.shape[1]}")

    print("\n=== [2/4] Partitioning Data (Train-Test Split) ===")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    print("[✓] 80/20 train-test split complete.")
    print(f"    -> Training samples: {len(X_train)}")
    print(f"    -> Testing samples : {len(X_test)}")

    print("\n=== [3/4] Training Simple Linear Regression Model ===")

    model = LinearRegression()
    model.fit(X_train, y_train)

    weight = model.coef_[0]
    bias = model.intercept_

    print("[✓] Model training complete.")
    print(f"    -> Weight (Slope): {weight:.4f}")
    print(f"    -> Bias (Intercept): {bias:.4f}")

    print("\nRegression Equation:")
    print(f"    Price = ({weight:.4f} × Rooms) + ({bias:.4f})") 

    print("\n=== [4/4] Evaluating Model ===")

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("[✓] Evaluation complete.")
    print(f"    -> Mean Squared Error (MSE): {mse:.4f}")
    print(f"    -> R² Score: {r2:.4f}")

    print("\nSample Predictions")
    print("-" * 35)

    for actual, predicted in zip(y_test[:5], y_pred[:5]):
        print(f"Actual: {actual:8.2f} | Predicted: {predicted:8.2f}")

    print("\n⚡ Pipeline Complete & Verified ⚡")

    return model, mse, r2


if __name__ == "__main__":
    run_regression_pipeline()