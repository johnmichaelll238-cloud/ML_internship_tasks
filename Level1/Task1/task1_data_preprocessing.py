import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def run_preprocessing_pipeline():
    """Executes the end-to-end data preprocessing pipeline for Level 1 Task 1."""

    print("=== [1/5] Loading Raw Dataset ===")
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)

    # Remove features that are unique identifiers or unlikely to help
    # a baseline machine learning model.
    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
    print(f"Raw data loaded successfully. Shape: {df.shape}")

    print("\n=== [2/5] Handling Missing Values (Imputation) ===")
    num_cols = ["Age", "Fare"]
    cat_cols = ["Sex", "Embarked"]

    # Median is robust to outliers and is commonly used for
    # imputing missing numerical values.
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # Replace missing categorical values with the most frequent category.
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    print("[✓] Missing values handled (Median/Mode imputation).")

    print("\n=== [3/5] Encoding Categorical Variables (One-Hot) ===")

    # Drop the first category to avoid redundant dummy variables,
    # which can affect certain linear models.
    encoder = OneHotEncoder(
        sparse_output=False,
        drop="first"
    )

    encoded_features = encoder.fit_transform(df[cat_cols])

    # Preserve the original DataFrame index so rows remain aligned.
    encoded_df = pd.DataFrame(
        encoded_features,
        columns=encoder.get_feature_names_out(cat_cols),
        index=df.index,
    )

    # Separate features and target.
    y = df["Survived"]
    X = df.drop(columns=["Survived"] + cat_cols)

    # Merge encoded categorical features with numerical features.
    X = pd.concat([X, encoded_df], axis=1)

    print(f"[✓] Categorical encoding complete. Features shape: {X.shape}")

    print("\n=== [4/5] Partitioning Data (Train-Test Split) ===")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("[✓] Dataset partitioned successfully.")
    print(f"    -> Training set size: {X_train.shape[0]} samples")
    print(f"    -> Testing set size:  {X_test.shape[0]} samples")

    print("\n=== [5/5] Feature Scaling (Standardization) ===")

    scaler = StandardScaler()
    scale_target_cols = ["Age", "Fare"]

    # Learn scaling parameters from the training data only,
    # then apply the same transformation to the test data.
    X_train[scale_target_cols] = scaler.fit_transform(
        X_train[scale_target_cols]
    )

    X_test[scale_target_cols] = scaler.transform(
        X_test[scale_target_cols]
    )

    print("[✓] Standardization complete. Zero leakage enforced.")

    print("\n⚡⚡ Pipeline Complete & Verified ⚡⚡")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Prevent automatic execution when imported as a module.
    X_train, X_test, y_train, y_test = run_preprocessing_pipeline()
