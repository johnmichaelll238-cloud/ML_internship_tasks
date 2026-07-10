"""
Logistic Regression Churn Training Pipeline

Author: Chinedu Ogbunobi

Description:
A modular machine learning pipeline that loads pre-split churn datasets,
preprocesses numerical features without data leakage, trains a Logistic
Regression model, evaluates its performance, and exports the ROC curve and
trained artifacts for future inference.
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

def load_data(base_dir: Path, 
    train_filename: str, 
    test_filename: str,
    )-> tuple[pd.DataFrame, pd.DataFrame]:
    """
Load the pre-split training and testing datasets.

Args:
    base_dir: Directory containing the dataset files.
    train_filename: Name of the training CSV.
    test_filename: Name of the testing CSV.

Returns:
    A tuple containing the training and testing DataFrames.

Raises:
    FileNotFoundError: If either dataset cannot be found.
"""
    train_path = base_dir / "data" / train_filename
    test_path = base_dir / "data" / test_filename
    
    if not train_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {train_path}")

    if not test_path.exists():
        raise FileNotFoundError(f"Test dataset not found: {test_path}")
        
    return pd.read_csv(train_path), pd.read_csv(test_path)

def preprocess_and_scale(train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    target_col: str
)-> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    StandardScaler,
]:
    """Separate features from the target column and standardize numerical features.

       The scaler is fitted only on the training data to prevent data leakage,
       then applied to both the training and testing sets.

       Returns:
         X_train_scaled
         X_test_scaled
         y_train
         y_test
         scaler
    """
    X_train = train_df.drop(columns=[target_col, "State"])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col, "State"])
    y_test = test_df[target_col]

    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(
        include=["object", "bool", "str"]
    ).columns.tolist()

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])

    X_train_scaled = pd.get_dummies(
        X_train_scaled, columns=categorical_features, drop_first=True
    )
    X_test_scaled = pd.get_dummies(
        X_test_scaled, columns=categorical_features, drop_first=True
    )
    X_test_scaled = X_test_scaled.reindex(columns=X_train_scaled.columns, fill_value=0)

    return (
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test,
    scaler,
)

def train_and_interpret(X_train: pd.DataFrame, 
    y_train: pd.Series,
    )-> LogisticRegression:
    """Train a Logistic Regression classifier and display the 
    learned coefficients and odds ratios for feature interpretation.
    """
    model = LogisticRegression(
        random_state=42, 
        max_iter=1000, ## Increase the iteration limit to help ensure convergence.
    )
    model.fit(X_train, y_train)
    
    print("\n=== MODEL COEFFICIENTS & ODDS RATIOS ===")
    print(f"Intercept (Base Log-Odds): {model.intercept_[0]:.4f}\n")
    for column, coefficient in zip(X_train.columns, model.coef_[0]):
        print(f"{column:<25} | Log-Odds: {coefficient:<8.4f} | Odds Ratio: {np.exp(coefficient):.4f}")
    print("=" * 60)
    
    return model

def evaluate_and_export(
    model: LogisticRegression,
    X_test: pd.DataFrame, 
    y_test: pd.Series,
    scaler: StandardScaler, 
    base_dir: Path
    )-> None:
    """Runs predictions, evaluates performance, and outputs plots to storage."""
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]

    # Save the trained model and fitted scaler for future inference.
    model_path = base_dir / "models" /  "logistic_model.pkl"
    scaler_path = base_dir / "models" / "standard_scaler.pkl"

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n=== MODEL PERFORMANCE ===")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    
    print("\n=== CONFUSION MATRIX ===")
    print(confusion_matrix(y_test, y_pred))
    
    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, y_pred))
    
    # ROC Plot Engine
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    print(f"\nROC AUC Score: {roc_auc:.4f}")
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    output_path = base_dir / "images" / "roc_curve.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"\n==== Artifact Successfully Exported ====")
    print(f"ROC Curve : {output_path}")
    print(f"Model     : {model_path}")
    print(f"Scaler    : {scaler_path}")

def main():
    # Execution setup
    BASE_DIR = Path.cwd()
    TARGET_COLUMN = "Churn"
    
    # Run entire modular pipeline
    train_df, test_df = load_data(BASE_DIR, 
    "churn_bigml_80.csv", 
    "churn_bigml_20.csv",
    )
    X_train, X_test, y_train, y_test, scaler = preprocess_and_scale(train_df, 
    test_df, 
    TARGET_COLUMN
    )
    model = train_and_interpret(X_train, y_train)
    evaluate_and_export(
        model,
        X_test,
        y_test,
        scaler,
        BASE_DIR,
    )

if __name__ == "__main__":
    main()
