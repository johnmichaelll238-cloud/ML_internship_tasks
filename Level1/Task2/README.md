# Housing Price Linear Regression (Task 2)

## Overview

This project implements a simple linear regression pipeline to predict housing prices from the number of rooms per dwelling. It loads a space-separated housing dataset, trains a `LinearRegression` model with scikit-learn, and evaluates performance using Mean Squared Error (MSE) and R².

---

## Dataset

- **File:** `housing_prices.csv`
- **Format:** Space-separated values, no header (506 samples × 14 features)
- **Feature (X):** Column 5 — average number of rooms per dwelling
- **Target (y):** Column 13 — median home value (in $1,000s)

The file follows the classic Boston Housing dataset layout.

---

## Objectives

The pipeline performs the following steps:

1. Load the dataset with pandas
2. Extract the rooms feature and price target
3. Split data into 80% training / 20% testing (`random_state=42`)
4. Train a simple linear regression model
5. Evaluate with MSE and R²
6. Print sample predictions vs. actual values

---

## Project Structure

```text
Task2/
│
├── task2_linear_regression.py   # Main regression script
├── task2_linear_regression.ipynb # Jupyter notebook version (optional)
├── housing_prices.csv            # Dataset
├── requirements.txt              # Dependencies
└── README.md                     # Project documentation
```

---

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
cd ~/ML_projects
python -m venv .venv
source .venv/bin/activate
pip install -r Level1/Task2/requirements.txt
```

---

## Usage

Run the script from the **project root** (`ML_projects`), since the data path is relative to that directory:

```bash
python Level1/Task2/task2_linear_regression.py
```

Or open and run `task2_linear_regression.ipynb` in Jupyter Notebook.

---

## Model

The trained model fits a line of the form:

```text
Price = (weight × Rooms) + bias
```

Example output from a successful run:

```text
Regression Equation:
    Price = (9.3483 × Rooms) + (-36.2463)

Mean Squared Error (MSE): 46.1448
R² Score: 0.3708
```

R² ≈ 0.37 indicates that rooms alone explain a moderate share of price variation; additional features would likely improve the model.

---

## Dependencies

| Package        | Purpose                          |
|----------------|----------------------------------|
| pandas         | Load and manipulate the dataset  |
| numpy          | Numerical arrays (used by pandas/sklearn) |
| scikit-learn   | Train/test split, regression, metrics |
| notebook       | Optional — run the `.ipynb` file |
| ipykernel      | Optional — Jupyter kernel support |
