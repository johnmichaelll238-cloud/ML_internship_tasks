# Iris Decision Tree with Cost-Complexity Pruning (Task 2)

## Overview

This project builds a modular decision tree classification pipeline on the Iris dataset. It trains a baseline tree, generates candidate pruned models using cost-complexity pruning (CCP), evaluates them on a held-out test set, selects the best model, saves the artifact, and exports a visualization of the final tree.

---

## Dataset

- **File:** `iris.csv`
- **Samples:** 150 (50 per class)
- **Features:** `sepal_length`, `sepal_width`, `petal_length`, `petal_width`
- **Target:** `species` (`setosa`, `versicolor`, `virginica`)

---

## Objectives

The pipeline performs the following steps:

1. Load and split the Iris dataset (80/20, stratified, `random_state=42`)
2. Train an unpruned baseline decision tree
3. Generate candidate `ccp_alpha` values from the pruning path
4. Train one pruned tree per candidate alpha
5. Evaluate train/test accuracy for each candidate
6. Select the model with the best test accuracy
7. Save the model with `joblib`
8. Export a PNG visualization of the selected tree

---

## Project Structure

```text
Task2/
│
├── main.py                       # End-to-end pipeline entry point
├── task2_decision_tree.ipynb     # Jupyter notebook version (optional)
├── iris.csv                      # Dataset
├── requirements.txt              # Dependencies
├── README.md                     # Project documentation
├── decision_tree2_model.joblib   # Saved model (generated after run)
├── pruned_tree2.png              # Saved tree plot (generated after run)
└── src/
    ├── config.py                 # Paths and hyperparameters
    ├── data_loader.py            # Load data and train/test split
    ├── trainer.py                # Baseline, pruning, candidate training
    ├── evaluation.py             # Model evaluation and selection
    ├── visual.py                 # Tree visualization
    └── experiments.py            # Optional experimentation script
```

---

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
cd ~/ML_projects
python -m venv .venv
source .venv/bin/activate
pip install -r Level2/Task2/requirements.txt
```

---

## Usage

Run the full pipeline from the Task2 directory:

```bash
cd ~/ML_projects/Level2/Task2
python main.py
```

Example output:

```text
Pipeline executed successfully.
Best Test Accuracy: 0.9667
```

Generated artifacts:

- `decision_tree2_model.joblib` — trained pruned decision tree
- `pruned_tree2.png` — visualization of the selected tree

Optional: open and run `task2_decision_tree.ipynb` in Jupyter Notebook.

---

## Configuration

Key settings in `src/config.py`:

| Setting | Value | Purpose |
|---------|-------|---------|
| `TEST_SIZE` | `0.2` | Hold-out test fraction |
| `RANDOM_STATE` | `42` | Reproducible split and training |
| `CRITERION` | `"gini"` | Split quality metric |
| `DATA_PATH` | `iris.csv` | Input dataset |
| `MODEL_FILENAME` | `decision_tree2_model.joblib` | Saved model path |
| `TREE_IMAGE_FILENAME` | `pruned_tree2.png` | Tree plot output path |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| pandas | Load and manipulate the dataset |
| numpy | Numerical arrays (used by pandas/sklearn) |
| scikit-learn | Decision tree, splitting, metrics |
| matplotlib | Tree visualization |
| joblib | Model serialization |
| notebook | Optional — run the `.ipynb` file |
| ipykernel | Optional — Jupyter kernel support |
