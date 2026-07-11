# Support Vector Machine (SVM) Classification Pipeline

## Overview

This project implements a complete machine learning classification pipeline using a **Support Vector Machine (SVM)** classifier. The pipeline covers data preprocessing, feature scaling, model training, evaluation, and visualization, while following a modular software design for improved readability and maintainability.

The project demonstrates how Support Vector Machines can be used to build accurate classification models using structured datasets.

---

## Features

* Data loading and preprocessing
* Feature scaling using `StandardScaler`
* Train-test data splitting
* Linear Support Vector Machine training
* Model evaluation using multiple performance metrics
* Confusion matrix visualization
* Class distribution visualization
* Modular project architecture

---

## Project Structure

```text
project/
│
├── config.py          # Configuration variables
├── data.py            # Data loading and preprocessing
├── model.py           # Model creation, training, and prediction
├── visual.py          # Visualization utilities
├── main.py            # Project entry point
│
├── data/              # Dataset
├── outputs/           # Saved plots and results
└── README.md
```

---

## Workflow

1. Load the dataset.
2. Clean and preprocess the data.
3. Split the dataset into training and testing sets.
4. Standardize numerical features using `StandardScaler`.
5. Train a Linear Support Vector Machine classifier.
6. Evaluate the trained model.
7. Generate and save visualizations.

---

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

---

## Model Evaluation

The classifier is evaluated using:

* Accuracy Score
* Classification Report
* Confusion Matrix

These metrics provide a comprehensive assessment of the model's performance on unseen data.

---

## Visualizations

The project automatically generates visualizations including:

* Class Distribution
* Confusion Matrix

All visualizations are saved for later analysis.

---

## How to Run

### Clone the repository

```bash
git clone <repository-url>
cd <repository-name>
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the project

```bash
python main.py
```

---

## Learning Objectives

This project demonstrates practical experience with:

* Machine Learning pipelines
* Support Vector Machines (SVM)
* Data preprocessing
* Feature scaling
* Model evaluation
* Data visualization
* Modular Python development

---

## Future Improvements

* Hyperparameter tuning using GridSearchCV
* Cross-validation
* Support for additional SVM kernels
* Model persistence with Joblib
* Automated experiment tracking

---

## License

This project is intended for educational purposes and as a demonstration of machine learning concepts and software engineering practices.
