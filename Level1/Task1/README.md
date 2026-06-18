# Titanic Data Preprocessing Pipeline (Task 1)

## Overview
This project implements an end-to-end data preprocessing pipeline using the Titanic dataset.  
The goal is to prepare raw data for machine learning by handling missing values, encoding categorical variables, splitting the dataset, and scaling numerical features.

---

## Dataset
The dataset used is the classic Titanic dataset:

- Source: https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv  
- Target variable: `Survived`

---

## Objectives

The pipeline performs the following steps:

1. Load raw dataset
2. Drop irrelevant or high-cardinality features
3. Handle missing values
4. Encode categorical variables
5. Split data into training and testing sets
6. Standardize numerical features

---

## Project Structure

```text
Task_1/
│
├── task1.py              # Main preprocessing script
├── task1.ipynb           # Jupyter notebook version (optional)
├── requirements.txt      # Dependencies
└── README.md             # Project documentation