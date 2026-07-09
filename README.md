<div align="center">

# Employee Promotion Prediction

**An end-to-end machine learning pipeline that predicts whether an employee will be promoted — using ensemble models, SMOTE balancing, and a full evaluation framework.**

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-enabled-006400?style=flat)](https://xgboost.readthedocs.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-enabled-9ACD32?style=flat)](https://lightgbm.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat)](LICENSE)

</div>

---

## Overview

HR teams at large organizations manually review thousands of employees for promotion — a slow, inconsistent process. This project automates that decision using historical employee data and machine learning.

The pipeline handles real-world challenges like **class imbalance** (only ~8.5% of employees are promoted) using SMOTE, trains and compares **6 different models**, and outputs a promotion probability for any individual employee.

---

## Project Structure

```
employee-promotion-prediction/
│
├── data/
│   ├── train.csv                  # 54,808 employee records with labels
│   └── test.csv                   # 23,490 records for inference
│
├── data_preprocessing.py          # Cleaning, encoding, scaling, SMOTE
├── train_models.py                # Model definitions (LR, RF, Ada, XGB, LGB, Ensemble)
├── evaluate.py                    # Metrics, confusion matrix plots
├── predict.py                     # Single-employee promotion prediction
├── main.py                        # Full pipeline runner
│
├── requirements.txt
└── README.md
```

---

## Dataset

| Property | Value |
|---|---|
| Training records | 54,808 |
| Test records | 23,490 |
| Features | 11 |
| Target | `is_promoted` (binary) |
| Class ratio | 91.5% not promoted / 8.5% promoted |

**Features include:** department, region, education, gender, recruitment channel, number of trainings, age, previous year rating, length of service, awards won, average training score.

---

## Pipeline

```
Raw CSV
  │
  ▼
data_preprocessing.py
  ├── Drop employee_id
  ├── Fill missing values (mode for categorical, median for numeric)
  ├── Label encode all categorical columns
  ├── Stratified train/test split (75/25)
  ├── StandardScaler
  └── SMOTE (applied only on training data)
  │
  ▼
train_models.py  ──►  evaluate.py
  ├── Logistic Regression          Accuracy
  ├── Random Forest                Precision
  ├── AdaBoost                     Recall
  ├── XGBoost                      F1 Score
  ├── LightGBM                     Confusion Matrix
  └── Ensemble (LGB + XGB)
  │
  ▼
main.py
  ├── Bar chart: model accuracy comparison
  └── Best model selected automatically
  │
  ▼
predict.py
  └── Promotion probability + decision for any employee
```

---

## Models

| Model | Accuracy |
|---|---|
| Logistic Regression | 67.98% |
| AdaBoost | 71.02% |
| Random Forest | 90.16% |
| XGBoost | 93.96% |
| Ensemble (LGB + XGB) | 94.08% |
| **LightGBM** | **94.09% ✅ Best** |

---



## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/employee-promotion-prediction.git
cd employee-promotion-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the dataset

Place `train.csv` and `test.csv` inside the `data/` folder.

### 4. Run the pipeline

```bash
python main.py
```

This will train all 6 models, print metrics, plot confusion matrices, display a comparison bar chart, and run a sample prediction.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.8+ |
| ML Models | scikit-learn, XGBoost, LightGBM |
| Imbalance | imbalanced-learn (SMOTE) |
| Data | pandas, NumPy |
| Visualization | matplotlib, seaborn |

---

## Author

**Dharshan**
Built as a portfolio project demonstrating a complete, production-style ML pipeline on a real-world HR dataset.
Updated by collaborator Suriya on 09-07-2026

---

## License

This project is licensed under the [MIT License](LICENSE).
