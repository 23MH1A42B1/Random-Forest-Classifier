# Random Forest Classifier Hyperparameter Optimization with GridSearchCV

![Python Version](https://img.shields.io/badge/Python-3.14%2B-blue.svg)
![Scikit-Learn Version](https://img.shields.io/badge/Scikit--Learn-1.3.0-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Build Status](https://img.shields.io/badge/Pipeline-Passed-brightgreen.svg)

An end-to-end, production-grade machine learning classification pipeline built with **Scikit-Learn**, **Pandas**, and **GridSearchCV**. This project demonstrates best practices in avoiding **Data Leakage**, building modular preprocessing pipelines using `ColumnTransformer`, and systematically optimizing hyperparameter search spaces using 5-fold cross-validation.

---

## 📁 Project Structure

```
project-root/
├── data/
│   └── winequality-red.csv               # Local copy of UCI Red Wine Quality dataset
├── notebooks/
│   ├── hyperparameter_optimization.ipynb # Fully executed Jupyter Notebook artifact
│   ├── confusion_matrix.png              # Visual diagnostic plot
│   └── feature_importance.png           # Feature importance bar chart
├── LICENSE                               # MIT Open Source License
├── README.md                             # Comprehensive project documentation
└── requirements.txt                      # Explicitly versioned dependency list
```

---

## 📊 Dataset & Task Description

- **Dataset**: [UCI Wine Quality (Red)](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- **Sample Count**: 1,599 instances
- **Features**: 11 physicochemical attributes (`fixed acidity`, `volatile acidity`, `citric acid`, `residual sugar`, `chlorides`, `free sulfur dioxide`, `total sulfur dioxide`, `density`, `pH`, `sulphates`, `alcohol`)
- **Target Variable**: Binary classification `quality_class`:
  - `1` (Good Quality): Wine rating `quality >= 6` (53.47% of samples)
  - `0` (Average/Low Quality): Wine rating `quality < 6` (46.53% of samples)
- **Categorical Feature**: `alcohol_group` (`Low`, `Medium`, `High`) engineered via quantile binning (`pd.qcut`) to validate multi-modal `ColumnTransformer` preprocessing.

---

## 🛡️ Preventing Data Leakage

A critical flaw in applied data science is fitting transformers (e.g., standard scalers, imputers, or one-hot encoders) on the entire dataset prior to splitting. This allows statistical properties of the test set (mean, variance, category frequencies) to leak into the training phase.

### Our Architectural Defense:
1. **Strict Holdout Split**: The dataset is split into **80% Training Data (`X_train`)** and **20% Test Data (`X_test`)** *before* any preprocessing using `train_test_split(..., test_size=0.20, random_state=42)`.
2. **Encapsulated Pipelines**: All transformations are encapsulated within a `ColumnTransformer` and passed into Scikit-Learn `Pipeline` objects.
3. **Cross-Validation Security**: During 5-Fold Cross-Validation, transformers are fit exclusively on the $k-1$ training folds during each iteration, completely isolating the validation fold.

---

## ⚙️ Hyperparameter Search Space

We tune the base `RandomForestClassifier` using `GridSearchCV` across a candidate hyperparameter grid:

| Pipeline Parameter | Search Values | Description |
| :--- | :--- | :--- |
| `classifier__n_estimators` | `[50, 100, 150]` | Number of trees in the forest ensemble |
| `classifier__max_depth` | `[None, 10, 20]` | Maximum depth allowable per tree |
| `classifier__min_samples_split` | `[2, 5]` | Minimum samples required to split an internal node |

*Total Fits*: $3 \text{ (n\_estimators)} \times 3 \text{ (max\_depth)} \times 2 \text{ (min\_samples\_split)} \times 5 \text{ (folds)} = 90 \text{ fit operations}$.

> **Note on Syntax**: Pipeline estimators require the double-underscore prefix (`classifier__<parameter_name>`) to instruct `GridSearchCV` which pipeline step is being parameterized.

---

## 📈 Quantitative Performance & Analytical Findings

### Holdout Test Set Evaluation Results (`X_test`, `y_test`)

| Metric | Baseline Model | Tuned Model | Difference |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **81.56%** | **80.94%** | **-0.62%** |
| **Precision (Weighted)** | **81.63%** | **80.97%** | **-0.66%** |
| **Recall (Weighted)** | **81.56%** | **80.94%** | **-0.62%** |
| **F1-Score (Weighted)** | **81.58%** | **80.95%** | **-0.63%** |

### Key Optimization Output:
- **Optimal Hyperparameters**:
  - `classifier__n_estimators`: `100`
  - `classifier__max_depth`: `20`
  - `classifier__min_samples_split`: `2`
- **Optimal Mean 5-Fold Cross-Validation Accuracy**: `80.69%`

### Analytical Discussion:
The baseline Random Forest model achieved **81.56%** accuracy on the holdout test set, while the 5-fold cross-validated model selected `n_estimators=100`, `max_depth=20`, `min_samples_split=2` with a cross-validation score of **80.69%** and a test accuracy of **80.94%**. 

This minor variance between cross-validation score and single holdout test set accuracy is a classic data science phenomenon:
1. **Validation Realism**: The 5-fold CV score (`80.69%`) provides a more realistic, variance-reduced estimate of generalization performance across the dataset distribution than a single random test split.
2. **Baseline Optimism**: Default baseline parameters can occasionally yield a slightly higher score on a specific holdout partition by chance, whereas cross-validated tuning guards against variance across different subsets of data.

---

## 🚀 Quickstart & Reproduction Guide

### Prerequisites
- Python 3.9+ (Tested on Python 3.14)
- Jupyter Notebook environment

### Environment Setup & Single-Command Run
```bash
# 1. Clone or navigate to the project directory
cd "c:/pdf/GPP/Random Forest Classifier"

# 2. Install dependencies & launch notebook
pip install -r requirements.txt && jupyter notebook notebooks/hyperparameter_optimization.ipynb
```

### Data Acquisition Script (Optional )
If downloading the dataset manually:
```bash
python -c "import os, urllib.request; os.makedirs('data', exist_ok=True); urllib.request.urlretrieve('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', 'data/winequality-red.csv')"
```


---

## 📜 License & Acknowledgments

This project is licensed under the **MIT License** - see the [`LICENSE`](file:///c:/pdf/GPP/Random%20Forest%20Classifier/LICENSE) file for details.


Dataset sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality). Developed as part of the Scikit-Learn Model Optimization Program.
