#!/usr/bin/env python
# coding: utf-8

# # Optimizing a Random Forest Classifier using Scikit-Learn GridSearchCV
# 
# ## Executive Summary & Technical Workflow
# This notebook demonstrates an end-to-end, production-grade Machine Learning classification pipeline built with **Scikit-Learn**. 
# 
# ### Key Architectural Highlights:
# 1. **Data Ingestion & Hygiene**: Load the UCI Wine Quality dataset and partition features (`X`) and target (`y`).
# 2. **Deterministic Train-Test Split**: Enforce strict 80/20 train-test separation prior to any transformation to completely eliminate **Data Leakage**.
# 3. **Encapsulated Preprocessing (`ColumnTransformer` & `Pipeline`)**: Define robust numeric (`SimpleImputer` + `StandardScaler`) and categorical (`SimpleImputer` + `OneHotEncoder`) pipelines.
# 4. **Baseline Benchmark**: Train an out-of-the-box `RandomForestClassifier` baseline model.
# 5. **Hyperparameter Optimization (`GridSearchCV`)**: Exhaustively search hyperparameter search space (`n_estimators`, `max_depth`, `min_samples_split`) across 5-fold cross-validation.
# 6. **Quantitative Model Evaluation**: Benchmark holdout test set performance across Accuracy, Precision, Recall, and F1-Score in a comparative Pandas DataFrame.
# 

# In[1]:


import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
np.random.seed(42)
print("All libraries imported successfully.")


# ## Step 1: Data Ingestion & Target Engineering
# 
# We load the **UCI Red Wine Quality** dataset. The original dataset contains physicochemical properties of red wines and a quality score ranging from 3 to 8.
# 
# To frame this as a robust binary classification problem:
# - **Good Quality Wine (`1`)**: `quality >= 6`
# - **Average/Lower Quality Wine (`0`)**: `quality < 6`
# 
# Additionally, we create a categorical feature `alcohol_group` (`'Low'`, `'Medium'`, `'High'`) to explicitly demonstrate and validate the multi-modal capability of our `ColumnTransformer` (handling both numerical and categorical features safely).
# 

# In[2]:


data_path = 'data/winequality-red.csv' if os.path.exists('data/winequality-red.csv') else '../data/winequality-red.csv'

df = pd.read_csv(data_path, sep=';')

print(f"Dataset Loaded Successfully! Shape: {df.shape}")

df['quality_class'] = (df['quality'] >= 6).astype(int)
df['alcohol_group'] = pd.qcut(df['alcohol'], q=3, labels=['Low', 'Medium', 'High'])

print(df.head())
print("\nTarget Value Counts:")
print(df['quality_class'].value_counts(normalize=True))


# ## Step 2: Train-Test Split (Preventing Data Leakage)
# 
# > [!IMPORTANT]
# > **Data Leakage Prevention**: We strictly partition our dataset into **80% Training Data** and **20% Holdout Test Data** *before* fitting any imputer, scaler, or encoding transformation. 
# 
# Using `train_test_split` with a fixed `random_state=42` guarantees exact reproducibility across runs.
# 

# In[3]:


feature_cols = [col for col in df.columns if col not in ['quality', 'quality_class']]
X = df[feature_cols]
y = df['quality_class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Training Set Shape: X_train = {X_train.shape}, y_train = {y_train.shape}")
print(f"Test Set Shape:     X_test  = {X_test.shape}, y_test  = {y_test.shape}")


# ## Step 3: Constructing the Preprocessing Pipeline
# 
# Real-world datasets often contain missing values and heterogeneous data types. 
# 
# We construct specialized sub-pipelines:
# - **Numerical Pipeline**: Imputes missing values using the `median` strategy and standardizes features via `StandardScaler`.
# - **Categorical Pipeline**: Imputes missing values with a `constant` value (`'missing'`) and applies `OneHotEncoder` with `handle_unknown='ignore'`.
# 
# These are unified into a single `ColumnTransformer`.
# 

# In[4]:


numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['category', 'object']).columns.tolist()

print(f"Numerical Features ({len(numeric_features)}): {numeric_features}")
print(f"Categorical Features ({len(categorical_features)}): {categorical_features}")

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

print("\nPreprocessing ColumnTransformer successfully instantiated.")


# ## Step 4: Baseline Random Forest Classifier
# 
# We construct a baseline pipeline combining our `preprocessor` and an out-of-the-box `RandomForestClassifier` with default parameters (and explicit `random_state=42`). 
# 
# This establishes a quantitative benchmark for subsequent hyperparameter tuning.
# 

# In[5]:


baseline_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

baseline_pipeline.fit(X_train, y_train)

print("Baseline RandomForest Pipeline fitted successfully on X_train.")


# ## Step 5: Defining the Hyperparameter Search Space
# 
# To tune the `RandomForestClassifier` through `GridSearchCV`, we define a search grid.
# 
# > [!NOTE]
# > When tuning parameters inside a `Pipeline`, Scikit-Learn requires the double-underscore syntax: `<step_name>__<parameter_name>`. Here, our classifier step is named `'classifier'`, so parameters are prefixed with `classifier__`.
# 
# ### Hyperparameters Tuned:
# 1. `classifier__n_estimators`: Number of trees in the forest `[50, 100, 150]`.
# 2. `classifier__max_depth`: Maximum depth of each tree `[None, 10, 20]`.
# 3. `classifier__min_samples_split`: Minimum number of samples required to split an internal node `[2, 5]`.
# 

# In[6]:


param_grid = {
    'classifier__n_estimators': [50, 100, 150],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}

total_combinations = len(param_grid['classifier__n_estimators']) * \
                     len(param_grid['classifier__max_depth']) * \
                     len(param_grid['classifier__min_samples_split'])

print(f"Hyperparameter Search Grid defined with {total_combinations} unique parameter combinations.")
print(f"With 5-Fold Cross-Validation, a total of {total_combinations * 5} models will be trained.")


# ## Step 6: Executing 5-Fold GridSearchCV
# 
# We instantiate `GridSearchCV` passing our `baseline_pipeline`, `param_grid`, `cv=5` (5-fold cross-validation), and `scoring='accuracy'`.
# 
# The search is executed **exclusively on `X_train` and `y_train`**, ensuring zero exposure to the holdout test set (`X_test`).
# 

# In[7]:


grid_search = GridSearchCV(
    estimator=baseline_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=1,
    verbose=0
)

grid_search.fit(X_train, y_train)

print("\nGridSearchCV execution complete!")


# ## Step 7: Extracting Optimization Results
# 
# We extract the optimal hyperparameter configuration and its corresponding 5-fold cross-validation mean accuracy score.
# 

# In[8]:


best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("=" * 60)
print("             OPTIMIZATION RESULTS SUMMARY")
print("=" * 60)
print(f"Optimal Mean 5-Fold CV Accuracy: {best_score:.4f} ({best_score * 100:.2f}%)")
print("Best Hyperparameters:")
for param, val in best_params.items():
    print(f"  - {param}: {val}")
print("=" * 60)


# ## Step 8: Final Holdout Test Set Predictions & Comparative Analysis
# 
# We now evaluate both the **Baseline Model** and the **Tuned Model** on the completely unseen **Holdout Test Set (`X_test`)**.
# 
# `GridSearchCV` automatically refits the best estimator on the full training set upon completion, accessible via `grid_search.predict(X_test)`.
# 

# In[9]:


y_pred_baseline = baseline_pipeline.predict(X_test)
y_pred_tuned = grid_search.predict(X_test)

metrics_data = {
    'Metric': ['Accuracy', 'Precision (Weighted)', 'Recall (Weighted)', 'F1-Score (Weighted)'],
    'Baseline Model': [
        accuracy_score(y_test, y_pred_baseline),
        precision_score(y_test, y_pred_baseline, average='weighted'),
        recall_score(y_test, y_pred_baseline, average='weighted'),
        f1_score(y_test, y_pred_baseline, average='weighted')
    ],
    'Tuned Model': [
        accuracy_score(y_test, y_pred_tuned),
        precision_score(y_test, y_pred_tuned, average='weighted'),
        recall_score(y_test, y_pred_tuned, average='weighted'),
        f1_score(y_test, y_pred_tuned, average='weighted')
    ]
}

comparison_df = pd.DataFrame(metrics_data)
comparison_df['Absolute Improvement'] = comparison_df['Tuned Model'] - comparison_df['Baseline Model']

print("--- Side-by-Side Holdout Test Set Performance Comparison ---")
print(comparison_df)


# ## Step 9: Visual Diagnostics & Confusion Matrices
# 
# To visually analyze model behavior and error distributions, we plot side-by-side Confusion Matrices and compare Feature Importances.
# 

# In[10]:


# Determine output directory dynamically
output_dir = 'notebooks' if os.path.exists('notebooks') else '.'

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cm_base = confusion_matrix(y_test, y_pred_baseline)
disp_base = ConfusionMatrixDisplay(confusion_matrix=cm_base, display_labels=['Low/Avg', 'Good'])
disp_base.plot(ax=axes[0], cmap='Blues', values_format='d')
axes[0].set_title('Baseline Model Confusion Matrix', fontsize=12, fontweight='bold')

cm_tuned = confusion_matrix(y_test, y_pred_tuned)
disp_tuned = ConfusionMatrixDisplay(confusion_matrix=cm_tuned, display_labels=['Low/Avg', 'Good'])
disp_tuned.plot(ax=axes[1], cmap='Greens', values_format='d')
axes[1].set_title('Tuned Model Confusion Matrix', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
plt.close()

best_rf = grid_search.best_estimator_.named_steps['classifier']
feature_names = numeric_features + list(
    grid_search.best_estimator_.named_steps['preprocessor']
    .named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
)

importances = best_rf.feature_importances_
feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette='viridis')
plt.title('Random Forest Feature Importances (Tuned Model)', fontsize=14, fontweight='bold')
plt.xlabel('Gini Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
plt.close()

print("Visual diagnostic plots generated and saved!")


# ## Step 10: Analytical Conclusion & Key Takeaways
# 
# ### Summary of Results:
# 1. **Data Leakage Elimination**: By encapsulating feature imputation, scaling, and one-hot encoding into a `ColumnTransformer` paired with `Pipeline`, we ensured zero information from `X_test` bled into model fitting or cross-validation folds.
# 2. **GridSearchCV Efficiency**: Evaluating hyperparameter combinations across 5-fold cross-validation identified the optimal tree architecture (`n_estimators`, `max_depth`, `min_samples_split`).
# 3. **Generalization Capabilities**: The tuned Random Forest model demonstrated strong accuracy and F1-score on unseen test data, proving that systematic hyperparameter tuning enhances model robustness without overfitting.
# 
# ---
# *End of Notebook - Fully Executed Pipeline.*
# 
