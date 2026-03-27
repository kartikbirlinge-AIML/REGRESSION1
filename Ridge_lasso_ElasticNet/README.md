# Ridge, Lasso & ElasticNet Regression

A machine learning project demonstrating regularization techniques on the **Diabetes dataset** from scikit-learn.

## Overview

This notebook compares three regularized linear regression models — **Ridge**, **Lasso**, and **ElasticNet** — alongside a baseline **Linear Regression**, applied to a diabetes progression prediction task.

## Dataset

- **Source:** `sklearn.datasets.load_diabetes`
- **Samples:** 442
- **Features:** 10 (age, sex, bmi, bp, s1–s6) — all pre-normalized
- **Target:** A continuous measure of diabetes disease progression one year after baseline

### Feature Selection

Correlation analysis was performed to identify features with correlation > 0.4 with the target:

| Feature | Correlation |
|---------|-------------|
| bmi     | 0.586       |
| s5      | 0.566       |
| bp      | 0.441       |
| s4      | 0.430       |

Low-correlation features (age, sex, s1, s2, s3, s6) were dropped, and the model was trained on **bmi, s4, s5**.

## Pipeline

```
Load Data → Feature Selection → Train/Test Split (75/25) → StandardScaler → Model Training → Evaluation
```

## Models & Results

| Model                     | MSE     | MAE   | RMSE  | R²    |
|---------------------------|---------|-------|-------|-------|
| Linear Regression         | 2705.48 | 41.75 | 52.01 | 0.511 |
| Ridge (default)           | 2705.75 | 41.76 | 52.02 | 0.511 |
| Ridge (CV-tuned, α=10)    | 2709.28 | 41.83 | 52.05 | 0.510 |
| Lasso (default)           | 2713.22 | 41.89 | 52.09 | 0.509 |
| Lasso (CV-tuned, α≈0.046) | 2705.76 | 41.75 | 52.02 | 0.511 |
| ElasticNet (default)      | 2916.17 | 44.42 | 54.00 | 0.473 |
| ElasticNet (CV-tuned)     | 2712.57 | 41.89 | 52.08 | 0.509 |

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
```

## Usage

Open the notebook in Google Colab or Jupyter and run cells sequentially.