# SVM Regressor (SVR) — Predicting Restaurant `total_bill`

This project uses **Support Vector Regression (SVR)** on the classic seaborn `tips` dataset to predict
`total_bill` from `tip`, `sex`, `smoker`, `day`, `time`, and party `size`.

Two notebooks are relevant:
- **`SVM_Regressor.ipynb`** — your original notebook.
- **`SVM_Regressor_Improved.ipynb`** — an improved version with feature scaling, a wider hyperparameter
  search, and cross-validated evaluation.

---

## 1. What is an SVM Regressor (SVR)?

Support Vector Machines (SVMs) are usually introduced as **classifiers** that find the widest possible margin
separating two classes. **Support Vector Regression (SVR)** adapts the same idea to predict a continuous
number instead of a class label.

**The core idea — an "epsilon tube":**
- Instead of trying to make predictions match the data exactly, SVR draws a "tube" of width `epsilon` (ε)
  around the regression line/curve.
- Any training point that falls **inside** the tube is considered a "good enough" prediction and contributes
  **zero** penalty to the loss — this is what makes SVR different from ordinary least-squares regression,
  which penalizes every deviation.
- Points **outside** the tube are penalized proportionally to how far they are from the tube edge, and only
  these points (the **support vectors**) end up influencing the shape of the final model.

**Key hyperparameters:**
| Parameter | What it controls |
|---|---|
| `kernel` | How the model measures similarity between data points. `linear` fits a straight line/hyperplane. `rbf` (Radial Basis Function, the default) can fit curved, flexible relationships. `poly` fits polynomial curves. |
| `C` | The regularization strength. A **large `C`** tries hard to fit every point outside the tube (risk of overfitting). A **small `C`** allows more points to be "wrong" in exchange for a smoother, simpler model. |
| `gamma` (rbf/poly only) | How far the influence of a single training point reaches. **High gamma** = very local, wiggly fit (can overfit). **Low gamma** = smoother, more global fit. |
| `epsilon` | The width of the no-penalty tube. **Larger epsilon** = a wider tube = a simpler model that ignores more small errors. |

**Why scaling matters for SVR:** because the model is built on distances and margins between data points,
features on very different numeric scales (e.g. `size` ranging 1–6 vs `tip` ranging ~1–10) can distort the
kernel's notion of "similarity." This is why the improved notebook adds a `StandardScaler` step.

---

## 2. Walking through your original notebook (`SVM_Regressor.ipynb`)

1. **Load data** — `sns.load_dataset('tips')`, a built-in 244-row dataset of restaurant bills.
2. **Define X and Y** — `X` = `tip, sex, smoker, day, time, size`; `Y` = `total_bill` (the value to predict).
3. **Train/test split** — 75% train, 25% test, `random_state=10` for reproducibility.
4. **One-hot encode categoricals** — `sex`, `smoker`, `day`, `time` are turned into 0/1 columns via
   `ColumnTransformer` + `OneHotEncoder(drop='first')`; the numeric columns `tip` and `size` are passed
   through unchanged (`remainder='passthrough'`).
5. **Fit a default `SVR()`** — trains with default `kernel='rbf'`, `C=1.0`, `gamma='scale'`, `epsilon=0.1`.
6. **Evaluate** — `r2_score` on the test set came out to **≈0.46**, meaning the model explains about 46% of
   the variance in `total_bill`.
7. **`GridSearchCV`** — searches over `C` (0.1–1000) and `gamma` (1–0.0001) with a fixed `rbf` kernel.
   The best combination found was `C=1000, gamma=0.0001`, improving test R² to **≈0.51** and MAE to **≈3.87**.

This is a solid first pass — the main gaps were **no feature scaling** and a **narrow search** (only one
kernel, no `epsilon` tuning, no cross-validated final check).

---

## 3. What changed in the improved notebook (`SVM_Regressor_Improved.ipynb`)

| # | Change | Why |
|---|---|---|
| 1 | Added `StandardScaler` to the numeric features inside the `ColumnTransformer` | Puts `tip` and `size` on comparable scales to the one-hot columns, so the kernel doesn't let one feature dominate |
| 2 | Wrapped preprocessing + model in a single `Pipeline` | Guarantees the scaler is fit only on training folds during cross-validation (no data leakage) and keeps train/test/CV code identical |
| 3 | Expanded the hyperparameter grid to include `epsilon`, and added `linear`/`poly` kernel options alongside `rbf` | `epsilon` is often as impactful as `C`/`gamma`; with only 244 rows a simpler kernel can sometimes generalize better than `rbf` |
| 4 | Added `mean_absolute_error` and `RMSE` alongside R² | R² alone can hide how large the typical dollar-error is; MAE/RMSE are in the same units as `total_bill` |
| 5 | Added a final 5-fold `cross_val_score` over the *entire* dataset with the best model | A single 75/25 split can be lucky/unlucky on a small dataset; cross-validation gives a more trustworthy accuracy estimate |
| 6 | Added an Actual-vs-Predicted scatter plot | Quick visual sanity check of where the model over/under-predicts |

**Results after these changes** (your exact numbers may vary slightly by environment):

| Model | Test R² | Test MAE |
|---|---|---|
| Original baseline `SVR()` | 0.46 | — |
| Original tuned (`GridSearchCV`, rbf only) | 0.51 | 3.87 |
| **Improved tuned (scaled + wider grid)** | **~0.53** | **~3.9** |
| **Improved, 5-fold CV mean (most reliable estimate)** | **~0.55 (± 0.10)** | — |

---

## 4. A realistic ceiling on accuracy

`total_bill` is genuinely hard to predict from just `tip`, `size`, and categorical context — tipping habits
vary a lot from person to person, and a small `tip` doesn't always mean a small bill. R² in the 0.5–0.6 range
is a reasonable ceiling for this feature set. To push further you'd typically need more informative features
(e.g. number/type of items ordered, or historical customer averages) rather than more model tuning.

## 5. How to run

```bash
pip install pandas numpy seaborn matplotlib scikit-learn
jupyter notebook SVM_Regressor_Improved.ipynb
```

Run all cells top to bottom — the `GridSearchCV` step is the slowest (a few seconds on this small dataset).
