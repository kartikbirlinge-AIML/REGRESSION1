# 🔢 Polynomial Regression — Fitting Curves to Data

## What's This About?

Sometimes data just doesn't follow a straight line — and that's where **polynomial regression** comes in. This notebook demonstrates how to fit a **curved** model to data that has a non-linear (specifically quadratic) shape, while still leveraging the familiar `LinearRegression` engine under the hood.

Think of it as "linear regression, but with superpowers."

---

## The Data

Unlike the other notebooks, there's **no external CSV here** — the data is synthetically generated using NumPy. The formula used is:

```
y = -1.5 * x² + 1.5 * x + noise
```

This creates a classic **inverted parabola** shape with some random scatter added to make it realistic. 100 data points are generated with x values ranging from **-3 to 3**.

A green scatter plot is shown right away so you can see the curve shape before any modeling begins.

---

## What the Notebook Does

### Part 1 — Degree-2 Polynomial Regression (Manual)

1. **Generate Data** — A synthetic quadratic dataset is created and plotted.
2. **Train/Test Split** — 80% training, 20% testing (with `random_state=42` for reproducibility).
3. **Transform Features** — `PolynomialFeatures(degree=2)` transforms the raw `x` values into `[1, x, x²]` — giving the linear regression model the "curved" input it needs.
4. **Train the Model** — A standard `LinearRegression` is fitted on the transformed training data.
5. **Evaluate** — R² score is calculated on the test set.
6. **Visualize** — Two plots are generated:
   - Predictions vs. actual training points (scatter)
   - Full smooth prediction curve over the range [-3, 3] with training points (blue) and test points (green) overlaid.

### Part 2 — Pipeline Approach (Comparing Degrees)

A reusable function `poly_regression(degree)` is defined that:
- Creates a `Pipeline` combining `PolynomialFeatures` and `LinearRegression`
- Fits on training data
- Plots the resulting prediction curve for a given polynomial degree

The function is then called with **degree = 10**, showing how high-degree polynomials can overfit — the curve wiggles wildly to pass through training points but likely performs worse on unseen data.

---

## Model Performance

The R² score from the degree-2 model is computed and printed. Since the data was generated from a quadratic formula, a degree-2 polynomial fits it very well, typically scoring **close to 1.0**.

---

## Libraries Used

- `numpy` — data generation and numerical operations
- `matplotlib` — scatter plots and prediction curves
- `scikit-learn`
  - `PolynomialFeatures` — feature transformation
  - `LinearRegression` — the actual model
  - `train_test_split` — splitting data
  - `r2_score` — evaluation metric
  - `Pipeline` — chaining preprocessing + model together

---

## How to Run It

1. Install dependencies:
   ```
   pip install numpy matplotlib scikit-learn
   ```
2. No dataset file needed — the data is generated inside the notebook.
3. Open in Jupyter or Google Colab and run all cells in order.

> **Tip:** Try changing the `degree` parameter in `poly_regression()` to see how lower or higher degrees affect the fit. Degree 1 is just linear regression; degree 10 is likely overfitting.

---

## Key Takeaway

Polynomial regression is a powerful way to model **non-linear relationships** without leaving the linear regression framework. The trick is transforming your input features into polynomial terms first. Just watch out for **overfitting** — the higher the degree, the more the model memorizes noise rather than learning the true pattern.
