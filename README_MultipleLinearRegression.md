#  Multiple Linear Regression — Economic Index Predictor

## What's This About?

This notebook explores **Multiple Linear Regression** — an extension of simple linear regression where we use **more than one input variable** to predict an outcome. Here, the goal is to predict a stock/economic **index price** based on interest rates and unemployment rates over time.

---

## The Data

The dataset (`economic_index.csv`) contains **24 monthly records** spanning 2016–2017, with the following columns:

| Column               | Description                          |
|----------------------|--------------------------------------|
| `year`               | Year of the record                   |
| `month`              | Month of the record                  |
| `interest_rate`      | Monthly interest rate (%)            |
| `unemployment_rate`  | Monthly unemployment rate (%)        |
| `index_price`        | The target — economic index price    |

The index price ranges from around **704 to 1464**, and you can see a clear upward trend as interest rates rise.

---

## What the Notebook Does

1. **Import Libraries** — numpy, pandas, and matplotlib are loaded.
2. **Load the Data** — The CSV is read into a DataFrame and displayed in full.
3. **Visualize the Relationships** — Scatter plots are generated to understand how `interest_rate` and `unemployment_rate` each relate to `index_price` individually.
4. **Set Up Features & Target** — `interest_rate` and `unemployment_rate` are used as input features (X); `index_price` is the target (Y).
5. **Train/Test Split** — The data is split 75/25 into training and test sets.
6. **Train the Model** — A `LinearRegression` model is fitted on the training data.
7. **Predict & Evaluate** — Predictions are made on the test set, and performance is assessed using:
   - **R² Score**: `0.759` — the model explains ~76% of the variance in index price.
   - **Adjusted R² Score**: `0.599` — a more conservative estimate accounting for the number of features.
8. **OLS Regression (statsmodels)** — A more detailed statistical summary is generated using `statsmodels.OLS`, including p-values, confidence intervals, and goodness-of-fit metrics.

---

## Model Performance

| Metric             | Value  |
|--------------------|--------|
| R² Score           | 0.759  |
| Adjusted R²        | 0.599  |

The R² of ~0.76 is decent for a simple two-variable model. The OLS summary shows that the individual coefficients aren't statistically significant on their own (high p-values), suggesting potential **multicollinearity** between interest rate and unemployment rate — they tend to move together.

---

## Libraries Used

- `pandas` — data loading and manipulation
- `numpy` — numerical operations
- `matplotlib` — scatter plot visualizations
- `scikit-learn` — `LinearRegression`, `train_test_split`, `r2_score`
- `statsmodels` — OLS regression for detailed statistical output

---

## How to Run It

1. Install dependencies:
   ```
   pip install pandas numpy matplotlib scikit-learn statsmodels
   ```
2. Place `economic_index.csv` in the same directory as the notebook (or update the path).
3. Open in Jupyter or Google Colab and run all cells in order.

---

## Key Takeaway

Multiple linear regression lets you incorporate **several predictors** at once, which can improve model accuracy compared to using a single variable. However, when predictors are correlated with each other (like interest rates and unemployment often are), it's worth checking for multicollinearity — the OLS summary is a great tool for spotting this.
