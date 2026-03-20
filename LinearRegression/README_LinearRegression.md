# Simple Linear Regression — Height & Weight Predictor

## What's This About?

This notebook walks through a classic machine learning task: **predicting a person's height based on their weight**. It's a great starting point for understanding how linear regression works — the idea being that there's a straight-line relationship between two variables, and we want to find that line.

---

## The Data

We're working with a small, clean dataset of **23 people**, each with two measurements:

| Column   | Description              |
|----------|--------------------------|
| `Weight` | Weight of a person (kg)  |
| `Height` | Height of a person (cm)  |

The data ranges from lightweight individuals (45 kg / 120 cm) all the way up to heavier ones (105 kg / 183 cm). It's a nice, compact dataset — perfect for learning.

---

## What the Notebook Does

Here's the step-by-step flow:

1. **Import Libraries** — pandas, numpy, and matplotlib are loaded up.
2. **Load the Data** — The CSV file (`height-weight.csv`) is read into a DataFrame.
3. **Explore the Data** — The full table is displayed so you can see what you're working with.
4. **Visualize** — A scatter plot is generated to see the weight-height relationship visually.
5. **Prepare Features** — Weight is used as the input (X), and Height as the output (Y).
6. **Scale the Data** — `StandardScaler` is applied to normalize the weight values before training.
7. **Split the Data** — The dataset is divided into training and test sets.
8. **Train the Model** — A `LinearRegression` model from scikit-learn is fitted on the training data.
9. **Evaluate** — Predictions are made on the test set.
10. **Make a Custom Prediction** — For example, predicting the height of someone who weighs **72 kg** → the model returns approximately **155.98 cm**.

---

## Libraries Used

- `pandas` — data loading and manipulation
- `numpy` — numerical operations
- `matplotlib` — plotting
- `scikit-learn` — `LinearRegression`, `StandardScaler`, `train_test_split`

---

## How to Run It

1. Make sure you have Python installed along with the required libraries:
   ```
   pip install pandas numpy matplotlib scikit-learn
   ```
2. Place the `height-weight.csv` file in the same directory (or update the path in the notebook).
3. Open the notebook in Jupyter or Google Colab and run all cells top to bottom.

---

## Key Takeaway

Linear regression shines when your data has a **clear, roughly linear trend**. This height-weight example demonstrates that beautifully — as weight increases, height tends to increase too, and a straight line can capture that relationship pretty well.

