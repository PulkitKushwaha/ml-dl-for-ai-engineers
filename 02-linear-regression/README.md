# 02 — Linear Regression

> **Week 1 · Topic 2** · The simplest algorithm — and the one that teaches you how every algorithm actually works.

---

## The core idea

Linear regression finds the best straight line through your data to predict a continuous number. It is a supervised learning algorithm used for regression tasks — the output is always a number, not a category.

But the real reason to study linear regression deeply is not the algorithm itself — it is what learning it teaches you. Cost functions, gradient descent, learning rate, feature scaling, and model assumptions all appear here for the first time. These same mechanics power every algorithm from logistic regression all the way to GPT.

---

## The music journalist analogy

You are a music journalist trying to predict how many copies an album will sell based on one thing — how many weeks it spent in the top 10 before release. You have data from 100 past albums. You plot them: X axis is weeks in top 10, Y axis is albums sold.

You notice a pattern — more weeks in the top 10 generally means more sales. So you grab a ruler and draw the best fitting straight line through that scatter of points. Now when a new album drops, you find its weeks in the top 10 on the X axis, read off the line, and get your predicted sales number.

That is linear regression. A straight line that best summarises the relationship between input and output. The algorithm's job is to find the exact position and angle of that line that minimises prediction error across all your data points.

---

## The equation

Simple linear regression (one feature):

```
ŷ = w₁x + w₀
```

- **ŷ** — the predicted value (e.g. predicted album sales)
- **x** — the input feature (e.g. weeks in top 10)
- **w₁** — the weight (slope) — how much y changes per unit of x
- **w₀** — the bias term (intercept) — baseline value when x = 0

Multiple linear regression (many features):

```
ŷ = w₁x₁ + w₂x₂ + w₃x₃ + ... + w₀
```

Each feature gets its own weight. Training finds the best values of all weights simultaneously.

---

## The cost function — measuring how wrong the model is

The cost function (also called loss function or objective function) is a mathematical measure of how wrong the model's current predictions are across all training data. Training is entirely about minimising this number.

### Mean Squared Error (MSE) — the standard for regression

```
MSE = (1/n) × Σ(ŷᵢ − yᵢ)²
```

For every data point: calculate (predicted − actual)², sum them all, divide by n.

**Why squared and not just the raw difference?**

Two reasons:
1. Errors can be positive or negative — a prediction of 105 when the answer is 100 (+5) and a prediction of 95 when the answer is 100 (−5) would cancel out if summed directly. Squaring makes all errors positive so they accumulate honestly.
2. Squaring penalises large errors disproportionately — off by 10 contributes 100 to the loss, off by 2 contributes only 4. This pushes the model to fix its worst mistakes first.

### Other common loss functions

| Loss function | When to use |
|---|---|
| **MSE** (Mean Squared Error) | Default for regression. Sensitive to outliers due to squaring. |
| **RMSE** (Root MSE) | Same as MSE but in the same units as y — easier to interpret. |
| **MAE** (Mean Absolute Error) | More robust to outliers. Treats all errors linearly. |
| **Huber Loss** | MSE for small errors, MAE for large ones. Great when data has some outliers. |
| **R² (R-squared)** | Proportion of variance in y explained by the model. 1.0 = perfect, 0 = no better than the mean. |

**Interview tip:** Use MSE when large errors are unacceptable. Use MAE when data has genuine outliers you do not want to over-penalise.

---

## Gradient descent — how the model actually learns

### The blindfolded valley analogy

Imagine you are blindfolded on a hilly landscape and you want to reach the lowest valley. You cannot see the whole landscape — you can only feel which direction is downhill right under your feet. So you take a small step downhill, feel the ground again, take another step, and keep going until the ground feels completely flat. You have reached the bottom.

The landscape is the cost function plotted against all possible weight values. The valley is the minimum — the weights that produce the least error. Gradient descent is the algorithm that finds that valley one step at a time.

### The 4 steps — repeated thousands of times

```
1. Forward pass     → use current weights to make predictions on training data
2. Compute loss     → calculate MSE between predictions and actual values
3. Compute gradient → calculate the slope of the loss at current weights (which direction is uphill?)
4. Update weights   → nudge each weight slightly downhill (opposite of gradient). Repeat.
```

### The weight update rule

```
w = w − learning_rate × gradient
```

If the gradient is positive (uphill to the right), subtract — move left. If negative, subtract a negative — move right. Always downhill.

---

## Learning rate — the size of each step

The learning rate is a hyperparameter you set before training. It controls how large each gradient descent step is.

| Learning rate | Symptom | Result |
|---|---|---|
| Too high | Loss goes up or oscillates wildly | Overshoots valley, may diverge |
| Just right | Loss steadily decreases | Finds minimum efficiently |
| Too low | Loss decreases agonisingly slowly | May get stuck, takes forever |

Common starting values: 0.1, 0.01, 0.001 — found by experimentation or a learning rate scheduler.

### Optimisers — smarter ways to run gradient descent

| Optimiser | How it works | When to use |
|---|---|---|
| **SGD** | Updates weights using one mini-batch at a time. Noisier but faster per step. | Simple models, large datasets |
| **SGD + Momentum** | Adds velocity — keeps moving in direction of recent gradients. Like a ball rolling downhill building speed. | Better than plain SGD in most cases |
| **AdaGrad** | Adapts learning rate per parameter — larger updates for infrequent features. | NLP, sparse features |
| **RMSProp** | Like AdaGrad but decays old gradient history — prevents learning rate shrinking to zero. | RNNs, non-stationary problems |
| **Adam** | Combines momentum + RMSProp. Adapts learning rate per parameter. Robust and fast out of the box. | Default for almost everything in deep learning |

**Why Adam dominates:** rarely needs tuning, handles sparse gradients well, converges faster than plain SGD in most scenarios. When in doubt, start with Adam.

---

## Feature scaling — why some algorithms need it

### The problem without scaling

Two features — weeks in top 10 (range: 1–52) and singles released (range: 1–5). During gradient descent, different scales warp the loss landscape — gradient descent zigzags inefficiently instead of going straight to the minimum.

**Algorithms that require scaling:**
- Linear regression, logistic regression — gradient descent sensitive to magnitude
- k-NN — uses distances; large-magnitude features dominate
- SVM — kernel functions depend on distances
- Neural networks — gradient-based, same issue
- PCA — variance-based; large-magnitude features dominate

**Algorithms that do NOT require scaling:**
- Decision trees, random forests, XGBoost — split on values, not magnitudes

### Scaling methods

**Normalisation (Min-Max Scaling)**
```
x_scaled = (x − min) / (max − min)
```
Maps values to [0, 1]. Use when you know the data bounds. Sensitive to outliers.

**Standardisation (Z-score Scaling)**
```
x_scaled = (x − mean) / std
```
Transforms to mean=0, std=1. More robust — the safer default choice.

**Robust Scaling**
```
x_scaled = (x − median) / IQR
```
Uses median and interquartile range. Highly robust to outliers.

**Critical rule:** Always fit the scaler on training data only. Never on the full dataset — that leaks test statistics into training and contaminates your evaluation.

---

## Feature engineering — creating better inputs

Feature engineering creates new, more informative features from raw data. Often more impactful than switching algorithms.

| Technique | Example |
|---|---|
| **Interaction terms** | weeks_in_top10 × singles_released |
| **Polynomial features** | Add x² to capture non-linear relationships |
| **Binning** | Convert continuous age into young / mid / senior |
| **Log transform** | Apply log to skewed features like income or sales |
| **Date decomposition** | Extract day_of_week, month, is_weekend from a timestamp |
| **Domain-specific** | BMI = weight / height² |

---

## Assumptions of linear regression

| Assumption | What it means | Fix when violated |
|---|---|---|
| **Linearity** | Relationship between X and y is a straight line | Polynomial features, non-linear model |
| **Independence** | Data points do not influence each other | Time-series models for temporal data |
| **Homoscedasticity** | Error variance constant across all values of X | Log-transform y, weighted regression |
| **Normality of errors** | Prediction errors follow a normal distribution | Transform features, remove outliers |
| **No multicollinearity** | Input features not highly correlated | Remove correlated features, PCA, Ridge regression |

**Interview tip:** When asked "when would you not use linear regression?" — these assumptions and what breaks them is the answer.

---

## Linear vs Polynomial regression

When the relationship is curved, linear regression will underfit regardless of data volume. Add polynomial features (x², x³) to fit a curve. But too many polynomial features → high variance → overfitting. Regularisation (Topic 10) is the fix.

---

## ⚠️ Common confusions

**Not scaling features:** model trains but gradient descent is slow and unstable.

**Always using MSE:** if data has genuine outliers, MSE distorts training. Use MAE or Huber loss.

**Fitting scaler on full dataset:** always fit on training data only. Fitting on full data leaks test statistics into training.

**Using linear regression on non-linear data:** if residuals show a clear pattern, linearity is violated. Switch to polynomial features or a non-linear model.

---

## Interview-ready summary

> "Linear regression finds the best straight line through data to predict a continuous output. It learns by minimising a cost function — typically MSE — using gradient descent, which iteratively nudges weights downhill on the loss landscape. The learning rate controls step size — too high diverges, too low is slow, and Adam is usually the best default optimiser. Features must be scaled before training because gradient descent is sensitive to feature magnitude, though tree-based models do not need this. Linear regression assumes a linear relationship, independent errors, constant variance, and no multicollinearity — when these break down I would switch to polynomial features, regularisation, or a non-linear model entirely."

---

## Resources
- **Udemy:** Machine Learning A-Z — Kirill Eremenko (Part 2: Regression)
- **YouTube:** StatQuest — "Linear Regression, Clearly Explained"
- **YouTube:** StatQuest — "Gradient Descent, Step by Step"
- **YouTube:** StatQuest — "Mean Squared Error and R-squared"

---

*Part of [ml-dl-for-ai-engineers](https://github.com/PulkitKushwaha/ml-dl-for-ai-engineers) — a learning journal built while targeting Agentic AI Engineer roles at product companies.*
