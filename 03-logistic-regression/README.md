# 03 — Logistic Regression

> **Week 1 · Topic 3** · The name says regression. The job is classification. Here is why.

---

## The core idea

Logistic regression is a classification algorithm — not a regression algorithm. The word "regression" refers to the internal mathematical technique it uses, not what it predicts. It predicts categories, not numbers. This is one of the most commonly tested name confusions in ML interviews.

It takes the linear regression equation, squashes its output into a probability between 0 and 1 using the sigmoid function, and uses that probability to classify inputs into categories.

---

## The A&R scout analogy

Imagine you are a record label A&R scout. Your job: listen to a demo and decide — sign this band or pass? Yes or no. Binary decision.

You have a mental scorecard — streams on the lead single, social followers, sold-out shows. You weigh them up and arrive at a gut feeling: "I am about 80% sure this band is worth signing." That is a probability. Then you apply your rule: if I am more than 50% sure, I sign them. That 50% threshold is your decision boundary.

That is exactly what logistic regression does:
1. Takes input features and computes a weighted sum — just like linear regression
2. Squashes that sum into a probability between 0 and 1 using the sigmoid function
3. Applies a decision boundary (usually 0.5) — above it: class 1, below it: class 0

---

## The equation

```
z = w₁x₁ + w₂x₂ + ... + w₀        (linear combination — same as linear regression)
ŷ = σ(z) = 1 / (1 + e⁻ᶻ)          (sigmoid applied to squash z into 0–1)
```

- **z** — the raw weighted sum (can be any number)
- **σ(z)** — the probability output (always between 0 and 1)
- **ŷ ≥ 0.5** → predict class 1 · **ŷ < 0.5** → predict class 0

---

## The sigmoid function

```
σ(z) = 1 / (1 + e⁻ᶻ)
```

The sigmoid takes any number z (from −∞ to +∞) and squashes it into a value between 0 and 1.

- Very large positive z → output close to 1 (highly confident: class 1)
- Very large negative z → output close to 0 (highly confident: class 0)
- z = 0 → output = 0.5 (completely uncertain — sits on the decision boundary)

The S-shaped curve is why the output is always a valid probability — it can never go below 0 or above 1.

---

## The decision boundary

The default threshold is 0.5 — but this is a hyperparameter you can and should adjust based on the problem:

| Threshold | Effect | Use when |
|---|---|---|
| Lower (e.g. 0.2–0.3) | Predicts class 1 more often → higher recall, lower precision | Missing a positive is costly (e.g. cancer detection, fraud) |
| Default (0.5) | Balanced trade-off | Classes are balanced and both errors equally costly |
| Higher (e.g. 0.7–0.8) | Predicts class 1 less often → higher precision, lower recall | False positives are costly (e.g. spam filter blocking real emails) |

---

## The loss function — Binary Cross-Entropy

### Why not MSE for classification?

MSE assumes errors are symmetric and continuous — it works for predicting numbers. For classification, applying MSE produces a non-convex loss landscape with many local minima where gradient descent gets stuck. Binary cross-entropy is convex, so gradient descent reliably finds the global minimum.

Three reasons to never use MSE for classification:
1. Non-convex loss landscape — gradient descent gets stuck in local minima
2. Output is not bounded to a probability
3. No natural decision threshold on a raw number

### Binary Cross-Entropy (BCE)

```
Loss = −[y·log(ŷ) + (1−y)·log(1−ŷ)]
```

- When y=1 (actual positive): loss = −log(ŷ) → confident correct prediction (ŷ≈1) gives loss≈0, confident wrong prediction (ŷ≈0) gives massive loss
- When y=0 (actual negative): loss = −log(1−ŷ) → same logic, penalises confident wrong predictions heavily

The model is punished hardest when it is confidently wrong.

---

## Linear vs Logistic — key differences

| | Linear Regression | Logistic Regression |
|---|---|---|
| **Task** | Regression — predict a number | Classification — predict a category |
| **Output** | Any number (−∞ to +∞) | Probability (0 to 1) |
| **Activation** | None (identity) | Sigmoid function |
| **Loss function** | MSE / RMSE / MAE | Binary cross-entropy |
| **Decision** | The number itself is the answer | Threshold applied to probability |
| **Feature scaling** | Required | Required |

---

## Multiclass classification — beyond binary

When you have more than 2 classes (e.g. classify a song as rock / pop / jazz / metal):

**One-vs-Rest (OvR)**
Train one binary classifier per class — "is this rock vs everything else?", "is this pop vs everything else?" etc. Pick the class with the highest probability. Simple and works well in practice.

**Softmax (multinomial logistic regression)**
Extends sigmoid to multiple classes — outputs a probability for every class simultaneously, all summing to 1. The class with the highest probability wins.

```
softmax(zᵢ) = e^zᵢ / Σe^zⱼ
```

Softmax is what neural networks use in their output layer for multiclass problems. Use BCE for binary output, use Softmax + categorical cross-entropy for multiclass.

---

## ⚠️ Class imbalance — a critical real-world issue

Class imbalance occurs when one class heavily outnumbers the other. Example: fraud detection where only 0.1% of transactions are fraudulent.

**The trap:** a model that predicts "not fraud" for everything achieves 99.9% accuracy — but catches zero fraudsters. Accuracy becomes a useless metric.

**How to detect it:** check if training accuracy is suspiciously high but the model never predicts the minority class.

**Fixes:**

| Fix | How it works |
|---|---|
| **Lower the decision threshold** | Instead of 0.5, use 0.2 or 0.1 — model flags more positives, catches more real cases at the cost of more false alarms |
| **Change the metric** | Use Precision, Recall, F1, and AUC-ROC instead of accuracy → covered in depth in Topic 9 |
| **Oversample minority class** | Duplicate or synthetically generate minority class examples (SMOTE) |
| **Undersample majority class** | Remove some majority class examples to rebalance |
| **Class weights** | Tell the algorithm to penalise misclassifying the minority class more heavily |

**The key insight:** in fraud detection, missing actual fraud (false negative) is far more costly than a false alarm (false positive). Lower the threshold, accept more false alarms, catch more real fraud. The right threshold depends entirely on the cost of each type of error in your specific problem.

→ Precision, Recall, F1, and AUC-ROC covered in depth in `09-eval-metrics`

---

## Assumptions of logistic regression

| Assumption | What it means | Fix when violated |
|---|---|---|
| **Linearity of log-odds** | Linear relationship between features and log of the odds | Add polynomial features, use tree-based model |
| **Independence** | Observations do not influence each other | Use models designed for correlated data |
| **No multicollinearity** | Features not highly correlated with each other | Remove correlated features, PCA, Ridge/Lasso |
| **Large sample size** | Needs sufficient data per class to estimate probabilities reliably | Use simpler model or get more data |

---

## ⚠️ Common confusions

**Mistake: calling logistic regression a regression algorithm.**
It is a classification algorithm. The "regression" refers to the internal technique, not the output type. Output is always a probability converted to a class.

**Mistake: using MSE as the loss function.**
Always use binary cross-entropy for binary classification. MSE produces a non-convex landscape that gradient descent cannot reliably optimise.

**Mistake: never adjusting the decision threshold.**
The default 0.5 threshold is rarely optimal for real-world problems. Always think about what type of error is more costly in your specific problem and adjust the threshold accordingly.

**Mistake: using accuracy on imbalanced datasets.**
Accuracy is misleading when classes are imbalanced. Always check class distribution first and use precision, recall, and F1 as your primary metrics.

---

## Interview-ready summary

> "Logistic regression is a classification algorithm that works by computing a linear weighted sum of input features, passing it through the sigmoid function to get a probability between 0 and 1, then applying a decision threshold to assign a class. It uses binary cross-entropy as its loss function — not MSE — because BCE is convex and produces reliable gradient descent. The decision threshold is a hyperparameter — lower it when missing positives is costly (fraud, cancer), raise it when false positives are costly (spam filters). For multiclass problems, extend using One-vs-Rest or Softmax. Always check for class imbalance — accuracy is meaningless on imbalanced datasets."

---

## Resources
- **Udemy:** Machine Learning A-Z — Kirill Eremenko (Part 3: Classification — Logistic Regression section)
- **YouTube:** StatQuest — "Logistic Regression, Clearly Explained"
- **YouTube:** StatQuest — "Sigmoid Function"

---

*Part of [ml-dl-for-ai-engineers](https://github.com/PulkitKushwaha/ml-dl-for-ai-engineers) — a learning journal built while targeting Agentic AI Engineer roles at product companies.*
