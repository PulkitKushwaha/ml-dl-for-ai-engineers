# 09 — Evaluation Metrics

> **Week 1 · Topic 9** · Accuracy looks great. Your model is useless. Here is why — and how to actually measure whether a model works.

---

## The core idea

Accuracy tells you how often your model is right — but the metrics that actually matter tell you which kind of right and which kind of wrong your model produces, and whether those trade-offs are acceptable for your specific problem.

Evaluation metrics are not the same as training loss. Training loss (MSE, cross-entropy) drives backpropagation automatically during training. Evaluation metrics (F1, AUC-ROC, precision, recall) measure real-world model quality after training — they guide your decisions, not the model's weight updates.

---

## Evaluation metrics vs training loss — a critical distinction

```
Training loss      → used DURING training
                     drives backpropagation and weight updates automatically
                     must be mathematically differentiable
                     examples: MSE, binary cross-entropy, categorical cross-entropy

Evaluation metrics → used AFTER training
                     measure real-world quality for humans and business decisions
                     do not need to be differentiable — just meaningful
                     examples: F1, AUC-ROC, precision, recall, accuracy
                     the model does NOT optimise these automatically
```

The complete workflow:

```
Train model       → minimise loss (automatic, backprop)
                          ↓
Evaluate model    → measure F1, AUC-ROC (manual, post-training)
                          ↓
Make decisions    → tune hyperparameters, adjust threshold,
                    change architecture (human-driven)
                          ↓
Retrain           → repeat
```

Evaluation metrics refine the model — but through your decisions, not automatically. Training loss is what the model uses to learn. Evaluation metrics are what you use to judge.

---

## The music critic analogy

You are a music critic reviewing albums. Your job: flag every great album for the magazine to feature.

**Approach 1 — Flag everything:** never miss a great album (perfect recall) but also flag every mediocre one (terrible precision). The magazine loses credibility featuring garbage.

**Approach 2 — Only flag when certain:** everything you flag is genuinely great (perfect precision) but you miss half the great albums (terrible recall). The magazine misses Radiohead's best album.

**The real job:** find the balance — flag most great albums (high recall) while keeping your flagging credible (high precision). That balance is F1. The threshold at which you flag is your decision boundary. AUC-ROC tells you how well your model performs across every possible threshold.

---

## Why accuracy fails on imbalanced data

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

If 99.9% of transactions are legitimate, predicting "not fraud" always gives 99.9% accuracy — with TP=0. The model catches zero criminals. Accuracy completely hides this failure because TN dominates the calculation.

---

## The confusion matrix — foundation of everything

```
                    Predicted: Positive    Predicted: Negative
Actual: Positive    TP (True Positive)     FN (False Negative)
                    Predicted fraud.       Predicted not fraud.
                    Actually fraud.        Actually fraud.
                    Correct — caught.      Missed — walks free.

Actual: Negative    FP (False Positive)    TN (True Negative)
                    Predicted fraud.       Predicted not fraud.
                    Actually not fraud.    Actually not fraud.
                    False alarm.           Correct — legitimate.
```

---

## Precision — when you flag something, how often are you right?

```
Precision = TP / (TP + FP)
```

Of all the things the model flagged as positive — how many were actually positive?

**Music critic:** of all the albums you gave 5 stars to, what fraction were genuinely great?

**When to prioritise precision:** when false positives are costly.
- Spam filter — you would rather miss some spam than block legitimate emails
- Recommendation system — fewer but better recommendations beats flooding users with irrelevant ones
- Content removal — wrongly removing legitimate content causes user frustration and appeals

---

## Recall — of all real positives, how many did you catch?

```
Recall = TP / (TP + FN)
```

Of all the actual positives in the dataset — how many did the model find?

**Music critic:** of all the genuinely great albums released this year, what fraction did you catch and review?

**When to prioritise recall:** when false negatives are costly.
- Cancer detection — missing a real case is far worse than a false alarm that triggers further testing
- Fraud detection — missing a real fraudster causes financial loss that outweighs investigating a false alarm
- Security systems — missing a real threat is more dangerous than a false alert

---

## The precision-recall trade-off

Precision and recall move in opposite directions as you change the decision threshold:

| Threshold | Effect | Use when |
|---|---|---|
| Low (e.g. 0.1) | Flag almost everything → high recall, low precision | False negatives are costly (fraud, cancer) |
| Default (0.5) | Balanced trade-off | Classes balanced, equal error cost |
| High (e.g. 0.9) | Only flag when certain → high precision, low recall | False positives are costly (spam, content removal) |

---

## F1 Score — balancing precision and recall

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

The harmonic mean of precision and recall. Ranges from 0 (worst) to 1 (best).

### Why harmonic mean and not arithmetic mean?

Arithmetic mean can be misleadingly high when one metric is very high and the other very low.

Example: Precision=1.0, Recall=0.01
- Arithmetic mean = (1.0 + 0.01) / 2 = 0.505 → looks okay
- Harmonic mean (F1) = 2 × (1.0 × 0.01) / (1.0 + 0.01) = 0.02 → correctly shows the model is terrible

The harmonic mean punishes extreme imbalances between the two — it forces both to be meaningfully high before the score looks good.

### F-beta score — weighted trade-off

When you want to weight recall more than precision (or vice versa):

```
F_β = (1 + β²) × (Precision × Recall) / (β² × Precision + Recall)
```

- β=1 → standard F1 (equal weight)
- β=2 → recall weighted twice as much (use for cancer detection, fraud)
- β=0.5 → precision weighted twice as much (use for spam filters)

---

## AUC-ROC — threshold-independent evaluation

The ROC curve plots True Positive Rate (Recall) vs False Positive Rate at every possible decision threshold from 0 to 1. AUC (Area Under the Curve) summarises this into one number.

```
AUC = 1.0  → perfect model
AUC = 0.5  → random classifier — no better than flipping a coin
AUC = 0.0  → perfectly wrong — flip all predictions
```

### Intuitive interpretation

AUC is the probability that the model ranks a random positive example higher than a random negative example. AUC=0.85 means: given one fraud transaction and one legitimate transaction, the model assigns a higher fraud probability to the real fraud 85% of the time.

### Why AUC-ROC over accuracy?

- Evaluates model quality across all thresholds — not just the default 0.5
- Unaffected by class imbalance
- Tells you how well the model discriminates between classes regardless of threshold
- Useful for comparing models independently of the operating threshold you will use in production

### What AUC=0.5 means in practice

The model has zero discriminative power — it ranks positives and negatives randomly. Diagnose by checking: data leakage, severe class imbalance in training, feature engineering problems, or fundamentally wrong model choice for the task.

---

## Class imbalance — the full solution

| Fix | How it works | Use when |
|---|---|---|
| **Change the metric** | Switch from accuracy to Precision, Recall, F1, AUC-ROC | Always — first step |
| **Lower decision threshold** | Reduce from 0.5 to 0.2/0.1 — catches more positives at cost of more false alarms | False negatives are costly |
| **Class weights** | Penalise minority class misclassification more heavily. `class_weight='balanced'` in scikit-learn | Quick fix, always worth trying |
| **SMOTE** | Synthetic Minority Oversampling — generates synthetic minority examples by interpolating between existing ones | When you need more minority data without simple duplication |
| **Undersample majority** | Randomly remove majority examples to rebalance | When majority data is abundant and redundant |
| **Stratified splits** | Preserve class ratio in every train/val/test split | Always — use StratifiedKFold |

### Full production workflow for imbalanced data

```
1. Recognise imbalance — check class distribution first
2. Drop accuracy as your primary metric immediately
3. Apply class_weight='balanced' during training
4. Use StratifiedKFold cross-validation
5. Evaluate with F1 and AUC-ROC
6. Plot precision-recall curve — find threshold where recall
   is acceptably high and precision is still manageable
7. Apply SMOTE if class weights alone are insufficient
8. Set operating threshold based on cost of each error type
```

### Real example — content moderation (1M normal posts, 1K hate speech)

Imbalance: 0.1% positive class. Accuracy = useless.

Primary metric: **Recall** — missing hate speech causes real harm. Track Precision and F1 alongside to ensure moderation team is not overwhelmed with false alarms.

Cross-validation: **Stratified k-Fold (k=5)** — preserves 0.1%/99.9% ratio in every fold.

Imbalance fixes: class_weight='balanced' + SMOTE + lower threshold post-training.

Threshold decision: based on moderation team capacity — how many flagged posts can they review per day?

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, recall_score

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    print(f"Fold {fold}: Recall={recall_score(y_val, y_pred):.3f}, "
          f"F1={f1_score(y_val, y_pred):.3f}, "
          f"AUC={roc_auc_score(y_val, y_prob):.3f}")
```

---

## Cross-validation — evaluating reliably

| Method | How it works | Use when |
|---|---|---|
| **k-Fold CV** | Split into k folds. Train on k-1, test on remaining. Repeat k times. Average results. | Standard evaluation — k=5 or 10 |
| **Stratified k-Fold** | Same as k-fold but each fold preserves original class distribution | Always for classification — especially imbalanced |
| **Leave-One-Out (LOO)** | k = n_samples. Each sample is its own test set. Very accurate but extremely slow. | Only on tiny datasets |

**Rule:** always use Stratified k-Fold for classification. Never use regular k-Fold on imbalanced data — folds may contain almost no minority class examples.

---

## Regression metrics — quick reference

| Metric | Formula | Use when |
|---|---|---|
| **MSE** | mean of (ŷ − y)² | Default. Penalises large errors heavily. Sensitive to outliers. |
| **RMSE** | √MSE | Same as MSE but in same units as target — easier to interpret. |
| **MAE** | mean of |ŷ − y| | Outliers present — treats all errors linearly. |
| **R²** | 1 − SS_res/SS_tot | Need normalised, interpretable score for stakeholders. |

---

## Metric selection — the cheat sheet

| Situation | Metric |
|---|---|
| Balanced classes, equal error cost | Accuracy |
| Imbalanced classes | F1 or AUC-ROC |
| False positives costly (spam, content removal) | Precision |
| False negatives costly (cancer, fraud, security) | Recall |
| Need threshold-independent evaluation | AUC-ROC |
| Regression with outliers | MAE or Huber |
| Regression, large errors unacceptable | RMSE |
| Need interpretable regression score | R² |

---

## ⚠️ Common confusions

**Confusion: evaluation metrics and training loss are the same thing.**
They serve different purposes. Training loss (MSE, cross-entropy) is differentiable and drives backpropagation automatically during training. Evaluation metrics (F1, AUC-ROC) are not differentiable — the model cannot optimise them directly. They guide your hyperparameter and architecture decisions between training runs, not the model's weight updates during training.

**Confusion: high accuracy means the model is good.**
Only when classes are balanced and errors are equally costly. On imbalanced data, a model predicting the majority class always achieves high accuracy while being completely useless. Always check class distribution before choosing accuracy as your metric.

**Confusion: F1 is just the average of precision and recall.**
F1 uses the harmonic mean, not arithmetic mean. Arithmetic mean hides extreme imbalance: Precision=1.0, Recall=0.01 gives arithmetic mean=0.505 (looks okay) but F1=0.02 (correctly shows the model is nearly useless). The harmonic mean forces both metrics to be meaningfully high.

**Confusion: AUC-ROC of 0.5 means 50% accuracy.**
AUC=0.5 means the model has zero discriminative power — it cannot distinguish between classes at all. Given a random positive and random negative, it ranks them correctly only by chance. It says nothing about accuracy directly.

**Confusion: use regular k-Fold for imbalanced classification.**
Regular k-Fold may put almost all minority class examples in one fold — other folds have essentially no positive examples and evaluation is meaningless. Always use Stratified k-Fold for classification, especially on imbalanced data.

---

## Interview-ready summary

> "Evaluation metrics are distinct from training loss — training loss drives backpropagation automatically, while evaluation metrics guide human decisions after training. Accuracy fails on imbalanced data because it is dominated by the majority class. The confusion matrix decomposes predictions into TP, FP, FN, TN. Precision measures how trustworthy your positive flags are; recall measures how many real positives you caught. Prioritise precision when false positives are costly (spam filters), recall when false negatives are costly (cancer, fraud). F1 uses the harmonic mean to force both to be high — arithmetic mean hides extreme imbalance. AUC-ROC evaluates model quality across all thresholds and is the probability the model ranks a random positive above a random negative. For imbalanced data: drop accuracy, use class weights and SMOTE, lower the decision threshold, and always use Stratified k-Fold cross-validation."

---

## Resources
- **Udemy:** Machine Learning A-Z — Kirill Eremenko (model evaluation sections throughout)
- **YouTube:** StatQuest — "ROC and AUC, Clearly Explained"
- **YouTube:** StatQuest — "Precision and Recall"
- **YouTube:** StatQuest — "The Confusion Matrix"

---

*Part of [ml-dl-for-ai-engineers](https://github.com/PulkitKushwaha/ml-dl-for-ai-engineers) — a learning journal built while targeting Agentic AI Engineer roles at product companies.*
