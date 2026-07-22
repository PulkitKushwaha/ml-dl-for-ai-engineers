# 05 — Random Forests

> **Week 1 · Topic 5** · What happens when you stop trusting one tree and ask the whole forest instead.

---

## The core idea

A random forest trains hundreds of decision trees on random subsets of data and features, then combines their predictions — eliminating the high variance problem of a single tree while keeping all its strengths.

It is an ensemble method that uses a technique called **bagging** (Bootstrap Aggregating). The key insight: individual trees overfit differently from each other. When you average their predictions, the errors cancel out and the signal remains.

---

## The 100 scouts analogy

You are trying to decide if a new band is worth signing. You ask one scout — your best one — and they say yes. Do you sign them? Maybe. But what if that scout saw them on a bad night? What if they have a personal bias toward a certain sound?

Now imagine you send 100 different scouts to 100 different shows. Each scout sees a slightly different set of performances, talks to different fans, pays attention to different things. Some scouts will be wrong. But when you collect all 100 opinions and take the majority vote — "73 out of 100 scouts say sign them" — that collective judgment is far more reliable than any single expert.

That is a random forest:
- Each scout = one decision tree
- Different shows and fans each scout sees = bootstrap sampling
- Each scout paying attention to different things = feature randomness
- The majority vote = aggregation

The magic: individual trees overfit (each scout has quirks and biases) but their errors are *different* from each other. When averaged, errors cancel — signal remains.

---

## How a random forest is built — the 4 steps

```
Step 1 — Bootstrap sampling
For each tree, randomly sample n rows from training data WITH replacement.
Each tree gets a slightly different dataset — some rows appear multiple times,
others not at all. ~63% of unique rows are used per tree, ~37% left out.

Step 2 — Feature randomness
At each node split, only consider a random subset of features.
Default: √n_features for classification, n_features/3 for regression.
Forces trees to be different — they cannot all pick the same dominant feature.

Step 3 — Grow many trees
Repeat steps 1 and 2 for each tree (typically 100–500 trees).
Each tree is grown deep — low bias, high variance individually.
That high variance is fine because of step 4.

Step 4 — Aggregate predictions
Classification → majority vote across all trees
Regression → average of all tree outputs
Individual errors cancel. Signal remains. Variance drops dramatically.
```

---

## Why this works — the variance reduction explained

A single decision tree overfits because it memorises specific patterns in its training data — its errors are **systematic**.

When you train 100 trees on different random subsets of data and features, each tree overfits **differently**. Their errors are **uncorrelated**.

When you average uncorrelated errors, they cancel each other out — like noise-cancelling headphones. What is left is the signal — the genuine pattern in the data.

**The key condition that makes bagging work:**
The errors across trees must be uncorrelated. If all trees saw the same data and the same features at every split, they would make the same mistakes and averaging would change nothing. Bootstrap sampling and feature randomness are specifically engineered to ensure uncorrelated errors. Without this condition, bagging does not work.

**Mathematically:** if each tree has variance σ² and trees are uncorrelated, the variance of the average of n trees is σ²/n. More trees = less variance. This is why adding more trees always helps — up to a point of diminishing returns.

---

## Out-of-bag (OOB) evaluation — free validation

Because each tree uses bootstrap sampling (~63% of data), about 37% of rows are left out of each tree's training. These are called **out-of-bag samples**.

The exact number: when sampling n rows with replacement from n rows, the probability of any single row being left out is (1 − 1/n)ⁿ which converges to 1/e ≈ **37%** as n grows large. So OOB is approximately a 63/37 split, not 67/33.

Each training row is out-of-bag for roughly 1/3 of all trees. The forest evaluates each row using only the trees that never saw it during training — giving a valid, unbiased performance estimate without needing a separate validation set.

```python
model = RandomForestClassifier(n_estimators=100, oob_score=True)
model.fit(X_train, y_train)
print(model.oob_score_)   # free validation accuracy
```

**When to use OOB evaluation:**
- When your dataset is small and you cannot afford to hold out a validation set
- As a quick sanity check during training without extra computation
- When you want a rough estimate of generalisation without full cross-validation

---

## Hyperparameters — deep dive with examples

Understanding what each hyperparameter actually does — and which direction to move it when something is wrong — is what separates a confident interview answer from a vague one.

### n_estimators — number of trees

```python
RandomForestClassifier(n_estimators=100)   # default in scikit-learn
```

Controls how many decision trees are in the forest.

- More trees = lower variance = better generalisation — up to a point
- After ~200–300 trees, returns diminish significantly — adding more trees costs training time but barely improves accuracy
- More trees never causes overfitting — unlike deeper trees or more features
- **If your model is overfitting:** increasing n_estimators will not help. It reduces variance but only up to its limit. The overfitting is coming from somewhere else.
- **Start with:** 100. If variance is still high after tuning other params, try 300–500.

```
n_estimators too low  → high variance, unstable predictions
n_estimators too high → wasted compute, no meaningful improvement
```

### max_features — features considered per split

```python
RandomForestClassifier(max_features="sqrt")    # default for classification
RandomForestClassifier(max_features="log2")    # alternative
RandomForestClassifier(max_features=0.5)       # use 50% of features
RandomForestClassifier(max_features=5)         # use exactly 5 features
```

Controls how many features are randomly considered at each node split. This is the parameter that creates uncorrelated trees.

- Lower max_features = more randomness = trees are more different from each other = lower variance but higher bias
- Higher max_features = trees are more similar = higher variance (closer to a single tree) but lower bias
- **If your model is overfitting:** reduce max_features — forces more randomness, trees become less correlated
- **If your model is underfitting:** increase max_features — gives each tree more information per split

```
max_features too low  → trees too random, high bias, underfitting
max_features too high → trees too similar, high variance, overfitting
```

### max_depth — maximum depth of each tree

```python
RandomForestClassifier(max_depth=None)   # default — grows full trees
RandomForestClassifier(max_depth=10)     # limit to 10 levels
RandomForestClassifier(max_depth=5)      # more aggressive limit
```

Controls how deep each individual tree grows.

- Default is None — trees grow until all leaves are pure or contain min_samples_leaf samples
- In random forests, deep trees are less dangerous than in single trees because bagging handles the variance — but depth can still be limited for speed or when variance remains high
- **CRITICAL: to fix overfitting, DECREASE max_depth — do not increase it**
- Increasing max_depth makes each tree more complex = higher individual variance = worse overfitting even after averaging
- **Start with:** None (default). If still overfitting after reducing max_features and increasing min_samples_leaf, try max_depth=10, then 5.

```
max_depth too high → trees memorise noise, high variance even after averaging
max_depth too low  → trees underfit, high bias
```

### min_samples_leaf — minimum samples per leaf node

```python
RandomForestClassifier(min_samples_leaf=1)    # default
RandomForestClassifier(min_samples_leaf=5)    # moderate smoothing
RandomForestClassifier(min_samples_leaf=20)   # aggressive smoothing
```

Every leaf node must contain at least this many training samples. If a proposed split would create a leaf with fewer samples, that split is rejected.

- Increasing this forces the model to generalise more — leaves must represent a meaningful group of samples, not just 1–2 outliers
- Very effective at smoothing decision boundaries and reducing variance
- **If your model is overfitting:** increase min_samples_leaf — this is often the most effective single fix for random forest overfitting
- **If your model is underfitting:** decrease min_samples_leaf — allow finer splits

```
min_samples_leaf=1  → leaves can represent single samples → high variance
min_samples_leaf=20 → leaves must represent 20+ samples → much smoother
```

### bootstrap — whether to use bootstrap sampling

```python
RandomForestClassifier(bootstrap=True)    # default — uses bagging
RandomForestClassifier(bootstrap=False)   # each tree sees all data
```

- Default True — each tree is trained on a bootstrap sample (~63% of data)
- Setting False removes the bagging benefit entirely — trees become more correlated, variance increases
- Setting False also disables OOB evaluation
- **Almost never change this from True** unless you have a specific reason

### Hyperparameter tuning — the right order

When overfitting, tune in this order:
```
1. Decrease max_features      → most impactful for variance
2. Increase min_samples_leaf  → smooths decision boundaries
3. Decrease max_depth         → if still overfitting
4. Increase n_estimators      → after other params are tuned

Never increase max_depth to fix overfitting — it makes things worse.
```

---

## Random Forest vs Single Decision Tree

| | Single Decision Tree | Random Forest |
|---|---|---|
| **Variance** | High — overfits easily | Low — averaging reduces variance |
| **Bias** | Low (deep tree) | Slightly higher — averaging smooths |
| **Interpretability** | High — can draw and explain | Low — black box of hundreds of trees |
| **Speed** | Fast | Slower — trains many trees |
| **Accuracy** | Lower on complex data | Much higher — generalises well |
| **Feature scaling** | Not needed | Not needed |
| **Use when** | Interpretability is critical | Accuracy matters more than explainability |

**When to prefer a single decision tree:**
- A doctor, judge, or product manager needs to understand and trust every individual decision
- Regulations require model explainability (finance, healthcare, legal)
- Data is simple enough that a single tree captures it well
- You need to draw the model on a whiteboard and walk a non-technical audience through it

---

## Random Forest vs XGBoost — preview of Topic 6

| | Random Forest (Bagging) | XGBoost (Boosting) |
|---|---|---|
| **Tree building** | Parallel — trees built independently | Sequential — each tree corrects the last |
| **Individual trees** | Deep, high variance | Shallow, high bias (weak learners) |
| **Solves** | High variance | High bias |
| **Speed** | Faster — parallelisable | Slower — sequential by design |
| **Tuning complexity** | Simpler | More hyperparameters |
| **Best for** | When a single tree overfits badly | When models are too simple to capture patterns |

---

## Key properties of random forests

| Property | Detail |
|---|---|
| **No feature scaling needed** | Inherits from decision trees — threshold splits, not magnitude-sensitive |
| **Handles missing values** | Can handle missing data better than most algorithms |
| **Built-in feature importance** | Aggregated across all trees — more reliable than single tree importance |
| **Robust to outliers** | Threshold-based splits are not distorted by extreme values |
| **Parallelisable** | Trees are independent — can be trained simultaneously on multiple cores |
| **Black box** | Cannot explain individual predictions as easily as a single tree |

---

## ⚠️ Common confusions

**Confusion: the key condition for bagging to work is just "more trees."**
More trees help, but the real requirement is that tree errors must be **uncorrelated**. If all trees saw identical data and features, averaging 1000 trees would produce the same result as one tree — the errors would be identical and nothing would cancel. Bootstrap sampling and feature randomness are the mechanisms that create uncorrelated errors. This is the actual reason random forests work, not simply the number of trees.

**Mistake: increasing max_depth to fix overfitting.**
This is the opposite of what you want. Deeper trees have higher individual variance — increasing max_depth makes each tree memorise more noise. Even after averaging, the overfitting worsens. To fix overfitting: decrease max_depth, decrease max_features, or increase min_samples_leaf. Always ask yourself: does this change increase or decrease model complexity? Overfitting = too complex. Fix = reduce complexity.

**Mistake: thinking more n_estimators always fixes overfitting.**
Increasing n_estimators reduces variance — but only up to its natural limit. If your forest is still overfitting at 500 trees, adding more trees will not help. The overfitting is coming from individual tree complexity (max_depth, max_features, min_samples_leaf) not from having too few trees.

**Mistake: OOB split is 67/33.**
The correct numbers are approximately **63/37**. The mathematical reason: sampling n rows with replacement from n rows, the probability of any row being excluded is (1 − 1/n)ⁿ → 1/e ≈ 37% as n grows. This is a commonly tested detail.

**Mistake: random forests always beat single decision trees.**
Not when interpretability matters. In regulated industries (healthcare, finance, legal), a model that cannot be explained to a regulator or patient is not deployable regardless of accuracy. A single decision tree you can draw on a whiteboard often wins over a black-box forest.

---

## Interview-ready summary

> "A random forest trains hundreds of decision trees on bootstrap samples of data and random subsets of features, then aggregates their predictions via majority vote or averaging. The key insight is that individual trees overfit differently — their errors are uncorrelated — so averaging cancels errors and leaves the signal. This is bagging. The critical condition: errors must be uncorrelated, which bootstrap sampling and feature randomness ensure. Key hyperparameters: n_estimators (more trees = lower variance up to diminishing returns), max_features (lower = more randomness = lower variance), min_samples_leaf (higher = smoother predictions), max_depth (lower = less complex trees). To fix overfitting: decrease max_features and increase min_samples_leaf first — never increase max_depth. Random forests are slower and less interpretable than single trees but dramatically more accurate on complex data. OOB evaluation gives a free validation score using the ~37% of rows each tree never saw during training."

---

## Resources
- **Udemy:** Machine Learning A-Z — Kirill Eremenko (Part 3: Classification — Random Forest section)
- **YouTube:** StatQuest — "Random Forests, Clearly Explained"
- **YouTube:** StatQuest — "Bagging"

---

*Part of [ml-dl-for-ai-engineers](https://github.com/PulkitKushwaha/ml-dl-for-ai-engineers) — a learning journal built while targeting Agentic AI Engineer roles at product companies.*
