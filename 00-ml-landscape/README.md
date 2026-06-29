# 00 — ML Landscape

> **Week 1 · Topic 0** · The foundation everything else builds on.

---

## The core idea

Traditional programming gives the computer rules and data, and it produces answers:

```
Rules + Data → Answers
```

Machine learning flips this entirely. You give it data and the correct answers, and it figures out the rules on its own:

```
Data + Answers → Rules
```

Those learned rules are what we call a **model**. Feed it enough examples, and it learns the underlying pattern — including patterns no human programmer would have thought to write explicitly.

---

## How every model learns — the universal loop

Every ML model, regardless of type, follows this same loop:

```
Input data
    ↓
Make a prediction
    ↓
Measure how wrong it was  ← this is the "loss"
    ↓
Adjust internal parameters slightly
    ↓
Repeat millions of times
```

The mechanism that does the adjusting is called **gradient descent** — covered in Topic 1.

---

## The 3 types of ML

### The one question that unlocks everything

> **"Do I have labelled data?"**
> - **Yes** → Supervised learning
> - **No** → Unsupervised learning
> - **Reward-based environment** → Reinforcement learning

---

### 1. Supervised learning
**What it is:** You train the model with labelled examples — every input has a correct answer attached. The model learns to map inputs → outputs.

**Key requirement:** Someone has to have already labelled every training example.

**Two subtypes:**

| Subtype | Output | Example question |
|---------|--------|-----------------|
| Classification | A category | "Will this user churn — yes or no?" |
| Regression | A number | "What will this house sell for?" |

**Algorithms:**

| Subtype | Algorithms |
|---------|-----------|
| Classification | Logistic regression, Decision tree, Random forest, SVM, k-NN, XGBoost, Neural networks |
| Regression | Linear regression, Decision tree, Random forest, XGBoost, SVR |

---

### 2. Unsupervised learning
**What it is:** No labels. The model finds hidden structure, patterns, or groupings in raw data entirely on its own.

**Key requirement:** Just raw data — no tagging needed. But results are harder to evaluate since there's no "correct answer" to check against.

**Two subtypes:**

| Subtype | Goal | Example question |
|---------|------|-----------------|
| Clustering | Group similar data points | "Segment our users into personality types" |
| Dimensionality reduction | Compress data, remove noise | "Visualise these 100 features in 2D" |

**Algorithms:**

| Subtype | Algorithms |
|---------|-----------|
| Clustering | k-Means, DBSCAN, Hierarchical clustering |
| Dimensionality reduction | PCA, t-SNE, UMAP |

---

### 3. Reinforcement learning
**What it is:** An agent learns by trial and error — taking actions in an environment and receiving rewards or penalties. No labels, no pre-given structure. Pure exploration guided by a reward signal.

**Key requirement:** A defined environment and reward signal.

**Key concepts:** Agent · Environment · State · Action · Reward · Policy

**Algorithms:** Q-learning · PPO · DQN

**Why this matters for AI engineers:**
RLHF (Reinforcement Learning from Human Feedback) is built directly on RL principles. When human reviewers rate an LLM's responses as good or bad, those ratings are the reward signal — the model adjusts based on that feedback. This is how ChatGPT, Claude, and most production LLMs are aligned. We go deep on this in Week 3.

---

## Algorithm map — full reference

```
ML
├── Supervised
│   ├── Classification
│   │   └── Logistic regression, Decision tree, Random forest,
│   │       SVM, k-NN, XGBoost, Neural networks
│   └── Regression
│       └── Linear regression, Decision tree, Random forest,
│           XGBoost, SVR
├── Unsupervised
│   ├── Clustering
│   │   └── k-Means, DBSCAN, Hierarchical clustering
│   └── Dimensionality reduction
│       └── PCA, t-SNE, UMAP
└── Reinforcement
    └── Q-learning, PPO, DQN
        → leads to RLHF, DPO (Week 3)
```

---

## Interview tips

**When given a real-world problem, ask yourself:**
1. Do I have labelled data? → determines supervised vs unsupervised
2. Am I predicting a category or a number? → classification vs regression
3. Is the system learning from rewards in an environment? → reinforcement learning

**k-NN caveat:** k-NN is valid for classification but breaks at scale — millions of data points means very slow prediction time. For large datasets, prefer Random Forest or XGBoost. Know *why* you'd move away from k-NN, not just that you would.

**Common interview scenario:**
> "We have user listening history and skip/listen labels — predict if a new user skips a song."
→ Supervised, classification. Algorithms: start with Logistic regression as baseline, then Random Forest or XGBoost for production scale.

> "Same data, no labels — find listener personality types."
→ Unsupervised, clustering. Algorithm: k-Means as starting point.

> "LLM trained using human ratings of its responses."
→ Reinforcement learning. Specific technique: RLHF.

---

## Resources
- **Udemy:** Machine Learning A-Z — Kirill Eremenko (Part 1: Data Preprocessing, intro sections)
- **YouTube:** StatQuest — "Machine Learning Fundamentals"
- **YouTube:** 3Blue1Brown — "But what is a neural network?"

---

*Part of [ml-dl-for-ai-engineers](https://github.com/your-username/ml-dl-for-ai-engineers) — a learning journal.*
