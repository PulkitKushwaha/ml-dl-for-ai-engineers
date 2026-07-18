# Week 1 — Interview Questions & Answers

> Use this file for rapid revision before interviews. Every question here was tested during the learning sessions — answers include the key nuances that interviewers are actually looking for.

---

## Topic 0 — ML Landscape

**Q: You have a dataset of house prices (input: size, location, age → output: price). What type of ML is this and what subcategory?**

A: Supervised learning, regression. The output is a continuous number (price), not a category. Algorithms to consider: linear regression as a baseline, random forest or XGBoost for production scale.

---

**Q: You want to detect unusual transactions in banking data and you have no examples of what fraud looks like. What type of ML would you use?**

A: Unsupervised learning, specifically anomaly detection via clustering. Without labels you cannot use supervised learning. k-Means or DBSCAN can identify unusual clusters. Note: if you later get labels, switch to supervised classification.

---

**Q: An LLM is trained using human ratings of its responses — thumbs up or thumbs down. What type of ML is this and what is the specific technique called?**

A: Reinforcement learning. The specific technique is RLHF — Reinforcement Learning from Human Feedback. Human ratings are the reward signal. The model adjusts to maximise positive ratings over time.

---

## Topic 1 — Bias-Variance Tradeoff

**Q: Your model scores 97% on training data but only 58% on the test set. What is the problem and name two fixes.**

A: High variance — overfitting. The model memorised training data instead of learning general patterns. Fixes: add regularisation (L1/L2), get more training data, simplify the model, use ensemble methods (bagging), or add dropout if it is a neural network.

---

**Q: Your model scores 52% on training data and 51% on test data. Both are bad. What is the problem and what is your first move?**

A: High bias — underfitting. The model is too simple to capture the pattern. The small gap between training and test error confirms it is not a variance issue. First move: increase model complexity — add more features, use a more powerful algorithm, or reduce regularisation. More training data will not help here.

---

**Q: A teammate says "let's just add more training data to fix our model." When does this actually help and when does it not?**

A: More data helps high variance (overfitting) — it gives the model more signal to learn real patterns instead of memorising noise. It does not help high bias (underfitting) — if the model is too simple to capture the pattern, more data is irrelevant. A straight line will never fit a curve regardless of how many points you add. Bias is a capability problem, not a data problem.

---

**Q: If you had infinite data and a perfect model, could you reach 0% error?**

A: No. Total error = Bias² + Variance + Irreducible Error. Irreducible error is noise inherent in the data itself — two near-identical houses selling for different prices due to factors not captured in your features. No model can eliminate this.

---

## Topic 2 — Linear Regression

**Q: You are predicting song popularity from two features — weeks in top 10 (range 1–52) and number of singles (range 1–5). Should you scale? Which method?**

A: Yes, scaling is required for linear regression because gradient descent is sensitive to feature magnitude. Weeks in top 10 would dominate during training due to its larger range. Use standardisation (Z-score) as the safer default — it handles the different distributions well and is robust to outliers. Normalisation works but is sensitive to extreme values.

---

**Q: What is the cost function doing and why do we square the errors?**

A: The cost function measures how wrong the model's current predictions are across all training data — it gives a single number that training tries to minimise. We square errors for two reasons: (1) errors can be positive or negative and would cancel out if simply summed — squaring makes all errors positive so they accumulate honestly, (2) squaring penalises large errors disproportionately more, pushing the model to fix its worst mistakes first.

---

**Q: Explain gradient descent in plain English — no math, no jargon.**

A: Gradient descent is an algorithm that finds the lowest point in a hilly landscape while blindfolded. You can only feel which direction is downhill right under your feet. You take a small step downhill, feel again, take another step, and repeat until the ground feels flat — you have reached the bottom. In ML: the landscape is the loss function, the valley is the minimum error, and each step is a weight update. Gradient descent is the algorithm that navigates there step by step.

---

**Q: Your model's loss keeps going up during training instead of down. What is the most likely cause and fix?**

A: High learning rate — the model is taking steps that are too large and overshooting the minimum, bouncing around or diverging entirely. Fix: decrease the learning rate. Secondary check: ensure features are scaled — unscaled features can also destabilise gradient descent. Consider switching to Adam optimiser which adapts the learning rate automatically.

---

**Q: When would you use MAE instead of MSE as your loss function?**

A: When your data has genuine outliers that you do not want to over-penalise. MSE squares errors so a single extreme outlier contributes massively to the loss and distorts training. MAE treats all errors linearly so outliers have proportional influence. Example: predicting album sales where one viral album sold 100x more than any other — MAE handles this better. Use Huber loss as a middle ground when you have some outliers but still want MSE behaviour for small errors.

---

**Q: When would you not use linear regression?**

A: When any of its assumptions are violated: (1) the relationship is non-linear — switch to polynomial features or tree-based models, (2) features are highly correlated (multicollinearity) — weights become unstable, use Ridge regression or PCA, (3) error variance is not constant (heteroscedasticity) — try log-transforming the target, (4) data points are not independent (e.g. time series) — use time-series specific models.

---

## Topic 3 — Logistic Regression

**Q: Is logistic regression a regression or classification algorithm? Why the confusing name?**

A: Classification algorithm. The "regression" in the name refers to the internal mathematical technique it uses (a linear weighted sum), not what it outputs. The output is always a probability converted to a class — never a continuous number.

---

**Q: What does the sigmoid function do and why does logistic regression need it?**

A: The sigmoid squashes any number (from −∞ to +∞) into a value between 0 and 1, making it interpretable as a probability. Logistic regression needs it because the raw linear weighted sum can produce any number, which cannot be interpreted as a probability or used directly for classification. The sigmoid bounds the output and gives the model a natural way to express confidence.

---

**Q: Your fraud detection model predicts "not fraud" for everything and gets 99.9% accuracy. What is wrong and how do you fix it?**

A: Class imbalance. When 99.9% of transactions are not fraud, predicting "not fraud" always gives 99.9% accuracy but catches zero fraudsters — accuracy is a meaningless metric here. Fixes: (1) lower the decision threshold from 0.5 to 0.1 or 0.2 so the model flags more transactions as potentially fraudulent — you get more false alarms but catch more real fraud, (2) switch metrics to Precision, Recall, F1, and AUC-ROC — Recall specifically measures "of all actual fraud cases, how many did we catch?", (3) use class weights to penalise misclassifying the minority class more heavily, (4) oversample the minority class using SMOTE or undersample the majority class.

---

**Q: Why should you never use MSE as the loss function for logistic regression?**

A: Three reasons: (1) MSE applied to classification produces a non-convex loss landscape with multiple local minima — gradient descent gets stuck and cannot reliably find the global minimum, (2) the output is not bounded to a probability so MSE has no natural interpretation, (3) there is no natural decision threshold on a raw MSE output. Always use binary cross-entropy for binary classification — it is convex, gradient descent reliably finds the minimum, and it directly measures the quality of probability predictions.

---

**Q: When would you use Softmax instead of sigmoid / BCE?**

A: When you have more than two classes (multiclass classification). Sigmoid and BCE are for binary problems — output is a single probability for one class. Softmax outputs a probability for every class simultaneously, all summing to 1, and uses categorical cross-entropy as the loss function. Example: classifying a song as rock / pop / jazz / metal — four classes, use Softmax. Classifying a song as "hit or not" — two classes, use sigmoid + BCE.

---

*Part of [ml-dl-for-ai-engineers](https://github.com/PulkitKushwaha/ml-dl-for-ai-engineers) — a learning journal built while targeting Agentic AI Engineer roles at product companies.*
