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

A: Yes, scaling is required for linear regression because gradient descent is sensitive to feature magnitude. Weeks in top 10 would dominate during training due to its larger range. Use standardisation (Z-score) as the safer default — it handles different distributions well and is robust to outliers. Normalisation works but is sensitive to extreme values.

---

**Q: What is the cost function doing and why do we square the errors?**

A: The cost function measures how wrong the model's current predictions are across all training data — it gives a single number that training tries to minimise. We square errors for two reasons: (1) errors can be positive or negative and would cancel out if simply summed — squaring makes all errors positive so they accumulate honestly, (2) squaring penalises large errors disproportionately more, pushing the model to fix its worst mistakes first.

---

**Q: Explain gradient descent in plain English — no math, no jargon.**

A: Gradient descent is an algorithm that finds the lowest point in a hilly landscape while blindfolded. You can only feel which direction is downhill right under your feet. You take a small step downhill, feel again, take another step, and repeat until the ground feels flat — you have reached the bottom. In ML: the landscape is the loss function, the valley is the minimum error, and each step is a weight update.

---

**Q: Your model's loss keeps going up during training instead of down. What is the most likely cause and fix?**

A: High learning rate — the model is taking steps that are too large and overshooting the minimum. Fix: decrease the learning rate (not increase it). Secondary check: ensure features are scaled. Consider switching to Adam optimiser which adapts the learning rate automatically.

---

**Q: When would you use MAE instead of MSE as your loss function?**

A: When your data has genuine outliers you do not want to over-penalise. MSE squares errors so a single extreme outlier contributes massively to the loss and distorts training. MAE treats all errors linearly so outliers have proportional influence. Use Huber loss as a middle ground when you have some outliers but still want MSE behaviour for small errors.

---

**Q: When would you not use linear regression?**

A: When its assumptions are violated: (1) non-linear relationship — switch to polynomial features or tree-based models, (2) multicollinearity — weights become unstable, use Ridge regression or PCA, (3) heteroscedasticity — try log-transforming the target, (4) non-independent data (e.g. time series) — use time-series specific models.

---

## Topic 3 — Logistic Regression

**Q: Is logistic regression a regression or classification algorithm? Why the confusing name?**

A: Classification algorithm. The "regression" refers to the internal mathematical technique (a linear weighted sum), not the output type. Output is always a probability converted to a class — never a continuous number.

---

**Q: What does the sigmoid function do and why does logistic regression need it?**

A: The sigmoid squashes any number (from −∞ to +∞) into a value between 0 and 1, making it interpretable as a probability. Logistic regression needs it because the raw linear weighted sum can produce any number, which cannot be used directly for classification.

---

**Q: Your fraud detection model predicts "not fraud" for everything and gets 99.9% accuracy. What is wrong and how do you fix it?**

A: Class imbalance. When 99.9% of transactions are not fraud, predicting "not fraud" always achieves 99.9% accuracy but catches zero fraudsters — accuracy is meaningless here. Fixes: (1) lower the decision threshold from 0.5 to 0.1–0.2 to flag more potential fraud, (2) switch metrics to Precision, Recall, F1, AUC-ROC, (3) use class weights to penalise minority class misclassification more heavily, (4) oversample minority class using SMOTE or undersample majority class.

---

**Q: Why should you never use MSE as the loss function for logistic regression?**

A: Three reasons: (1) MSE produces a non-convex loss landscape for classification — gradient descent gets stuck in local minima, (2) output is not bounded to a probability, (3) no natural decision threshold on a raw number. Always use binary cross-entropy — it is convex and directly measures probability prediction quality.

---

**Q: When would you use Softmax instead of sigmoid / BCE?**

A: When you have more than two classes. Sigmoid and BCE are for binary problems. Softmax outputs a probability for every class simultaneously, all summing to 1, and uses categorical cross-entropy. Example: classifying a song as rock / pop / jazz / metal — four classes, use Softmax.

---

## Topic 4 — Decision Trees

**Q: Your decision tree has 100% training accuracy and 61% test accuracy. What happened and name two specific fixes.**

A: High variance — overfitting. The tree grew too deep and memorised training data including noise. Specific fixes for decision trees: (1) pre-pruning — set max_depth=5, increase min_samples_leaf to force leaves to represent more data, (2) post-pruning — grow the full tree then tune ccp_alpha to remove branches that do not justify their complexity.

---

**Q: What is Gini impurity measuring in plain English?**

A: The probability of being wrong if you randomly picked a sample from a node and randomly assigned it a label based on the class distribution there. A perfectly pure node (all one class) has Gini = 0 — you are never wrong. A perfectly mixed node has Gini = 0.5 — you are wrong half the time. The tree picks splits that reduce Gini the most.

---

**Q: Do you need to scale features before training a decision tree? What about logistic regression on the same data?**

A: No scaling for decision trees — they ask threshold questions on raw values so magnitude is irrelevant. "BPM > 140" works identically whether BPM is normalised or not. Logistic regression absolutely needs scaling — gradient descent is sensitive to feature magnitude and large-range features will dominate training.

---

**Q: Are Gini and entropy the same thing as gradient descent for trees?**

A: They serve the same purpose — guiding the algorithm toward a better model — but through a completely different mechanism. Gradient descent works on continuous parameters, makes small incremental adjustments using calculus, and runs iteratively thousands of times. Gini and entropy work on discrete decisions (which feature, which threshold), make one greedy choice per node by evaluating every possible split, and never revisit that decision. This is called greedy search — locally optimal at each node, not guaranteed globally optimal. Gradient descent is a compass continuously adjusting your heading. Gini/entropy is a scout who picks the best path from your current spot — one decision, no going back.

---

**Q: A non-technical product manager asks why the model predicted "Metal" for a particular song. You used a decision tree. How do you explain it?**

A: Walk through the actual path the song took through the tree as a natural conversation: "First we asked: does this song have heavy guitar? Yes. Then: is the tempo fast? Yes. Then: are the vocals aggressive or melodic? Aggressive. That combination led us to Metal." This is the core advantage of decision trees — complete transparency. Every prediction has a human-readable explanation.

---

## Topic 5 — Random Forests

**Q: Why does averaging many high-variance trees produce a low-variance model? What is the key condition?**

A: Individual trees overfit differently — each is trained on a different bootstrap sample and considers different features at each split, so their errors are uncorrelated. When you average uncorrelated errors, they cancel each other out leaving the genuine signal. The key condition is that errors must be uncorrelated. If all trees saw identical data and features, averaging would change nothing — the errors would be identical and would not cancel. Bootstrap sampling and feature randomness are specifically designed to ensure uncorrelated errors.

---

**Q: Your random forest has 500 trees and is still overfitting. Increasing n_estimators is not helping. What do you tune?**

A: n_estimators has hit its variance-reduction limit — the overfitting is coming from individual tree complexity, not the number of trees. Tune in this order: (1) decrease max_features — less features per split = more randomness = more uncorrelated trees = lower variance, (2) increase min_samples_leaf — forces leaves to represent more samples, smooths decision boundaries significantly, (3) decrease max_depth if still overfitting. Never increase max_depth to fix overfitting — it increases individual tree complexity and makes things worse.

---

**Q: When would you prefer a single decision tree over a random forest?**

A: When interpretability is critical. In regulated industries (healthcare, finance, legal) a model must often be explainable to regulators, doctors, or patients — a single tree you can draw on a whiteboard wins over a black box of 500 trees regardless of accuracy. Also when data is simple enough that a single tree captures it well and you need fast training and inference.

---

**Q: What is out-of-bag evaluation and when would you use it?**

A: Because each tree uses bootstrap sampling (~63% of data), about 37% of rows are left out of each tree's training. The forest evaluates each row using only the trees that never saw it — giving a valid performance estimate without a separate validation set. The correct split is ~63/37 (not 67/33) — mathematically derived from (1 − 1/n)ⁿ → 1/e ≈ 37%. Use OOB when your dataset is small and you cannot afford to hold out a validation set, or as a quick sanity check during training.

---

*Part of [ml-dl-for-ai-engineers](https://github.com/PulkitKushwaha/ml-dl-for-ai-engineers) — a learning journal built while targeting Agentic AI Engineer roles at product companies.*
