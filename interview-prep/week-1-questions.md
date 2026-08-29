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

A: High bias — underfitting. Model too simple to capture the pattern. Small gap confirms it is not a variance issue. First move: increase model complexity — add features, use a more powerful algorithm, or reduce regularisation. More data will not help.

---

**Q: A teammate says "let's just add more training data to fix our model." When does this actually help and when does it not?**

A: More data helps high variance — gives the model more signal to learn real patterns instead of memorising noise. Does not help high bias — if the model is too simple to capture the pattern, more data is irrelevant. Bias is a capability problem, not a data problem.

---

**Q: If you had infinite data and a perfect model, could you reach 0% error?**

A: No. Total error = Bias² + Variance + Irreducible Error. Irreducible error is noise in the data itself that no model can eliminate.

---

## Topic 2 — Linear Regression

**Q: You are predicting song popularity from weeks in top 10 (range 1–52) and number of singles (range 1–5). Should you scale? Which method?**

A: Yes — gradient descent is sensitive to feature magnitude. Weeks in top 10 would dominate. Use standardisation (Z-score) as the safer default — robust to outliers and different distributions.

---

**Q: What is the cost function doing and why do we square the errors?**

A: Cost function measures how wrong current predictions are — gives a single number training tries to minimise. Square errors for two reasons: (1) positive and negative errors would cancel if summed directly, (2) squaring penalises large errors disproportionately, pushing the model to fix worst mistakes first.

---

**Q: Explain gradient descent in plain English.**

A: Finding the lowest point in a hilly landscape while blindfolded. Feel which direction is downhill, take a small step, feel again, repeat until the ground is flat. In ML: landscape = loss function, valley = minimum error, each step = weight update.

---

**Q: Your model's loss keeps going up during training. What is the cause and fix?**

A: High learning rate — overshooting the minimum. Fix: decrease the learning rate (never increase it). Secondary check: ensure features are scaled. Consider Adam optimiser which adapts learning rate automatically.

---

**Q: When would you use MAE instead of MSE?**

A: When data has genuine outliers you do not want to over-penalise. MSE squares errors so one extreme outlier dominates training. MAE treats all errors linearly. Use Huber loss as a middle ground.

---

**Q: When would you not use linear regression?**

A: When assumptions are violated: non-linear relationship, multicollinearity, heteroscedasticity, or non-independent data (time series).

---

## Topic 3 — Logistic Regression

**Q: Is logistic regression a regression or classification algorithm?**

A: Classification. The "regression" refers to the internal mathematical technique, not the output type. Output is always a probability converted to a class.

---

**Q: What does the sigmoid function do and why does logistic regression need it?**

A: Squashes any number into 0–1, making it interpretable as a probability. The raw linear weighted sum can produce any number — sigmoid bounds it for classification use.

---

**Q: Your fraud detection model gets 99.9% accuracy but catches zero fraudsters. What is wrong?**

A: Class imbalance. Accuracy is meaningless when 99.9% of data is one class. Fixes: lower decision threshold, switch to Precision/Recall/F1/AUC-ROC metrics, use class weights, oversample minority class with SMOTE.

---

**Q: Why should you never use MSE for logistic regression?**

A: Three reasons: (1) produces non-convex loss landscape — gradient descent gets stuck, (2) output not bounded to a probability, (3) no natural decision threshold on a raw number. Use binary cross-entropy.

---

**Q: When would you use Softmax instead of sigmoid?**

A: When you have more than two classes. Sigmoid and BCE are for binary problems. Softmax outputs a probability for every class simultaneously, all summing to 1, with categorical cross-entropy as the loss.

---

## Topic 4 — Decision Trees

**Q: Your decision tree has 100% training accuracy and 61% test accuracy. What happened and name two specific fixes.**

A: High variance — overfitting. Tree grew too deep and memorised training data. Fixes: (1) pre-pruning — set max_depth, increase min_samples_leaf, (2) post-pruning — tune ccp_alpha to remove branches that do not justify their complexity.

---

**Q: What is Gini impurity measuring in plain English?**

A: The probability of being wrong if you randomly picked a sample from a node and randomly assigned it a label based on the class distribution there. Pure node = Gini 0. Perfectly mixed = Gini 0.5.

---

**Q: Do you need to scale features before training a decision tree? What about logistic regression?**

A: No scaling for decision trees — threshold questions on raw values, magnitude irrelevant. Logistic regression absolutely needs scaling — gradient descent sensitive to feature magnitude.

---

**Q: Are Gini and entropy the same thing as gradient descent for trees?**

A: Same purpose, completely different mechanism. Gradient descent: continuous parameters, incremental adjustments, iterative, uses calculus. Gini/entropy: discrete decisions, one greedy choice per node, evaluates every possible split, never revisits. This is called greedy search.

---

**Q: A non-technical product manager asks why the model predicted Metal for a particular song.**

A: Walk through the actual path: "First we asked does this song have heavy guitar — yes. Then is the tempo fast — yes. Then are the vocals aggressive — yes. That combination led us to Metal." Complete transparency — the core advantage of decision trees.

---

## Topic 5 — Random Forests

**Q: Why does averaging many high-variance trees produce a low-variance model?**

A: Individual trees overfit differently — trained on different bootstrap samples with different features at each split, so errors are uncorrelated. Averaging uncorrelated errors cancels them out. Key condition: errors must be uncorrelated. Bootstrap sampling and feature randomness ensure this.

---

**Q: Your random forest has 500 trees and is still overfitting. Increasing n_estimators is not helping. What do you tune?**

A: n_estimators has hit its variance limit. Tune in order: (1) decrease max_features, (2) increase min_samples_leaf, (3) decrease max_depth. Never increase max_depth to fix overfitting.

---

**Q: When would you prefer a single decision tree over a random forest?**

A: When interpretability is critical — regulated industries (healthcare, finance, legal) where a model must be explainable to regulators or patients.

---

**Q: What is out-of-bag evaluation and when would you use it?**

A: Bootstrap sampling uses ~63% of data per tree, leaving ~37% out-of-bag. Free validation score without a separate validation set. Correct split is ~63/37 — from (1−1/n)ⁿ → 1/e ≈ 37%. Use when dataset is small.

---

## Topic 6 — Gradient Boosting & XGBoost

**Q: What is each tree in a gradient boosting model actually learning?**

A: The residuals — actual minus predicted from the current combined model. Residuals are the negative gradient of MSE loss, so fitting a tree to residuals is one gradient descent step in function space.

---

**Q: Your XGBoost model is overfitting. Reducing n_estimators is not enough. What else do you tune?**

A: Decrease max_depth (keep shallow, 3–6), decrease learning_rate (never increase it), reduce subsample and colsample_bytree, increase gamma, increase reg_lambda/reg_alpha.

---

**Q: What conditions would make you push for XGBoost over random forest?**

A: Maximum predictive accuracy needed, baseline is underfitting, data has missing values, time to tune properly. Workflow: random forest as quick baseline first, XGBoost to push further.

---

**Q: Why can adding more trees cause overfitting in XGBoost but not in random forests?**

A: Random forests average independent trees — more averaging always helps or is neutral. XGBoost trees fit residuals sequentially — later trees fit increasingly noisy residuals. Always use early stopping.

---

## Topic 7 — k-NN & SVM

**Q: You have 500 features and 10,000 samples. A teammate suggests k-NN. What do you tell them?**

A: k-NN breaks in high dimensions due to the curse of dimensionality — all points become approximately equidistant. Also slow at prediction on large datasets — computes distances to every training point. Apply PCA first, then reconsider.

---

**Q: What is a support vector in plain English?**

A: An actual training data point — the samples sitting closest to the decision boundary on each side. The hyperplane is positioned to maximise distance to these points. Every other training point is irrelevant. Support vectors are points, not lines.

---

**Q: Your SVM is overfitting. Which hyperparameter do you tune and in which direction?**

A: Decrease C — high C forces classification of every training point correctly, creating a complex boundary. Lowering C allows some misclassification, widens the margin, better generalisation. If using RBF, also decrease gamma.

---

**Q: Both k-NN and SVM require feature scaling. Why?**

A: Both are distance-based. Without scaling, large-range features dominate distance calculations. If house size ranges 100–5000 and rooms 1–10, house size contributes 1,000,000 to squared distance vs 25 for rooms. Large-range features drown out small-range ones entirely.

---

## Topic 8 — k-Means Clustering

**Q: You want to cluster 10M users with 50 features. Walk through your approach including choosing k.**

A: Standardise all features, apply PCA to reduce dimensions, run k-Means for k=2 to 15 plotting inertia (elbow method), validate with silhouette score, use k-Means++ initialisation, interpret clusters by examining average feature values per cluster.

---

**Q: Your k-Means produces one cluster with 9.8M users and two with 100K each. What went wrong?**

A: Almost certainly a feature scaling problem — one large-range feature dominates distance calculations pulling most users to one centroid. Standardise all features and re-run. Check whether imbalance is genuine before assuming it is a bug.

---

**Q: A teammate suggests using k-Means for fraud detection. What is the problem?**

A: k-Means assigns every point to a cluster — no concept of "this point does not belong anywhere." Fraud cases are rare, diverse, and do not form dense clusters — they get absorbed into legitimate transaction clusters. Use DBSCAN (labels outliers as noise) or Isolation Forest instead. Switch to supervised classification if labelled examples are available.

---

**Q: What is inertia and why can you not use it alone to choose k?**

A: Inertia is the sum of squared distances from every point to its assigned centroid — measures cluster compactness. Cannot use alone because inertia always decreases as k increases — at k=n_samples, inertia=0. Use the elbow method (find point of diminishing returns) and validate with silhouette score.

---

## Topic 9 — Evaluation Metrics

**Q: Your cancer detection model has 99% recall and 30% precision. Is this good or bad?**

A: Recall=99% means the model catches almost every real cancer case — missing very few. Precision=30% means 70% of its positive flags are false alarms. For cancer detection this is a good trade-off — missing a real case is far more dangerous than a false alarm that leads to further testing. The threshold is deliberately set low.

---

**Q: Why does F1 use harmonic mean instead of arithmetic mean?**

A: Arithmetic mean hides extreme imbalance. Precision=1.0, Recall=0.01 → arithmetic mean=0.505 (looks okay), F1=0.02 (correctly shows the model is nearly useless). Harmonic mean punishes extreme imbalance between the two metrics — forces both to be meaningfully high before the score looks good.

---

**Q: Your model has AUC-ROC of 0.5. What does this tell you?**

A: The model has zero discriminative power — no better than random. Given a random positive and random negative, it ranks them correctly only by chance. Diagnose: check for data leakage, class imbalance in training, feature engineering problems, or wrong model choice.

---

**Q: Content moderation system — 1M normal posts, 1K hate speech. Full evaluation strategy.**

A: Accuracy is useless at 0.1% positive class. Primary metric: Recall — missing hate speech causes real harm. Also track F1 and AUC-ROC. Use Stratified k-Fold (k=5) — preserves 0.1%/99.9% ratio in every fold. Imbalance fixes: class_weight='balanced', SMOTE, lower decision threshold post-training. Set final threshold based on moderation team capacity — how many flagged posts can they review per day.

---

## Topic 10 — Regularisation

**Q: 200 features, most suspected noise. L1 or L2?**

A: L1 (Lasso) — it drives irrelevant feature weights to exactly zero, automatically selecting only the meaningful ones. L2 would keep all 200 features with reduced weights — not what you want when most are noise. L1 gives a sparse, interpretable model. L2 gives a dense, stable one.

---

**Q: Your Ridge regression model is underfitting. What do you do to lambda?**

A: Decrease lambda — it is too high, penalising the model so heavily that weights are driven near zero and the model is too simple to capture real patterns. Reducing lambda weakens the penalty and allows the model to express more complexity.

---

**Q: What is the key difference between L1 and L2?**

A: L1 drives weights to exactly zero (feature selection — sparse model). L2 shrinks all weights toward zero but never reaches zero (all features retained — dense stable model). L1 is unstable under multicollinearity — arbitrarily picks one correlated feature and zeros others. L2 distributes weight across correlated features. Use L1 when most features are noise. Use L2 as the default.

---

**Q: Does increasing lambda increase bias, variance, or both?**

A: Increasing lambda increases bias and decreases variance. Stronger penalty = simpler model = less expressive = more bias, less variance. Too high lambda = underfitting. The goal is the sweet spot — not maximum regularisation. If already underfitting, decrease lambda or change the model entirely.

---

*Part of [ml-dl-for-ai-engineers](https://github.com/PulkitKushwaha/ml-dl-for-ai-engineers) — a learning journal built while targeting Agentic AI Engineer roles at product companies.*
