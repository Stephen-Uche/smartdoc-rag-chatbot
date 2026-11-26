# AI Machine-Learning and Deep-Learning

## Fundamental Concepts

### **1. Explain what AI, ML, and DL are, respectively, and how these concepts relate to one another.**

- **AI (Artificial Intelligence):** Technology that enables computers to perform tasks that normally require human intelligence, such as decision-making and problem-solving.
- **ML (Machine Learning):** A branch of AI where computers learn patterns from data to make predictions without being explicitly programmed for every task.
- **DL (Deep Learning):** A branch of ML that uses artificial neural networks with multiple layers to learn complex patterns, such as recognizing images and speech.

**Relationship:** AI > ML > DL
ML is a subset of AI
DL is a subset of M

### **2. Julia splits her data into training and test sets. On the training data, she trains three models: “Linear Regression,” “Lasso Regression,” and a “Random Forest model.” How should she choose which of the three models to continue using when she has not created an explicit validation dataset?**

Julia should use **K-Fold Cross-Validation** on the training data.

1. Divide the training data into K parts (e.g., 5).
2. Train and validate each model 5 times.
3. Compare their average performance using RMSE or MAE.
4. Choose the model with the lowest average error.
5. Finally, evaluate the selected model on the test set.
Important: The test set should only be used for final evaluation, not for choosing the model.

### **3. What is a “regression problem”? Can you give examples of models used and potential areas of application?**

A **regression problem** in Machine Learning involves predicting a **continuous numerical value** based on input data.

**Examples of regression models:**

1. Linear Regression
2. Lasso Regression
3. Random Forest Regression

**Areas of application:**

- **House prices:** Predicting property prices.
- **Weather:** Predicting temperature.
- **Business:** Predicting sales or revenue.
- **Healthcare:** Predicting medical costs.

**In short:** Regression is used to predict numerical values rather than categories.

### **4. Assume we have the following regression model: Y = β₀ + β₁X + ε. What are Y, X, βᵢ, and ε called?**

In the regression model **Y = β₀ + β₁X + ε**:

1. **Y:** Dependent variable (the value we want to predict).
2. **X:** Independent variable (input or predictor).
3. **β₀ (Beta 0):** Intercept (value of Y when X = 0).
4. **β₁ (Beta 1):** Regression coefficient (how much Y changes when X increases by 1).
5. **ε (Epsilon):** Error term (unexplained variation).

**In short:** X predicts Y, β values describe the relationship, and ε represents the error.

### **5. How can you interpret RMSE, and what is it used for?**

**RMSE (Root Mean Squared Error)** measures the average size of prediction errors in a regression model, giving more weight to larger errors.

- **Lower RMSE:** Better predictions.
- **Higher RMSE:** Larger prediction errors.
- **Used for:** Evaluating and comparing regression models.

**Example:** If a house price prediction model has an RMSE of €10,000, its typical prediction error is approximately €10,000.

**In short:** RMSE measures how far predicted values are from actual values.

### **6. What is a “classification problem”? Can you give examples of models used and potential areas of application?**

A **classification problem** in Machine Learning involves predicting a **category or class** rather than a numerical value.

**Examples of classification models:**

1. Logistic Regression
2. Decision Tree
3. Random Forest Classifier
4. Support Vector Machine (SVM)

**Areas of application:**

- **Email:** Spam or not spam.
- **Healthcare:** Disease or no disease.
- **Banking:** Fraudulent or legitimate transactions.
- **Image recognition:** Identifying cats, dogs, or birds.

**In short:** Classification predicts categories, while regression predicts numerical values.

### **7. What is a "Confusion Matrix"?**

A **Confusion Matrix** is a table used to evaluate the performance of a classification model by comparing predicted results with actual results.

It contains four outcomes:

1. **True Positive (TP):** Correctly predicted positive.
2. **True Negative (TN):** Correctly predicted negative.
3. **False Positive (FP):** Incorrectly predicted positive.
4. **False Negative (FN):** Incorrectly predicted negative.

**Example:** A model predicting whether an email is spam or not spam.

**In short:** A Confusion Matrix shows how many predictions are correct or incorrect and helps calculate accuracy, precision, and recall.

### **8.  How is Precision defined? How is Precision interpreted?**

**Precision** measures how many predicted positive cases are actually correct.

**Formula:**

\[ \text{Precision}=\frac{TP}{TP+FP} \]

- **TP (True Positive):** Correct positive predictions.
- **FP (False Positive):** Incorrect positive predictions.

**Interpretation:**

- **High Precision:** Few false positive predictions.
- **Low Precision:** Many false positive predictions.

**Example:** If a model predicts 100 emails as spam and 90 are actually spam, Precision = 90%.

**In short:** Precision tells us how reliable a model's positive predictions are.

### **9. How is Recall defined? How is Recall interpreted?**

**Recall** measures how many actual positive cases are correctly identified by a classification model.

**Formula:**

\[ \text{Recall}=\frac{TP}{TP+FN} \]

- **TP (True Positive):** Correctly predicted positive cases.
- **FN (False Negative):** Positive cases incorrectly predicted as negative.

**Interpretation:**

- **High Recall:** Few actual positive cases are missed.
- **Low Recall:** Many actual positive cases are missed.

**Example:** If 100 patients have a disease and the model correctly identifies 90, Recall = 90%.

**In short:** Recall measures how well a model identifies all actual positive cases.

### **10. Göran claims that data is either “ordinal” or “nominal.” Julia says that this must be interpreted in context. She gives the example that colors such as {red, green, blue} generally have no inherent order (nominal), but if you are wearing a red shirt, you are the best-looking person at the party (ordinal). Who is right?**

**Julia is right.** Data can be nominal or ordinal depending on the context.

- **Nominal data:** Categories without a natural order. Example: Red, Green, Blue.
- **Ordinal data:** Categories with a meaningful order or ranking. Example: Gold, Silver, Bronze.

**Example:** Shirt colors are normally nominal, but if they are ranked by attractiveness, they become ordinal.

**In short:** The same data can be nominal or ordinal depending on how it is interpreted and used.

### **11. What is the difference between parameters and hyperparameters in a machine learning model Give one example of each and explain why they cannot be optimized in the same way.**

**Parameters:** Values learned automatically by the model during training.

\- **Example:** Weights (coefficients) in Linear Regression.

**Hyperparameters:** Settings chosen before training that control how the model learns.

\- **Example:** Number of trees in a Random Forest.

**Why are they optimized differently?**

- **Parameters** are optimized during training by minimizing the loss function.
- **Hyperparameters** are optimized by testing different settings using methods like Grid Search and Cross-Validation.

**In short:** Parameters are learned from data, while hyperparameters control the learning process.

### **12. Explain the difference between overfitting and underfitting in a machine learning model. Also describe how each can be detected and addressed.**

**Overfitting:** The model learns the training data too well, including noise, and performs poorly on new data.

- **Detection:** High training accuracy but low validation/test accuracy.
- **Solution:** Use regularization, simplify the model, or add more training data.

**Underfitting:** The model is too simple to learn important patterns in the data.

- **Detection:** Poor performance on both training and validation/test data.
- **Solution:** Use a more complex model, add useful features, or reduce regularization.

**In short:**

- **Overfitting:** Model learns too much detail and fails to generalize.
- **Underfitting:** Model learns too little and makes poor predictions.

### **13. How do the OvO and OvR algorithms work?**

**OvO (One-vs-One):** Trains a separate classifier for every pair of classes.

- **Example:** For 3 classes (A, B, C), trains A vs B, A vs C, and B vs C.
- **Prediction:** The class receiving the most votes wins.

**OvR (One-vs-Rest):** Trains one classifier for each class against all other classes.

- **Example:** For 3 classes, trains A vs (B, C), B vs (A, C), and C vs (A, B).
- **Prediction:** The class with the highest classifier score wins.

**In short:**

- **OvO:** Compares classes in pairs.
- **OvR:** Compares each class against all others.

Both are used for **multiclass classification**.

### **14. What is the intuition behind regularizing a model?**

**Regularization** is a technique used to prevent overfitting by reducing model complexity.

- **Purpose:** Help the model generalize better to new data.
- **How it works:** Adds a penalty for large model coefficients to the loss function.
- **L1 (Lasso):** Can reduce some coefficients to zero.
- **L2 (Ridge):** Shrinks coefficients toward zero.

**Example:** Regularization prevents a model from learning unnecessary details or noise in training data.

**In short:** Regularization helps balance model complexity and accuracy to improve predictions on unseen data.

### **15. What is the Bias-Variance Trade-off?**

The **Bias-Variance Trade-off** is the balance between a model being too simple and too complex.

- **High Bias:** Model is too simple and misses important patterns (**underfitting**).
- **High Variance:** Model is too complex and learns noise from training data (**overfitting**).
- **Good Balance:** Model learns important patterns and performs well on new data.

**Example:**

- Simple Linear Regression may have high bias.
- A very deep Decision Tree may have high variance.

**In short:** The goal is to balance bias and variance to minimize prediction errors on unseen data.

### **16. Why are more complex models not always better?**

More complex models are not always better because they can **overfit** the training data and perform poorly on new data.

- **Overfitting:** Complex models may learn noise instead of important patterns.
- **Higher computational cost:** They require more time and resources.
- **Harder to interpret:** Their predictions can be difficult to explain.
- **More data needed:** Complex models often require larger datasets.

**Example:** A simple Linear Regression model may perform better than a complex Decision Tree on a small dataset.

**In short:** The best model balances complexity, accuracy, and generalization to new data.

### **17. Can you briefly explain how the fixed but unknown parameters β₀, β₁, …, βₚ are estimated using MSE?**

The unknown parameters **β₀, β₁, …, βₚ** are estimated by minimizing the **Mean Squared Error (MSE)**.

1. **Make predictions:** Use the regression model with initial parameter values.
2. **Calculate errors:** Find the differences between actual and predicted values.
3. **Calculate MSE:** Square the errors and calculate their average.
4. **Minimize MSE:** Adjust the parameters to find the smallest possible MSE.

**Formula:**

\[ MSE=\frac{1}{n}\sum\_{i=1}^{n}(y_i-\hat{y}\_i)^2 \]

**In short:** The best parameter estimates are those that minimize the average squared difference between actual and predicted values.

### **18. In the question above, why is MSE used rather than RMSE? What is the difference?**

**MSE (Mean Squared Error)** and **RMSE (Root Mean Squared Error)** both measure prediction errors.

- **MSE:** Calculates the average squared prediction errors.
- **RMSE:** Calculates the square root of MSE.

**Why use MSE instead of RMSE?**

1. **Easier optimization:** MSE is mathematically simpler to differentiate and minimize.
2. **Same result:** Minimizing MSE also minimizes RMSE.
3. **Interpretation:** RMSE is easier to interpret because it uses the same units as the target variable.

**Formulas:**

\[ MSE=\frac{1}{n}\sum\_{i=1}^{n}(y_i-\hat y_i)^2 \]

\[ RMSE=\sqrt{MSE} \]

**In short:** MSE is preferred for training and optimization, while RMSE is useful for interpreting model performance.

### **19. What do supervised and unsupervised machine learning mean, respectively?**

**Supervised Learning:** A machine learning method where the model learns from **labeled data** (inputs with known correct outputs).

\- **Example:** Predicting house prices or classifying emails as spam.

**Unsupervised Learning:** A machine learning method where the model learns from **unlabeled data** to discover hidden patterns or groups.

\- **Example:** Grouping customers based on shopping behavior (clustering).

**In short:**

- **Supervised:** Learns from labeled data to make predictions.
- **Unsupervised:** Finds patterns in data without predefined labels.

### **20. What are prediction intervals and confidence intervals? Which interval is wider, and why?**

**Confidence Interval (CI):** Estimates a range where the true population mean or average predicted value is likely to fall.

**Prediction Interval (PI):** Estimates a range where a new individual observation is likely to fall.

**Which is wider?**

- **Prediction Interval** is wider because it includes both uncertainty in the estimated mean and variability of individual observations.
- **Confidence Interval** is narrower because it only accounts for uncertainty in estimating the mean.

**Example:**

- 95% CI for average house price: €190,000–€210,000.
- 95% PI for an individual house price: €150,000–€250,000.

**In short:** Confidence intervals estimate the mean, while prediction intervals estimate individual outcomes and are wider.
