# Regression-github

# Logistic Regression :

##  Repository Structure

logistic_regression.py
── log1-Social_Network_Ads_dataset.csv
── log2-Future prediction1-data.csv

---

## Project Description

Implemented a Logistic Regression classification model using Python and scikit-learn.

The model predicts user purchase behavior based on Age and Estimated Salary.

Evaluated the model using Accuracy, Confusion Matrix, Classification Report, ROC Curve, and AUC score.

Also performed future prediction on new unseen data and saved results to a CSV file.

---

## Dataset Used

* Social_Network_Ads_dataset.csv

 Features: Age, Estimated Salary

 Target: Purchased (0 / 1)

* Future prediction1_data.csv

 Used for predicting future outcomes

 ---

## Data Preprocessing

Selected relevant feature columns using iloc

Split dataset into Training (80%) and Testing (20%)

Applied Feature Scaling using StandardScaler to normalize data

---

 ## Model Building

Used Logistic Regression from sklearn.linear_model

Trained the model on scaled training data

Generated predictions on test data

---

## Model Evaluation

Confusion Matrix to analyze prediction results

Accuracy Score to measure overall performance

Classification Report for precision, recall, and F1-score

Calculated:

Bias → Training accuracy

Variance → Testing accuracy

---

## ROC Curve & AUC Score

Computed predicted probabilities using predict_proba

Plotted ROC Curve to visualize TPR vs FPR

Calculated AUC Score (~0.98) indicating excellent model performance

Compared model against a random classifier baseline

---

## Future Prediction

Applied trained Logistic Regression model on a new dataset

Generated predictions for future data

Saved predicted results to final1.csv

---

## Visualization

Plotted ROC Curve

Visualized decision boundary for training data using meshgrid

Used different colors to represent different classes

---


## Tools

Python

NumPy

Pandas

Matplotlib

Scikit-learn

---

## How to Run

pip install numpy pandas matplotlib scikit-learn

python logistic_regression.py

---

## Key Learnings

Understanding Logistic Regression for binary classification

Importance of feature scaling

Model evaluation using ROC-AUC

Difference between bias and variance

Visual interpretation of decision boundaries

