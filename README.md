 1) Logistic Regression 
 2) K-Nearest Neighbors (KNN)



# 1) Logistic Regression :

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

----------------------------------------------------------------------------------------------------------------------------------------

## 2) K-Nearest Neighbors (KNN) :

## Project Description


This project implements a K-Nearest Neighbors (KNN) classification algorithm using Python and scikit-learn.  
The goal is to predict whether a user will purchase a product based on their Age and Estimated Salary.

---

## Dataset Information


Dataset Name:
- Social_Network_Ads.csv
 
- Features Used:
  
- Age
  
- Estimated Salary
  
- Target Variable:
  
- Purchased (0 = No, 1 = Yes)

---

## Libraries Used

- NumPy
  
- Pandas
   
- Matplotlib
  
- scikit-learn

---

##  Machine Learning Workflow

1. Imported required libraries  

2. Loaded the dataset  

3. Selected input features and target variable  

4. Split data into training and test sets (80%–20%)  

5. Applied Feature Scaling using `StandardScaler`  

6. Trained KNN Classifier on training data  

7. Predicted results on test data  

8. Evaluated the model using:
    - Confusion Matrix  

    - Accuracy Score  

    - Classification Report  

10. Visualized decision boundaries for:
    - Training set  

    - Test set
   
  ---
  
## Model Configuration

- Algorithm: K-Nearest Neighbors (KNN)

- Number of Neighbors (k): 4

- Distance Metric: Manhattan Distance (`p = 1`)

- Test Size: 20%

- Random State: 0

---

## Model Evaluation
- Confusion Matrix to analyze prediction results  

- Accuracy Score to measure overall performance  

- Classification Report for precision, recall, and F1-score

---

## Visualization
- Decision boundary plots for training and test datasets  

- Scatter plots showing class separation based on Age and Salary

---

## Key Learnings

Importance of feature scaling in distance-based algorithms

Working of KNN classification

Effect of k value on model performance

Visualization of decision boundaries

Understanding bias and variance

--------

## Code Explanation Line by Line

- - - - - - - - - - - - - - - - - - -

## Importing Library

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
-  -  
numpy → numerical computations and array handling

matplotlib.pyplot → data visualization and decision boundary plots

pandas → dataset loading and manipulation

---

## Loading the Dataset

dataset = pd.read_csv("Social_Network_Ads.csv")

- -

Loads the Social Network Ads dataset into a pandas DataFrame for analysis and modeling.

---

## Selecting Features and Target Variable

X = dataset.iloc[:, [2, 3]].values

y = dataset.iloc[:, -1].values

- -

X → Independent variables:

Column 2: Age

Column 3: Estimated Salary

y → Dependent variable:

Purchased (0 = No, 1 = Yes)

These numerical features are selected because KNN is a distance-based algorithm.

---

## Splitting dataset into training and test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

- -

80% data used for training

20% data used for testing

random_state=0 ensures reproducible results

---

## Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

- -

Feature scaling standardizes values to mean = 0 and standard deviation = 1

This step is crucial for KNN because it relies on distance calculations

Scaling prevents features like salary from dominating age

---

## Training the K-Nearest Neighbors Model

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=4, p=1)
classifier.fit(X_train, y_train)

- -

n_neighbors=4 → model considers 4 nearest neighbors

p=1 → Manhattan distance is used

Model learns patterns from the scaled training data

---

## Predicting Test Set Results

y_pred = classifier.predict(X_test)

- -

Predicts whether a user will purchase or not for unseen test data

---

## Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

- -

Displays:

True Positives

True Negatives

False Positives

False Negatives

Helps analyze model errors

---

## Accuracy Score

from sklearn.metrics import accuracy_score

ac = accuracy_score(y_test, y_pred)

- -

Measures overall correctness of the model

Accuracy = Correct Predictions / Total Predictions

## Classification Report

from sklearn.metrics import classification_report

cr = classification_report(y_test, y_pred)

- -

Provides precision, recall, F1-score, and support

Useful for class-wise performance evaluation

## Bias and Variance Analysis

bias = classifier.score(X_train, y_train)

variance = classifier.score(X_test, y_test)

- -

Training accuracy represents bias

Test accuracy represents variance

Similar values indicate good generalization

---

## Visualizing Training Set Decision Boundary

plt.contourf(...)

plt.scatter(...)

- -

Creates a mesh grid to visualize decision regions

Shows how the KNN model classifies training data

Red and green regions represent different classes

---

## Visualizing Test Set Decision Boundary

plt.contourf(...)

plt.scatter(...)

- -

Visualizes how well the model generalizes to unseen data

Confirms whether learned patterns hold on test data

---

## Summary

This project demonstrates a complete KNN classification workflow including data preprocessing, feature scaling, model training, evaluation, bias-variance analysis, and visualization. Each step is designed to ensure accurate predictions and proper model generalization.
