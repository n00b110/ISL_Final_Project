import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Set random seed
np.random.seed(6727)

# Load and preprocess the dataset
df = pd.read_csv("mushrooms.csv")
df = df.apply(LabelEncoder().fit_transform)  # convert all categorical data

X = df.drop("class", axis=1)
y = df["class"]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=6727)

# -------------------------------
# Logistic Regression
# -------------------------------
logreg = LogisticRegression(max_iter=2000, random_state=6727)
logreg.fit(X_train, y_train)
y_pred_log = logreg.predict(X_test)

print("Logistic Regression Classification Report:\n")
print(classification_report(y_test, y_pred_log))

# Plot confusion matrix for Logistic Regression
ConfusionMatrixDisplay.from_estimator(logreg, X_test, y_test)
plt.title("Logistic Regression Confusion Matrix")
plt.show()

# -------------------------------
# Linear Discriminant Analysis
# -------------------------------
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
y_pred_lda = lda.predict(X_test)

print("LDA Classification Report:\n")
print(classification_report(y_test, y_pred_lda))

# Plot confusion matrix for LDA
ConfusionMatrixDisplay.from_estimator(lda, X_test, y_test)
plt.title("LDA Confusion Matrix")
plt.show()
