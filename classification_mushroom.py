
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report

np.random.seed(6727)
df = pd.read_csv("mushrooms.csv")
df = df.apply(LabelEncoder().fit_transform)

X = df.drop("class", axis=1)
y = df["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=6727)

logreg = LogisticRegression(max_iter=2000, random_state=6727)
logreg.fit(X_train, y_train)
print("Logistic Classification Report:\n", classification_report(y_test, logreg.predict(X_test)))

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
print("LDA Classification Report:\n", classification_report(y_test, lda.predict(X_test)))
