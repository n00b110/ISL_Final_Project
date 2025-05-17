
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

np.random.seed(6727)
df = pd.read_csv("mushrooms.csv")
df = df.apply(LabelEncoder().fit_transform)
X = df.drop("class", axis=1)
y = df["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=6727)

tree = DecisionTreeClassifier(random_state=6727)
tree.fit(X_train, y_train)
print("Decision Tree:\n", classification_report(y_test, tree.predict(X_test)))

rf = RandomForestClassifier(n_estimators=100, random_state=6727)
rf.fit(X_train, y_train)
print("Random Forest:\n", classification_report(y_test, rf.predict(X_test)))

gb = GradientBoostingClassifier(random_state=6727)
gb.fit(X_train, y_train)
print("Gradient Boosting:\n", classification_report(y_test, gb.predict(X_test)))

svm = SVC(kernel='rbf', random_state=6727)
svm.fit(X_train, y_train)
print("SVM RBF:\n", classification_report(y_test, svm.predict(X_test)))
