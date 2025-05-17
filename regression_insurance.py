
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

np.random.seed(6727)
df = pd.read_csv('insurance.csv')
df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})
X = df[['age', 'bmi', 'smoker']]
y = df['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=6727)

# Linear regression
X_train_const = sm.add_constant(X_train)
model = sm.OLS(y_train, X_train_const).fit()
print("Linear Model Summary:\n", model.summary())

# Polynomial regression on 'bmi'
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(df[['bmi']])
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_poly, y, test_size=0.3, random_state=6727)
model_poly = sm.OLS(y_train_p, sm.add_constant(X_train_p)).fit()
y_pred_poly = model_poly.predict(sm.add_constant(X_test_p))
print("Polynomial R2:", r2_score(y_test_p, y_pred_poly))
