
import pandas as pd
from patsy import dmatrix
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(6727)
df = pd.read_csv("insurance.csv")
x = df["age"]
y = df["charges"]

spline = dmatrix("bs(age, df=4, include_intercept=False)", {"age": x}, return_type='dataframe')
model = sm.OLS(y, spline).fit()
pred = model.predict(spline)

plt.scatter(x, y, edgecolor='b', alpha=0.5)
plt.plot(x, pred, color='r')
plt.title("Natural Spline (df=4)")
plt.show()
