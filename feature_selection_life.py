
import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

np.random.seed(6727)
df = pd.read_csv("Life Expectancy Data.csv")
df.dropna(inplace=True)

X = df[['Adult Mortality', ' BMI ', 'Schooling', 'Income composition of resources']]
y = df['Life expectancy ']

ridge = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=10)
ridge.fit(X, y)
print("Ridge Coefficients:", ridge.coef_)

lasso = LassoCV(cv=10, random_state=6727)
lasso.fit(X, y)
print("Lasso Coefficients:", lasso.coef_)

X_scaled = StandardScaler().fit_transform(X)
pca = PCA()
pca.fit(X_scaled)

plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Components')
plt.ylabel('Explained Variance')
plt.title('PCA Scree Plot')
plt.grid(True)
plt.show()
