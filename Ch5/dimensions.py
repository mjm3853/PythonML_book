# pylint: disable=C0103

"""Dimensionality Reduction Examples from Chapter 5"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df_wine = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
print('Wine Raw:\n', df_wine[:3])
print("----------------------------------------")

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)
print('\nX Trained and Standardized:\n\n', X_train_std[:3])
print('\nX Tested and Standardized:\n\n', X_test_std[:3])

cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print('\nEigenvalues:\n\n%s' % eigen_vals)
