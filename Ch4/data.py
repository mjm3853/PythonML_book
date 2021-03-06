# pylint: disable=C0103

"""Data Processing Examples from Chapter 4"""
from io import StringIO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sequential_backward_selection import SBS


print("----------------------------------------")
print('Loaded Data')
print("----------------------------------------")
csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''
df = pd.read_csv(StringIO(csv_data))
print(df)

print("----------------------------------------")
print('Imputer')
print("----------------------------------------")
imr = Imputer(missing_values="NaN", strategy="mean", axis=0)
imr = imr.fit(df)
imputed_data = imr.transform(df.values)
print(imputed_data)

print("----------------------------------------")
print('Categorical Data')
print("----------------------------------------")
dx = pd.DataFrame([
    ['green', 'M', 10.1, 'class1'],
    ['red', 'L', 13.5, 'class2'],
    ['blue', 'XL', 15.3, 'class1']
])
dx.columns = ['color', 'size', 'price', 'classlabel']
print(dx)

print("----------------------------------------")
print('Ordinal Features')
print("----------------------------------------")
size_mapping = {
    'XL': 3,
    'L': 2,
    'M': 1
}
dx['size'] = dx['size'].map(size_mapping)
print(dx)

'''
print("----------------------------------------")
print('Reverse Ordinal Features')
print("----------------------------------------")
inv_size_mapping = {
    v: k for k, v in size_mapping.items()
}
dx['size'] = dx['size'].map(inv_size_mapping)
print(dx)
'''

print("----------------------------------------")
print('Encoding Class Labels')
print("----------------------------------------")
class_mapping = {
    label: idx for idx, label in
    enumerate(np.unique(dx['classlabel']))
}
print(class_mapping)

print("----------------------------------------")
print('Transform Class Labels')
print("----------------------------------------")
dx['classlabel'] = dx['classlabel'].map(class_mapping)
print(dx)

print("----------------------------------------")
print('Inverse Class Mapping')
print("----------------------------------------")
inv_class_mapping = {
    v: k for k, v in class_mapping.items()
}
dx['classlabel'] = dx['classlabel'].map(inv_class_mapping)
print(dx)

print("----------------------------------------")
print('Class Encoder')
print("----------------------------------------")
class_le = LabelEncoder()
y = class_le.fit_transform(dx['classlabel'].values)
print(y)

print("----------------------------------------")
print('Inverse Class Encoder')
print("----------------------------------------")
class_le.inverse_transform(y)
print(y)

print("----------------------------------------")
print('Color Encoder')
print("----------------------------------------")
X = dx[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
print(X)

print("----------------------------------------")
print('One Hot Encoding')
print("----------------------------------------")
ohe = OneHotEncoder(categorical_features=[0])
print(ohe.fit_transform(X).toarray())

print("----------------------------------------")
print('Get Dummies')
print("----------------------------------------")
print(pd.get_dummies(dx[['price', 'color', 'size']]))

print("----------------------------------------")
print('Load Wine Dataset')
print("----------------------------------------")
df_wine = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
df_wine.columns = ['Class label',
                   'Alcohol',
                   'Malic acid',
                   'Ash',
                   'Alcalinity of ash',
                   'Magnesium',
                   'Total phenols',
                   'Flavanoids',
                   'Nonflavanoid phenols',
                   'Proanthocyanins',
                   'Color intensity',
                   'Hue',
                   'OD280/OD315 of diluted wines',
                   'Proline']
print('Class labels', np.unique(df_wine['Class label']))
print("----------------------------------------")
print(df_wine.head())

print("----------------------------------------")
print('Partitioned Wine Dataset')
print("----------------------------------------")
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)
print("Training Data---------------------------")
print(X_train[:3])
print("Test Data-------------------------------")
print(X_test[:3])

print("----------------------------------------")
print('MinMaxScaler')
print("----------------------------------------")
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)
print("Training Data Normalized----------------")
print(X_train_norm[:3])
print("Test Data Normalized--------------------")
print(X_test_norm[:3])

print("----------------------------------------")
print('StandardScaler')
print("----------------------------------------")
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)
print("Training Data Standardized--------------")
print(X_train_std[:3])
print("Test Data Standardized------------------")
print(X_test_std[:3])

print("----------------------------------------")
print('L1 Regularization')
print("----------------------------------------")
lr = LogisticRegression(penalty='l1', C=0.1)
lr.fit(X_train_std, y_train)
print('Training accuracy:', lr.score(X_train_std, y_train))
print('Test accuracy:', lr.score(X_test_std, y_test))
print('Intercept:', lr.intercept_)
print('Coefficients:', lr.coef_)

print("----------------------------------------")
print('Plot Regularization Path')
print("----------------------------------------")
fig = plt.figure()
ax = plt.subplot(111)
colors = ['blue',
          'green',
          'red',
          'cyan',
          'magenta',
          'yellow',
          'black',
          'pink',
          'lightgreen',
          'lightblue',
          'gray',
          'indigo',
          'orange']
weights, params = [], []
for c in np.arange(-4, 6, dtype=float):
    lr = LogisticRegression(penalty='l1', C=10**c, random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)
weights = np.array(weights)
for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:, column],
             label=df_wine.columns[column + 1], color=color)
plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim([10**(-5), 10**5])
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center', bbox_to_anchor=(
    1.32, 1.03), ncol=1, fancybox=True)
plt.show()
print('>>> Regularization Path Plotted')

print("----------------------------------------")
print('Run SBS')
print("----------------------------------------")
knn = KNeighborsClassifier(n_neighbors=2)
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)
k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.1])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.show()
print('>>> Sequential Backward Selection Run')
print("----------------------------------------")
k5 = list(sbs.subsets_[8])
print('Important features:', df_wine.columns[1:][k5])

print("----------------------------------------")
print('Feature Importance with Random Forest')
print("----------------------------------------")
feat_labels = df_wine.columns[1:]
forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" %
          (f + 1, 30, feat_labels[indices[f]], importances[[f]]))
plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]), importances[indices],
        color='lightblue', align='center')
plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()
