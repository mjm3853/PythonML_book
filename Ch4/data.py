import pandas as pd
import numpy as np
from io import StringIO


csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''
df = pd.read_csv(StringIO(csv_data))
print(df)
print("----------------------------------------")

# Imputer
from sklearn.preprocessing import Imputer
imr = Imputer(missing_values="NaN", strategy="mean", axis=0)
imr = imr.fit(df)
imputed_data = imr.transform(df.values)
print(imputed_data)
print("----------------------------------------")

# Categorical Data
dx = pd.DataFrame([
    ['green', 'M', 10.1, 'class1'],
    ['red', 'L', 13.5, 'class2'],
    ['blue', 'XL', 15.3, 'class1']
])
dx.columns = ['color', 'size', 'price', 'classlabel']
print(dx)
print("----------------------------------------")

# Ordinal Features
size_mapping = {
    'XL': 3,
    'L': 2,
    'M': 1
}
dx['size'] = dx['size'].map(size_mapping)
print(dx)
print("----------------------------------------")

'''
# Reverse Ordinal Features
inv_size_mapping = {
    v: k for k, v in size_mapping.items()
}
dx['size'] = dx['size'].map(inv_size_mapping)
print(dx)
print("----------------------------------------")
'''

# Encoding Class Labels
class_mapping = {
    label: idx for idx, label in
    enumerate(np.unique(dx['classlabel']))
}
print(class_mapping)
print("----------------------------------------")

# Transform Class Labels
dx['classlabel'] = dx['classlabel'].map(class_mapping)
print(dx)
print("----------------------------------------")

# Inverse Class Mapping
inv_class_mapping = {
    v: k for k, v in class_mapping.items()
}
dx['classlabel'] = dx['classlabel'].map(inv_class_mapping)
print(dx)
print("----------------------------------------")

# Class Encoder
from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
y = class_le.fit_transform(dx['classlabel'].values)
print(y)
print("----------------------------------------")

# Inverse Class Encoder
class_le.inverse_transform(y)
print(y)
print("----------------------------------------")

# Color Encoder
X = dx[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
print(X)
print("----------------------------------------")

# One Hot Encoding
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features=[0])
print(ohe.fit_transform(X).toarray())

# Get Dummies
print(pd.get_dummies(dx[['price', 'color', 'size']]))