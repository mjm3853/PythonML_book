import pandas as pd
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

