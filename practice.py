#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

# Handling missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])

# Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder = LabelEncoder()
X[:, 0] = label_encoder.fit_transform(X[:, 0])
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder="passthrough")
X = ct.fit_transform(X)
label_encoder_y = LabelEncoder()
Y = label_encoder_y.fit_transform(Y)

# Splitting into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(X, test_size=0.2, random_state=0)
Y_train, Y_test = train_test_split(Y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
ss_X = StandardScaler()
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.fit_transform(X_test)
print(X_train)