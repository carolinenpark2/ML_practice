#Convert categorical values to dummy values
pd.get_dummies(df['<column of interest>'])

#one-hot encoding 

#import necessary libraries/packages

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('data_set.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

dataset.isnull().sum()  # check for any missing values

# take care of the missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# encoding categorical data

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X =  np.array(ct.fit_transform(X))

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# split the data into test and training sets

from sklearn.model_selection import train_test_split
X_Train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# feature scaling - normalization and standardization

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_Train[:, 3:] = sc.fit_transform(X_Train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
