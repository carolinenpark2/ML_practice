# import necessary packages

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# load the dataset

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# show columns to see which columns will be x and y 

dataset.columns

# assign x and y 

X = dataset[['R&D Spend','Administration','Marketing Spend', 'State']].values
y = dataset['Profit'].values

# encoding the categorical data

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
