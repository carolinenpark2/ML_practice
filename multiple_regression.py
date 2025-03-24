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

# splitting the data into training and test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# train the model

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)

# making predictions

lm.intercept_

lm.coef_

y_pred = lm.predict(X_test)

# make a scatter plot of predictions

plt.scatter(y_test, y_pred)

# dist plot of the predictions

sns.distplot(y_test-y_pred)

# display an array of predicted and actual values

y_pred = lm.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# display metrics

from sklearn import metrics
metrics.mean_absolute_error(y_test, y_pred)
metrics.mean_squared_error(y_test, y_pred)
np.sqrt(metrics.mean_squared_error(y_test, y_pred))
metrics.r2_score(y_test, y_pred)
