# Import necessary libraries and packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate

df = pd.read_csv('iris.csv') # load csv file into a dataframe

# use sns to visualize the data – sepal_length vs sepal_width and plot petal_length vs petal_width with the three species as separator class

sns.regplot(data=df, x='sepal_width', y='sepal_length')

sns.scatterplot(data=df, x='sepal_width', y='sepal_length', hue='class')

# Split data into training and testing and scale all features

x = df.drop('class', axis = 1)
y = df['class']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

scaler = StandardScaler()
scaled_x_train = scaler.fit_transform(x_train)
scaled_x_test = scaler.transform(x_test)

# Build KNN classifier (K=1) and make predictions. Display confusion matrix and classification report

knn_model = KNeighborsClassifier(n_neighbors = 1)
knn_model.fit(scaled_x_train, y_train)

y_pred = knn_model.predict(scaled_x_test)
accuracy_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)
cm = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=knn_model.classes_)
cm.plot()

print(classification_report(y_test, y_pred))
