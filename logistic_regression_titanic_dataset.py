import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

df = pd.read_csv('titanic_mod.csv') # read csv into a pandas dataframe


# visualize any missing data
sns.heatmap(df.isnull(), yticklabels=False, cbar = False, cmap=('viridis'))

# countplot of the dependent value we will look at and try to predict
sns.set_style('whitegrid')
sns.countplot(x='Survived', data=df, palette='RdBu_r')

# explore sex differences in survival 
sns.set_style('whitegrid')
sns.countplot(x='Survived', hue='Sex', data=df, palette='RdBu_r')

# visualize economic differences in survival 
sns.set_style('whitegrid')
sns.countplot(x='Survived', hue = 'Pclass', data=df, palette='rainbow')

# plot the distribution of age 
sns.displot(df['Age'].dropna(), kde = True, color='darkred', bins=30)

# histogram of ticket prices
sns.histplot(data=df, x='Fare', bins=40)

# fill in missing ages with the mean age for the dataset, visualize missing values
df['Age'].fillna(df['Age'].mean(),inplace=True)
sns.heatmap(df.isnull(), yticklabels=False,cbar=False,cmap='viridis')

# drop the column labeled cabin, too many missing values
df.drop('Cabin',axis = 1, inplace=True)

# convert categorical features to dummy variables 
pd.get_dummies(df['Sex'])
newsex=pd.get_dummies(df['Sex'], drop_first=True)
embark=pd.get_dummies(df['Embarked'], drop_first=True)
df.drop(['PassengerId','Name','Sex','Ticket','Embarked'], axis=1, inplace=True)
df=pd.concat([df, newsex, embark], axis=1)

# Build a logistic regression model
x = df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'male', 'Q','S']]
y = df['Survived']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state=101)
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()
clf = logmodel.fit(x_train, y_train)
logmodel.coef_

sns.boxplot(x='Survived',y='Fare',data=df)

# make predictions
predictions = clf.predict(x_test)
clf.predict_proba(x_test)

from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
print(confusion_matrix(y_test, predictions))
cm = ConfusionMatrixDisplay(confusion_matrix(y_test, predictions), display_labels=clf.classes_)
cm.plot()
plt.show()

# accuracy and classification report 
print(accuracy_score(y_test, predictions))

from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))
