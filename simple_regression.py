x = df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'male', 'Q',
       'S']]
y = df['Survived'] # dependent/response variable is y, other columns become x

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression                 # download packages to build regression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state=101)  # creat training and test data, 30% test data

logmodel = LogisticRegression()
clf = logmodel.fit(x_train, y_train)

logmodel.coef_

predictions = clf.predict(x_test)  #show predictions model is making

clf.predict_proba(x_test)

print(confusion_matrix(y_test, predictions))  # confusion matrix of y_test and regression predictions

cm = ConfusionMatrixDisplay(confusion_matrix(y_test, predictions), display_labels=clf.classes_)

cm.plot()
plt.show()  # plot the confusion matrix for a visual representation

print(accuracy_score(y_test, predictions)) # show the accuracy score of the model

print(classification_report(y_test, predictions))  # classification report displaying recall, f1 score, accuracy and averages 
