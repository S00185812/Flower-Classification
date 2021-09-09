import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn

# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

#region Summarize Dataset

# Shape, counts instances and attributes
print(dataset.shape)

# Head, see the first 20 rows
print(dataset.head(20)) 

# Descriptions, get count, mean, min, max and percentages of each attribute
print(dataset.describe()) 

# Class distribution, get the number of instances that belong to each class in this case its 50 instances for each case
print(dataset.groupby('class').size()) 

#endregion

#region Univariate Plots

# Univariate plots used to better understand each attribute

# Box and whisker plots 
# plots of each individual variable 
# gives us idea of distribution of the input varibales
# in this case most sepal length is between 5 and 6.5 and the middle of this is 5.9

#dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False) 
#pyplot.show() 

# Histograms, get idea of distibution

#dataset.hist() 
#pyplot.show() 

#endregion

#region Multivariate Plots

# Multivariate plots to better understand the relationships between attributes

# Scatter plot matrix, spot structured relationships
# A diagonal grouping suggests high correlation and a predictable relationship

#scatter_matrix(dataset) 
#pyplot.show() 

#endregion

#region Evaluate Algorthims to use

# Split-out validation dataset 
# We are going to hold back some data that the algorithms will not get to see
# We will use this data to get a second and independent idea of how accurate the model is
# 80% of the data we have will be used for training and the other 20% will be used as the already valid values
array = dataset.values 
X = array[:,0:4] 
y = array[:,4] 
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1) 


# Spot Check Algorithms 
# Here we are testing 6 algorithims to see which will provide the best accuracy
models = [] 
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr'))) 
models.append(('LDA', LinearDiscriminantAnalysis())) 
models.append(('KNN', KNeighborsClassifier())) 
models.append(('CART', DecisionTreeClassifier())) 
models.append(('NB', GaussianNB())) 
models.append(('SVM', SVC(gamma='auto'))) 

# evaluate each model in turn 
# From our results we can see SVM/ Support Vector Machines hast the best accuracy with 98%
results = [] 
names = [] 
print('')
for name, model in models: 
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True) 
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy') 
	results.append(cv_results) 
	names.append(name) 
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std())) 

# Compare Algorithms, So we can get a better idea of the best model
#pyplot.boxplot(results, labels=names) 
#pyplot.title('Algorithm Comparison') 
#pyplot.show() 

#endregion

#region Make Predictions

# Make predictions on validation dataset
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# Evaluate predictions 
print('')
print('Evaluate Predictions')
# We can see the accuracy is 96% 
print('Accuracy score: ' , accuracy_score(Y_validation, predictions)) 
# confusion matrix shows amount of errors made
print('')
print('Confusion matrix')
print(confusion_matrix(Y_validation, predictions)) 
# classification report provides a breakdown of each class by precision, recall, f1-score and support
print('')
print('Classification report')
print(classification_report(Y_validation, predictions)) 

#endregion

