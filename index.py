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

# Shape, counts instances and attributes
print(dataset.shape)

# Head, see the first 20 rows
print(dataset.head(20)) 

# Descriptions, get count, mean, min, max and percentages of each attribute
print(dataset.describe()) 

# Class distribution, get the number of instances that belong to each class in this case its 50 instances for each case
print(dataset.groupby('class').size()) 

# Box and whisker plots 
# plots of each individual variable 
# gives us idea of distribution of the input varibales
# in this case most sepal length is between 5 and 6.5 and the middle of this is 5.9
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False) 
pyplot.show() 

# Histograms, get idea of distibution
dataset.hist() 
pyplot.show() 

# Scatter plot matrix, spot structured relationships
# A diagonal grouping suggests high correlation and a predictable relationship
scatter_matrix(dataset) 
pyplot.show() 