#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 13:05:26 2021

@author: rosaliesherry
"""

import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

wine = np.array(pd.read_csv('wine.csv'))

X = wine[:, :-1] # all but last column for X
Y = wine[:, -1] #Last column is feature, who is the cultivator

#Goal is to classify by cultivator!!!


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .25)

#Gausian Distribution
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, Y_train)
estimatedY = gnb.predict(X_test)

print('Gaussian Train Score: %.2f'
      % gnb.fit(X_train, Y_train).score(X_train, Y_train))
print('Gaussian Test Score: %.2f'
      % gnb.fit(X_train, Y_train).score(X_test, Y_test))


#Decision Tree
from sklearn.tree import DecisionTreeClassifier

#DIRECT STEAL FROM LAB 3

def runDecisionTree(criterion, maxdepth):
    clf = DecisionTreeClassifier(criterion = criterion, max_depth = maxdepth)
    clf = clf.fit(X_train, Y_train)
    predict = clf.predict(X_test)
    #print('For', criterion, 'and max depth of', maxdepth, 'answer is', predict)
    print('Decision Tree', criterion, maxdepth,'Train Score: %.2f'
      % clf.fit(X_train, Y_train).score(X_train, Y_train))
    print('Decision Tree', criterion, maxdepth,'Test Score: %.2f'
      % clf.fit(X_train, Y_train).score(X_test, Y_test))


#For whatever reason I can only get the for-loop to work outside of the function and not in it?
for m in [None, 1, 2]:
    criterion = 'gini'
    runDecisionTree('gini', m)
    if m == None:
        criterion = 'entropy'
        runDecisionTree('entropy', m)


#SVC Model
from sklearn.svm import SVC

svc_model = SVC(kernel = 'sigmoid')
svc_model.fit(X_train, Y_train)
Y_pred = svc_model.predict(X_test)

#print('Coefficients: \n', svc_model.coef_)
# The mean squared error
'''print('Mean squared error: %.2f'
      % mean_squared_error(Y_test, Y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(Y_test, Y_pred))'''
print('SVC Train Score: %.2f'
      % svc_model.fit(X_train, Y_train).score(X_train, Y_train))
print('SVC Test Score: %.2f'
      % svc_model.fit(X_train, Y_train).score(X_test, Y_test))


#Polynominal Regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
for i in range(1, 11, 1):
    polyregr = Pipeline([('poly', PolynomialFeatures(i)), 
                     ('linear', linear_model.LinearRegression(fit_intercept=False))])

# Train the model using the training sets
    polyregr.fit(X_train, Y_train)

# Make predictions using the testing set
    diabetes_y_pred2 = polyregr.predict(X_test)

    print('Polynomial', i,'degree Train Score: %.2f'
          % polyregr.fit(X_train, Y_train).score(X_train, Y_train))
    print('Polynomial', i,'degree Test Score: %.2f'
      % polyregr.fit(X_train, Y_train).score(X_test, Y_test))

#Fuck It Lasso, all these classifiers so far are super bad
for t in range(1, 11, 1):
    lassie = linear_model.Lasso(alpha = t)

# Train the model using the training sets
    lassie.fit(X_train, Y_train)

# Make predictions using the testing set
    pred2 = lassie.predict(X_test)
    print('Lasso Alpha =',t,'Train Score: %.2f'
      % lassie.fit(X_train, Y_train).score(X_train, Y_train))
    print('Lasso Alpha =',t,'Test Score: %.2f'
      % lassie.fit(X_train, Y_train).score(X_test, Y_test))