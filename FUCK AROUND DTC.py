#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 14:51:15 2021

@author: rosaliesherry
"""

import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt


wine = np.array(pd.read_csv('wine.csv'))

X = wine[:, :-1] # all but last column for X
Y = wine[:, -1] #Last column is feature, who is the cultivator

for r in [.33, .25, .10, .05]:
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = r)
    
    if r == .33:
        #alright this is where you are currently, be better tomorrow
        test_score_third = {}
        train_score_third = {}
        for j in range(100):
        
    #Gausian Distribution
            gnb = GaussianNB()
            gnb.fit(X_train, Y_train)
            estimatedY = gnb.predict(X_test)
            NB_three = []
            NB_three.append(gnb.fit(X_train, Y_train).score(X_train, Y_train))
            NB_test_three = []
            NB_test_three.append(gnb.fit(X_train, Y_train).score(X_test, Y_test))
            
    
    #Decision Tree
            clf = DecisionTreeClassifier()
            clf = clf.fit(X_train, Y_train)
            predict = clf.predict(X_test)
            DT_tree = []
            DT_tree.append(clf.fit(X_train, Y_train).score(X_train, Y_train))
            DT_test_tree = []
            DT_test_tree.append(clf.fit(X_train, Y_train).score(X_test, Y_test))
            
    #SVC Model
            svc_model = SVC(kernel = 'sigmoid')
            svc_model.fit(X_train, Y_train)
            Y_pred = svc_model.predict(X_test)
            vect = []
            vect_test = []
            vect.append(svc_model.fit(X_train, Y_train).score(X_train, Y_train))
            vect_test.append(svc_model.fit(X_train, Y_train).score(X_test, Y_test))
    
    #Polynominal Regression
            polyregr = Pipeline([('poly', PolynomialFeatures(2)), 
                         ('linear', linear_model.LinearRegression(fit_intercept=False))])
            polyregr.fit(X_train, Y_train)
            diabetes_y_pred2 = polyregr.predict(X_test)
            plr = []
            plr_test = []
            plr.append(polyregr.fit(X_train, Y_train).score(X_train, Y_train))
            plr_test.append(polyregr.fit(X_train, Y_train).score(X_test, Y_test))
    
    #Fuck It Lasso, all these classifiers so far are super bad
            lassie = linear_model.Lasso(alpha = 2)
            lassie.fit(X_train, Y_train)
            phoenix = []
            phoenix_test = []
            phoenix.append(lassie.fit(X_train, Y_train).score(X_train, Y_train))
            phoenix_test.append(lassie.fit(X_train, Y_train).score(X_test, Y_test))
        
        #WOOHOO AVERAGES
        train_score_third['Naive Bayes'] = (sum(NB_three) / len(NB_three))
        test_score_third['Naive Bayes'] = (sum(NB_test_three) / len(NB_test_three))
        train_score_third['Decision Tree'] = (sum(DT_tree) / len(DT_tree))
        test_score_third['Decision Tree'] = (sum(DT_test_tree) / len(DT_test_tree))
        train_score_third['SVC'] = (sum(vect) / len(vect))
        test_score_third['SVC'] = (sum(vect_test) / len(vect_test))
        train_score_third['Poly'] = (sum(plr) / len(plr))
        test_score_third['Poly'] = (sum(plr_test) / len(plr_test))
        train_score_third['Lasso'] = (sum(phoenix) / len(phoenix))
        test_score_third['Lasso'] = (sum(phoenix_test) / len(phoenix_test))
        
        
    elif r == .25:
        test_score_quart = {}
        train_score_quart = {}
        
        for t in range(100):
        
    #Gausian Distribution
            gnb = GaussianNB()
            gnb.fit(X_train, Y_train)
            estimatedY = gnb.predict(X_test)
            NB_three = []
            NB_three.append(gnb.fit(X_train, Y_train).score(X_train, Y_train))
            NB_test_three = []
            NB_test_three.append(gnb.fit(X_train, Y_train).score(X_test, Y_test))
            
    
    #Decision Tree
            clf = DecisionTreeClassifier()
            clf = clf.fit(X_train, Y_train)
            predict = clf.predict(X_test)
            DT_tree = []
            DT_tree.append(clf.fit(X_train, Y_train).score(X_train, Y_train))
            DT_test_tree = []
            DT_test_tree.append(clf.fit(X_train, Y_train).score(X_test, Y_test))
            
    #SVC Model
            svc_model = SVC(kernel = 'sigmoid')
            svc_model.fit(X_train, Y_train)
            Y_pred = svc_model.predict(X_test)
            vect = []
            vect_test = []
            vect.append(svc_model.fit(X_train, Y_train).score(X_train, Y_train))
            vect_test.append(svc_model.fit(X_train, Y_train).score(X_test, Y_test))
    
    #Polynominal Regression
            polyregr = Pipeline([('poly', PolynomialFeatures(2)), 
                         ('linear', linear_model.LinearRegression(fit_intercept=False))])
            polyregr.fit(X_train, Y_train)
            diabetes_y_pred2 = polyregr.predict(X_test)
            plr = []
            plr_test = []
            plr.append(polyregr.fit(X_train, Y_train).score(X_train, Y_train))
            plr_test.append(polyregr.fit(X_train, Y_train).score(X_test, Y_test))
    
    #Fuck It Lasso, all these classifiers so far are super bad
            lassie = linear_model.Lasso(alpha = 2)
            lassie.fit(X_train, Y_train)
            phoenix = []
            phoenix_test = []
            phoenix.append(lassie.fit(X_train, Y_train).score(X_train, Y_train))
            phoenix_test.append(lassie.fit(X_train, Y_train).score(X_test, Y_test))
        
        #WOOHOO AVERAGES
        train_score_quart['Naive Bayes'] = (sum(NB_three) / len(NB_three))
        test_score_quart['Naive Bayes'] = (sum(NB_test_three) / len(NB_test_three))
        train_score_quart['Decision Tree'] = (sum(DT_tree) / len(DT_tree))
        test_score_quart['Decision Tree'] = (sum(DT_test_tree) / len(DT_test_tree))
        train_score_quart['SVC'] = (sum(vect) / len(vect))
        test_score_quart['SVC'] = (sum(vect_test) / len(vect_test))
        train_score_quart['Poly'] = (sum(plr) / len(plr))
        test_score_quart['Poly'] = (sum(plr_test) / len(plr_test))
        train_score_quart['Lasso'] = (sum(phoenix) / len(phoenix))
        test_score_quart['Lasso'] = (sum(phoenix_test) / len(phoenix_test))
    
    elif r == .10:
        test_score_ten = {}
        train_score_ten = {}
    
        for s in range(100):
        
    #Gausian Distribution
            gnb = GaussianNB()
            gnb.fit(X_train, Y_train)
            estimatedY = gnb.predict(X_test)
            NB_three = []
            NB_three.append(gnb.fit(X_train, Y_train).score(X_train, Y_train))
            NB_test_three = []
            NB_test_three.append(gnb.fit(X_train, Y_train).score(X_test, Y_test))
            
    
    #Decision Tree
            clf = DecisionTreeClassifier()
            clf = clf.fit(X_train, Y_train)
            predict = clf.predict(X_test)
            DT_tree = []
            DT_tree.append(clf.fit(X_train, Y_train).score(X_train, Y_train))
            DT_test_tree = []
            DT_test_tree.append(clf.fit(X_train, Y_train).score(X_test, Y_test))
            
    #SVC Model
            svc_model = SVC(kernel = 'sigmoid')
            svc_model.fit(X_train, Y_train)
            Y_pred = svc_model.predict(X_test)
            vect = []
            vect_test = []
            vect.append(svc_model.fit(X_train, Y_train).score(X_train, Y_train))
            vect_test.append(svc_model.fit(X_train, Y_train).score(X_test, Y_test))
    
    #Polynominal Regression
            polyregr = Pipeline([('poly', PolynomialFeatures(2)), 
                         ('linear', linear_model.LinearRegression(fit_intercept=False))])
            polyregr.fit(X_train, Y_train)
            diabetes_y_pred2 = polyregr.predict(X_test)
            plr = []
            plr_test = []
            plr.append(polyregr.fit(X_train, Y_train).score(X_train, Y_train))
            plr_test.append(polyregr.fit(X_train, Y_train).score(X_test, Y_test))
    
    #Fuck It Lasso, all these classifiers so far are super bad
            lassie = linear_model.Lasso(alpha = 2)
            lassie.fit(X_train, Y_train)
            phoenix = []
            phoenix_test = []
            phoenix.append(lassie.fit(X_train, Y_train).score(X_train, Y_train))
            phoenix_test.append(lassie.fit(X_train, Y_train).score(X_test, Y_test))
        
        #WOOHOO AVERAGES
        train_score_ten['Naive Bayes'] = (sum(NB_three) / len(NB_three))
        test_score_ten['Naive Bayes'] = (sum(NB_test_three) / len(NB_test_three))
        train_score_ten['Decision Tree'] = (sum(DT_tree) / len(DT_tree))
        test_score_ten['Decision Tree'] = (sum(DT_test_tree) / len(DT_test_tree))
        train_score_ten['SVC'] = (sum(vect) / len(vect))
        test_score_ten['SVC'] = (sum(vect_test) / len(vect_test))
        train_score_ten['Poly'] = (sum(plr) / len(plr))
        test_score_ten['Poly'] = (sum(plr_test) / len(plr_test))
        train_score_ten['Lasso'] = (sum(phoenix) / len(phoenix))
        test_score_ten['Lasso'] = (sum(phoenix_test) / len(phoenix_test))

    else:
        test_score_final = {}
        train_score_final = {}
        for q in range(100):
        
    #Gausian Distribution
            gnb = GaussianNB()
            gnb.fit(X_train, Y_train)
            estimatedY = gnb.predict(X_test)
            NB_three = []
            NB_three.append(gnb.fit(X_train, Y_train).score(X_train, Y_train))
            NB_test_three = []
            NB_test_three.append(gnb.fit(X_train, Y_train).score(X_test, Y_test))
            
    
    #Decision Tree
            clf = DecisionTreeClassifier()
            clf = clf.fit(X_train, Y_train)
            predict = clf.predict(X_test)
            DT_tree = []
            DT_tree.append(clf.fit(X_train, Y_train).score(X_train, Y_train))
            DT_test_tree = []
            DT_test_tree.append(clf.fit(X_train, Y_train).score(X_test, Y_test))
            
    #SVC Model
            svc_model = SVC(kernel = 'sigmoid')
            svc_model.fit(X_train, Y_train)
            Y_pred = svc_model.predict(X_test)
            vect = []
            vect_test = []
            vect.append(svc_model.fit(X_train, Y_train).score(X_train, Y_train))
            vect_test.append(svc_model.fit(X_train, Y_train).score(X_test, Y_test))
    
    #Polynominal Regression
            polyregr = Pipeline([('poly', PolynomialFeatures(2)), 
                         ('linear', linear_model.LinearRegression(fit_intercept=False))])
            polyregr.fit(X_train, Y_train)
            diabetes_y_pred2 = polyregr.predict(X_test)
            plr = []
            plr_test = []
            plr.append(polyregr.fit(X_train, Y_train).score(X_train, Y_train))
            plr_test.append(polyregr.fit(X_train, Y_train).score(X_test, Y_test))
    
    #Fuck It Lasso, all these classifiers so far are super bad
            lassie = linear_model.Lasso(alpha = 2)
            lassie.fit(X_train, Y_train)
            phoenix = []
            phoenix_test = []
            phoenix.append(lassie.fit(X_train, Y_train).score(X_train, Y_train))
            phoenix_test.append(lassie.fit(X_train, Y_train).score(X_test, Y_test))
        
        #WOOHOO AVERAGES
        train_score_final['Naive Bayes'] = (sum(NB_three) / len(NB_three))
        test_score_final['Naive Bayes'] = (sum(NB_test_three) / len(NB_test_three))
        train_score_final['Decision Tree'] = (sum(DT_tree) / len(DT_tree))
        test_score_final['Decision Tree'] = (sum(DT_test_tree) / len(DT_test_tree))
        train_score_final['SVC'] = (sum(vect) / len(vect))
        test_score_final['SVC'] = (sum(vect_test) / len(vect_test))
        train_score_final['Poly'] = (sum(plr) / len(plr))
        test_score_final['Poly'] = (sum(plr_test) / len(plr_test))
        train_score_final['Lasso'] = (sum(phoenix) / len(phoenix))
        test_score_final['Lasso'] = (sum(phoenix_test) / len(phoenix_test))


#Hahaha Graphing this HELL

#THIRDS
swag = sorted(train_score_third.items()) # sorted by key, return a list of tuples

x_swag, y_swag = zip(*swag) # unpack a list of pairs into two tuples

yeehaw = sorted(test_score_third.items()) # sorted by key, return a list of tuples

x_yeehaw, y_yeehaw = zip(*yeehaw)
x_swag = x_yeehaw

plt.bar(x_swag, y_swag, color = 'coral', label = 'Train Error 2/3')
plt.title('Training Error at 2/3 Train Data')
plt.xlabel('Classifier Type')
plt.ylabel('Percentage Error')
for x_swag,y_swag in zip(x_swag, y_swag):

    label = "{:.2f}".format(y_swag)

    plt.annotate(label, # this is the text
                 (x_swag,y_swag), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,-10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.show()
plt.show()

plt.bar(x_yeehaw, y_yeehaw, color = 'turquoise', label = 'Test Error 1/3')
plt.title('Testing Error at 1/3 Test Data')
plt.xlabel('Classifier Type')
plt.ylabel('Percentage Error')
for x_yeehaw,y_yeehaw in zip(x_yeehaw, y_yeehaw):

    label = "{:.2f}".format(y_yeehaw)

    plt.annotate(label, # this is the text
                 (x_yeehaw,y_yeehaw), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,-10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center


plt.show()


#QUARTERS
pain = sorted(train_score_quart.items()) # sorted by key, return a list of tuples

x_pain, y_pain = zip(*pain) # unpack a list of pairs into two tuples

sad = sorted(test_score_quart.items()) # sorted by key, return a list of tuples

x_sad, y_sad = zip(*sad)

plt.bar(x_pain, y_pain, color = 'palevioletred', label = 'Train Error 3/4')
plt.title('Training Error at 3/4 Train Data')
plt.xlabel('Classifier Type')
plt.ylabel('Percentage Error')
for x_pain,y_pain in zip(x_pain, y_pain):

    label = "{:.2f}".format(y_pain)

    plt.annotate(label, # this is the text
                 (x_pain,y_pain), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,-10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.show()


plt.bar(x_sad, y_sad, color = 'cornflowerblue', label = 'Test Error 1/4')
plt.title('Testing Error at 1/4 Test Data')
plt.xlabel('Classifier Type')
plt.ylabel('Percentage Error')
for x_sad,y_sad in zip(x_sad, y_sad):

    label = "{:.2f}".format(y_sad)

    plt.annotate(label, # this is the text
                 (x_sad,y_sad), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,-10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.show()

#TENTH
jb = sorted(train_score_ten.items()) # sorted by key, return a list of tuples

x_jb, y_jb = zip(*jb) # unpack a list of pairs into two tuples

sg = sorted(test_score_ten.items()) # sorted by key, return a list of tuples

x_sg, y_sg = zip(*sg)

plt.bar(x_jb, y_jb, color = 'firebrick', label = 'Train Error 9/10')
plt.title('Training Error at 9/10 Train Data')
plt.xlabel('Classifier Type')
plt.ylabel('Percentage Error')
for x_jb,y_jb in zip(x_jb, y_jb):

    label = "{:.2f}".format(y_jb)

    plt.annotate(label, # this is the text
                 (x_jb,y_jb), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,-10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.show()


plt.bar(x_sg, y_sg, color = 'cyan', label = 'Test Error 1/10')
plt.title('Testing Error at 1/10 Test Data')
plt.xlabel('Classifier Type')
plt.ylabel('Percentage Error')
for x_sg,y_sg in zip(x_sg, y_sg):

    label = "{:.2f}".format(y_sg)

    plt.annotate(label, # this is the text
                 (x_sg,y_sg), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,-10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.show()

#A TWENTIETH

sarah = sorted(train_score_final.items()) # sorted by key, return a list of tuples

x_sarah, y_sarah = zip(*sarah) # unpack a list of pairs into two tuples

tori = sorted(test_score_final.items()) # sorted by key, return a list of tuples

x_tori, y_tori = zip(*tori)

plt.bar(x_sarah, y_sarah, color = 'lightsalmon', label = 'Train Error 19/20')
plt.title('Training Error at 19/20 Train Data')
plt.xlabel('Classifier Type')
plt.ylabel('Percentage Error')
for x_sarah,y_sarah in zip(x_sarah, y_sarah):

    label = "{:.2f}".format(y_sarah)

    plt.annotate(label, # this is the text
                 (x_sarah,y_sarah), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,-10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.show()


plt.bar(x_tori, y_tori, color = 'deepskyblue', label = 'Test Error 1/20')
plt.title('Testing Error at 1/20 Test Data')
plt.xlabel('Classifier Type')
plt.ylabel('Percentage Error')
for x_tori,y_tori in zip(x_tori, y_tori):

    label = "{:.2f}".format(y_tori)

    plt.annotate(label, # this is the text
                 (x_tori,y_tori), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,-10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.show()
