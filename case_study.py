### -*- coding: utf-8 -*-

"""

Created on Thu Sep 26 16:02:27 2019

@author: Prateek's PC
"""
##=================================================================
## CLASSIFIYING PERSONAL INCOME 
##================================================================

##REQURIED PACKGES

## To Work with dataframe

import pandas as pd

##To work With Numarical Values
import numpy as np

##To Work With Visualizing Data
import seaborn as sns

##For Pratition Data
from sklearn.model_selection import train_test_split

## Importing library For Logistic Regression
from sklearn.linear_model import LogisticRegression

## For performence - accuracy score and confusion matrix
from sklearn.metrics import accuracy_score, confusion_matrix

## 

#####################################################################
## ==================================================================
## Importing Data
## ==================================================================

data_income = pd.read_csv('income(1).csv')

## Make a Copy of "data_income"
data= data_income.copy()

"""

## Exploratory data analysis:

#1. Geting to know the data
#2. Data preprocessing(missing value)
#3. cross tables and data visualization
"""

##======================================================================
## Geting to know data
##======================================================================
##
## ***** To Check the variables' data type
data.info()

## checking the missing values
data.isnull()
print('data null values are:: \n',data.isnull().sum())
##** No missing values!!



## summary of numarical values
summary_num=data.describe()
print(summary_num)
## summary of categorical values
summary_cate=data.describe(include='O')
print(summary_cate)

## checking for unique classes

print(np.unique(data['JobType']))
print(np.unique(data['occupation']))
"""THERE IS EXIST ' ?' instead 'nan' values(space before ?)


""go back and read the data by including 'nan_value['?']'"""

data= pd.read_csv('income(1).csv', na_values=[' ?'])

##============================================================
## Data pre-processing
##============================================================

data.isnull().sum()

missing= data[data.isnull().any(axis=1)]
## axis=1 to cosider atlest one coloumn value is missing
'''
#  points to be noted that::
1. missing values in "JobType" is 1809 
2. missing values in "occupation" is 1816 
## 1816-1809=7 
   * seven unfilled coloumn  of occupation is null because JobType is never work
   
'''
data2 = data.dropna(axis=0)
corelation = data2.corr()
 
##============================================================
## Cross table and  data visualization
##===========================================================
## extraction the coloumn name

data2.columns


##===================================================================
## Gender proportion  table
##===================================================================

gender = pd.crosstab(index=data2["gender"],columns= "counts", normalize = True)
 
print(gender)

##====================================================================
##  Gender vs salary status
##=========================================================================

gender_salsat= pd.crosstab(index= data2['gender'], columns= data2['SalStat'] , margins= True, normalize='index')

print(gender_salsat)

##==========================================================================
## Frequency disrubtion of salary status
##==============================================================================

SalStat= sns.countplot(data2['SalStat'])


##=========================================================================
## Hisogram of Age
##=============================================================================

sns.distplot(data2['age'], bins=10, kde=False)
## people with 20-40 age is high

############# Boxplot - age vs salary status##########################
sns.boxplot('SalStat', "age" ,data= data2)
#######################################################################

####################################################################
## LOGISTI REGRESSION
##############################################################

# Reindexing the salary status y nme to 0, 1

data2['SalStat']= data2['SalStat'].map({' less than or equal to 50,000': 0, ' greater than 50,000': 1})
print(data2['SalStat'])

# dummies is for changing categorical variables into dummies variabe called one hot encoding
new_data= pd.get_dummies(data2, drop_first=True) 

# Dividing the column in dependent variable(Y) and indepenent variable(X)
#
# Shorting the column names
column_list=list(new_data.columns)
print(column_list)

#separating the input names from data
# substract 'SalStat'column from column_list
features= list(set(column_list)-set(["SalStat"]))
print(features)

# Shorting the output values in Y
# .values gives all the values stored in 'SalStat' column
y=new_data['SalStat'].values
print(y)

# Shorting the output values in x
x= new_data[features].values
print(x)


# Splitting the data into train and test 
train_x, test_x,train_y,test_y = train_test_split(x,y,test_size=0.3, random_state=0)

# Makr an intance of the Model
logistic= LogisticRegression()

# Fitting the values for x and y
logistic.fit(train_x,train_y)
logistic.coef_
logistic.intercept_

# Prediction from test data set
prediction= logistic.predict(test_x)
print(prediction)


# Confusion matrix
confusion_matrix= confusion_matrix(test_y,prediction)
print(confusion_matrix)

# Calculating the accuracy
accuracy_score= accuracy_score(test_y,prediction)
print(accuracy_score)
#accuracy_score=0.8423030169079456

# Printing Misscalssified values from prediction
print('Missclassified samples: %d'%(test_y!= prediction).sum())
# 1427 Missclassified samples




######################################################
## KNN
###########################################################

# Importng the KNN librar
from sklearn.neighbors import KNeighborsClassifier
# 

# importing library for ploting
import matplotlib.pyplot as plt


# Shorting the KNN classifier
KNN_classifier= KNeighborsClassifier(n_neighbors= 5)


# Fitting the values for x And y
KNN_classifier.fit(train_x,train_y)

# prediction the test vales with model
prediction= KNN_classifier.predict(test_x)

# Performace matrix check
confusion_matrix= confusion_matrix(test_y,prediction)
print("\t ", "prediction values")
print("original values \n",confusion_matrix)

# calculation the accuracy
accuracy_score= accuracy_score(test_y, prediction)
print(accuracy_score)

print('misclassified sample : %d' %(test_y != prediction.sum()) 

# Effect of k value on classifier 


 Misclassified_sample1 = []

# Calculating erro for k values between 1 to 20

for i in range(1,20):
    knn= KNeighborsClassifier( n_neighbors= i)
    knn.fit(train_x,train_y)
    pred_i= knn.predict(test_x)
    Misclassified_sample1.append((test_y != pred_i).sum())
    
print(Misclassified_sample1)    
    
###########################################################################
# END SRIPT
########################################################################    

