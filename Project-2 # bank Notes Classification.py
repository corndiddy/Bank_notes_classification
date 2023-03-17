#!/usr/bin/env python
# coding: utf-8

# # Project-2
# # Bank Notes Classification

# ## problem statement

# Banknotes are one of the most important assets of a country. Some criminals introduce fake notes which bear a resemblance to original note to create discrepancies of the money in the financial market. It is difficult for humans to tell true and fake banknotes apart especially because they have a lot of similar features. Fake notes are created with precision, hence there is need for an efficient algorithm which accurately predicts whether a banknote is genuine or not.

# ## dataset

# These datasets have different features and the goal of my analysis is to classify genuine (label 0) and counterfeit (label 1) banknotes comparing different features through Machine Learning algorithms.

# ## Banknote Authentication Data from UCI:

# ## Tool and technlogies used:
# ### Python
# ### pandas 
# ### matplotlib and seaborn
# ### numpy 
# ### tkinter
# ### flask

# In[1]:


#import dataset


# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[3]:


df=pd.read_csv('banknotes.csv')


# In[ ]:


df.shape


# In[ ]:


df.head()


# The data file banknote_authentication.csv is the source of information for the classification problem. The number of instances (rows) in the data set is 1372, and the number of variables (columns) is 5.
variance_of_wavelet_transformed, used as input.
skewness_of_wavelet_transformed, used as input.
curtosis_of_wavelet_transformed, used as input.
entropy_of_image, used as input.
class, used as the target. It can only have two values: 0 (non-counterfeit) or 1 (counterfeit).
# In[ ]:


s1=df['Class'].value_counts()


# In[ ]:


s1


# In[ ]:


s1.plot(kind='bar')


# In[ ]:


# dataset is balanced
# count 1...610...fake class
#count 0..762...real class


# In[ ]:


#null values


# In[ ]:


df.isnull().sum()


# In[ ]:


# no missing values present in dataset


# In[ ]:


df.duplicated().sum()


# In[ ]:


df.drop_duplicates(inplace=True)


# In[ ]:


# we have 24 duplicates sample ..drop duplicates row
#no duplicate columns are present


# In[ ]:


df.shape


# In[ ]:


# statistical information


# In[ ]:


df.describe()


# In[ ]:


# variance   -7.0 to 6.824
# skewness.-13.77 to 12.95
#curtosis..-5.2 to 17.92
#Entropy..-8.5 to 2.44


# In[ ]:


# histogram


# In[ ]:


df.hist(figsize=(10,10),bins=50)
plt.show()


# In[ ]:


# input features follows normal distribution


# In[ ]:


# box plot


# In[ ]:


df.plot(kind ='box',subplots = True, layout =(2,3),sharex = False)
plt.show()


# In[ ]:


# in curtosis and entropy outliers are present
# but may be for fake note


# In[ ]:


df.plot(kind ='density',subplots = True, layout =(2,3),sharex = False)
plt.show()


# In[ ]:


# violin plot


# In[ ]:


sns.violinplot(x='Class', y='Variance', data=df)


# In[ ]:


# variance for real notes ...-4.2 to 6.8
# variance for fake notes...-8 to 2.39


# In[ ]:


pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)


# In[ ]:


df.groupby('Class').describe()


# In[ ]:


# skewness for real notes ...-6.9 to 12.95
# skewness for fake notes...-13.77 to 9.6

# curtosis for real notes ...-4.9 to 8.82
# curtoiss for fake notes...--5.2 to 17.92

# Entropy for real notes ...-8.54 to 2.44
# Entrop for fake notes...--7.5 to 2.13


# In[ ]:


sns.pairplot(df)


# In[ ]:


sns.scatterplot(x='Curtosis',y='Class',data=df)


# In[ ]:


sns.scatterplot(x='Entropy',y='Class',data=df)


# In[ ]:


sns.scatterplot(x='Skewness',y='Class',data=df)


# In[ ]:


#countplot


# In[ ]:


sns.countplot(x=df['Variance'])
plt.show()


# In[ ]:


# feature selection


# In[ ]:


#correlation and multi colinearity


# In[ ]:


cm=df.corr()
sns.heatmap(cm,annot=True)


# In[ ]:


# all features has strong correlation with target variable
# multi colinearity....(strong correation among input features)


# In[ ]:


X=df.drop('Class',axis=1)
y=df['Class']


# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier
et = ExtraTreesClassifier()
et.fit(X,y)


# In[ ]:


a=et.feature_importances_
s2=pd.Series(a,index=X.columns)
s2.plot(kind='barh')


# In[ ]:


# all fetures has good score


# In[ ]:


# encoding...get_dummies,one hot encoder,label encoder,map,replace


# In[ ]:


#scaling..


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_sc=sc.fit_transform(X)


# In[ ]:


import pickle


# In[ ]:


# f=open('bankscaler','wb')
# pickle.dump(sc,f)
# f.close()


# In[ ]:


#cross validation


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_sc,y,
                                        test_size=0.2,random_state=7)


# In[ ]:


#precision recall
# High precision


# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


def evaluate(model, X_train, X_test, y_train, y_test):
    y_test_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    print("TRAINIG RESULTS: \n===============================")
    clf_report = pd.DataFrame(classification_report(y_train, y_train_pred, output_dict=True))
    print(f"CONFUSION MATRIX:\n{confusion_matrix(y_train, y_train_pred)}")
    print(f"ACCURACY SCORE:\n{accuracy_score(y_train, y_train_pred):.4f}")
    print(f"CLASSIFICATION REPORT:\n{clf_report}")

    print("TESTING RESULTS: \n===============================")
    clf_report = pd.DataFrame(classification_report(y_test, y_test_pred, output_dict=True))
    print(f"CONFUSION MATRIX:\n{confusion_matrix(y_test, y_test_pred)}")
    print(f"ACCURACY SCORE:\n{accuracy_score(y_test, y_test_pred):.4f}")
    print(f"CLASSIFICATION REPORT:\n{clf_report}")


# In[ ]:


# logistic regression


# In[ ]:


from sklearn.linear_model import LogisticRegression
model1 = LogisticRegression(penalty='l2',C=1.0)
model1.fit(X_train,y_train)


# In[ ]:


evaluate(model1, X_train, X_test, y_train, y_test)


# In[ ]:


d={}


# In[ ]:


d['Logistic']={'Train':99.07,'Test':1.0,'precision':1.0}


# In[ ]:


d


# In[ ]:


# knn


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
model2=KNeighborsClassifier(n_neighbors=10)
model2.fit(X_train,y_train)


# In[ ]:


evaluate(model2, X_train, X_test, y_train, y_test)


# In[ ]:


d['knn']={'Train':99.7,'Test':99.63,'precision':0.99}


# In[ ]:


d


# In[ ]:


# naive bays


# In[ ]:


from sklearn.naive_bayes import GaussianNB
model3=GaussianNB()
model3.fit(X_train,y_train)


# In[ ]:


evaluate(model3, X_train, X_test, y_train, y_test)


# In[ ]:


d['NB']={'Train':84.32,'Test':83.33,'precision':0.83}


# In[ ]:


d


# In[ ]:


# random forest classifier


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
model4=RandomForestClassifier(n_estimators=100)
model4.fit(X_train,y_train)


# In[ ]:


evaluate(model4, X_train, X_test, y_train, y_test)


# In[ ]:


d['Random Forest']={'Train':100,'Test':100,'precision':100}


# In[ ]:


d


# In[ ]:


#SVM


# In[ ]:


from sklearn.svm import SVC
model5=SVC(C=1.0,kernel='rbf')
model5.fit(X_train,y_train)


# In[ ]:


evaluate(model5, X_train, X_test, y_train, y_test)


# In[ ]:


d['SVC']={'Train':100,'Test':100,'precision':100}


# In[ ]:


#boosting


# In[ ]:


get_ipython().system('pip install xgboost')


# In[ ]:


from xgboost import XGBClassifier
model6=XGBClassifier(n_estimators=100,learning_rate=1.0,random_state=11)
model6.fit(X_train,y_train)
evaluate(model6, X_train, X_test, y_train, y_test)


# In[ ]:


d['XGB']={'Train':100,'Test':99.63,'precision':100}


# In[ ]:


d


# In[ ]:


d={'Logistic': {'Train': 99.07, 'Test': 100, 'precision': 100},
 'knn': {'Train': 99.7, 'Test': 99.63, 'precision': 99},
 'NB': {'Train': 84.32, 'Test': 83.33, 'precision': 83},
 'Random Forest': {'Train': 100, 'Test': 100, 'precision': 100},
 'SVC': {'Train': 100, 'Test': 100, 'precision': 100},
 'XGB': {'Train': 100, 'Test': 99.63, 'precision': 100}}


# In[ ]:


df1=pd.DataFrame(d)


# In[ ]:


df1.plot(kind='barh',figsize=(20,15))


# In[ ]:


# conclusion
# RF gives best accuracy with best precision
# random forest is best model here..its bagging type ML algorithm
# it avoids overfitting
# testing score 100%..precision 100 %


# In[ ]:


# model4 is final model


# In[ ]:


import pickle
f=open('bankmodel','wb')
pickle.dump(model4,f)
f.close()


# In[ ]:


a=[3.4,2.5,11.3,7.8]


# In[ ]:


a_sc=sc.transform([a])


# In[ ]:


model4.predict(a_sc)


# In[ ]:


# 0 real note
# 1 fake note

