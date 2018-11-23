# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 04:00:04 2018

@author: Mossy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn.neural_network import MLPClassifier

# =============================================================================
# from sklearn.linear_model import LogisticRegression
# =============================================================================


from sklearn.cross_validation import train_test_split
#matplotlib inline
data = pd.read_csv('iris.csv')
print(data.columns)

print("dimension of Iris data: {}".format(data.shape))


print(data.groupby('species').size())


data.info()


feature_names = ['sepal_length','sepal_width','petal_length','petal_width']
X = data[feature_names]
y = data.species




#check for null values
null_columns=data.columns[data.isnull().any()]
data[null_columns].isnull().sum()




### Importing Classification Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# =============================================================================
# from sklearn.naive_bayes import GaussianNB
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# 
# =============================================================================
models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('SVM', SVC()))
models.append(('LR', LogisticRegression()))
models.append(('DT', DecisionTreeClassifier()))
# =============================================================================
# models.append(('GNB', GaussianNB()))
# models.append(('RF', RandomForestClassifier()))
# models.append(('GB', GradientBoostingClassifier()))
# 
# =============================================================================



####
#from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold



X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = data.species, random_state=0)

#Visualizing the Data Set
data.groupby('species').hist(figsize=(9, 9))

#Training the Model
names = []
scores = []
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))
    names.append(name)
tr_split = pd.DataFrame({'Name': names, 'Score': scores})
print(tr_split)


#### after application of Kfold Cross Validation 
names = []
scores = []
for name, model in models:
    
    kfold = KFold(n_splits=10, random_state=10) 
    score = cross_val_score(model, X, y, cv=kfold, scoring='accuracy').mean()
    
    names.append(name)
    scores.append(score)
kf_cross_val = pd.DataFrame({'Name': names, 'Score': scores})
print(kf_cross_val)


#Showing Bar plots of different Classifier Accuracies
axis = sns.barplot(x = 'Name', y = 'Score', data = kf_cross_val)
axis.set(xlabel='Classifier', ylabel='Accuracy')
for p in axis.patches:
    height = p.get_height()
    axis.text(p.get_x() + p.get_width()/2, height + 0.005, '{:1.4f}'.format(height), ha="center") 
    
plt.show()


##

#Compute the correlation matrix
corr = data.corr()


# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True


# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))


# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)


# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
sns.pairplot(data)