import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
import pickle
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import sklearn.metrics as met

#print("BEFORE")
train_data=preprocessing.scale(np.loadtxt('imputed_train_median.csv', delimiter=','))
#test_data=preprocessing.scale(np.loadtxt('../DS3/test.csv', delimiter=','))

train_labels=np.loadtxt('train_labels.csv', delimiter=',')-1
X_train, X_validate, y_train, y_validate = train_test_split(train_labels, y,stratify=y, test_size=0.25)
#test_labels=np.loadtxt('../DS3/test_labels.csv', delimiter=',')-1

lda =LinearDiscriminantAnalysis(n_components=500)
chosen_feature=lda.fit_transform(X_train,y_train)

x_test_chosen=lda.transform(test_data)

