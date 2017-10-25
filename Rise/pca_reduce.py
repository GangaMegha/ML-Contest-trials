import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
import pickle
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.decomposition import PCA
import sklearn.metrics as met

def csv_writer(data,filename,mode):
		with open(filename,mode) as fname:
			np.savetxt(fname,data,delimiter=',',newline='\n')
#print("BEFORE")
train_data=preprocessing.scale(np.loadtxt('imputed_train_median.csv', delimiter=','))
test_data=preprocessing.scale(np.loadtxt('imputed_test_median.csv', delimiter=','))

train_labels=np.loadtxt('train_labels.csv', delimiter=',')
X_train, X_validate, Y_train, Y_validate = train_test_split(train_data, train_labels,stratify=train_labels, test_size=0.25)
#test_labels=np.loadtxt('../DS3/test_labels.csv', delimiter=',')-1

pca =PCA(n_components=2000)

X_train=pca.fit_transform(X_train)
X_validate=pca.transform(X_validate)
X_test=pca.transform(test_data)

csv_writer(X_train,"X_train_p.csv","wb")
csv_writer(X_test,"X_test_p.csv","wb")
csv_writer(X_validate,"X_validate_p.csv","wb")
csv_writer(Y_validate,"Y_validate.csv","wb")
csv_writer(Y_train,"Y_train.csv","wb")


