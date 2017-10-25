import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

def csv_writer(data,filename,mode):
		with open(filename,mode) as fname:
			np.savetxt(fname,data,delimiter=',',newline='\n')


# Read data from files
data = np.genfromtxt('../Dataset/imputed_train_mean.csv', delimiter=',')
test = np.genfromtxt("../Dataset/imputed_test_mean.csv", delimiter=',')
labels = np.genfromtxt("../Dataset/train_labels.csv")

data = preprocessing.scale(data)
test = preprocessing.scale(test)

lda = LinearDiscriminantAnalysis(n_components=1000)
features_train = lda.fit_transform(data, labels)
features_test = lda.transform(test)

print(features_train.shape)

csv_writer(features_train,"../Dataset/train_mean_1000_lda.csv","wb")
csv_writer(features_test,"../Dataset/test_mean_1000_lda.csv","wb")

pca = PCA(n_components=750)
features_train = pca.fit_transform(data, labels)
features_test = pca.transform(test)

print(features_train.shape)

csv_writer(features_train,"../Dataset/train_mean_750_pca.csv","wb")
csv_writer(features_test,"../Dataset/test_mean_750_pca.csv","wb")
