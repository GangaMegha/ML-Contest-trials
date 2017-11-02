import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

def csv_writer(data,filename,mode):
		with open(filename,mode) as fname:
			np.savetxt(fname,data,delimiter=',',newline='\n')


# Read data from files
data = np.genfromtxt('../Dataset/imputed_train_mean.csv', delimiter=',')
test_set = np.genfromtxt("../Dataset/imputed_test_mean.csv", delimiter=',')
labels = np.genfromtxt("../Dataset/train_labels.csv")

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, stratify=labels, test_size=0.20, shuffle=True)

train_data = preprocessing.scale(train_data)
test_data = preprocessing.scale(test_data)
test_set = preprocessing.scale(test_set)
data = preprocessing.scale(data)

# LDA
lda = LinearDiscriminantAnalysis(n_components=100)
train_data = lda.fit_transform(train_data, train_labels)
test_data = lda.transform(test_data)
test = lda.transform(test_set)

X = lda.fit_transform(data, labels)

train_data = preprocessing.scale(train_data)
test_data = preprocessing.scale(test_data)
test = preprocessing.scale(test)
X = preprocessing.scale(X)

print(train_data.shape)

csv_writer(train_data,"../Dataset/train_mean_28_lda.csv","wb")
csv_writer(test_data,"../Dataset/validation_mean_28_lda.csv","wb")
csv_writer(train_labels,"../Dataset/train_labels_mean_28_lda.csv","wb")
csv_writer(test_labels,"../Dataset/validation_labels_mean_28_lda.csv","wb")
csv_writer(test,"../Dataset/test_mean_28_lda.csv","wb")
csv_writer(X,"../Dataset/X_mean_28_lda.csv","wb")


#PCA
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, stratify=labels, test_size=0.20, shuffle=True)

pca = PCA(n_components=500)
train_data = pca.fit_transform(train_data, train_labels)
test_data = pca.transform(test_data)
test = pca.transform(test_set)

X = pca.fit_transform(data, labels)

train_data = preprocessing.scale(train_data)
test_data = preprocessing.scale(test_data)
test = preprocessing.scale(test)
X = preprocessing.scale(X)

print(train_data.shape)

csv_writer(train_data,"../Dataset/train_mean_500_pca.csv","wb")
csv_writer(test_data,"../Dataset/validation_mean_500_pca.csv","wb")
csv_writer(train_labels,"../Dataset/train_labels_mean_500_pca.csv","wb")
csv_writer(test_labels,"../Dataset/validation_labels_mean_500_pca.csv","wb")
csv_writer(test,"../Dataset/test_mean_500_pca.csv","wb")
csv_writer(X,"../Dataset/X_mean_500_pca.csv","wb")
