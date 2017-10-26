import csv
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def csv_writer(data,filename,mode):
		with open(filename,mode) as fname:
			fname.write("id,label\n")
			np.savetxt(fname,data,delimiter=',',newline='\n')

# Read data from files
# data = np.genfromtxt('../Dataset/train_mean_750_pca.csv', delimiter=',')
data = np.genfromtxt('../Dataset/train_mean_1000_lda.csv', delimiter=',')
labels = np.genfromtxt('../Dataset/train_labels.csv', delimiter=',')
test = np.genfromtxt("../Dataset/test_mean_1000_lda.csv", delimiter=',')

print(data.shape)
print(labels.shape)

# Validation
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, stratify=labels, test_size=0.20, shuffle=True)

rf = RandomForestClassifier(n_estimators=1000, max_depth=20, min_samples_split=4, min_samples_leaf=2)
rf.fit(train_data,train_labels)

y_prediction = rf.predict(test_data)

print(classification_report(test_labels, y_prediction))

# Test
rf = RandomForestClassifier(n_estimators=1000, max_depth=20, min_samples_split=4, min_samples_leaf=2)
rf.fit(data, labels)

y_prediction = rf.predict(test)
item = np.column_stack((np.arange(len(y_prediction)), y_prediction))

csv_writer(item, "../Dataset/result.csv","wb")

