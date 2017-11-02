import csv
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold


def csv_writer(data,filename,mode):
		with open(filename,mode) as fname:
			fname.write("id,label\n")
			np.savetxt(fname,data,delimiter=',',newline='\n')

# Read data from files
train_data = np.genfromtxt('../Dataset/temp/X_train_p.csv', delimiter=',')
train_labels = np.genfromtxt('../Dataset/temp/Y_train.csv', delimiter=',')
validation_data = np.genfromtxt("../Dataset/temp/X_validate_p.csv", delimiter=',')
validation_labels = np.genfromtxt("../Dataset/temp/Y_validate.csv", delimiter=',')
test_data = np.genfromtxt("../Dataset/test_mean_500_pca.csv", delimiter=',')


# Validation
print("\n\n\nRunning Validation..................................\n")

# rf = RandomForestClassifier(n_estimators=6000, max_depth=7, criterion='gini', min_samples_split=2, min_samples_leaf=1, n_jobs=-1, class_weight="balanced_subsample")
nb = GaussianNB()
# nb.fit(train_data,train_labels)
nb.partial_fit(train_data, train_labels, np.unique(train_labels))

y_prediction = nb.predict(validation_data)

item = np.column_stack((np.arange(len(y_prediction)), y_prediction))

csv_writer(item, "../Dataset/result_rf_valid2.csv","wb")

print(classification_report(validation_labels, y_prediction))

#Train
y_prediction = nb.predict(train_data)

item = np.column_stack((np.arange(len(y_prediction)), y_prediction))

csv_writer(item, "../Dataset/result_rf_train2.csv","wb")

# Test
print("\n\nRunning test........................")
# rf = RandomForestClassifier(n_estimators=3000, max_depth=25, min_samples_split=4, min_samples_leaf=2)
# rf.fit(X, labels)
# rf.fit(train_data,train_labels)

y_prediction = nb.predict(test_data)
item = np.column_stack((np.arange(len(y_prediction)), y_prediction))

csv_writer(item, "../Dataset/result_rf_test2.csv","wb")

print("\n\nDone :)")
