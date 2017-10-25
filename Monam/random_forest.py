import csv
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# # Read data from files
# data = np.array(pd.read_csv("../Dataset/imputed_train_mean.csv"))
# labels = np.array(pd.read_csv("../Dataset/train_labels.csv"))
data = np.genfromtxt('../Dataset/train_mean_750_pca.csv', delimiter=',')
labels = np.genfromtxt('../Dataset/train_labels.csv', delimiter=',')
print(data.shape)
print(labels.shape)

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, stratify=labels, test_size=0.20, shuffle=True)

rf = RandomForestClassifier(n_estimators=1000, max_depth=20, min_samples_split=4, min_samples_leaf=2)
rf.fit(train_data,train_labels)

y_prediction = rf.predict(test_data)

print(classification_report(test_labels, y_prediction))
