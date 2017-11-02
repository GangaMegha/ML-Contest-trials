import csv
import numpy as np

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.model_selection import StratifiedKFold


def csv_writer(data,filename,mode):
		with open(filename,mode) as fname:
			fname.write("id,label\n")
			np.savetxt(fname,data,delimiter=',',newline='\n')

train_data=np.genfromtxt('../Dataset/train.csv', delimiter=',',missing_values="NaN",skip_header=1)
test=np.genfromtxt('../Dataset/test.csv', delimiter=',',missing_values="NaN",skip_header=1)
labels = np.genfromtxt("../Dataset/train_labels.csv", delimiter=',')

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, stratify=labels, test_size=0.20, shuffle=True)

# Create linear regression object
regr = linear_model.LinearRegression()

for i in range(30):
	for j in range(2600):
		# Train the model using the training sets
		regr.fit(train_data, diabetes_y_train)

		# Make predictions using the testing set
		diabetes_y_pred = regr.predict(diabetes_X_test)