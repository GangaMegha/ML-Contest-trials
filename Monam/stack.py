import csv
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report
from sklearn import linear_model
from sklearn.model_selection import StratifiedKFold


def csv_writer(data,filename,mode):
		with open(filename,mode) as fname:
			fname.write("id,label\n")
			np.savetxt(fname,data,delimiter=',',newline='\n')


train_rf = np.genfromtxt("../Dataset/result_rf_train.csv", delimiter=',')
train_gb = np.genfromtxt("../Dataset/result_gb_train.csv", delimiter=',')
train_ab = np.genfromtxt("../Dataset/result_ab_train.csv", delimiter=',')

valid_rf = np.genfromtxt("../Dataset/result_rf_valid.csv", delimiter=',')
valid_gb = np.genfromtxt("../Dataset/result_gb_valid.csv", delimiter=',')
valid_ab = np.genfromtxt("../Dataset/result_ab_valid.csv", delimiter=',')

test_rf = np.genfromtxt("../Dataset/result_rf_test.csv", delimiter=',')
test_gb = np.genfromtxt("../Dataset/result_gb_test.csv", delimiter=',')
test_ab = np.genfromtxt("../Dataset/result_ab_test.csv", delimiter=',')

train_labels = np.genfromtxt("../Dataset/temp/Y_train.csv", delimiter=',')
valid_labels = np.genfromtxt("../Dataset/temp/Y_validate.csv", delimiter=',')

logreg = linear_model.LogisticRegression(C=1e5)