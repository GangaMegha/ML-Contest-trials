import numpy as np 
from sklearn.preprocessing import Imputer
from sklearn import linear_model


def csv_writer(data,filename,mode):
		with open(filename,mode) as fname:
			np.savetxt(fname,data,delimiter=',',newline='\n')


train_data=np.genfromtxt('train.csv', delimiter=',',missing_values="NaN",skip_header=1)
test=np.genfromtxt('test.csv', delimiter=',',missing_values="NaN",skip_header=1)


print(train_data.shape)
print(test.shape)

impute=Imputer(missing_values='NaN',strategy='median',axis=0)
train_data[:,:-1]=impute.fit_transform(train_data[:,:-1])
test=impute.fit_transform(test)

csv_writer(train_data[:,1:-1],"imputed_train_median.csv","wb")
csv_writer(test[:,1:],"imputed_test_median.csv","wb")

csv_writer(train_data[:,-1],"train_labels.csv","wb")
print("Done")


