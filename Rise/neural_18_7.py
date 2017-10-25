from sklearn.neural_network import MLPClassifier
import numpy as np



X_train=np.loadtxt('X_train_p.csv', delimiter=',')
X_validate=np.loadtxt('X_validate_p.csv', delimiter=',')
X_test=np.loadtxt('X_test_p.csv', delimiter=',')
Y_train=np.loadtxt('Y_train.csv', delimiter=',') 
Y_validate=np.loadtxt('Y_validate.csv', delimiter=',')

Y_out=np.zeros([X_test.shape[0],2])
Y_out[:,0]=np.arange(X_test.shape[0])
mlp=MLPClassifier(hidden_layer_sizes=(500,200 ), activation='relu', solver='adam', 
	alpha=0.00005, batch_size='auto', learning_rate='constant', learning_rate_init=0.0005, 
	power_t=0.5, max_iter=7500, shuffle=True, random_state=None, tol=0.0001, verbose=False, 
	warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, 
	validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

mlp.fit(X_train,Y_train)

print(mlp.score(X_validate,Y_validate))

Y_out[:,1]=mlp.predict(X_test)

np.savetxt("Y_out.csv", Y_out, fmt='%d',delimiter=',', newline='\n', header='id,label',comment=None)