import numpy as np


Y_out=np.genfromtxt('Y_out.csv', delimiter=',',skip_header=1)

np.savetxt("Y_out.csv", Y_out, fmt='%d',delimiter=',', newline='\n', header='id,label')