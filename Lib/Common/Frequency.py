import numpy as np
import matplotlib.pyplot as plt

def period_calc(data):
	
	period = np.diff(np.nonzero(data[:-1] != data[1:]))
	return period

def digitized(data, limit = 0):

	if limit == 0:
		limit = 0.1*data.max()
	
	# data_out = np.array([1 if np.all(data[i-5:i+1]>=limit) else -1 if np.all(data[i-5:i+1]<-limit) else 0 for i in range(5,data.size)])
	data_out = np.zeros(data.size, dtype='int')

	for i in range(5,data.size):
		if np.all(data[i-5:i+1] >= limit):
			data_out[i] = 1
		elif np.all(data[i-5:i+1] < -limit):
			data_out[i] = 0
		else:
			data_out[i] = data_out[i-1]

	return data_out

