import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

def Early_late(data, period, delta=1):

	sampled_data = np.empty(data.size)
	bit_out = np.empty(0, dtype='int')
	
	index = int(1.5*period)
	while index<data.size:
		x = (data[index-period]>=0) == (data[index-period//2]>=0)
		y = (data[index]>=0) == (data[index-period//2]>=0)
		if x and not(y):
			index += delta
		elif not(x) and y:
			index -= delta
		
		bit_out = np.append(bit_out, int(data[index]>=0))
		sampled_data[index] = data[index]

		index += period

	# plt.plot(data)
	# plt.plot(sampled_data, '.')
	# plt.show()

	return bit_out, sampled_data

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