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
