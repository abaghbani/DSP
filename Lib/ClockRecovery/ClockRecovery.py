import numpy as np
import matplotlib.pyplot as plt

def EarlyLate(data, period, gap=0, delta=1, plot_data=False):

	bit_out = np.empty(0, dtype=type(data[0]))
	bit_index = np.empty(0, dtype=np.uint32)
	
	index = 1*period
	while index<(data.size-delta):
		if gap == 0:
			x = (data[index-period]>=0) == (data[index-period//2]>=0)
			y = (data[index]>=0) == (data[index-period//2]>=0)
		else:
			x = np.abs(data[index-period]-data[index-period//2]) < gap
			y = np.abs(data[index]-data[index-period//2]) < gap
		
		if x and not(y):
			index += delta
		elif not(x) and y:
			index -= delta
		
		bit_out = np.append(bit_out, data[index])
		bit_index = np.append(bit_index, index)

		index += period

	if plot_data:
		plt.plot(data)
		plt.plot(bit_index, bit_out, '.')
		plt.legend(['raw', 'sample'])
		plt.show()

	return bit_out, bit_index

def CrossZero(data, period, Level_low):

	counter = np.zeros(data.size, dtype=np.uint8)
	bit_out = np.empty(0, dtype=np.int8)
	bit_index = np.empty(0, dtype=np.uint32)

	for i in range(data.size):
		counter[i] = 0 if (data[i]>=0) != (data[i-1]>=0) or (counter[i-1] >= period-1) else counter[i-1]+1
		if counter[i] == period//2:
			bit_out = np.append(bit_out, 1 if (data[i]>=Level_low) else -1 if (data[i] <= -Level_low) else 0)
			bit_index = np.append(bit_index, i)

	return bit_out, bit_index

def Earlylate_withOffset(data, period, delta=1, plot_data=False):

	sampled_data = np.zeros(data.size, dtype = type(data[0]))
	sampled_dis = np.zeros(data.size, dtype = type(data[0]))
	bit_out = np.empty(1, dtype='int')
	bit_index = np.empty(0, dtype='int32')
	
	dis = 1
	count = 0
	index = int(2*period)
	while index<(data.size-delta):
		x = (np.abs(data[index-period]-data[index-period//2]) < dis)
		y = (np.abs(data[index]       -data[index-period//2]) < dis)
		if x and not(y):
			index += delta
		elif not(x) and y:
			index -= delta
		
		if(data[index]-data[index-period] > dis):
			bit = 1
			dis = (data[index]-data[index-period]) // 2
			count = 0
		elif(data[index-period]-data[index] > dis):
			bit = 0
			dis = (data[index-period]-data[index]) // 2
			count = 0
		else:
			bit = bit_out[-1]
			count = count + 1

		if count >= 3:
			dis = 1
			count = 0

		bit_out = np.append(bit_out, bit)
		bit_index = np.append(bit_index, index)
		sampled_data[index] = data[index]
		sampled_dis[index] = dis

		index += period

	if plot_data:
		plt.plot(data)
		plt.plot(sampled_data, '.')
		plt.plot(sampled_dis)
		plt.legend(['raw', 'sample'])
		plt.show()

	return bit_out, bit_index

def EarlyLate_noninteger(data, period, delta):
	
	bit_out = np.empty(0, dtype='int')
	bit_index = np.empty(0, dtype='int32')

	period_low = int(np.floor(period))
	period_high = int(np.ceil(period))
	print(period_low, period_high)

	index = 1*period_high
	flag = False
	period_now = period_high
	while index < (data.size-delta):
		x = (data[index-period_now]>=0) == (data[index-period_now//2]>=0)
		y = (data[index]>=0) == (data[index-period_now//2]>=0)
		
		if x and not(y):
			index += delta
		elif not(x) and y:
			index -= delta
		
		bit_out = np.append(bit_out, np.floor(data[index]+0.5))
		bit_index = np.append(bit_index, index)

		period_now = period_high if flag else period_low
		index += period_now
		flag = ~flag

	return bit_out, bit_index

