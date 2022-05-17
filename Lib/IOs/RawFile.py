import numpy as np

def readRawFile(fileName, bitvalid_number=12):
	#adcData = np.memmap(fileName, mode='r', dtype=np.dtype('<h'))
	adcData = np.fromfile(fileName, dtype='int16')
	adcData = (adcData<<(16-bitvalid_number))>>(16-bitvalid_number)
	print('ADC Data: {} samples'.format(adcData.size))
	print('ADC Data Min/Max: ',adcData.min(),adcData.max(), type(adcData[0]))
	return adcData

def writeRawFile(fileName, data):
	fp = np.memmap(fileName, mode='w+', dtype=np.dtype('<h'), shape=(1, data.size))
	fp[:] = data[:]


