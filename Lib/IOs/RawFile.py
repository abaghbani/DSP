import numpy as np

def readRawFile(fileName):
	adcData = np.memmap(fileName, mode='r', dtype=np.dtype('<h'))
	adcData = (adcData<<(16-12))>>(16-12)
	print('ADC Data: {} samples'.format(adcData.size))
	print('ADC Data Min/Max: ',adcData.min(),adcData.max(), type(adcData[0]))
	return adcData

def writeRawFile(fileName, data):
	fp = np.memmap(fileName, mode='w+', dtype=np.dtype('<h'), shape=(1, data.size))
	fp[:] = data[:]

