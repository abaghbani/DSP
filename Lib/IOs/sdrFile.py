import numpy as np

def readSdrFile(fileName):
	adcData = np.fromfile(fileName, dtype='int16')
	## data order is : I0, Q0, I1, Q0 (samples are 16-bits with 12-bits meaningful, it means samples are between +-2**11)
	print('ADC Data: {} samples'.format(adcData.size))
	print('ADC Data Min/Max: ',adcData.min(),adcData.max(), type(adcData[0]))
	return np.array([adcData[0::4]+1j*adcData[1::4], adcData[2::4]+1j*adcData[3::4]])

def writeSdrFile(fileName, data):
	writeData = np.hstack([(d0.real, d0.imag, d1.real, d1.imag) for d0, d1 in zip(data[0], data[1])])
	fp = np.memmap(fileName, mode='w+', dtype=np.dtype('<h'), shape=(1, writeData.size))
	fp[:] = writeData[:]


