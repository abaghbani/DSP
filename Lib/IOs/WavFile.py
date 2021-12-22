import numpy as np
import scipy.io.wavfile as wavfile

def readWaveFile(fileName):
	[fs, readdata] = wavfile.read(fileName)
	dataI = readdata[:,0]
	dataQ = readdata[:,1]
	return (fs, dataI, dataQ)

def writeWaveFile(fileName, data, fs):
	wavfile.write(fileName, fs, np.array([(data.real).astype('float'), (data.imag).astype('float')]).reshape(-1,2))

