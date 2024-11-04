import numpy as np
import scipy.io as sio

def readMatFile(fileName, streamI, streamQ):
	readdata = sio.loadmat(fileName)
	print(readdata.keys())
	print(readdata['__header__'])
	dataI = np.hstack(readdata[streamI])
	dataQ = np.hstack(readdata[streamQ])
	fs = 1 # fixme!!! still is not clear how to read, this info is not available in mat file.
	return (dataI, dataQ, fs)


