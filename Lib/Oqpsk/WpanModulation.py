import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

from Oqpsk.Constant import Constant as C

def WpanModulation(payload):
	pi= np.pi
	overSampling = 30
	# Convert input data to chirp (evry 4 bit convert to 32-bits chirp PN[0:15])
	chirpStream = np.zeros(0)
	for i in range(payload.size):
		chirpStream = np.concatenate((chirpStream, C.WpanPN[int(payload[i])]))
	symbolStream = np.concatenate((C.WpanPN[0]*10, chirpStream), axis=None)

	##################################
	## Modulation
	##################################
	n = np.arange(overSampling)
	pt = np.sin(pi*n/overSampling)
	dataI = np.zeros(overSampling*20)
	dataQ = np.zeros(overSampling*20)
	for i in range(symbolStream.size//2):
		dataI = np.concatenate((dataI, (1 if symbolStream[2*i]==1   else -1)*pt), axis=None)
		dataQ = np.concatenate((dataQ, (1 if symbolStream[2*i+1]==1 else -1)*pt), axis=None)

	dataI = np.concatenate((dataI, np.zeros(50*overSampling)), axis=None)
	dataQ = np.concatenate((dataQ, np.zeros(50*overSampling)), axis=None)

	## imply Offset-QPSK
	basebandSig = dataI+1j*np.roll(dataQ, overSampling//2)
	
	fs = overSampling * (C.WpanChirpBitRate/2)  # chirp is divided to two data (I and Q)
	bw = 2.0	# baseband bandwidth is 2MHz
		
	##################################
	## Frequency offset and drift (BUGBUG: this approach for o-qpsk is not correct)
	##################################
	# offset+drift should be less than +-75KHz
	# drift should be less than 10KHz
	#offset = 1.0e3/1.0e6
	#drift = 5.0e3/1.0e6
	#frequencyOffset = offset+drift*np.linspace(0, 1, basebandSig.size)
	#baseband = basebandSig*np.exp(1j*2*pi*np.cumsum(frequencyOffset/fs))
	baseband = basebandSig

	#plt.plot(basebandSig.real)
	#plt.plot(basebandSig.imag)
	#plt.plot(baseband.real)
	#plt.plot(baseband.imag)
	#plt.legend(['real', 'imag', 'realOut', 'imagOut'], loc='best')
	#plt.grid()
	#plt.show()

	#plt.plot(baseband.real, baseband.imag)
	#plt.title('Constellation diagram of Offset-QPSK')
	#plt.grid()
	#plt.show()

	return baseband, fs, bw
