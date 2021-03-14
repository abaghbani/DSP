import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

from Spectrum.ModemLib import ModemLib
myLib = ModemLib(0)

######################################
# symb-time = (2**SF)/Bw
# symb-rate = Bw/(2**SF)
# bit-rate = SF*Bw/(2**SF)*CR
######################################

def SomfyFilterBank(dataI, dataQ, fs, Bw, fMix, downSamplingRate):
	n=np.arange(dataI.size)
	cosMix = np.cos((n*2*np.pi*fMix/fs)+0.06287)
	sinMix = np.sin((n*2*np.pi*fMix/fs)+0.06287)
	dataMixI = np.multiply(dataI, cosMix) - np.multiply(dataQ, sinMix)
	dataMixQ = np.multiply(dataQ, cosMix) + np.multiply(dataI, sinMix)
	myLib.fftPlot(dataMixI+1j*dataMixQ, n=1, fs=fs)

	d = signal.firwin(301, cutoff = (1.2*Bw)/(fs/2.), window = 'blackmanharris')
	dataFltI = np.convolve(d, dataMixI)
	dataFltQ = np.convolve(d, dataMixQ)
	dataFltI = dataFltI[::downSamplingRate]
	dataFltQ = dataFltQ[::downSamplingRate]
	fs /= downSamplingRate

	myLib.fftPlot(dataFltI+1j*dataFltQ, n=1, fs=fs)
	#myLib.specPlot(dataFltI+1j*dataFltQ, fs=fs)
	
	return dataFltI, -dataFltQ, fs

def SomfyDemodulation(dataI, dataQ, fs, Bw, SF):
	
	dI = dataI.astype('int64')
	dQ = dataQ.astype('int64')
	magSample = (dI*dI)+(dQ*dQ)
	
	dataLength = dataI.size
	mag = np.zeros(dataLength)
	for i in range(29, dataLength):
		mag[i] = 10.0*np.log10(np.sum(magSample[i-29:i+1])/30)
	
	mag[mag < 0] = 0
	plt.plot(magSample, '-.')
	plt.plot(mag)
	plt.legend(['raw', 'avg'], loc='best')
	plt.grid()
	plt.show()
