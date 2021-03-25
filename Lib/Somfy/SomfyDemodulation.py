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

	# myLib.fftPlot(dataFltI+1j*dataFltQ, n=1, fs=fs)
	# myLib.specPlot(dataFltI+1j*dataFltQ, fs=fs)
	
	return dataFltI, dataFltQ, fs

def decod_manchester(data):
	data_out = np.empty(0, dtype='int')
	firstEdge = 0
	for i in range(data.size):
		if data[i] != data[i-1]:
			if firstEdge == 0:
				firstEdge = 1
				period = 0.0
				last_edge = i
				data_out = np.append(data_out, int(data[i-1]))
			elif period == 0 or ((i-last_edge)>(0.8*period) and (i-last_edge)<(1.2*period)):
				period =i-last_edge
				last_edge = i
				data_out = np.append(data_out, int(data[i-1]))
			elif (i-last_edge)>(1.5*period):
				period =i-last_edge
				data_out = np.empty(0, dtype='int')
				print('reset detection = ',i)
				i = 0
	return data_out


def SomfyDemodulation(dataI, dataQ, fs, Bw, SF):
	
	# dI = dataI.astype('int64')
	# dQ = dataQ.astype('int64')
	dI = dataI
	dQ = dataQ
	magSample = (dI*dI)+(dQ*dQ)
	
	plt.plot(magSample)
	plt.show()

	data = magSample[73000: 180000]
	print('max data is: ', data.max())
	high_threshold = int(0.5*data.max())
	data[data<high_threshold] = 0
	data[data!=0] = 1
	dataout = decod_manchester(data)
	print('data extract is: ')
	print(dataout)
	plt.plot(data)
	# plt.legend(['raw', 'avg'], loc='best')
	plt.show()
	
	plt.plot(dataout)
	plt.show()
	