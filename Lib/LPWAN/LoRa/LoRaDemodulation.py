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

def LoRaFilterBank(dataI, dataQ, fs, Bw, fMix, downSamplingRate):
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

def FrequencyCalculation(dataI, dataQ, fs):
	freq = np.diff(np.arctan2(dataQ, dataI))
	freq[freq>np.pi] -= 2*np.pi
	freq[freq<-np.pi] += 2*np.pi
	freq *= fs/(2*np.pi)
	return freq

def LoRaChirpDetection(freq, fs):

	len = freq.size
	freqAvg = np.zeros(len, dtype=float)

	for i in range(19, len):
		freqAvg[i] = np.sum(freq[i-19:i+1])/20
	
	plt.plot(freq)
	plt.plot(freqAvg)
	#plt.plot(np.diff(freqAvg))
	plt.show()

	#fmax = 0.0
	#fmaxIndex = 0
	#fmin = 0.0
	#fminIndex = 0
	#for i in range(freq.size):
	#	if(freq[i]>fmax):
	#		fmax = freq[i]
	#		fmaxIndex = i
	#		print('max=', fmaxIndex, fmax)
	#	if(freq[i]<fmin):
	#		fmin = freq[i]
	#		fminIndex = i
	#		print('min=', fminIndex, fmin)
	#	if(fmaxIndex-fminIndex < 200 and fmaxIndex-fminIndex > 20):
	#		fmin = 1e10
	#		fmax = -1e10
	#		print('reset')


	startSample = 8525
	symbolLength = 315

	fHigh = 10400.0
	fLow = -10400.0
	step = (fHigh-fLow)/symbolLength

	baseChirp = np.exp(-2j*np.pi*np.cumsum(np.arange(fLow, fHigh, step)/fs))


	return startSample, symbolLength, baseChirp

def LoRaDemodulation(dataI, dataQ, fs, Bw, SF):
	
	freq = FrequencyCalculation(dataI, dataQ, fs)
	plt.plot(freq)
	plt.show()

	#[startSample, symbolLength, baseChirp] = LoRaChirpDetection(freq, fs)
	#symbolTime = int(np.exp2(SF) / Bw * fs)
	#symTime = 315 # (2**6)/20800 * 1024000
	##start1 = 4822
	#start0 = 4680
	#startK = 8525
	#n = np.arange(symTime)
	#dechirp = np.exp((1j*2*np.pi*n*2000/fs)+0.012)
	#test1 = np.zeros(40*symbolLength, dtype=complex)
	
	#symTime = int(np.exp2(SF)*(Bw/fs))
	symTime = int(np.exp2(SF))
	start0 = 1020
	startK = int(7*symTime + start0)
	startP = int(9.25*symTime + start0)
	test1 = np.zeros(10*symTime, dtype=complex)
	for i in range(2):
		test1[i*symTime:(i+1)*symTime] = np.multiply(dataI[i*symTime+startK:(i+1)*symTime+startK]+1j*dataQ[i*symTime+startK:(i+1)*symTime+startK], dataI[start0:start0+symTime]-1j*dataQ[start0:start0+symTime])
	for i in range(2,10):
		test1[i*symTime:(i+1)*symTime] = np.multiply(dataI[i*symTime+startP:(i+1)*symTime+startP]+1j*dataQ[i*symTime+startP:(i+1)*symTime+startP], dataI[start0:start0+symTime]-1j*dataQ[start0:start0+symTime])
	
	#symTime = int(np.exp2(SF))*2
	#start0 = 2216
	#startK = int(9.25*symTime + start0)
	#test1 = np.zeros(5*symTime, dtype=complex)
	#for i in range(5):
	#	test1[i*symTime:(i+1)*symTime] = np.multiply(dataI[i*symTime+startK:(i+1)*symTime+startK]+1j*dataQ[i*symTime+startK:(i+1)*symTime+startK], dataI[start0:start0+symTime]-1j*dataQ[start0:start0+symTime])
	
	
	#for i in range(10):
	#	test1[i*symbolLength:(i+1)*symbolLength] = np.multiply(dataI[i*symbolLength+startSample:(i+1)*symbolLength+startSample]+1j*dataQ[i*symbolLength+startSample:(i+1)*symbolLength+startSample], baseChirp)
	#for i in range(10):
	#	test1[i*symTime:(i+1)*symTime] = np.multiply(dataI[i*symTime+start0:(i+1)*symTime+start0]+1j*dataQ[i*symTime+start0:(i+1)*symTime+start0], dataI[start0:start0+symTime]-1j*dataQ[start0:start0+symTime])
	#for i in range(10,40):
	#	test1[i*symTime:(i+1)*symTime] = np.multiply(dataI[i*symTime+startK:(i+1)*symTime+startK]+1j*dataQ[i*symTime+startK:(i+1)*symTime+startK], dataI[start0:start0+symTime]-1j*dataQ[start0:start0+symTime])
		
	#plt.plot(test1.real)
	#plt.plot(test1.imag)
	#plt.show()
	#myLib.fftPlot(test1, fs=fs)
	#plt.specgram(test1)
	#plt.show()
	
	freq = FrequencyCalculation(test1.imag, test1.real, fs)
	freq2 = np.zeros(freq.size)

	for i in range(freq.size):
		if freq[i] >= 0:
			freq2[i] = Bw/2 - freq[i]
		else:
			freq2[i] = -Bw/2 - freq[i]

	rxData = freq2/Bw * np.exp2(SF)
	rxData[rxData<0] += np.exp2(SF)
	dataOut = np.zeros(0, dtype=int)
	for i in range(int(symTime/2), freq.size, symTime):
		dataOut = np.append(dataOut, int(rxData[i]))
	print(dataOut)
	
	plt.plot(freq)
	plt.plot(freq2)
	plt.legend(['freq', 'freq2'], loc='best')
	plt.show()
