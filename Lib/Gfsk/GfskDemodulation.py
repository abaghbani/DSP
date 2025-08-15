import numpy as np
import matplotlib.pyplot as plt

import Spectrum as sp
import ClockRecovery as cr

from .Constant import *
C = Constant()

def ModeDetection(freq, fs):
	if fs==7.5:
		return np.zeros(freq.size, dtype=np.uint8)
	else:
	mode = np.zeros(6, dtype=np.uint8)
	mode_avg = np.zeros(freq.size, dtype=np.uint8)
	for i in range(2, freq.size):
		if (freq[i]>=0) != (freq[i-1]>=0):
			mode = np.roll(mode, 1)
			mode[0] = (np.abs(freq[i]-freq[i-2]) >= C.ModeRateThreshold)
		mode_avg[i] = 1 if np.all(mode == 1) else 0 if np.all(mode == 0) else mode_avg[i-1]

	return mode_avg

def SignalDetection(freq):
	level = np.array([(np.abs(val)<50) for val in freq])
	
	mode = np.zeros(6, dtype=np.uint8)
	detection = np.zeros(freq.size, dtype=np.uint8)
	for i in range(19, freq.size):
		if (freq[i]>=0) != (freq[i-1]>=0):
			mode = np.roll(mode, 1)
			mode[0] = (np.abs(freq[i]-freq[i-2]) <= 22)
		detection[i] = 1 if np.all(level[i-19:i+1] == 1) and np.all(mode==1) else 0 if np.all(level[i-2:i+1] == 0) or np.all(mode==0) else detection[i-1]
	
	return detection

def FrequencyOffsetCalc(freq, mode, Fs, plot_enable=False):
	dataLength = freq.size
	SymbolCount = Fs/2.0
	print("SymbolCount: ", SymbolCount, "Fs: ", Fs)
	freqAvrage = np.zeros(dataLength, dtype='int')
	syncDetection = np.zeros(dataLength, dtype=bool)
	syncDetectionHalf = np.zeros(dataLength, dtype=bool)
	xcorr = np.zeros(dataLength, dtype='int')
	xcorrHalf = np.zeros(dataLength, dtype='int')
	offset = np.zeros(dataLength, dtype='int')
	# LrDet = np.zeros(dataLength, dtype=bool)
	for i in range(int(4*SymbolCount), dataLength):
		freqAvrage[i] = sum(freq[i-int(4*SymbolCount):i])/int(4*SymbolCount)
		syncDetection[i] = 1 if (freq[i]>0 and freq[i-int(1*SymbolCount)]<0 and  freq[i-int(2*SymbolCount)]>0 and  freq[i-int(3*SymbolCount)]<0 and  freq[i-int(4*SymbolCount)]>0) or \
								(freq[i]<0 and freq[i-int(1*SymbolCount)]>0 and  freq[i-int(2*SymbolCount)]<0 and  freq[i-int(3*SymbolCount)]>0 and  freq[i-int(4*SymbolCount)]<0) else 0
		
		syncDetectionHalf[i] = 1 if (freq[i]>0 and freq[i-int(2*SymbolCount)]<0 and  freq[i-int(4*SymbolCount)]>0 and  freq[i-int(6*SymbolCount)]<0 and  freq[i-int(8*SymbolCount)]>0) or \
								(freq[i]<0 and freq[i-int(2*SymbolCount)]>0 and  freq[i-int(4*SymbolCount)]<0 and  freq[i-int(6*SymbolCount)]>0 and  freq[i-int(8*SymbolCount)]<0) else 0
		
		xcorr[i]     = freq[i]-freq[i-int(1*SymbolCount)]+freq[i-int(2*SymbolCount)]-freq[i-int(3*SymbolCount)]
		xcorrHalf[i] = freq[i]-freq[i-int(2*SymbolCount)]+freq[i-int(4*SymbolCount)]-freq[i-int(6*SymbolCount)]
		
		if mode[i] == 0:
			offset[i] = freqAvrage[i] if np.abs(freqAvrage[i]) < C.FrequencyAvrageMaximum and syncDetectionHalf[i] and syncDetectionHalf[i-1] and syncDetectionHalf[i-2] and ((xcorrHalf[i-4]>=0) ^ (xcorrHalf[i-5]>=0)) else offset[i-1]
		else:
			offset[i] = freqAvrage[i] if np.abs(freqAvrage[i]) < C.FrequencyAvrageMaximum and syncDetection[i] and syncDetection[i-1] and syncDetection[i-2] and ((xcorr[i-4]>=0) ^ (xcorr[i-5]>=0)) else offset[i-1]
		
		# Limit = 12
		# LrDet[i] = 1 if ((freq[i]>Limit and freq[i-1*SymbolCount]>Limit and freq[i-3*SymbolCount]<-Limit and freq[i-4*SymbolCount]<-Limit) or \
						# (freq[i]<-Limit and freq[i-1*SymbolCount]<-Limit and freq[i-3*SymbolCount]>Limit and freq[i-4*SymbolCount]>Limit)) and \
	
	if plot_enable:
		plt.plot(freq)
		plt.plot(freqAvrage, '.')
		# plt.plot(syncDetection*10)
		plt.plot(syncDetectionHalf)
		# plt.plot(xcorr)
		plt.plot(xcorrHalf)
		plt.plot(offset)
		plt.plot(mode)
		plt.legend(['freq', 'avg', 'syncdet', 'xcorr', 'offset', 'mode'], loc='best')
		plt.grid()
		plt.show()
					# ( (freq[i-2*SymbolCount]>=0) ^ (freq[i-2*SymbolCount-1]>=0) ) else 0
	return offset

def phase_correction(phase):
	return (phase+np.pi)%(2*np.pi)-np.pi

def Demodulation(data, Fs):
	
	#sp.fftPlot(data, fs = Fs)
	mag = np.abs(data)
	mag = np.array([20.0*np.log10(np.sum(mag[i-29:i+1])/30) for i in range(29, mag.size)])
	mag = np.insert(mag, 0, [0]*28)
	
	phase = np.angle(data)
	freq = phase_correction(np.diff(phase))*128.0/np.pi
	#phase = (np.arctan2(data.imag, data.real) * 128.0/np.pi).astype(np.int8)
	#freq = np.diff(phase)
	
	# avraging over 4 samples
	freq = np.convolve([1, 1, 1, 1], freq, 'same')

	mode = ModeDetection(freq, Fs)
	present = SignalDetection(freq)
	present_mag = (mag>55)

	offset = FrequencyOffsetCalc(freq, mode, Fs, False)
	freq_sync = freq - offset
	bit, bit_index = cr.EarlyLate(freq_sync, Fs)
	bit = np.floor(bit+0.5)
	
	plt.plot(bit_index, bit, 'bo')
	plt.plot(freq)
	plt.plot(mode)
	plt.plot(present*present_mag*100)
	#plt.legend(['bit', 'freq'], loc='best')
	plt.grid()
	plt.show()
	bit = np.where(bit>=0, 1, -1)

	return freq, mag, bit

class unused:
	def ClockRecovery(freq, mode, Fs):
		dataLength = freq.size
		SymbolCount = Fs/2.0
		clkRecCount = np.zeros(dataLength, dtype='int8')
		valid = np.zeros(dataLength, dtype=bool)
		data = np.zeros(dataLength, dtype=np.int8)
		correction = 0
		for i in range(dataLength):
			clkRecCount[i] = 0 if ((freq[i]>=0) ^ (freq[i-1]>=0)) or clkRecCount[i-1] == int(SymbolCount*2)-1 or correction == 1 else clkRecCount[i-1]+1
			correction = 1 if ((freq[i]>=0) ^ (freq[i-1]>=0)) and (clkRecCount[i-1] == 3 or clkRecCount[i-1] == 4 or clkRecCount[i-1] == 10 or clkRecCount[i-1] == 11) else 0
			if mode[i]==0: ## halfrate mode (1mb/s)
				valid[i] = 1 if clkRecCount[i] == int(SymbolCount) else 0
			elif mode[i]==1: ## fullrate (2Mb/s)
				valid[i] = 1 if (clkRecCount[i]%int(SymbolCount)) == int(SymbolCount/2) else 0
			data[i] = (freq[i]>=0)
		
		rxData = data[np.nonzero(valid)]
		print(rxData)
			
		plt.plot(freq)
		#plt.plot(offset)
		#plt.plot(valid, '.')
		#plt.plot(data)
		#plt.plot(clkRecCount)
		#plt.legend(['freq', 'offset', 'valid', 'data', 'counter'], loc='best')
		plt.grid()
		plt.show()

		return rxData

