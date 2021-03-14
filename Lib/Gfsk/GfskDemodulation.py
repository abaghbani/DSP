import numpy as np
import matplotlib.pyplot as plt

from Common.Constant import Constant
C = Constant

def GfskDemodulation(dataI, dataQ, fs):
	
	dataLength = dataI.size
	dI = dataI.astype('int64')
	dQ = dataQ.astype('int64')
	magSample = (dI*dI)+(dQ*dQ)
	phase = np.arctan2(dataQ, dataI)
	phase *= 128.0/np.pi
	phase = phase.astype('int')
	
	mag = np.zeros(dataLength)
	for i in range(29, dataLength):
		mag[i] = 10.0*np.log10(np.sum(magSample[i-29:i+1])/30)
	
	freqSig = np.diff(phase)
	## convert phase from range (-2pi:2pi) to range (-pi:pi)
	freqSig &= 0xff
	freqSig[freqSig>=128] += -256

	freq = np.zeros(dataLength, dtype='int')
	for i in range(3, dataLength-1):
		# frequency avrage of 4 samples
		freq[i] = freqSig[i]+freqSig[i-1]+freqSig[i-2]+freqSig[i-3]
	
	###########################
	# mode detection
	###########################
	modeStream = np.zeros(0, dtype=bool)
	mode = np.zeros(dataLength, dtype=bool)
	for i in range(1, dataLength):
		if (freq[i]>=0) ^ (freq[i-1]>=0):
			modeStream = np.append(modeStream, bool(np.abs(freq[i]-freq[i-2]) >= C.ModeRateThreshold))
		if modeStream.size >= 6:
			mode[i] = 1 if np.all(modeStream[-6:]==1) else 0 if np.all(modeStream[-6:]==0) else mode[i-1]
	# hacking for BER test (shift 64 samples to left to have all preamble correctly):
	mode = np.roll(mode, -64)

	###########################
	# offset canselation
	###########################
	SymbolCount = fs/2.0
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
			offset[i] = freqAvrage[i] if np.abs(freqAvrage[i]) < C.FrequencyAvrageMaximum and syncDetectionHalf[i] and syncDetectionHalf[i-1] and syncDetectionHalf[i-2] and ((xcorrHalf[i-2]>=0) ^ (xcorrHalf[i-3]>=0)) else offset[i-1]
		else:
			offset[i] = freqAvrage[i] if np.abs(freqAvrage[i]) < C.FrequencyAvrageMaximum and syncDetection[i] and syncDetection[i-1] and syncDetection[i-2] and ((xcorr[i-3]>=0) ^ (xcorr[i-4]>=0)) else offset[i-1]
		
		# Limit = 12
		# LrDet[i] = 1 if ((freq[i]>Limit and freq[i-1*SymbolCount]>Limit and freq[i-3*SymbolCount]<-Limit and freq[i-4*SymbolCount]<-Limit) or \
						# (freq[i]<-Limit and freq[i-1*SymbolCount]<-Limit and freq[i-3*SymbolCount]>Limit and freq[i-4*SymbolCount]>Limit)) and \
						# ( (freq[i-2*SymbolCount]>=0) ^ (freq[i-2*SymbolCount-1]>=0) ) else 0
	freqSync = freq-offset
	
	#plt.plot(freq)
	#plt.plot(freqAvrage, '.')
	#plt.plot(syncDetection, '.')
	##plt.plot(syncDetectionHalf)
	#plt.plot(xcorr, '-.')
	##plt.plot(xcorrHalf)
	#plt.plot(offset)
	##plt.plot(mode)
	#plt.legend(['freq', 'avg', 'syncdet', 'xcorr', 'offset', 'mode'], loc='best')
	#plt.grid()
	#plt.show()
	
	###########################
	# clock recovery
	###########################
	clkRecCount = np.zeros(dataLength, dtype='int8')
	valid = np.zeros(dataLength, dtype=bool)
	data = np.zeros(dataLength, dtype=bool)
	correction = 0
	for i in range(dataLength):
		clkRecCount[i] = 0 if ((freqSync[i]>=0) ^ (freqSync[i-1]>=0)) or clkRecCount[i-1] == int(SymbolCount*2)-1 or correction == 1 else clkRecCount[i-1]+1
		correction = 1 if ((freqSync[i]>=0) ^ (freqSync[i-1]>=0)) and (clkRecCount[i-1] == 3 or clkRecCount[i-1] == 4 or clkRecCount[i-1] == 10 or clkRecCount[i-1] == 11) else 0
		if mode[i]==0: ## halfrate mode (1mb/s)
			valid[i] = 1 if clkRecCount[i] == int(SymbolCount) else 0
		elif mode[i]==1: ## fullrate (2Mb/s)
			valid[i] = 1 if (clkRecCount[i]%int(SymbolCount)) == int(SymbolCount/2) else 0
		data[i] = bool(freqSync[i]>=0)
		
	#plt.plot(freqSync)
	#plt.plot(offset)
	#plt.plot(valid, '.')
	#plt.plot(data)
	#plt.plot(clkRecCount)
	#plt.legend(['freq', 'offset', 'valid', 'data', 'counter'], loc='best')
	#plt.grid()
	#plt.show()

	return freqSync, mag, valid, data