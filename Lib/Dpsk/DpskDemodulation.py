import numpy as np
import matplotlib.pyplot as plt

from .Constant import *
C = Constant()

def DpskDemodulation(dataI, dataQ, fs, modulation_type):
	
	dataLength = dataI.size
	SymbolCount = fs/SymbolRate(modulation_type)
	# print SymbolCount
	
	dI = dataI.astype('int64')
	dQ = dataQ.astype('int64')
	magSample = (dI*dI)+(dQ*dQ)
	angle = np.arctan2(dataQ, dataI)
	angle = (angle*256.0/np.pi).astype(int)
	angle = angle.astype('int16')
	
	##################################
	# phase calculation:
	##################################
	mag = np.zeros(dataLength)
	for i in range(29, dataLength):
		mag[i] = 10.0*np.log10(np.sum(magSample[i-29:i+1])/30)
	
	angleMid = np.zeros(dataLength, dtype='int16')
	for i in range(1, dataLength):
		angleMid[i] = (angle[i-1]+angle[i])/2
		if np.abs(angle[i-1]) > 128 and np.abs(angle[i]) > 128 and (angle[i-1]>=0)^(angle[i]>=0):
			if np.abs(angle[i-1]) > np.abs(angle[i]):
				angleMid[i] += ((angle[i]>=0)*2-1)*256
			else:
				angleMid[i] += ((angle[i-1]>=0)*2-1)*256
	
	phase = np.zeros(dataLength, dtype='int16')
	for i in range(int(SymbolCount), dataLength):
		if int(SymbolCount) == SymbolCount:
			phase[i] = angle[i]-angle[i-int(SymbolCount)]
		else:
			phase[i] = angle[i]-angleMid[i-int(SymbolCount)]
	## convert phase from range (-2pi:2pi) to range (-pi:pi)
	phase &= 0x1ff
	phase[phase>=256] += -512
	
	plt.plot(mag)
	plt.plot(phase)
	plt.legend(['mag', 'phase'], loc='best')
	plt.grid()
	plt.show()

	# ##################################
	# # sync detection and offset canselation
	# ##################################
	
	freqAvrage = np.zeros(dataLength, dtype='int16')
	xcorr = np.zeros(dataLength, dtype='int16')
	syncDet = np.zeros(10, dtype=bool)
	offset = np.zeros(dataLength, dtype='int16')
	syncEnable = np.zeros(dataLength, dtype=bool)
	
	for i in range(int(9*SymbolCount), dataLength):
		for j in range(10):
			syncDet[j] = np.abs(phase[i-int((9-j)*SymbolCount)]-(C.DpskSync[j]*64)) <= 32
		xcorr[i] = phase[i-int(5*SymbolCount)]-phase[i-int(6*SymbolCount)]+phase[i-int(7*SymbolCount)]-phase[i-int(8*SymbolCount)]
		freqAvrage[i] = sum(phase[i-int(8*SymbolCount):i-int(4*SymbolCount)])/int(4*SymbolCount)
		if np.all(syncDet) and ((xcorr[i-int(SymbolCount/2)]>=0) ^ (xcorr[i-int(SymbolCount/2)-1]>=0)):
			print('Sync is detected, sample number = ',i)
			offset[i:] = np.full_like(offset[i:], freqAvrage[i])
			syncEnable[i] = 1
			break
	
	phaseSync = phase - offset
	
	#plt.plot(phase)
	#plt.plot(freqAvrage)
	#plt.plot(syncEnable)
	#plt.plot(xcorr)
	#plt.plot(offset)
	#plt.legend(['phase', 'avg', 'sync', 'xcorr', 'offset'], loc='best')
	#plt.grid()
	#plt.show()
	
	# ##################################
	# # clock recovery
	# ##################################
	
	correction = np.zeros(dataLength, dtype='int16')
	phaseDiff = np.zeros(dataLength, dtype='int16')
	clkRecCount = np.zeros(dataLength, dtype='int16')
	valid = np.zeros(dataLength, dtype='int8')
	data = np.zeros(dataLength, dtype='int8')
	phaseData = np.zeros(dataLength, dtype='int16')
	data2 = np.zeros(dataLength, dtype='int8')

	for i in range(1, dataLength-1):
		phaseDiff[i] = phaseSync[i+1]-phaseSync[i-1]
		correction[i] = 1 if np.abs(phaseDiff[i]) >= np.abs(phaseDiff[i-1]) else -1
		clkRecCount[i] =	int((SymbolCount+1)*32) if syncEnable[i] == 1 else \
							clkRecCount[i-1]-int((SymbolCount-1)*32)+correction[i] if clkRecCount[i-1] >= int(SymbolCount+0.5)*32 else \
							clkRecCount[i-1]+32
		valid[i] = 1 if clkRecCount[i] >= int(SymbolCount+0.5)*32 else 0
		
		#clkRecCount[i] = 0 if (clkRecCount[i-1] == 14) else 4 if syncEnable[i] else clkRecCount[i-1]+1
		#valid[i] = 1 if (clkRecCount[i] == 4 or clkRecCount[i] == 11) else 0
		
		data[i] = phaseSync[i]>>5
		
		## for debugging
		if valid[i] == 1:
			phaseData[i] = phaseSync[i]
			data2[i] = C.TableRxPhase4DQPSK[(data[i]+16)%16]
	
	plt.plot(phaseSync)
	#plt.plot(phaseDiff, '-o')
	#plt.plot(clkRecCount)
	#plt.plot(correction)
	plt.plot(syncEnable)
	plt.plot(valid)
	plt.plot(phaseData, '.')
	plt.plot(data2, '.')
	#plt.legend(['phase', 'diff', 'count', 'sync', 'valid'], loc='best')
	plt.grid()
	plt.show()
	
	
	return mag, syncEnable, valid, data

def CompareData(txData, rssi, sync, valid, data, modType):
	dataLength = data.size
	detected = np.zeros(dataLength, dtype=bool)
	demodData = np.zeros(0, dtype='int')
	demodDataRaw = np.zeros(0, dtype='int')
	for i in range(1, dataLength):
		if sync[i] == 1:
			detected[i] = 1
		else:
			detected[i] = detected[i-1]
	
		if detected[i] == 1 and valid[i] == 1 and sync[i] == 0:
			if modType == C.DpskModulationType.Edr2:
				demodData = np.append(demodData, C.TableRxPhase4DQPSK[(data[i]+16)%16]) 
			else:
				demodData = np.append(demodData, C.TableRxPhase8DQPSK[(data[i]+16)%16]) 
			demodDataRaw = np.append(demodDataRaw, data[i]) 

	print('received bit number=',demodData.size)
	if demodData.size >= txData.size:
		ber = 0
		errorIndex = []
		for i in range(txData.size):
			if demodData[i] != txData[i]:
				ber += 1
				errorIndex.append(i)
				# print 'error data in: ', i, txData[i], demodData[i], demodDataRaw[i], ber
		print('test is done and BER={0}/{1}'.format(ber, txData.size))
		print('first index: ',errorIndex[:20])
		print('last index: ',errorIndex[-20:])
	else:
		print('Not enough data is received')
