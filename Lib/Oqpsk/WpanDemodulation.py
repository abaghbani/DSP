import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

def WpanDemodulation(dataI, dataQ, fs=30, symboleRate=1.0):
	
	dataLength = dataI.size
	SymbolCount = float(fs/symboleRate)
	# print SymbolCount
	
	# dataI = np.roll(dataI, 15)
	# dataQ = np.roll(dataQ, -15)
	# dataI = signal.lfilter(np.array([0.5, 0.5]), np.array([1]), dataI)
	# dataQ = signal.lfilter(np.array([0.5, 0.5]), np.array([1]), dataQ)
	# rangestart = 800
	# rangeend = 1100
	# plt.plot(dataI[rangestart:rangeend], dataQ[rangestart:rangeend])
	# # dataI -= 500
	# # dataQ -= 500
	# # plt.plot(dataQ)
	# plt.grid()
	# plt.show()
	# return 0

	magSample = dataI*dataI+dataQ*dataQ

	phase = np.arctan2(dataQ, dataI)
	phase *= 128.0/np.pi
	phase = phase.astype('int')
	
	##################################
	# phase calculation:
	##################################
	mag = np.zeros(dataLength)
	for i in range(7, dataLength):
		mag[i] = 10.0*np.log10(np.sum(magSample[i-7:i])) if np.sum(magSample[i-7:i]) > 1.0e-2 else mag[i-1]
	
	phaseDiff = np.diff(phase)
	## convert phase from range (-2pi:2pi) to range (-pi:pi)
	phaseDiff &= 0xff
	phaseDiff[phaseDiff>=128] += -256

	freq = np.zeros(dataLength, dtype='int')
	for i in range(3, dataLength-1):
		# frequency avrage of 4 samples
		freq[i] = phaseDiff[i]+phaseDiff[i-1]+phaseDiff[i-2]+phaseDiff[i-3]
	
	# ##################################
	# # sync detection and offset canselation
	# ##################################
	
	# syncDetection = np.zeros(dataLength, dtype='int32')
	# syncDet = np.zeros(6, dtype=bool)
	# for i in range(int(15*SymbolCount), dataLength):
	# 	for j in range(syncDet.size):
	# 		syncDet[j] = np.abs(phase[i-int((syncDet.size-1-j)*SymbolCount)]-(C.WpanPN0Phase[j]*64)) <= 64
	# 	if np.all(syncDet):
	# 		syncDetection[i] = 200
	# 		print "Sync is detected, sample number = ",i
	
	valid = 0
	data = 0
	
	plt.plot(mag)
	plt.plot(freq)
	plt.legend(['mag', 'freq'], loc='best')
	plt.grid()
	plt.show()
	
	# plt.plot(dataI)
	# plt.plot(dataQ)
	# plt.legend(['real', 'imag'], loc='best')
	# # plt.plot(basebandSig.real)
	# # plt.plot(basebandSig.imag)
	# plt.grid()
	# plt.show()
	
	# plt.plot(mag)
	# plt.plot(syncDetection)
	
	return freq, mag, valid, data
