import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

from Ofdm.Constant import Constant as C

def OfdmQamModulation(inputStream):
	
	# wifi.Rev-g: 48  data subcarriers channels and 4 pilot channels
	allCarriers = np.arange(64)
	pilotCarriers = np.array([-21, -7, 7, 21])+32
	nullCarriers = np.array([-32, -31, -30, -29, -28, -27, 0, 27, 28, 29, 30, 31])+32
	dataCarriers = np.delete(allCarriers, np.hstack([pilotCarriers, nullCarriers]))

	#print(allCarriers.size, pilotCarriers.size, nullCarriers.size, dataCarriers.size)
	#print ("allCarriers:   %s" % allCarriers)
	#print ("pilotCarriers: %s" % pilotCarriers)
	#print ("dataCarriers:  %s" % dataCarriers)
	#print ("dataCarriersUpSample:  %s" % dataCarriersUpSample)
	#plt.plot(pilotCarriers, np.zeros_like(pilotCarriers), 'bo', label='pilot')
	#plt.plot(dataCarriers, np.zeros_like(dataCarriers), 'ro', label='data')
	#plt.plot(nullCarriers, np.zeros_like(nullCarriers), 'ko', label='null')
	#plt.show()

	fs = 64/3.2	# 64 samples during 3.2us (plus 16 samples during 0.8us as CP)
	bw = 16.875/2	# 20MHz bandwidth with some null subcarriers so we have (54/64)*20 = 16.875MHz

	##################################
	## Modulation
	##################################
	def modulationQAM16(inputData):
		chipStream = inputData[:(inputData.size//4)*4].reshape((-1,4))
		return np.array([C.QAM16_table[tuple(b)] for b in chipStream])

	modulatedData = modulationQAM16(inputStream)
	# plt.plot(modulatedData.real, modulatedData.imag, 'bo')
	# plt.grid()
	# plt.show()
	
	##################################
	## OFDM
	##################################
	def ofdm(totalSubC, pilotSubC, dataSubC, cp, UpSamplingRate, dataStream):
		ofdmData = dataStream[:(dataStream.size//dataSubC.size)*dataSubC.size].reshape([-1, dataSubC.size])
		ofdmFreq = np.zeros(totalSubC.size, dtype=complex)
		ofdmFreqUp = np.zeros(totalSubC.size*UpSamplingRate, dtype=complex)
		ofdmTime = np.zeros((ofdmData.shape[0],(totalSubC.size+cp)*UpSamplingRate), dtype=complex)
		ofdmFreq[pilotSubC] = [3+3j]
		for i in range(ofdmData.shape[0]):
			ofdmFreq[dataSubC] = ofdmData[i]
			ofdmFreqUp[int(totalSubC.size*((UpSamplingRate-1)/2)):int(totalSubC.size*((UpSamplingRate+1)/2))] = ofdmFreq
			tempData = np.fft.ifft(np.hstack([ofdmFreqUp[ofdmFreqUp.size//2:], ofdmFreqUp[:ofdmFreqUp.size//2]]))
			ofdmTime[i] = np.hstack([tempData[-cp*UpSamplingRate:], tempData])*UpSamplingRate
		return ofdmTime.reshape(-1)
	
	baseband = ofdm(allCarriers, pilotCarriers, dataCarriers, 16, int(C.AdcSamplingFrequency/fs), modulatedData)

	return baseband, C.AdcSamplingFrequency, bw
