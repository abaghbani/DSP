import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

from Common.Constant import Constant
from Common.ModemLib import ModemLib
C = Constant
myLib = ModemLib(0)

def payloadModulation(data, fLow, fHigh, SF, step):
	dataOut = []
	for i in range(data.size):
		if data[i]<np.exp2(SF-1):
			fStart = data[i]/np.exp2(SF)*(fHigh-fLow)
		else:
			fStart = (data[i]/np.exp2(SF)*(fHigh-fLow))-(fHigh-fLow)
		print(data[i], fStart)
		dataOut = np.append(dataOut, np.concatenate((np.arange(fStart, fHigh, step), np.arange(fLow, fStart, step)), axis=None))
	return dataOut

def LoRaModulation(payload, fLow, fHigh, SF, Fs):
	chirpRate = float((fHigh-fLow)/np.exp2(SF))
	step = chirpRate*((fHigh-fLow)/Fs)

	preamble1 = np.tile(np.arange(fLow, fHigh, step), 8)
	preamble2 = np.tile(np.arange(fLow, fHigh, step), 2)
	preamble3 = np.tile(np.arange(fHigh, fLow, -step), 2)
	preamble4 = np.arange(fHigh, 0.5*fHigh, -step)

	modulatedData = payloadModulation(payload, fLow, fHigh, SF, step)
	# plt.plot(modulatedData)
	# plt.show()
	freq = np.concatenate((np.zeros(50), preamble1, preamble2, preamble3, preamble4, modulatedData, np.zeros(50)), axis=None)
	baseband = np.exp(2j*np.pi*np.cumsum(freq/Fs))

	#plt.plot(freq)
	#plt.plot(baseband.real)
	#plt.plot(baseband.imag)
	#plt.show()

	#fftPlot(baseband)
	#plt.specgram(baseband)
	#plt.show()

	return baseband

