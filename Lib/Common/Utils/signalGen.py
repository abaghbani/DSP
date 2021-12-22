import sys
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

sys.path.insert(1, sys.path[0]+'\..')
from ModemLib			import ModemLib
myLib = ModemLib(0)

from Constant			import Constant
C = Constant

pi = np.pi
fs = 240.0		# sampling frequency = 240Ms/s
f = np.arange(20,100)	# frequency is from 22MHz to 101 MHz, so 80 channel is mapped to this range
n = np.arange(10*fs)

data = np.zeros((f.size,n.size), dtype=complex)
gain = np.random.rand(f.size)/2.0+0.5
for i in range(f.size):
	data[i] = gain[i]*np.exp((2j*pi*n*f[i]/fs))

# myLib.fftPlot(data[0], data[1], n=2, fs=fs)
# myLib.fftPlot(data[2], data[3], n=2, fs=fs)

dataTx = sum(data[m] for m in range(f.size))
myLib.fftPlot(dataTx.real, dataTx.imag, n=2, fs=fs)
# plt.plot(dataTx.imag)
# plt.show()

IfSignal = dataTx*np.exp(2j*pi*n*(-120.0)/fs)
transmitSig = IfSignal.real*(2**4)
adcData = transmitSig.astype('int16')
fp = np.memmap('./Samples/noiseGen.bttraw', mode='w+', dtype=np.dtype('<h'),shape=(1,adcData.size))
fp[:] = adcData[:]
