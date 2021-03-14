import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

from Common.Constant import Constant
from Common.ModemLib import ModemLib

C = Constant
myLib = ModemLib(0)

def OfdmRfTransceiver(baseband, basebandFS, basebandBW, channel, SNRdb = 25, ch_res=False):
	pi= np.pi
	adcFS = C.AdcSamplingFrequency
	
	## upsampling and flitering
	# upsampling is implemented before OFDM (upsampling in frequency domain)
	basebandFlt = baseband

	## mix to correct channel and then mix to have IF signal
	IfSignal = basebandFlt*np.exp((np.arange(basebandFlt.size)*2j*pi*(C.IfMixerFrequency+channel)/adcFS))
	# IfSignal = basebandFlt*np.exp((np.arange(basebandFlt.size)*2j*pi*(C.IfMixerFrequency+channel)/adcFS)+(6j*pi/8))
	
	# adding white noise (some zero at the beginning to pass preamble/sync without noise)
	signal_power = np.mean(abs(IfSignal**2))
	sigma2 = signal_power * 10**(-SNRdb/10)  # calculate noise power based on signal power and SNR
	print ("RX Signal power: %.4f, Noise power: %.4f" % (signal_power, sigma2))
	noiseSignal = np.sqrt(sigma2/2) * (np.random.randn(basebandFlt.size)-0.5)
	# noiseSignal = np.sqrt(sigma2/2) * np.concatenate((np.zeros((50+16)*int(adcFS/basebandBW)), np.random.rand(basebandFlt.size-(50+16)*int(adcFS/basebandBW))-0.5), axis=None)
	
	#myLib.fftPlot(baseband.real, basebandFlt.real, n=2)
	#myLib.fftPlot(baseband.real, baseband.imag, n=2)
	#myLib.fftPlot(basebandFlt.real, basebandFlt.imag, n=2)
	#myLib.fftPlot(IfSignal.real, IfSignal.imag, n=2)
	#myLib.fftPlot(IfSignal.real, noiseSignal, n=2)
	
	# adding wireless channel response to the main signal
	if ch_res == True:
		channelResponse = np.array([1, 0, 0.3+0.3j])  # the impulse response of the wireless channel
		IfSignal = np.convolve(IfSignal, channelResponse)

	return IfSignal.real+noiseSignal
