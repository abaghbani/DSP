import numpy as np
import scipy.signal as signal

from Spectrum.Constant import Constant
from Spectrum.ModemLib import ModemLib

C = Constant
myLib = ModemLib(0)

def RfTransceiver(baseband, basebandFS, basebandBW, channel, SNRdb = 25, ch_res=False):
	pi= np.pi
	adcFS = C.AdcSamplingFrequency
	
	## upsampling and flitering
	basebandUpsampled = (basebandBW*adcFS/basebandFS)*baseband.repeat(adcFS/basebandFS)
	b = signal.remez(140+1, [0, basebandBW, np.min([basebandFS/2-1, 20]), (adcFS/2)], [1,0], fs=adcFS)
	basebandFlt = signal.lfilter(b, 1, basebandUpsampled)
	
	## mix to correct channel and then mix to have IF signal
	IfSignal = basebandFlt*np.exp((np.arange(basebandFlt.size)*2j*pi*(C.IfMixerFrequency+channel)/adcFS))
	# IfSignal = basebandFlt*np.exp((np.arange(basebandFlt.size)*2j*pi*(C.IfMixerFrequency+channel)/adcFS)+(6j*pi/8))
	
	# adding white noise (some zero at the beginning to pass preamble/sync without noise)
	signal_power = np.mean(abs(IfSignal**2))
	sigma2 = signal_power * 10**(-SNRdb/10)  # calculate noise power based on signal power and SNR
	print ("RX Signal power: %.4f, Noise power: %.4f" % (signal_power, sigma2))
	noiseSignal = np.sqrt(sigma2/2) * (np.random.randn(basebandFlt.size)-0.5)
	# noiseSignal = np.sqrt(sigma2/2) * np.concatenate((np.zeros((50+16)*int(adcFS/basebandBW)), np.random.rand(basebandFlt.size-(50+16)*int(adcFS/basebandBW))-0.5), axis=None)
	# myLib.fftPlot(IfSignal.real, noiseSignal, n=2)
	
	# adding wireless channel response to the main signal
	if ch_res == True:
		channelResponse = np.array([1, 0, 0.3+0.3j])  # the impulse response of the wireless channel
		IfSignal = np.convolve(IfSignal, channelResponse)

	return IfSignal.real+noiseSignal
