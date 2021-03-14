import numpy as np
import scipy.signal as signal

from Common.Constant import Constant
from Common.ModemLib import ModemLib

C = Constant
myLib = ModemLib(0)

def OfdmChannelFilter(adcData, channel):

	pi= np.pi
	adcFS = C.AdcSamplingFrequency

	# input data : [-98M (2402) ... -19M (2481)],   [[2402:2481]-2260(rf_mixer)]sampling@240MHz => nyquist1 left side = -98:-19

	# mixer: (I+jQ)*exp(j*2pi*fmix*n) = [I*cos(2*pi*fmix*n)-Q*sin(2pi*fmix*n)]+j[I*sin(2pi*fmix*n)+Q*cos(2pi*fmix*n)]
	# mixed_I = I*cos(fmix) - Q*sin(fmix)
	# mixed_Q = Q*cos(fmix) + I*sin(fmix)

	# stage 0 : Fs = 240MHz, Fmix = 58.5MHz, Fc=40.2MHz channel=1 [output = -39.5M:39.5M]
	sigMix0 = adcData*np.exp((np.arange(adcData.size)*(-2j)*pi*(C.IfMixerFrequency+channel-0.016)/adcFS))*2
	
	b = signal.remez(146+1, [0, 20, 30, 120], [1,0], fs=adcFS)
	sigFlt0 = np.convolve(b, sigMix0,'same')
	fs = adcFS

	#myLib.fftPlot(adcData.real, adcData.imag, n=2, fs=240.0)
	#myLib.fftPlot(sigMix0.real, sigMix0.imag, n=2, fs=240.0)
	#myLib.fftPlot(sigFlt0.real, sigFlt0.imag, n=2, fs=240.0)
	
	return sigFlt0, fs
