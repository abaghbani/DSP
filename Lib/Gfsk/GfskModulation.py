import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

from Spectrum.Constant import Constant
C = Constant

def GfskModulation(payload, modType):

	pi= np.pi
	overSampling = 15
	symbolStream = np.concatenate((np.zeros(50), C.GfskPreamble if payload[0]==1 else -1*C.GfskPreamble , payload, np.zeros(100)), axis=None)

	## Upsampling
	freqStream= symbolStream.repeat(overSampling)

	##################################
	## Gaussian filter
	##################################
	t = np.linspace(-3, 3, 6*int(overSampling)+1)
	alpha = np.sqrt(np.log(2)/2)/C.GfskBT
	gaussianFlt = (np.sqrt(pi)/alpha)*np.exp(-(t*pi/alpha)**2)
	gaussianFlt /= np.sum(gaussianFlt)

	##################################
	## Modulation
	##################################
	filteredStream = np.convolve(gaussianFlt, freqStream, 'same')
	phaseSig = signal.lfilter(np.array([0.5, 0.5]), np.array([1, -1]), filteredStream*((C.GfskBleModulationIndex[0]+C.GfskBleModulationIndex[1])/4)*2*pi/overSampling)
	basebandSig = np.exp(1j*phaseSig)

	##################################
	## filter output
	##################################
	(b, a) = signal.butter(7, 1.0, fs=overSampling)
	basebandFlt = signal.lfilter(b, a, basebandSig)

	# b = signal.remez(25, [0, 1.0, 7.5, 120], [1,0], fs=240.0)
	# basebandFlt2 = signal.lfilter(b, 1, basebandSig)

	fs = overSampling * (1.0 if modType == C.GfskModulationType.Gfsk1M else 2.0)
	bw = (1.0 if modType == C.GfskModulationType.Gfsk1M else 2.0)

	##################################
	## Frequency offset and drift
	##################################
	# offset+drift should be less than +-150KHz (BLE), +-75KHz+25/40KHz (Br)
	# drift should be less than 50KHz (BLE), 25/40KHz (Br)
	offset = 15.0e3/1.0e6
	drift = 10.0e3/1.0e6
	frequencyOffset = offset+drift*np.linspace(0, 1, basebandFlt.size)
	baseband = basebandFlt*np.exp (1j*2*pi*np.cumsum(frequencyOffset/overSampling))
	
	return baseband, fs, bw
