import numpy as np
import scipy.signal as signal

from Common.Constant import Constant
C = Constant

def DpskModulation(payload):
	pi= np.pi
	overSampling = 15
	symbolStream = np.concatenate((np.zeros(50), C.DpskSync, payload, np.zeros(100)), axis=None)

	##################################
	## Modulation
	##################################
	phaseSig = symbolStream*pi/4
	basebandSig = np.exp(1j*np.cumsum(phaseSig))
	
	##################################
	## upsampling
	##################################
	basebandUp = np.zeros(overSampling*basebandSig.size,dtype=basebandSig.dtype)
	basebandUp[overSampling*np.arange(basebandSig.size)] = overSampling*basebandSig[np.arange(basebandSig.size)]
	b = signal.remez(100+1, [0., .25, 0.85, 0.5*overSampling], [1,0], fs=overSampling)
	basebandFlt = signal.lfilter(b, 1, basebandUp)
	
	fs = overSampling * 1.0		# phase sample rate is 1Msps
	bw = 1.0	# baseband width is 1MHz

	##################################
	## Frequency offset and drift
	##################################
	# offset+drift should be less than +-75KHz
	# drift should be less than 10KHz
	offset = 15.0e3/1.0e6
	drift = 5.0e3/1.0e6
	frequencyOffset = offset+drift*np.linspace(0, 1, basebandFlt.size)
	baseband = basebandFlt*np.exp (1j*2*pi*np.cumsum(frequencyOffset/overSampling))
	
	return baseband, fs, bw
