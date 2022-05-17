import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import logging as log

from Spectrum import freqPlot as fp
from RfModel import RfTransceiver as rfTrx
from ClockRecovery import ClockRecovery as cr

## y(t) = exp(1/2) * (t/σ) * exp(−((t/σ)**2)/2),  where σ=tc/2=1/(2πfc)
def gmonopuls(fc, fs):

	t = np.arange(-1/fc, 1/fc, 1/fs);
	sigma = 1/(2*np.pi*fc);
	sig = np.exp(1/2) * (t/sigma) * np.exp(-((t/sigma)**2)/2);

	#plt.plot(t, sig)
	#plt.legend(['gauss mono pulse'], loc='best')
	#plt.grid()
	#plt.show()

	return t, sig

def uwb_transmit(fp, fs, Nf, Np, snrdb):
	Tc=10e-9		# Chip Width
	Tf=200			# Frame duration in Nanoseconds
	Ni=1000			# Number of information symbols
	Ns=Np+Ni		# Total number of symbols
	
	# PULSE GENERATION
	t, p = gmonopuls(fp, fs)
	
	#chip = np.concatenate((p, np.zeros(5)), axis=None)
	chip = p
	
	# UWB CHANNEL
	#uwb_h=uwb_channel(CMN,num_channels)		# Reference 21
	
	# FRAME GENERATION
	#frame1=conv(uwb_h(:,num_channels),chip)
	#frame=frame1(1:1200)
	#N=length(frame)
	frame = chip

	sig = np.hstack([frame]*Nf)

	# modulation and adding pilot
	Mod = (np.random.rand(sig.size) >= 0.5).astype('int')
	s3 = np.multiply(sig, Mod)
	sig = np.concatenate((np.hstack([sig]*Np), np.hstack([s3]*Ni)), axis=None)
	#plt.plot(sig)

	# adding noise (rf channel model)
	signal_power = np.mean(abs(sig**2))
	sigma2 = signal_power * 10**(snrdb/10)  # calculate noise power based on signal power and SNR
	print ("RX Signal power: %.4f, Noise power: %.4f" % (signal_power, sigma2))
	noiseSignal = np.sqrt(sigma2/2) * (np.random.randn(sig.size)-0.5)
	sig = sig + noiseSignal
	
	#plt.plot(sig)
	#plt.show()
	#freqPlt.fftPlot(sig, fs=fs)

	return sig