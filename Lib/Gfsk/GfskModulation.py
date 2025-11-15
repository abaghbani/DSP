import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

import Spectrum as sp

from .Constant import *
C = Constant()

def Modulation(payload):

	crc = CrcCalculation(payload)
	access_address = C.GfskAccessAddress_Adv
	packet_frame = np.concatenate((C.GfskPreamble if (access_address[0] & 0x01) else ~C.GfskPreamble, access_address , payload, crc))
	bit_frame = np.unpackbits(packet_frame, bitorder='little').astype(np.int8)
	
	## convert bit to frequency sample
	frequency_sample = bit_frame*2-1
	frequency_sample = np.concatenate((np.zeros(4*8, dtype=np.int8), frequency_sample, np.zeros(4*8, dtype=np.int8) ))

	## Upsampling
	Fs = 15.0
	over_sample_rate = int(Fs/C.bit_rate)
	frequency_sample = frequency_sample.repeat(over_sample_rate)

	##################################
	## Gaussian filter
	##################################
	t = np.linspace(-3, 3, 6*int(over_sample_rate)+1)
	alpha = np.sqrt(np.log(2)/2)/C.GfskBT
	gaussianFlt = (np.sqrt(np.pi)/alpha)*np.exp(-(t*np.pi/alpha)**2)
	gaussianFlt /= np.sum(gaussianFlt)

	##################################
	## Modulation
	##################################
	frequency_sig = np.convolve(gaussianFlt, frequency_sample, 'same')
	phase_sig = signal.lfilter(np.array([0.5, 0.5]), np.array([1, -1]), frequency_sig*((C.GfskBleModulationIndex[0]+C.GfskBleModulationIndex[1])/4)*2*np.pi/over_sample_rate)
	baseband_sig = np.exp(1j*phase_sig)
	baseband_sig = np.concatenate((np.zeros(20, dtype=type(baseband_sig[0])), baseband_sig, np.zeros(20, dtype=type(baseband_sig[0])) ))

	#plt.plot(frequency_sample)
	#plt.plot(frequency_sig)
	#plt.plot(phase_sig)
	#plt.show()

	##################################
	## filter output
	##################################
	(b, a) = signal.butter(7, 1.0, fs=Fs)
	basebandFlt = signal.lfilter(b, a, baseband_sig)
	
	#sp.fftPlot(basebandFlt, fs= Fs)
	#plt.plot(basebandFlt.real, basebandFlt.imag)
	#plt.show()

	# b = signal.remez(25, [0, 1.0, 7.5, 120], [1,0], fs=240.0)
	# basebandFlt2 = signal.lfilter(b, 1, basebandSig)

	##################################
	## Frequency offset and drift
	##################################
	# offset+drift should be less than +-150KHz (BLE), +-75KHz+25/40KHz (Br)
	# drift should be less than 50KHz (BLE), 25/40KHz (Br)
	offset = 15.0e3/1.0e6
	drift = 10.0e3/1.0e6
	frequencyOffset = offset+drift*np.linspace(0, 1, basebandFlt.size)
	baseband = basebandFlt*np.exp (1j*2*np.pi*np.cumsum(frequencyOffset/Fs))
	
	return baseband_sig, Fs

def Modulation_new(payload, fs, rate):

	fs /= 1.0e6
	crc = CrcCalculation(payload)
	access_address = ~C.GfskAccessAddress_Adv
	packet_frame = np.concatenate((C.GfskPreamble if (access_address[0] & 0x01) else ~C.GfskPreamble, access_address , payload, crc))
	bit_frame = np.unpackbits(packet_frame, bitorder='little').astype(np.int8)
	
	## convert bit to frequency sample
	frequency_sample = bit_frame*2-1
	frequency_sample = np.concatenate((np.zeros(4*8, dtype=np.int8), frequency_sample, np.zeros(4*8, dtype=np.int8) ))

	## Upsampling
	over_sample_rate = int(fs/rate)
	frequency_sample = frequency_sample.repeat(over_sample_rate)

	##################################
	## Gaussian filter
	##################################
	t = np.linspace(-3, 3, 6*int(over_sample_rate)+1)
	alpha = np.sqrt(np.log(2)/2)/C.GfskBT
	gaussianFlt = (np.sqrt(np.pi)/alpha)*np.exp(-(t*np.pi/alpha)**2)
	gaussianFlt /= np.sum(gaussianFlt)

	##################################
	## Modulation
	##################################
	frequency_sig = np.convolve(gaussianFlt, frequency_sample, 'same')
	phase_sig = signal.lfilter(np.array([0.5, 0.5]), np.array([1, -1]), frequency_sig*((C.GfskBleModulationIndex[0]+C.GfskBleModulationIndex[1])/4)*2*np.pi/over_sample_rate)
	baseband_sig = np.exp(1j*phase_sig)
	baseband_sig = np.concatenate((np.zeros(20, dtype=type(baseband_sig[0])), baseband_sig, np.zeros(20, dtype=type(baseband_sig[0])) ))

	# plt.plot(frequency_sample)
	# plt.plot(frequency_sig)
	# plt.plot(phase_sig)
	# plt.show()

	##################################
	## filter output
	##################################
	(b, a) = signal.butter(7, 1.0, fs=fs)
	basebandFlt = signal.lfilter(b, a, baseband_sig)
	
	#sp.fftPlot(basebandFlt, fs= Fs)
	#plt.plot(basebandFlt.real, basebandFlt.imag)
	#plt.show()

	# b = signal.remez(25, [0, 1.0, 7.5, 120], [1,0], fs=240.0)
	# basebandFlt2 = signal.lfilter(b, 1, basebandSig)

	##################################
	## Frequency offset and drift
	##################################
	# offset+drift should be less than +-150KHz (BLE), +-75KHz+25/40KHz (Br)
	# drift should be less than 50KHz (BLE), 25/40KHz (Br)
	offset = 15.0e3/1.0e6
	drift = 10.0e3/1.0e6
	frequencyOffset = offset+drift*np.linspace(0, 1, basebandFlt.size)
	baseband = basebandFlt*np.exp (1j*2*np.pi*np.cumsum(frequencyOffset/fs))
	
	return baseband_sig

def GaussianFunction(Td, BT, h, fs):
					
	# Td: Symbol period
	# BT: Bandwidth.Symboltime
	# h: modulation index
	# fs: sampling frequency

	t = np.arange(-Td/2, Td/2, 1/fs)
	alpha = 0.833*Td / (BT*2*np.pi)
	G = h/(2*fs) * (1/(np.sqrt(2*np.pi)*alpha)) * np.exp(-1/2 * (t/alpha)**2)
	
	return G