import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt

import RfModel as rf
import Filter as fd
import ChannelFilter as cf
import Spectrum as sp

from .modulation import *
from .demodulation import *
from .constant import *
C = Constant()

def wpan_channel(channel):
	# wpan channel is in range 11 to 26
	return 3+5*(channel-11)

def transmitter(channel, payload, snr):
	data_modulated, fs = modulation(payload)
	
	## upsampling and then implying Offset to IQ (O-QPSK)
	data_upsampled = data_modulated.repeat(8)
	fs *= 8
	b = fd.rrc(141, 0.4, 2.4, fs)
	data_upsampled = np.convolve(b, data_upsampled, 'same')
	baseband = data_upsampled.real+1j*np.roll(data_upsampled.imag, int(fs/2))

	fs_RF = cf.Constant.AdcSamplingFrequency
	b = signal.firls(141, np.array([0., 4.0-0.5, 4.0+0.5, fs_RF/2]), [1, 1.4, 0, 0], fs=fs_RF)
	tx_upsampled = np.convolve(b, baseband.repeat(fs_RF//fs), mode='same')
	# tx_upsampled = rf.UpSampling(baseband, int(Fs_RF/fs))
	
	freq_offset = 0 # (np.random.random(1)-0.5)/5	# offset -0.1 to +0.1 (-100KHz to +100KHz)
	phase_offset = 0 # (np.random.random(1)-0.5)*2*np.pi # offset -pi to +pi
	tx_mixer = rf.Mixer(tx_upsampled, cf.Constant.IfMixerFrequency+wpan_channel(channel)+freq_offset, phase_offset, fs_RF)
	tx_sig = tx_mixer.real + rf.WhiteNoise(tx_mixer, snr)

	print(f'transmitter: payload={payload.size=} bytes, freq_offset={float(freq_offset*1000):.3f} KHz')

	
	# sp.fftPlot(baseband.real, baseband.imag, n=2, fs=fs)
	# sp.fftPlot(tx_mixer.real, tx_mixer.imag, n=2, fs=fs_RF)
	# sp.fftPlot(tx_sig.real, fs=fs_RF)

	return tx_sig

def receiver(adcSamples, channel):
	(data4M, data2M, data1M) = cf.ChannelDecimate(adcSamples)
	(dataI, dataQ, fs) = cf.ChannelFilter(data4M, data2M, data1M, wpan_channel(channel), cf.Constant.ChannelFilterType.Gfsk2M)
	#extracted_data = Demodulation(dataI+1j*dataQ, fs*1.0e6)
	extracted_data = demodulation(dataI+1j*dataQ, fs)
	return extracted_data 
	
def modem(channel, byte_number, snr):
	payload = (np.random.rand(byte_number)*256).astype(np.uint8)
	IfSig = transmitter(channel, payload, snr)

	lnaGain = 0.9*(2**15)/np.abs(IfSig).max()
	adcData = (IfSig*lnaGain).astype('int16')
	
	print(f'ADC Data: {adcData.size} samples, Min = {adcData.min()}, Max = {adcData.max()}, type={type(adcData[0])}')

	extracted_data  = receiver(adcData, channel)

def modem_baseband(byte_number, snr):
	payload = (np.random.rand(byte_number)*256).astype(np.uint8)
	data_modulated, fs = modulation(payload)
	
	## upsampling and then implying Offset to IQ (O-QPSK)
	fs *= 8
	# b = fd.rrc(141, 0.4, 2.4, fs)
	# data_upsampled = np.convolve(b, data_modulated.repeat(8), 'same')
	data_upsampled = rf.UpSampling(data_modulated, 8)

	baseband = data_upsampled.real+1j*np.roll(data_upsampled.imag, int(fs/2))

	sp.fftPlot(baseband.real, baseband.imag, n=2, fs=fs)
	plt.plot(data_upsampled.real, data_upsampled.imag, 'b.')
	plt.plot(baseband.real, baseband.imag, 'r.')
	plt.show()

	# freq_offset = (np.random.random(1)-0.5)/5	# offset -0.1 to +0.1 (-100KHz to +100KHz)
	# phase_offset = (np.random.random(1)-0.5)*2*np.pi # offset -pi to +pi

	# tx_mixer = rf.Mixer(tx_upsampled, freq_offset, phase_offset, fs)
	# noise = rf.WhiteNoise(tx_mixer, snr)
	# sig = tx_mixer + (noise+1j*noise)

	# print(f'transmitter: payload={payload.size=} bytes, freq_offset={float(freq_offset*1000):.3f} KHz, {lts_seq=}')
	# demodulation(data_upsampled, fs)
