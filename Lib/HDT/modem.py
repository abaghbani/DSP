import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

import Spectrum as sp
import ChannelFilter as cf
import Filter as fd
import RfModel as rf

from .modulation import *
from .demodulation import *
from .constant import *
C = Constant()

def transmitter(channel, payload, block_number, mod_type, snr):
	lts_seq = int(np.random.random(1)*16)
	txBaseband, _ = modulation_old(payload, block_number, mod_type, lts_seq)
	print(txBaseband.size)
	# first step of upsampling (2Msps -> 16Msps)
	# b = fd.rrc(141, 0.4, 2.4, 16.0)
	fs=16.0
	b = fd.rrc_filter(0.4, 64, fs/2)
	txBaseband = signal.upfirdn(b, txBaseband, up=fs//2)

	# second step of upsampling (16Msps -> 240Msps)
	fs_RF = cf.Constant.AdcSamplingFrequency
	b = signal.firls(141, np.array([0., 4.0-0.5, 4.0+0.5, fs_RF/2]), [1, 1.4, 0, 0], fs=fs_RF)
	tx_upsampled = signal.upfirdn(b, txBaseband, up=fs_RF//fs)
	
	freq_offset = (np.random.random(1)-0.5)/5	# offset -0.1 to +0.1 (-100KHz to +100KHz)
	phase_offset = (np.random.random(1)-0.5)*2*np.pi # offset -pi to +pi
	tx_mixer = rf.Mixer(tx_upsampled, cf.Constant.IfMixerFrequency+channel+freq_offset, phase_offset, fs_RF)
	tx_sig = tx_mixer.real + np.real(rf.WhiteNoise(tx_mixer, snr))

	print(f'transmitter: payload={payload.size=} bytes, freq_offset={float(freq_offset*1000):.3f} KHz, phase_offset={phase_offset*180/np.pi} Deg, {lts_seq=}')
	#plt.plot(txBaseband.real, txBaseband.imag, 'bo')
	#plt.grid()
	#plt.show()
	
	# sp.fftPlot(txBaseband.real, txBaseband.imag, n=2, fs=fs)
	# sp.fftPlot(tx_upsampled.real, tx_upsampled.imag, n=2, fs=fs_RF)
	# sp.fftPlot(tx_mixer.real, tx_mixer.imag, n=2, fs=fs_RF)
	# sp.fftPlot(tx_sig.real, fs=fs_RF)

	return tx_sig

def receiver(adcSamples, channel, modulation_type):
	(data4M, data2M, data1M) = cf.ChannelDecimate(adcSamples)
	(dataI, dataQ, fs) = cf.ChannelFilter(data4M, data2M, data1M, channel, cf.Constant.ChannelFilterType.Hdt2M)

	demodulation(dataI+1j*dataQ, fs, modulation_type)
	
def modem(channel, byte_number, block_number, modulation_type, snr):
	payload = (np.random.rand(byte_number)*256).astype(np.uint8)
	IfSig = transmitter(channel, payload, block_number, modulation_type, snr)

	lnaGain = 0.9*(2**15)/np.abs(IfSig).max()
	adcData = (IfSig*lnaGain).astype('int16')
	
	print(f'ADC Data: {adcData.size} samples, Min = {adcData.min()}, Max = {adcData.max()}, type={type(adcData[0])}')

	#print(f'transmit payload : {[hex(val)[2:] for val in payload]}')

	receiver(adcData, channel, modulation_type)

	return adcData

def modem_baseband(byte_number, block_number, mod_type, snr):
	payload = (np.random.rand(byte_number)*256).astype(np.uint8)
	lts_seq = int(np.random.random(1)*16)
	txBaseband, fs = modulation(payload, block_number, mod_type, lts_seq)

	fs *= 15
	tx_upsampled = rf.UpSampling(txBaseband, 15)

	freq_offset = (np.random.random(1)-0.5)/5	# offset -0.1 to +0.1 (-100KHz to +100KHz)
	phase_offset = (np.random.random(1)-0.5)*2*np.pi # offset -pi to +pi

	tx_mixer = rf.Mixer(tx_upsampled, freq_offset, phase_offset, fs)
	noise = rf.WhiteNoise(tx_mixer, snr)
	sig = tx_mixer + noise

	print(f'transmitter: payload={payload.size=} bytes, freq_offset={float(freq_offset*1000):.3f} KHz, {lts_seq=}')
	demodulation(sig, fs, mod_type)

	return sig
