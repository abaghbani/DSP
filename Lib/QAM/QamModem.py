import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

import Spectrum as sp
import ChannelFilter as cf
import Filter as fd
import RfModel as rf

from .QamModulation import *
from .QamDemodulation import *
from .Constant import *
C = Constant()

def QamTransmitter(channel, payload, block_number, mod_type, snr):
	lts_seq = int(np.random.random(1)*16)
	txBaseband, fs = modulation(payload, block_number, mod_type, lts_seq)
	
	fs_RF = cf.Constant.AdcSamplingFrequency
	tx_upsampled = rf.UpSampling(txBaseband, int(fs_RF/fs))
	
	freq_offset = (np.random.random(1)-0.5)/5	# offset -0.1 to +0.1 (-100KHz to +100KHz)
	phase_offset = (np.random.random(1)-0.5)*2*np.pi # offset -pi to +pi
	tx_mixer = rf.Mixer(tx_upsampled, cf.Constant.IfMixerFrequency+channel+freq_offset, phase_offset, fs_RF)
	tx_sig = tx_mixer.real + rf.WhiteNoise(tx_mixer, snr)

	print(f'transmitter: payload={payload.size=} bytes, freq_offset={float(freq_offset*1000):.3f} KHz, phase_offset={phase_offset*180/np.pi} Deg, {lts_seq=}')
	#plt.plot(txBaseband.real, txBaseband.imag, 'bo')
	#plt.grid()
	#plt.show()
	
	#sp.fftPlot(txBaseband.real, txBaseband.imag, n=2)
	#sp.fftPlot(tx_upsampled.real, tx_upsampled.imag, n=2)
	#sp.fftPlot(tx_mixer.real, tx_mixer.imag, n=2)
	#sp.fftPlot(tx_sig.real)

	return tx_sig

def QamReceiver_simple(adcSamples, channel, modulation_type):
	#freq_offset = np.random.random(1)/(25)
	#phase_offset = (np.random.random(1)-0.5)*2*np.pi
	#print(f'freq_offset= {freq_offset/240} {phase_offset=}')
	#rx_mixer = rf.Mixer(adcSamples, (-1*channel)+freq_offset, phase_offset, 240)
	rx_mixer = rf.Mixer(adcSamples, -1*channel, 0, 240)
	b = signal.remez(100+1, [0, 2.0/240, 10.0/240, 0.5], [1, 1e-4])
	baseband = np.convolve(b, rx_mixer, 'same')
	

	#sp.fftPlot(adcSamples)
	#sp.fftPlot(rx_mixer.real, rx_mixer.imag, n=2)
	#sp.fftPlot(baseband.real, baseband.imag, n=2)

	## first level of down_sampling (to remove mixer effect)
	baseband = baseband[4::10]
	#sp.fftPlot(baseband.real, baseband.imag, n=2)

	return Demodulation(baseband, 24, modulation_type)
	
def QamReceiver(adcSamples, channel, modulation_type):
	(data4M, data2M, data1M) = cf.ChannelDecimate(adcSamples)
	(dataI, dataQ, fs) = cf.ChannelFilter(data4M, data2M, data1M, channel, cf.Constant.ChannelFilterType.Dpsk4M.ch4M)
	return Demodulation(dataI+1j*dataQ, fs, modulation_type)
	
def QamModem(channel, byte_number, block_number, modulation_type, snr):
	payload = (np.random.rand(byte_number)*256).astype(np.uint8)
	IfSig = QamTransmitter(channel, payload, block_number, modulation_type, snr)

	lnaGain = 0.9*(2**15)/np.abs(IfSig).max()
	adcData = (IfSig*lnaGain).astype('int16')
	
	print(f'ADC Data: {adcData.size} samples, Min = {adcData.min()}, Max = {adcData.max()}, type={type(adcData[0])}')

	#print(f'transmit payload : {[hex(val)[2:] for val in payload]}')

	QamReceiver(adcData, channel, modulation_type)

	return adcData