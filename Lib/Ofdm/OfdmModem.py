import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

import Spectrum as sp
import ChannelFilter as cf
import RfModel as rf

from .OfdmModulation import *
from .OfdmDemodulation import *
from .Constant import *
C = Constant()

def OfdmTransmitter(channel, payload, modulation_type, snr):
	baseband, fs = Modulation(payload, modulation_type)

	fs_RF = cf.Constant.AdcSamplingFrequency
	tx_upsampled = rf.UpSampling(baseband, int(fs_RF/fs))
	
	freq_offset = (np.random.random(1)-0.5)/5	# freq offset -0.1 to +0.1 (-100KHz to +100KHz)
	phase_offset = (np.random.random(1)-0.5)*2*np.pi # phase offset -pi to +pi
	tx_mixer = rf.Mixer(tx_upsampled, cf.Constant.IfMixerFrequency+channel+freq_offset, phase_offset, fs_RF)
	tx_sig = tx_mixer.real + rf.WhiteNoise(tx_mixer, snr)

	print(f'freq_offset= {float(freq_offset*1000):.3f} KHz -- phase_offset= {float(phase_offset):.2f}')
	#sp.fftPlot(baseband.real, baseband.imag, n=2, fs=fs)
	#sp.fftPlot(tx_upsampled.real, tx_upsampled.imag, n=2, fs=fs_RF)
	#sp.fftPlot(tx_mixer.real, tx_mixer.imag, n=2, fs=fs_RF)
	#sp.fftPlot(tx_sig, fs=fs_RF)

	return tx_sig

def OfdmReceiver(adcSamples, channel, modulation_type=None):
	rx_mixer = rf.Mixer(adcSamples, -(cf.Constant.IfMixerFrequency+channel), 0, 240)
	b = signal.remez(100+1, [0, 10/240, 20/240, 0.5], [1, 1e-4])
	baseband = np.convolve(b, rx_mixer, 'same')
	
	#sp.specPlot(adcSamples)
	#sp.fftPlot(adcSamples, fs=240)
	#sp.fftPlot(rx_mixer.real, rx_mixer.imag, n=2, fs=240)
	#sp.fftPlot(baseband.real, baseband.imag, n=2, fs=240)

	return Demodulation(baseband, 240, modulation_type)

def OfdmModem(channel, byte_number, modulation_type, snr):
	payload = (np.random.rand(byte_number)*256).astype(np.uint8)
	IfSig = OfdmTransmitter(channel, payload, modulation_type, snr)

	lnaGain = 0.9*(2**15)/np.abs(IfSig).max()
	adcData = (IfSig*lnaGain).astype('int16')
	
	print(f'transmit {payload.size=}')
	print(f'ADC Data: {adcData.size} samples')
	print(f'ADC Data Min = {adcData.min()}, Max = {adcData.max()}, type={type(adcData[0])}')

	data_extract, data_len, crc = OfdmReceiver(adcData, channel, modulation_type)
	if crc==0:
		if payload.size==data_len :
			print(f'{[hex(val)[2:] for val in payload[np.nonzero(data_extract != payload)]]}, {data_len=}')
			print(f'{[hex(val)[2:] for val in data_extract[np.nonzero(data_extract != payload)]]}, {data_len=}')
		else:
			print(f'extracted data len is not matched with transmitted payload.')
	
	return 	adcData
