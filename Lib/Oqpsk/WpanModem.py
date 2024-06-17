import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt

import RfModel as rf
import ChannelFilter as cf
import Spectrum as sp

from .WpanModulation import *
from .WpanDemodulation import *
from .Constant import Constant as C

def WpanChannel(channel):
	# wpan channel is in range 11 to 26
	return 3+5*(channel-11)

def WpanTransmitter(channel, payload, snr):
	txBaseband, fs = Modulation(payload)
	
	Fs_RF = cf.Constant.AdcSamplingFrequency
	tx_upsampled = rf.UpSampling(txBaseband, int(Fs_RF/fs))
	phase_offset = np.random.random(1)*2*np.pi
	frequency_offset = np.random.random(1)/100
	tx_mixer = rf.Mixer(tx_upsampled, cf.Constant.IfMixerFrequency+WpanChannel(channel)+frequency_offset, phase_offset, Fs_RF)
	tx_sig = tx_mixer.real + rf.WhiteNoise(tx_mixer, snr)
	
	print('phase offset = ', phase_offset, ' frequency offset = ', frequency_offset)
	#sp.fftPlot(txBaseband.real, txBaseband.imag, n=2, fs=fs)
	#plt.plot(txBaseband.real, txBaseband.imag, '.')
	#plt.show()
	#sp.fftPlot(tx_mixer.real, tx_mixer.imag, n=2, fs=Fs_RF)
	#sp.fftPlot(tx_sig.real, fs=Fs_RF)

	return tx_sig

def WpanReceiver(adcSamples, channel):
	(data4M, data2M, data1M) = cf.ChannelDecimate(adcSamples)
	(dataI, dataQ, fs) = cf.ChannelFilter(data4M, data2M, data1M, WpanChannel(channel), cf.Constant.ChannelFilterType.Gfsk2M)
	#extracted_data = Demodulation(dataI+1j*dataQ, fs*1.0e6)
	extracted_data = Demodulation_nco(dataI+1j*dataQ, fs)
	return extracted_data 
	
def WpanModem(channel, byte_number, snr):
	payload = (np.random.rand(byte_number)*256).astype(np.uint8)
	IfSig = WpanTransmitter(channel, payload, snr)

	IfSig *= 1000
	adcData = IfSig.astype('int16')
	fp = np.memmap('WpanData.bttraw', mode='w+', dtype=np.dtype('<h'),shape=(1,adcData.size))
	fp[:] = adcData[:]

	print(f'transmit {payload.size=}')
	print('ADC Data: {} samples'.format(adcData.size))
	print('ADC Data Min/Max: ',adcData.min(),adcData.max(), type(adcData[0]))

	print([hex(dd) for dd in payload])
	extracted_data  = WpanReceiver(adcData, channel)
