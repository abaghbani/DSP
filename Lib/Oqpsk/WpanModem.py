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

def WpanDataGen(byte_number):
	payload = (np.random.rand(byte_number)*256).astype(np.uint8)
	return payload

def WpanTransmitter(channel, byte_number, snr):
	payload = WpanDataGen(byte_number)
	fs = 16.0e6
	txBaseband = Modulation(payload, fs)
	
	Fs_RF = cf.Constant.AdcSamplingFrequency
	tx_upsampled = rf.UpSampling(txBaseband, 5.0e6, int(Fs_RF/fs), Fs_RF)
	tx_mixer = rf.Mixer(tx_upsampled, cf.Constant.IfMixerFrequency+(WpanChannel(channel)*1.0e6), np.pi/4, Fs_RF)
	tx_sig = tx_mixer.real + rf.WhiteNoise(tx_mixer, snr)
	
	#sp.fftPlot(txBaseband.real, txBaseband.imag, n=2, fs=fs)
	#plt.plot(txBaseband.real, txBaseband.imag, '.')
	#plt.show()
	#sp.fftPlot(tx_mixer.real, tx_mixer.imag, n=2, fs=Fs_RF)
	#sp.fftPlot(tx_sig.real, fs=Fs_RF)

	return payload, tx_sig

def WpanReceiver(adcSamples, channel):
	(data4M, data2M, data1M) = cf.ChannelDecimate(adcSamples)
	(dataI, dataQ, fs) = cf.ChannelFilter(data4M, data2M, data1M, WpanChannel(channel), cf.Constant.ChannelFilterType.Dpsk4M.ch4M)
	extracted_data = Demodulation(dataI+1j*dataQ, fs*1.0e6)
	return extracted_data 
	
def WpanModem(channel, byte_number, snr):
	(txData, IfSig) = WpanTransmitter(channel, byte_number, snr)

	IfSig *= 1000
	adcData = IfSig.astype('int16')
	fp = np.memmap('WpanData.bttraw', mode='w+', dtype=np.dtype('<h'),shape=(1,adcData.size))
	fp[:] = adcData[:]

	print('transmit bit number=',txData.size, txData)
	print('ADC Data: {} samples'.format(adcData.size))
	print('ADC Data Min/Max: ',adcData.min(),adcData.max(), type(adcData[0]))

	extracted_data  = WpanReceiver(adcData, channel)
