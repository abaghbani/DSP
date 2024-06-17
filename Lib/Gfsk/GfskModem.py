import numpy as np
import matplotlib.pyplot as plt

import RfModel as rf
import ChannelFilter as cf

from .GfskModulation import *
from .GfskDemodulation import *
from .Constant import Constant as C

def GfskTransmitter(payload, channel, rate, snr):
	txBaseband, Fs_BB = Modulation(payload)
	
	Fs_RF = cf.Constant.AdcSamplingFrequency
	tx_upsampled = rf.UpSampling(txBaseband, int(Fs_RF/(Fs_BB*rate)))
	tx_mixer = rf.Mixer(tx_upsampled, cf.Constant().IfMixerFrequency+(channel), 0, Fs_RF)
	tx_sig = tx_mixer.real + rf.WhiteNoise(tx_mixer, snr)
	#IfSig = RfTransceiver(txBaseband, channel, rate, snr)
	
	return tx_sig

def GfskReceiver(adcSamples, channel, Channel_Filter):
	(data4M, data2M, data1M) = cf.ChannelDecimate(adcSamples)
	(dataI, dataQ, fs) = cf.ChannelFilter(data4M, data2M, data1M, channel, Channel_Filter)
	(freq, rssi, data) = Demodulation(dataI+1j*dataQ, fs)
	return freq, rssi, data

def CompareData(txData, freq, rssi, data):
	
	demodData = data
	#print(demodData)
	GfskPreamble = np.array([-1,1]*8)
	demodDataConv = np.convolve(demodData, GfskPreamble, mode='same')
	syncPosition = np.where(np.abs(demodDataConv)>=(0.5*len(GfskPreamble)))[0]
	if syncPosition.size != 0:
		syncPosition = syncPosition[0]  + 32 +15
		demodData = data[int(syncPosition):].astype(np.int8)
		print('Preamble is detected at position = ', syncPosition)
		print('received bit number=',demodData.size)
		ber = 0
		errorIndex = []
		txData = (np.unpackbits(txData, bitorder='little').astype(np.int8))*2-1
		print(txData)
		print(demodData)
		if demodData.size >= txData.size:
			for i in range(txData.size):
				if demodData[i] != txData[i]:
					ber += 1
					errorIndex.append(i)
					# print 'error data in: ', i, txData[i], demodData[i], ber
			print('test is done and BER={0}/{1}'.format(ber, txData.size))
			print('Error index: ',errorIndex[:20])
			print('Error index: ',errorIndex[-20:])
		else:
			print('Not enough data is received')
	else:
		print('Preamble is not detected')


def GfskModem(channel, bit_number, rate, snr, channel_filter, saving_enable=False):
	payload_len = bit_number//8
	payload = np.array(np.random.rand(payload_len)*256, dtype=np.uint8)

	IfSig = GfskTransmitter(payload, channel, rate, snr)

	lnaGain = (2**14-1)/np.abs(IfSig).max()
	adcData = (IfSig*lnaGain).astype('int16')

	if saving_enable:
		fp = np.memmap('gfskData.bttraw', mode='w+', dtype=np.dtype('<h'), shape=(1,adcData.size))
		fp[:] = adcData[:]
	
	print('transmit bit number=',payload.size)
	print('ADC Data: {} samples'.format(adcData.size))
	print('ADC Data Min/Max: ',adcData.min(),adcData.max(), type(adcData[0]))

	(freq, rssi, rx_data) = GfskReceiver(adcData, channel, channel_filter)

	CompareData(payload, freq, rssi, rx_data)
