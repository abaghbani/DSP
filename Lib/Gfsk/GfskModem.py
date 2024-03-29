import numpy as np
import matplotlib.pyplot as plt

import RfModel as rf
import ChannelFilter as cf

from .GfskModulation import *
from .GfskDemodulation import *
from .Constant import Constant as C

def GfskTransmitter(channel, bit_number, rate, snr):
	payload_len = bit_number//8
	Fs_BB = 15.0e6
	payload = np.array(np.random.rand(payload_len)*256, dtype=np.uint8)
	txBaseband = Modulation(payload, Fs_BB)
	
	Fs_RF = cf.Constant.AdcSamplingFrequency
	tx_upsampled = rf.UpSampling(txBaseband, rate*1.0e6, int(Fs_RF/(Fs_BB*rate)), Fs_RF)
	tx_mixer = rf.Mixer(tx_upsampled, cf.Constant().IfMixerFrequency+(channel*1.0e6), 0, Fs_RF)
	tx_sig = tx_mixer.real + rf.WhiteNoise(tx_mixer, snr)
	#IfSig = RfTransceiver(txBaseband, channel, rate, snr)
	
	return payload, tx_sig

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


def GfskModem(channel, bit_number, rate, snr, channel_filter):
	(payload, IfSig) = GfskTransmitter(channel, bit_number, rate, snr)

	adcData = IfSig.astype('int16')
	fp = np.memmap('gfskData.bttraw', mode='w+', dtype=np.dtype('<h'), shape=(1,adcData.size))
	fp[:] = adcData[:]
	
	print('transmit bit number=',payload.size)
	print('ADC Data: {} samples'.format(adcData.size))
	print('ADC Data Min/Max: ',adcData.min(),adcData.max(), type(adcData[0]))

	(freq, rssi, data) = GfskReceiver(adcData, channel, channel_filter)

	CompareData(payload, freq, rssi, data)
	
	# plt.plot(adcData[5000:8000])
	# plt.plot(freq)
	# plt.plot(valid)
	# plt.grid()
	# plt.show()
