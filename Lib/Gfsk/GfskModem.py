import numpy as np
import matplotlib.pyplot as plt

from RfModel.RfTransceiver import RfTransceiver
from ChannelFilter.ChannelDecimate import ChannelDecimate
from ChannelFilter.ChannelFilter import ChannelFilter
from Gfsk.GfskModulation import GfskModulation
from Gfsk.GfskDemodulation import GfskDemodulation
from Gfsk.Constant import Constant as C

def GfskTransmitter(channel, bit_number, rate, snr):
	payload = np.array((np.random.rand(bit_number) >= 0.5)*2-1)
	txBaseband = GfskModulation(payload)
	IfSig = RfTransceiver(txBaseband, channel, rate, snr)
	return payload, IfSig

def GfskReceiver(adcSamples, channel, Channel_Filter):
	(data4M, data2M, data1M) = ChannelDecimate(adcSamples)
	(dataI, dataQ, fs) = ChannelFilter(data4M, data2M, data1M, channel, Channel_Filter)
	(freq, rssi, valid, data) = GfskDemodulation(dataI, dataQ, fs)
	return freq, rssi, valid, data

def CompareData(txData, freq, rssi, valid, data):
	dataLength = freq.size
	detected = np.zeros(dataLength, dtype=bool)
	demodData = np.zeros(0, dtype='int')
	for i in range(1, dataLength):
		if valid[i] == 1:
			demodData = np.append(demodData, data[i]*2-1) 

	demodDataConv = np.convolve(demodData, C.GfskPreamble, mode='same')
	syncPosition = np.where(np.abs(demodDataConv)==len(C.GfskPreamble))[0]
	if syncPosition.size != 0:
		syncPosition = syncPosition[0] + len(C.GfskPreamble)/2
		demodData = demodData[int(syncPosition):]
		print('Preamble is detected.')
		print('received bit number=',demodData.size)
		ber = 0
		errorIndex = []
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

	(freq, rssi, valid, data) = GfskReceiver(adcData, channel, channel_filter)

	CompareData(payload, freq, rssi, valid, data)
	
	# plt.plot(adcData[5000:8000])
	# plt.plot(freq)
	# plt.plot(valid)
	# plt.grid()
	# plt.show()
