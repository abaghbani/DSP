import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt

from Oqpsk.Constant import Constant as C
from Oqpsk.WpanModulation import WpanModulation
from Oqpsk.WpanDemodulation import WpanDemodulation
from RfModel.RfTransceiver import RfTransceiver
from ChannelFilter.ChannelDecimate import ChannelDecimate
from ChannelFilter.ChannelFilter import ChannelFilter

def WpanChannel(channel):
	# wpan channel is in range 11 to 26
	return 3+5*(channel-11)

def WpanDataGen(bit_number):
	payload = np.random.rand(bit_number)*16
	payload = payload.astype('int')
	return payload

def WpanTransmitter(channel, bit_number, snr):
	payload = WpanDataGen(bit_number)
	(baseband, fs, bw) = WpanModulation(payload)
	IfSig = RfTransceiver(baseband, fs, bw, WpanChannel(channel), snr)
	return payload, IfSig

def WpanReceiver(adcSamples, channel):
	(data4M, data2M, data1M) = ChannelDecimate(adcSamples)
	(dataI, dataQ, fs) = ChannelFilter(data4M, data2M, data1M, WpanChannel(channel), C.ChannelFilterType.Dpsk4M.ch4M)
	(rssi, sync, valid, data) = WpanDemodulation(dataI, dataQ, fs)
	return rssi, sync, valid, data
	
def WpanModem(channel, bit_number, snr):
	(txData, IfSig) = WpanTransmitter(channel, bit_number, snr)

	#IfSig *= 10
	adcData = IfSig.astype('int16')
	fp = np.memmap('WpanData.bttraw', mode='w+', dtype=np.dtype('<h'),shape=(1,adcData.size))
	fp[:] = adcData[:]

	print('transmit bit number=',txData.size)
	print('ADC Data: {} samples'.format(adcData.size))
	print('ADC Data Min/Max: ',adcData.min(),adcData.max(), type(adcData[0]))

	(rssi, sync, valid, data) = WpanReceiver(adcData, channel)
