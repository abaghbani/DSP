import numpy as np
import matplotlib.pyplot as plt

from Dpsk.Constant import Constant as C
from Dpsk.DpskModulation import DpskModulation
from Dpsk.DpskDemodulation import DpskDemodulation
from RfModel.RfTransceiver import RfTransceiver
from ChannelFilter.ChannelDecimate import ChannelDecimate
from ChannelFilter.ChannelFilter import ChannelFilter

def DpskDataGen(bit_number, type):
	if type == C.DpskModulationType.Edr2:
		### Edr2 data = [-3, -1, 1, 3] * pi/4
		payload = np.random.rand(bit_number)*4
		payload = payload.astype('int')
		payload[payload == 0] = -3
		payload[payload == 1] = -1
		payload[payload == 2] = 1
		payload[payload == 3] = 3
	elif type == C.DpskModulationType.Edr3:
		#### Edr3 data = [-3, -2, -1, 0, 1, 2, 3, 4] * pi/4
		payload = np.random.rand(bit_number)*8
		payload = payload.astype('int')
		payload[payload == 0] = -3
		payload[payload == 1] = -2
		payload[payload == 2] = -1
		payload[payload == 3] = 0
		payload[payload == 4] = 1
		payload[payload == 5] = 2
		payload[payload == 6] = 3
		payload[payload == 7] = 4
	return payload

def DpskTransmitter(channel, bit_number, modType, rate, snr):
	payload = DpskDataGen(bit_number, modType)
	txBaseband = DpskModulation(payload)
	IfSig = RfTransceiver(txBaseband, channel, rate, snr)
	return payload, IfSig

def DpskReceiver(adcSamples, channel, Channel_Filter):
	(data4M, data2M, data1M) = ChannelDecimate(adcSamples)
	(dataI, dataQ, fs) = ChannelFilter(data4M, data2M, data1M, channel, Channel_Filter)
	(rssi, sync, valid, data) = DpskDemodulation(dataI, dataQ, fs)
	return rssi, sync, valid, data
	
def CompareData(txData, rssi, sync, valid, data, modType):
	dataLength = data.size
	detected = np.zeros(dataLength, dtype=bool)
	demodData = np.zeros(0, dtype='int')
	demodDataRaw = np.zeros(0, dtype='int')
	for i in range(1, dataLength):
		if sync[i] == 1:
			detected[i] = 1
		else:
			detected[i] = detected[i-1]
	
		if detected[i] == 1 and valid[i] == 1 and sync[i] == 0:
			if modType == C.DpskModulationType.Edr2:
				demodData = np.append(demodData, C.TableRxPhase4DQPSK[(data[i]+16)%16]) 
			else:
				demodData = np.append(demodData, C.TableRxPhase8DQPSK[(data[i]+16)%16]) 
			demodDataRaw = np.append(demodDataRaw, data[i]) 

	print('received bit number=',demodData.size)
	if demodData.size >= txData.size:
		ber = 0
		errorIndex = []
		for i in range(txData.size):
			if demodData[i] != txData[i]:
				ber += 1
				errorIndex.append(i)
				# print 'error data in: ', i, txData[i], demodData[i], demodDataRaw[i], ber
		print('test is done and BER={0}/{1}'.format(ber, txData.size))
		print('first index: ',errorIndex[:20])
		print('last index: ',errorIndex[-20:])
	else:
		print('Not enough data is received')

def DpskModem(channel, bit_number, modType, rate, snr, channel_filter):
	(txData, IfSig) = DpskTransmitter(channel, bit_number, modType, rate, snr)

	adcData = IfSig.astype('int16')
	fp = np.memmap('dpskData.bttraw', mode='w+', dtype=np.dtype('<h'),shape=(1,adcData.size))
	fp[:] = adcData[:]

	print('transmit bit number=',txData.size)
	print('ADC Data: {} samples'.format(adcData.size))
	print('ADC Data Min/Max: ',adcData.min(),adcData.max(), type(adcData[0]))

	(rssi, sync, valid, data) = DpskReceiver(adcData, channel, channel_filter)
	CompareData(txData, rssi, sync, valid, data, modType)
