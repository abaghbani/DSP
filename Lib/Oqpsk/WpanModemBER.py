import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt

from Oqpsk.WpanModulation import WpanModulation
from Oqpsk.WpanDemodulation import WpanDemodulation
from RfModel.RfTransceiver import RfTransceiver
from ChannelFilter.ChannelDecimate import ChannelDecimate
from ChannelFilter.ChannelFilter import ChannelFilter

from .Constant import *
C = Constant()

# wpan channel is in range 11 to 26
wpanChannel = 12
channel = 3+5*(wpanChannel-11)
Transmitter_Enable = True

## Edr2 data = [-3, -1, 1, 3] * pi/4
payload = np.random.rand(20)*4
payload = payload.astype('int')
payload[payload == 0] = -3
payload[payload == 1] = -1
payload[payload == 2] = 1
payload[payload == 3] = 3

#### Edr3 data = [-3, -2, -1, 0, 1, 2, 3, 4] * pi/4
# payload = np.random.rand(20)*8
# payload = payload.astype('int')
# payload[payload == 0] = -3
# payload[payload == 1] = -2
# payload[payload == 2] = -1
# payload[payload == 3] = 0
# payload[payload == 4] = 1
# payload[payload == 5] = 2
# payload[payload == 6] = 3
# payload[payload == 7] = 4

print('transmit bit number=',payload.size)
# DpskSymbolStream = np.concatenate((np.zeros(50), C.WpanPN0*10, np.zeros(100)), axis=None)
DpskSymbolStream = np.concatenate((C.WpanPN0Phase*10, np.zeros(100)), axis=None)
txBaseband = WpanModulation(C, DpskSymbolStream)
IfSig = RfTransceiver(C, txBaseband, channel, basebandFS=C.WpanBasebandFS, basebandBW=C.WpanBasebandBW, SNRdb=10)

#############################################
## data saving as binary (Ellisys format)
#############################################
if Transmitter_Enable:
	adcData = IfSig.astype('int16')
	fp = np.memmap('wpanData.bttraw', mode='w+', dtype=np.dtype('<h'),shape=(1,adcData.size))
	fp[:] = adcData[:]
else:
	adcData = np.memmap('./Samples/wpanData4.bttraw', mode='r', dtype=np.dtype('<h'))
	adcData = (adcData<<(16-12))>>(16-12)
	adcData = adcData[290000: 390000]
#############################################

# adcData = io.loadmat('RxOQPSK.mat')['RxOQPSK'][1]
# print adcData.size
# myLib.fftPlot(adcData)
# plt.plot(adcData)
# plt.grid()
# plt.show()


#################
# debugging
#################
# input data : [-98M (2402) ... -19M (2481)],   [[2402:2481]-2260(rf_mixer)]sampling@240MHz => nyquist1 left side = -98:-19

# Fs = 240.0
# fmix0 = -90.0
# dataMixI = adcData*np.cos((np.arange(adcData.size)*2*np.pi*fmix0/Fs)-np.pi)
# dataMixQ = adcData*np.sin((np.arange(adcData.size)*2*np.pi*fmix0/Fs))

# dataFltI = np.convolve(C.hbFlt0, dataMixI)[::2]
# dataFltQ = np.convolve(C.hbFlt0, dataMixQ)[::2]

# dataOut = np.convolve(C.chFltDpskHdr4M, dataFltI)


# plt.plot(dataOut)
# plt.grid()
# plt.show()



print('ADC Data: {} samples'.format(adcData.size))
print('ADC Data Min/Max: ',adcData.min(),adcData.max(), type(adcData[0]))
(data4M, data2M, data1M) = ChannelDecimate(C, adcData)
(dataI, dataQ) = ChannelFilter(C, data4M, data2M, data1M, channel, type='Dpsk4M4M')
(freq, rssi, valid, data) = WpanDemodulation(C, dataI, dataQ, symboleRate=1.0)

# dataLength = phase.size
# detected = np.zeros(dataLength, dtype=bool)
# demodData = np.zeros(0, dtype='int')
# demodDataRaw = np.zeros(0, dtype='int')
# for i in range(1, dataLength):
# 	if sync[i] == 1:
# 		detected[i] = 1
# 	else:
# 		detected[i] = detected[i-1]
	
# 	if detected[i] == 1 and valid[i] == 1 and sync[i] == 0:
# 		demodData = np.append(demodData, C.TableRxPhase4DQPSK[(data[i]+16)%16]) 
# 		demodDataRaw = np.append(demodDataRaw, data[i]) 

# print 'received bit number=',demodData.size
# if demodData.size >= payload.size:
# 	ber = 0
# 	errorIndex = []
# 	for i in range(payload.size):
# 		if demodData[i] != payload[i]:
# 			ber += 1
# 			errorIndex.append(i)
# 			# print 'error data in: ', i, payload[i], demodData[i], demodDataRaw[i], ber
# 	print 'test is done and BER={0}/{1}'.format(ber, payload.size)
# 	print 'first index: ',errorIndex[:20]
# 	print 'last index: ',errorIndex[-20:]
# else:
# 	print 'Not enough data is received'

	
# plt.plot(valid)
# plt.plot(phase)
# plt.plot(rssi)
# plt.grid()
# plt.show()
