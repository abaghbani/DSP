import numpy as np
import matplotlib.pyplot as plt

from Spectrum.Constant import Constant
from Spectrum.ModemLib import ModemLib
from Dpsk.DpskModulation import DpskModulation
from Dpsk.DpskDemodulation import DpskDemodulation
from RfModel.RfTransceiver import RfTransceiver
from ChannelFilter.ChannelDecimate import ChannelDecimate
from ChannelFilter.ChannelFilter import ChannelFilter

C = Constant
myLib = ModemLib(0)

channel = 20
Transmitter_Enable = True

### Edr2 data = [-3, -1, 1, 3] * pi/4
payload = np.random.rand(1000)*4
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
# DpskSymbolStream = np.array([0]*10+sync+payload.tolist()+[0]*10)
DpskSymbolStream = np.concatenate((np.zeros(50), C.DpskSync, payload, np.zeros(100)), axis=None)
txBaseband = DpskModulation(C, DpskSymbolStream)
IfSig = RfTransceiver(C, txBaseband, channel, basebandFS=C.Dpsk1MBasebandFS, basebandBW=C.Dpsk1MBasebandBW, SNRdb=10)
# IfSig = RfTransceiver(C, txBaseband, channel, basebandFS=C.Dpsk2MBasebandFS, basebandBW=C.Dpsk2MBasebandBW, noiseLevel=100)
# IfSig = RfTransceiver(C, txBaseband, channel, basebandFS=C.Dpsk4MBasebandFS, basebandBW=C.Dpsk4MBasebandBW, noiseLevel=20)

#############################################
## data saving as binary (Ellisys format)
#############################################
if Transmitter_Enable:
	adcData = IfSig.astype('int16')
	fp = np.memmap('dpskData.bttraw', mode='w+', dtype=np.dtype('<h'),shape=(1,adcData.size))
	fp[:] = adcData[:]
else:
	adcData = np.memmap('./Samples/dpskData.bttraw', mode='r', dtype=np.dtype('<h'))
	adcData = (adcData<<(16-12))>>(16-12)
#############################################

print('ADC Data: {} samples'.format(adcData.size))
print('ADC Data Min/Max: ',adcData.min(),adcData.max(), type(adcData[0]))
(data4M, data2M, data1M) = ChannelDecimate(C, adcData)
(dataI, dataQ) = ChannelFilter(C, data4M, data2M, data1M, channel, type='Dpsk4M1M')
(phase, rssi, sync, valid, data) = DpskDemodulation(C, dataI, dataQ, symboleRate=1.0)

dataLength = phase.size
detected = np.zeros(dataLength, dtype=bool)
demodData = np.zeros(0, dtype='int')
demodDataRaw = np.zeros(0, dtype='int')
for i in range(1, dataLength):
	if sync[i] == 1:
		detected[i] = 1
	else:
		detected[i] = detected[i-1]
	
	if detected[i] == 1 and valid[i] == 1 and sync[i] == 0:
		demodData = np.append(demodData, C.TableRxPhase4DQPSK[(data[i]+16)%16]) 
		demodDataRaw = np.append(demodDataRaw, data[i]) 

print('received bit number=',demodData.size)
if demodData.size >= payload.size:
	ber = 0
	errorIndex = []
	for i in range(payload.size):
		if demodData[i] != payload[i]:
			ber += 1
			errorIndex.append(i)
			# print 'error data in: ', i, payload[i], demodData[i], demodDataRaw[i], ber
	print('test is done and BER={0}/{1}'.format(ber, payload.size))
	print('first index: ',errorIndex[:20])
	print('last index: ',errorIndex[-20:])
else:
	print('Not enough data is received')

	
# plt.plot(valid)
# plt.plot(phase)
# plt.plot(rssi)
# plt.grid()
# plt.show()
