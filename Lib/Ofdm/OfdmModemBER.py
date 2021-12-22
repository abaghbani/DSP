import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

from Ofdm.OfdmModulation import OfdmQamModulation
from Ofdm.OfdmRfTransceiver import OfdmRfTransceiver
from Ofdm.OfdmChannelFilter import OfdmChannelFilter
from Ofdm.OfdmDemodulation import OfdmDemodulation
from Ofdm.Constant import Constant as C

channel = 20
Transmitter_Enable = True

if Transmitter_Enable:
	payload = np.random.rand(48*4*200) < 0.5
	# payload = np.random.choice(C.QAM16_table, 48*200)
	print('payload size=', payload.size)
	txBaseband = OfdmQamModulation(C, payload)
	IfSig = OfdmRfTransceiver(C, txBaseband, channel, basebandFS=C.ofdmBasebandFS, basebandBW=C.ofdmBasebandBw, SNRdb=5)
	############################################
	# data saving as binary (Ellisys format)
	############################################
	adcData = IfSig*200
	adcData = adcData.astype('int16')
	fp = np.memmap('./Samples/ofdmData.bttraw', mode='w+', dtype=np.dtype('<h'),shape=(1,adcData.size))
	fp[:] = adcData[:]
	adcData = IfSig
else:
	adcData = np.memmap('./Samples/ofdmData-24Mbps.bttraw', mode='r', dtype=np.dtype('<h'))
	adcData = (adcData<<(16-12))>>(16-12)
	adcData = adcData[80700:80700+12*80*20]
#############################################

print( 'ADC Data: {} samples'.format(adcData.size))
print( 'ADC Data Min/Max: ',adcData.min(),adcData.max(), type(adcData[0]))
rxBaseband = OfdmChannelFilter(C, adcData, channel)
print( 'base Data: {} samples'.format(rxBaseband.size))
print( 'base Data Min/Max: ',rxBaseband.min(),rxBaseband.max(), type(rxBaseband[0]))
rxData = OfdmDemodulation(C, rxBaseband)

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

	
# # plt.plot(valid)
# # plt.plot(phase)
# # plt.plot(rssi)
# # plt.grid()
# # plt.show()
