import numpy as np
import matplotlib.pyplot as plt

from RfModel.RfTransceiver import RfTransceiver
from ChannelFilter.ChannelDecimate import ChannelDecimate
from ChannelFilter.ChannelFilter import ChannelFilter
from Gfsk.GfskModulation import GfskModulation
from Gfsk.GfskDemodulation import GfskDemodulation
from Gfsk.Constant import Constant as C

channel = 20
Transmitter_Enable = True

payload = np.array((np.random.rand(200) >= 0.5)*2-1)
print('transmit bit number=',payload.size)
# print payload
GfskSymbolStream = np.concatenate((np.zeros(50), C.GfskPreamble , payload, np.zeros(100)), axis=None)
txBaseband = GfskModulation(GfskSymbolStream)
# IfSig = RfTransceiver(C, txBaseband, channel, basebandFS=C.Gfsk1MBasebandFS, basebandBW=C.Gfsk1MBasebandBW, noiseLevel=45)
# IfSig = RfTransceiver(C, txBaseband, channel, basebandFS=C.Gfsk2MBasebandFS, basebandBW=C.Gfsk2MBasebandBW, noiseLevel=25)
IfSig = RfTransceiver(txBaseband, channel, basebandFS=C.Gfsk2MBasebandFS, basebandBW=C.Gfsk2MBasebandBW, SNRdb=10)

#############################################
## data saving as binary file
#############################################
if Transmitter_Enable:
	adcData = IfSig.astype('int16')
	fp = np.memmap('gfskData.bttraw', mode='w+', dtype=np.dtype('<h'),shape=(1,adcData.size))
	fp[:] = adcData[:]
else:
	adcData = np.memmap('gfskData.bttraw', mode='r', dtype=np.dtype('<h'))
	adcData = (adcData<<(16-12))>>(16-12)
#############################################

print('ADC Data: {} samples'.format(adcData.size))
print('ADC Data Min/Max: ',adcData.min(),adcData.max(), type(adcData[0]))
(data4M, data2M, data1M) = ChannelDecimate(C, adcData)
(dataI, dataQ) = ChannelFilter(C, data4M, data2M, data1M, channel, type="Gfsk2M")
(freq, rssi, valid, data) = GfskDemodulation(C, dataI, dataQ)

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
	if demodData.size >= payload.size:
		for i in range(payload.size):
			if demodData[i] != payload[i]:
				ber += 1
				errorIndex.append(i)
				# print 'error data in: ', i, payload[i], demodData[i], ber
		print('test is done and BER={0}/{1}'.format(ber, payload.size))
		print('Error index: ',errorIndex[:20])
		print('Error index: ',errorIndex[-20:])
	else:
		print('Not enough data is received')
else:
	print('Preamble is not detected')

# plt.plot(adcData[5000:8000])
# plt.plot(freq)
# plt.plot(valid)
# plt.grid()
# plt.show()
