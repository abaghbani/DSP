import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

from Ofdm.OfdmRfTransceiver import OfdmRfTransceiver
from Ofdm.OfdmChannelFilter import OfdmChannelFilter
from Ofdm.OfdmModulation import OfdmQamModulation
from Ofdm.OfdmDemodulation import OfdmDemodulation
from Ofdm.Constant import Constant as C

def OfdmTransmitter(channel, bit_number, modType, snr):
	payload = np.array(np.random.rand(bit_number) >= 0.5)
	(baseband, fs, bw) = OfdmQamModulation(payload)
	IfSig = OfdmRfTransceiver(baseband, fs, bw, channel, snr)
	return payload, IfSig

def OfdmReceiver(adcSamples, channel):
	(baseband, fs) = OfdmChannelFilter(adcSamples, channel)
	#print('base Data: {} samples'.format(baseband.size))
	#print('base Data Min/Max: ',baseband.min(),baseband.max(), type(baseband[0]))
	data = OfdmDemodulation(baseband, fs, 'DownSampling')
	return data

def OfdmModem(channel, bit_number, snr):
	(payload, IfSig) = OfdmTransmitter(channel, bit_number, 0, snr)

	lnaGain = 200
	adcData = (IfSig*lnaGain).astype('int16')
	fp = np.memmap('ofdmData.bttraw', mode='w+', dtype=np.dtype('<h'), shape=(1,adcData.size))
	fp[:] = adcData[:]
	
	print('transmit bit number=',payload.size)
	print('ADC Data: {} samples'.format(adcData.size))
	print('ADC Data Min/Max: ',adcData.min(),adcData.max(), type(adcData[0]))

	adcDataNoGain = adcData/lnaGain
	rxData = OfdmReceiver(adcDataNoGain, channel)
	
def OfdmBaseband(channel, bit_number, snr):
	payload = np.array(np.random.rand(bit_number) >= 0.5)
	(baseband, fs, bw) = OfdmQamModulation(payload)
	data = OfdmDemodulation(baseband, 20)
	