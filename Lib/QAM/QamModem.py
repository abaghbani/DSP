import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

import Spectrum as sp
import Filter as fd
import RfModel as rf

from .QamModulation import *
from .QamDemodulation import *
from .Constant import *
C = Constant()


def QamTransmitter(channel, byte_number, type, upsample_rate, snr):
	payload = (np.random.rand(byte_number)*256).astype(np.uint8)
	txBaseband = modulation(payload, type)

	tx_upsampled = rf.UpSampling(txBaseband, 5/100, upsample_rate)
	tx_mixer = rf.Mixer(tx_upsampled, channel/100, 3*np.pi/8)
	## add noise on all sample
	#tx_sig = tx_mixer.real + rf.WhiteNoise(tx_mixer, snr)
	## add noise only over payload (to always detect preamble)
	tx_sig = tx_mixer.real + rf.WhiteNoise(tx_mixer, snr)*( [0]*(4+4+1)*2*upsample_rate + [1]*(tx_mixer.size-(4+4+1)*2*upsample_rate) )

	#plt.plot(txBaseband.real, txBaseband.imag, 'bo')
	#plt.grid()
	#plt.show()
	#sp.fftPlot(txBaseband.real, txBaseband.imag, n=2)
	#sp.fftPlot(tx_upsampled.real, tx_upsampled.imag, n=2)
	#sp.fftPlot(tx_mixer.real, tx_mixer.imag, n=2)
	#sp.fftPlot(tx_sig.real)

	return payload, tx_sig

def QamReceiver(adcSamples, channel, over_sample_rate):
	rx_mixer = rf.Mixer(adcSamples, -1*channel/100, 0*np.pi/8)
	b = signal.remez(100+1, [0, .1, 0.2, 0.5], [1, 1e-4])
	baseband = np.convolve(b, rx_mixer, 'same')
	## first level of down_sampling (to remove mixer effect)
	baseband = baseband[4::10]
	#sp.fftPlot(baseband.real, baseband.imag, n=2)

	data = Demodulation(baseband, over_sample_rate//10)
	
	return data

def QamModem(channel, byte_number, modulation_type, snr):
	upsample_rate = 100
	(payload, IfSig) = QamTransmitter(channel, byte_number, modulation_type, upsample_rate, snr)

	lnaGain = 0.9*(2**15)/np.abs(IfSig).max()
	adcData = (IfSig*lnaGain).astype('int16')
	fp = np.memmap('QamData.bin', mode='w+', dtype=np.dtype('<h'), shape=(1,adcData.size))
	fp[:] = adcData[:]
	
	print(f'transmit {payload.size=}')
	print(f'ADC Data: {adcData.size} samples')
	print(f'ADC Data Min = {adcData.min()}, Max = {adcData.max()}, type={type(adcData[0])}')

	print(f'transmit payload : {[hex(val)[2:] for val in payload]}')

	rxData = QamReceiver(adcData, channel, upsample_rate)
