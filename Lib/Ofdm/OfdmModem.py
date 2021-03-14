import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

from Ofdm.OfdmRfTransceiver import OfdmRfTransceiver
from Ofdm.OfdmChannelFilter import OfdmChannelFilter
from Ofdm.OfdmModulation import OfdmQamModulation
from Ofdm.OfdmDemodulation import OfdmDemodulation
from Common.Constant import Constant
from Common.ModemLib import ModemLib

C = Constant
myLib = ModemLib(0)

def OfdmTransmitter(channel, bit_number, modType, snr):
	payload = np.array(np.random.rand(bit_number) >= 0.5)
	(baseband, fs, bw) = OfdmQamModulation(payload)
	IfSig = OfdmRfTransceiver(baseband, fs, bw, channel, snr)
	return payload, IfSig

def OfdmReceiver2(adcSamples, channel):

	adcData = adcSamples
	pi= np.pi
	adcFS = C.AdcSamplingFrequency

	b = signal.remez(146+1, [0, 20, 30, 120], [1,0], fs=adcFS)
	downSamplingRate = 12
	maxValue = np.zeros(100)
	for k in range(maxValue.size):
		sigMix0 = adcData*np.exp((np.arange(adcData.size)*(-2j)*pi*(C.IfMixerFrequency+channel)/adcFS)+k*2j*pi/maxValue.size)*2
		sigFlt0 = np.convolve(b, sigMix0,'same')
		baseband = sigFlt0
		corrGuard = np.zeros(baseband.size)
		for i in range(80*downSamplingRate, 80*downSamplingRate+20):
			data = baseband[i-80*downSamplingRate:i:downSamplingRate]
			corrGuard[i] = np.correlate(data.real[:16], data.real[64:])
		maxValue[k] = np.max(corrGuard)
		#print(k, np.max(corrGuard))
		#plt.plot(corrGuard.real[:80*downSamplingRate+20])
		plt.plot(baseband.real[:80*downSamplingRate+20])
		plt.show()

	plt.plot(maxValue, 'bo')
	plt.show()

def OfdmReceiver3(adcSamples, channel):

	adcData = adcSamples
	pi= np.pi
	adcFS = C.AdcSamplingFrequency

	#plt.plot(adcData.real[:4000])
	#plt.plot(adcData.imag[:4000])

	b = signal.remez(146+1, [0, 20, 30, 120], [1,0], fs=adcFS)
	downSamplingRate = 12
	maxValue = np.zeros(100)
	indexValue = np.zeros(100)
	#for k in range(maxValue.size):
	for k in range(1):
		sigMix0 = adcData*np.exp((np.arange(adcData.size)*(-2j)*pi*(C.IfMixerFrequency+channel-0.016)/adcFS))*2
		sigFlt0 = np.convolve(b, sigMix0,'same')
		baseband = sigFlt0
		#plt.plot(baseband.real[:4000])
		#plt.plot(baseband.imag[:4000])
		#plt.legend(['real', 'imag'], loc='best')
		#plt.show()
		corrGuard = np.zeros(baseband.size)
		for i in range(4):
			data = baseband[i*1920:(i+1)*1920]
			corrGuard[i] = np.correlate(data.real[192*5:192*6], data.real[192*7:192*8])

			#ofdmNoCp = data[6*192:10*192]
			#ofdm = np.fft.fft(ofdmNoCp)
			#ofdm = np.hstack([ofdm[ofdm.size//2:], ofdm[:ofdm.size//2]])
			#ofdm = ofdm[ofdm.size//2-32:ofdm.size//2+32]/12
			#print(k)
			#plt.plot(ofdm.real, 'bo')
			#plt.plot(ofdm.imag, 'ro')
			#plt.grid()
			#plt.show()

			ofdmNoCp = data[6*192:10*192:12]
			ofdm = np.fft.fft(ofdmNoCp)
			ofdm = np.hstack([ofdm[ofdm.size//2:], ofdm[:ofdm.size//2]])
			#ofdm = ofdm[ofdm.size//2-32:ofdm.size//2+32]/12
			print(k)
			plt.plot(ofdm.real, 'bo')
			plt.plot(ofdm.imag, 'ro')
			#carrierIndex = np.array([2,6,10,14,18,22,30,34,38,42,46,50])+6
			#ofdm = ofdm[carrierIndex]*np.exp(1j*3*pi/4)
			#plt.plot(np.arctan2(ofdm.imag, ofdm.real)*128/pi, 'bo')
			plt.grid()
			plt.show()



		maxValue[k] = np.max(corrGuard)
		indexValue[k] = np.argmax(corrGuard)
		#print(k, np.max(corrGuard))
		#plt.plot(corrGuard.real[:80*downSamplingRate+20])
		#plt.plot(baseband.real[:80*downSamplingRate+20])
		#plt.show()

	plt.plot(maxValue, 'bo')
	plt.plot(indexValue, 'ro')
	plt.show()

def OfdmReceiver(adcSamples, channel):

	(baseband, fs) = OfdmChannelFilter(adcSamples, channel)
	#print('base Data: {} samples'.format(baseband.size))
	#print('base Data Min/Max: ',baseband.min(),baseband.max(), type(baseband[0]))
	data = OfdmDemodulation(baseband, fs, 'Direct')
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
	