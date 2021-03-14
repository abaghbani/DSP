import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import scipy.io as sio
import matplotlib.pyplot as plt
import msvcrt
#import adi

import sys
import os

sys.path.insert(1, 'C:/Users/Akbar/Documents/Projects/Python/DSPApp/Lib/Python')

from Common.ModemLib import ModemLib
from Common.Constant import Constant
from Common.DistortionCalc import Thd, ThdN
from Common.WriteToWav import WriteToWav
from Common.KarplusStrong import KarplusStrong
from Common.Histogram2jpeg import histogram2jpeg
from ChannelFilter.ChannelDecimate import ChannelDecimate
from ChannelFilter.ChannelFilter import ChannelFilter
from RfModel.RfTransceiver import RfTransceiver
from Gfsk.GfskModulation import GfskModulation
from Gfsk.GfskDemodulation import GfskDemodulation
from Gfsk.GfskModem import GfskModem, GfskReceiver
from Dpsk.DpskModem import DpskModem, DpskReceiver
from Oqpsk.WpanModem import WpanModem, WpanReceiver
from Ofdm.OfdmModem import OfdmModem, OfdmBaseband, OfdmReceiver
from LPWAN.LoRa.LoRaDemodulation import LoRaDemodulation, LoRaFilterBank
from LPWAN.LoRa.LoRaModulation import LoRaModulation
from Pluto.Pluto import Pluto


mLib = ModemLib(0)
C = Constant

def THDtest():
	fs = 1000
	t = np.arange(0,1, 1/fs)
	x = 2*np.cos(2*np.pi*100*t)+0.01*np.cos(2*np.pi*200*t)+0.005*np.cos(2*np.pi*300*t)

	#plt.plot(x)
	#plt.grid()
	#plt.show()

	#mLib.fftPlot(x, fs=1000)
	Thd(x, fs)
	ThdN(x, fs)

def rms_flat(a):
    return np.sqrt(np.mean(np.absolute(a)**2))

def find_range(f, x):
    for i in np.arange(x+1, len(f)):
        if f[i+1] >= f[i]:
            uppermin = i
            break
    for i in np.arange(x-1, 0, -1):
        if f[i] <= f[i-1]:
            lowermin = i + 1
            break
    return (lowermin, uppermin)

def rms_harmony(f, x):
    total_rms_harmony = 0
    for i in np.arange(2*x, len(f), x):
        total_rms_harmony += np.abs(f[i])**2
    
    return (np.sqrt(total_rms_harmony))

def WaveGen():
	sRate = 44100.0
	t = np.arange(0, 2.0, 1/sRate)
	samples = np.sin(2*np.pi*440.0*t)
	WriteToWav('test1.wav', samples, sRate)

	sampleRate = 44100
	samples = KarplusStrong(1*sampleRate, sampleRate, 391)
	WriteToWav('test2.wav', samples, sampleRate)


	samples -= np.mean(samples)
	windowed = samples * signal.blackmanharris(len(samples))

	f = np.fft.rfft(windowed)
	i = np.argmax(np.abs(f))
	ThdValue = rms_harmony(f, i) / np.abs(f[i])

	print('freq = %f and thd = %f' %(i, ThdValue))

	#Thd(samples, sampleRate)
	#ThdN(samples, sampleRate)

	plt.plot(np.abs(f))
	#plt.legend(['test1', 'test2'], loc='best')
	plt.grid()
	plt.show()

def readMatFile(fileName, streamI, streamQ):
	readdata = sio.loadmat(fileName)
	print(readdata.keys())
	print(readdata['__header__'])
	dataI = np.hstack(readdata[streamI])
	dataQ = np.hstack(readdata[streamQ])
	fs = 1 # fixme!!! still is not clear how to read, this info is not available in mat file.
	return (dataI, dataQ, fs)

def readWaveFile(fileName):
	[fs, readdata] = wavfile.read(fileName)
	dataI = readdata[:,0]
	dataQ = readdata[:,1]
	return (fs, dataI, dataQ)

def writeWaveFile(data, fs, fileName):
	wavfile.write(fileName, fs, np.array([(data.real).astype('float'), (data.imag).astype('float')]).reshape(-1,2))

def readRawFile(fileName):
	adcData = np.memmap(fileName, mode='r', dtype=np.dtype('<h'))
	adcData = (adcData<<(16-12))>>(16-12)
	print('ADC Data: {} samples'.format(adcData.size))
	print('ADC Data Min/Max: ',adcData.min(),adcData.max(), type(adcData[0]))
	return adcData

def writeRawFile(fileName, data):
	fp = np.memmap(fileName, mode='w+', dtype=np.dtype('<h'), shape=(1, data.size))
	fp[:] = data[:]

def generateGfsk(fileName, bitNumber):
	AccessAddress = np.array(list(bin(0x71764129)[2:]), dtype='int')*2-1
	payload = np.concatenate((np.flip(AccessAddress), np.array((np.random.rand(bitNumber) >= 0.5)*2-1)), axis=None)
	txBaseband = GfskModulation(payload)
	txData = np.zeros(0)
	for i in range(40):
		IfSig = RfTransceiver(txBaseband, i*2, C.SymboleRate.R1M, 20)
		txData = np.append(txData, IfSig)
	writeRawFile(fileName, txData.astype('int16'))

def PlutoCommand():
	print('R: Receiving')
	print('T: Transmitting')
	print('W: Receiving/Transmitting')
	print('d: debugging')
	print('X: Exit')
	print('>> ')

	mPluto = Pluto(0)
	while True:
		if msvcrt.kbhit():
			c = msvcrt.getch().decode("utf-8")
			print(c)
			c = c.lower()
			if c == 'r':
				[samples, fs] = mPluto.Read(5.0e6, 5.0e6, 868.0e6, 50.0, int(2e6))
				print('sample freq: ', fs, 'sample size: ', samples.size, 'sample min/max: ', samples.min(), samples.max())
				mLib.fftPlot(samples.real, n=1, fs=fs)
				np.save('testPluto', samples)

			elif c == 't':
				t = np.arange(10000)/10.0e6
				samples = 0.5*np.exp(2.0j*np.pi*100e3*t)
				samples *= 2**14 # The PlutoSDR expects samples to be between -2^14 and +2^14, not -1 and +1 like some SDRs

				fs = mPluto.Write(10.0e6, 10.0e6, 868.0e6, -30.0, samples, 3)
				mLib.fftPlot(samples.real, n=1, fs=fs)

			elif c == 'w':
				t = np.arange(10000)/3.0e6
				samples = 0.99*np.exp(2.0j*np.pi*120e3*t)
				#samples = np.array((np.random.rand(2000) >= 0.5)*0.5)
				#samples = samples.repeat(200)
				samples = np.concatenate((np.zeros(5000), samples, np.zeros(5000)), axis=None)
				
				samples *= 2**14 # The PlutoSDR expects samples to be between -2^14 and +2^14, not -1 and +1 like some SDRs

				[rxSamples, fs] = mPluto.ReadWrite(3.0e6, 3.0e6, 868.0e6, -30.0, 50.0, samples, int(2e6))
				mLib.fftPlot(rxSamples.real, n=1, fs=fs)
				mLib.specPlot(rxSamples, fs=fs)
				#writeWaveFile(rxSamples, int(fs), 'test1.wav')
				np.save('testPluto', rxSamples)

			elif c == 'd':
				rxSamples = np.load('testPluto.npy')
				print('data size = ', rxSamples.shape)
				fs = 3.0e6
				rxSamples = np.multiply(rxSamples, np.exp((np.arange(rxSamples.size)*(-2j)*np.pi*120.0e3/fs)+0.06287))

				mLib.fftPlot(rxSamples.real, n=1, fs=fs)
				mLib.specPlot(rxSamples, fs=fs)
				plt.plot(rxSamples[:10000])
				plt.show()

			elif c == 's':
				mPluto.DDS()
			elif c == 'p':
				mPluto.DDS_stop()

			elif c == 'x':
				break
			print('Press new command:')
	print('Exit from Pluto command.')

def debugging():
	
	adcData = readRawFile('data1.bttraw')
	adcData = adcData[int(36e5):int(37e5)]
	mLib.specPlot(adcData)
	(data4M, data2M, data1M) = ChannelDecimate(adcData)
	for i in range(40,50,2):
		(dataI, dataQ, fs) = ChannelFilter(data4M, data2M, data1M, i, C.ChannelFilterType.Gfsk2M)
		(freq, rssi, valid, data) = GfskDemodulation(dataI, dataQ, fs)

		dataLength = freq.size
		detected = np.zeros(dataLength, dtype=bool)
		demodData = np.zeros(0, dtype='int')
		demodRssi = np.zeros(0, dtype='int')
		for k in range(1, dataLength):
			if valid[k] == 1:
				demodData = np.append(demodData, data[k]*2-1) 
				demodRssi = np.append(demodRssi, rssi[k]) 

		demodDataConv = np.convolve(demodData, C.GfskPreamble[:8], mode='same')
		syncPosition = np.where(np.abs(demodDataConv)==8)[0]
		if syncPosition.size != 0:
			print('Preamble is detected.')
			print('ch=',i, 'rssi=', demodRssi[int(syncPosition[0])])
			plt.plot(freq)
			plt.plot(rssi)
			plt.grid()
			plt.show()

if __name__=="__main__":
	
	print('F: Gfsk modem')
	print('R: Gfsk receiver')
	print('P: Dpsk modem')
	print('T: Dpsk receiver')
	print('G: Generate Gfsk in all channel')
	print('S: Specgram of sampled data')
	print('H: Histogram to Jpeg of sampled data')
	print('A: FFT plot')
	print('Q: Wpan modem')
	print('E: Ofdm modem')	
	print('Z: Ofdm receiver')	
	print('L: LoRa receiver')
	print('M: LoRa modem')
	print('N: read matfile')
	print('B: Pluto Command')
	print('')
	print('C: THD test')
	print('W: Wav file generator')
	print('D: Debugging')
	print('X: Exit')
	print('>> ')

	while True:
		if msvcrt.kbhit():
			c = msvcrt.getch().decode("utf-8")
			print(c)
			c = c.lower()
			if c == 'f':
				GfskModem(20, 200, C.GfskModulationType.Gfsk2M, 25, C.ChannelFilterType.Gfsk2M)
			
			elif c == 'r':
				adcData = readRawFile('../Lib/Python/Samples/gfskData.bttraw')
				(freq, rssi, valid, data) = GfskReceiver(adcData, 20, C.ChannelFilterType.Gfsk2M)
				plt.plot(freq)
				plt.plot(valid)
				plt.grid()
				plt.show()
			
			elif c == 'p':
				#DpskModem(20, 200, C.DpskModulationType.Edr2, 5, C.ChannelFilterType.Dpsk4M.ch1M )
				DpskModem(20, 2000, C.DpskModulationType.Edr3, 15, C.ChannelFilterType.Dpsk1M)
			
			elif c == 't':
				adcData = readRawFile('../Lib/Python/Samples/dpskData.bttraw')
				(rssi, sync, valid, data) = DpskReceiver(adcData, 20, C.ChannelFilterType.Dpsk1M)
				plt.plot(data)
				plt.plot(valid)
				plt.plot(sync)
				plt.grid()
				plt.show()
	
			elif c == 'g':
				generateGfsk('gfskAllChannels.bttraw', 1600)

			elif c == 's':
				adcData = readRawFile('../Lib/Python/Samples/gfskData.bttraw')
				mLib.specPlot(adcData[:5000000])
				
			elif c == 'h':
				adcData = readRawFile('../Lib/Python/Samples/gfskData.bttraw')
				histogram2jpeg(adcData)
				
			elif c == 'a':
				adcData = readRawFile('../Lib/Python/Samples/gfskData.bttraw')
				adcData = np.multiply(adcData, np.cos((np.arange(adcData.size)*2*np.pi*120.0e+6/240.0e+6)+0.06287))
				mLib.fftPlot(adcData, fs = 240)
				
			elif c == 'q':
				WpanModem(15, 20, 50)
			
			elif c == 'e':
				OfdmModem(20, 2000, 50)
				#OfdmBaseband(20, 2000, 50)
			
			elif c == 'z':
				adcData = readRawFile('wifi-6Mbps-ch3-1.bttraw')
				lnaGain = 4000
				adcDataNoGain = adcData[67800+1034:80000]/lnaGain
				rxData = OfdmReceiver(adcDataNoGain, 20)
			
			elif c == 'l':
				[fs, dataI, dataQ] = readWaveFile('../Lib/Python/Samples/LoRa/SDRSharp_20160109_195047Z_869600kHz_IQ_BW21CR48SF6PL64x00.wav')
				[dataI, dataQ, fs] = LoRaFilterBank(dataI, dataQ, fs, Bw=20800.0, fMix=251500.0, downSamplingRate=10)
				LoRaDemodulation(dataI, dataQ, fs, Bw=20800, SF=6)
			
			elif c == 'm':
				SF = 7
				Bw = 20800.0
				fs = (1*Bw)

				payload = np.random.rand(5)*(np.exp2(SF))
				payload = payload.astype('int')
				baseband = LoRaModulation(payload, fLow=-Bw/2, fHigh=Bw/2, SF=SF, Fs=fs)
				LoRaDemodulation(baseband.real, baseband.imag, fs, Bw, SF )
			
			elif c == 'n':
				[dataI, dataQ, fs] = readMatFile('../Lib/Python/Samples/LoRa/pluto-capture-5.mat', 'cf_ad9361_lpc_voltage0', 'cf_ad9361_lpc_voltage1')
				print(dataI.shape, dataQ.shape)	
				fs = 16.0e+6
				Bw = 125000.0

				#dataI = dataI[int(1.5e6):int(2.2e6)]
				#dataQ = dataQ[int(1.5e6):int(2.2e6)]
				#dataI = dataI[int(17.2e6):int(17.4e6)]
				#dataQ = dataQ[int(17.2e6):int(17.4e6)]
				#dataI = dataI[:int(2.6e5)]
				#dataQ = dataQ[:int(2.6e5)]
				data = dataI+1j*dataQ
				
				mLib.fftPlot(data, n=1, fs=fs)
				mLib.specPlot(data, fs=fs)

				[dataI, dataQ, fs] = LoRaFilterBank(dataI, dataQ, fs, Bw=Bw, fMix=.992078e6, downSamplingRate=10) #int(fs/Bw)//2)
				LoRaDemodulation(dataI, dataQ, fs, Bw=Bw, SF=7)
		
			elif c == 'b':
				PlutoCommand()

			elif c == 'c':
				THDtest()
			
			elif c == 'w':
				WaveGen()
			
			elif c == 'd':
				debugging()
			
			elif c == 'x':
				break
			print('Press new command:')

	print('Exit')