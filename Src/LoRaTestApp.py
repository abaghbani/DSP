import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import msvcrt

from Spectrum.freqPlot import fftPlot, specPlot
from Spectrum.Histogram2jpeg import histogram2jpeg
from IOs.wavFile import readWaveFile, writeWaveFile
from IOs.rawFile import readRawFile, writeRawFile
from Pluto.Pluto import Pluto

from LPWAN.LoRa.LoRaDemodulation import LoRaDemodulation, LoRaFilterBank
from LPWAN.LoRa.LoRaModulation import LoRaModulation

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

if __name__=="__main__":
	
	print('L: LoRa receiver')
	print('M: LoRa modem')
	print('N: read matfile')
	print('B: Pluto Command')
	print('X: Exit')
	print('>> ')

	filename = '../Samples/LoRa/' + 'SDRSharp_20210316_152737Z_868000000Hz_IQ.wav'
	
	while True:
		if msvcrt.kbhit():
			c = msvcrt.getch().decode("utf-8")
			print(c)
			c = c.lower()
			if c == 'l':
				[fs, dataI, dataQ] = readWaveFile(filename)
				[dataI, dataQ, fs] = LoRaFilterBank(dataI, dataQ, fs, Bw=125.0e3, fMix=-100.0e3, downSamplingRate=1)
				specPlot(dataI+1j*dataQ, fs=fs)
				LoRaDemodulation(dataI, dataQ, fs, Bw=125.0e3, SF=7)
			
			elif c == 'm':
				SF = 7
				Bw = 20800.0
				fs = (1*Bw)

				payload = np.random.rand(5)*(np.exp2(SF))
				payload = payload.astype('int')
				baseband = LoRaModulation(payload, fLow=-Bw/2, fHigh=Bw/2, SF=SF, Fs=fs)
				LoRaDemodulation(baseband.real, baseband.imag, fs, Bw, SF )
			
			elif c == 'n':
				[dataI, dataQ, fs] = readMatFile('../Samples/LoRa/pluto-capture-5.mat', 'cf_ad9361_lpc_voltage0', 'cf_ad9361_lpc_voltage1')
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
				
				fftPlot(data, n=1, fs=fs)
				specPlot(data, fs=fs)

				[dataI, dataQ, fs] = LoRaFilterBank(dataI, dataQ, fs, Bw=Bw, fMix=.992078e6, downSamplingRate=10) #int(fs/Bw)//2)
				LoRaDemodulation(dataI, dataQ, fs, Bw=Bw, SF=7)
		
			elif c == 'b':
				PlutoCommand()

			elif c == 'x':
				break
			print('Press new command:')

	print('Exit')