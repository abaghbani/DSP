import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import msvcrt
import sys

sys.path.insert(1, './Lib')
import Spectrum.freqPlot as fp
import Spectrum.Histogram2jpeg as hp
from IOs.WavFile import readWaveFile, writeWaveFile
from IOs.MatFile import readMatFile
from LPWAN.LoRa.LoRaDemodulation import LoRaDemodulation, LoRaFilterBank
from LPWAN.LoRa.LoRaModulation import LoRaModulation


if __name__=="__main__":

	path		= '../../traces/lora/'
	fileName = path + 'test1_Lora.wav'

	print('L: LoRa receiver')
	print('M: LoRa modem')
	print('N: read matfile')
	print('S: Specgram of sampled data')
	print('H: Histogram to Jpeg of sampled data')
	print('A: FFT plot')
	print('X: Exit')
	print('>> ')

	while True:
		if msvcrt.kbhit():
			c = msvcrt.getch().decode("utf-8")
			print(c)
			c = c.lower()
			
			if c == 'l':
				[fs, dataI, dataQ] = readWaveFile(fileName)
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
				[dataI, dataQ, fs] = readMatFile(fileName)
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
				
				fp.fftPlot(data, n=1, fs=fs)
				fp.specPlot(data, fs=fs)

				[dataI, dataQ, fs] = LoRaFilterBank(dataI, dataQ, fs, Bw=Bw, fMix=.992078e6, downSamplingRate=10) #int(fs/Bw)//2)
				LoRaDemodulation(dataI, dataQ, fs, Bw=Bw, SF=7)
			
			
			elif c == 's':
				[fs, dataI, dataQ] = readWaveFile(fileName)
				fp.specPlot(dataI)
				
			elif c == 'h':
				[fs, dataI, dataQ] = readWaveFile(fileName)
				hp.histogram2jpeg(dataI)
				
			elif c == 'a':
				[fs, dataI, dataQ] = readWaveFile(fileName)
				# adcData = np.multiply(adcData, np.cos((np.arange(adcData.size)*2*np.pi*120.0e+6/240.0e+6)+0.06287))
				fp.fftPlot(dataI)
				
			elif c == 'x':
				break
			print('Press new command:')

	print('Exit')
