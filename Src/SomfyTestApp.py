import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import msvcrt
import sys

sys.path.insert(1, './Lib')
from Spectrum.ModemLib import ModemLib
from Spectrum.Histogram2jpeg import histogram2jpeg
from IOs.WavFile import readWaveFile, writeWaveFile
from Somfy.SomfyDemodulation import SomfyDemodulation, SomfyFilterBank

if __name__=="__main__":
	
	mLib = ModemLib(0)
	fileName = '../Samples/' + 'SDRSharp_20210321_171718Z_433200000Hz_IQ.wav'

	print('l: Somfy demod')
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
				print('Sampling freq = ', fs)
				print(dataI.shape, dataQ.shape)	
				#Bw = 20800.0
				Bw = 125000.0

				#dataI = dataI[int(1.5e6):int(2.2e6)]
				#dataQ = dataQ[int(1.5e6):int(2.2e6)]
				#dataI = dataI[int(17.2e6):int(17.4e6)]
				#dataQ = dataQ[int(17.2e6):int(17.4e6)]
				# dataI = dataI[int(5.6e6):int(6.5e6)]
				# dataQ = dataQ[int(5.6e6):int(6.5e6)]
				# dataI = dataI[int(16.0e6):int(17.0e6)]
				# dataQ = dataQ[int(16.0e6):int(17.0e6)]
				
				# data = dataI+1j*dataQ
				# mLib.fftPlot(data, n=1, fs=fs)
				# mLib.specPlot(data, fs=fs)

				Bw = 5.0e3
				[dataI, dataQ, fs] = SomfyFilterBank(dataI, dataQ, fs, Bw=Bw, fMix=-193.44e3, downSamplingRate=1)
				SomfyDemodulation(dataI, dataQ, fs, Bw=Bw, SF=7)
			
			elif c == 's':
				[fs, dataI, dataQ] = readWaveFile(fileName)
				mLib.specPlot(dataI)
				
			elif c == 'h':
				[fs, dataI, dataQ] = readWaveFile(fileName)
				histogram2jpeg(dataI)
				
			elif c == 'a':
				[fs, dataI, dataQ] = readWaveFile(fileName)
				# adcData = np.multiply(adcData, np.cos((np.arange(adcData.size)*2*np.pi*120.0e+6/240.0e+6)+0.06287))
				mLib.fftPlot(dataI)
				
			elif c == 'x':
				break
			print('Press new command:')

	print('Exit')
