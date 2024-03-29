import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import logging as log
import os
import sys
import msvcrt
import sys

sys.path.insert(1, './Lib')
import Spectrum.freqPlot as fp
import Spectrum.Histogram2jpeg as hp
import IOs.WavFile as wav

import Somfy.SomfyDemodulation as somfy
import ClockRecovery.ClockRecovery as cr

if __name__=="__main__":

	verbose = True
	if verbose:
		log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG)
		#log.info("Verbose output.")
		#log.info("===============")
	else:
		log.basicConfig(format="%(levelname)s: %(message)s")
	log.getLogger('matplotlib.font_manager').disabled = True
	log.getLogger('PIL').setLevel(log.INFO)

	print('R: Read sample file')
	print('K: Somfy demod')
	print('N: Hormann demod')
	print('M: Hormann extract data')
	
	print('')
	print('S: Spectogram of sampled data')
	print('H: Histogram to Jpeg of sampled data')
	print('A: FFT plot')
	print('X: Exit')
	print('>> ')

	path		= '../../traces/somfy/'
	path_out	= path + '../somfy_output/'
	filename	= path + 'SDRSharp_20210407_200432Z_433200000Hz_IQ.wav'
	auto_search	= False

	if auto_search:
		logFolderContent = os.listdir(path)
		logFolderContent.reverse()
		for i in range(len(logFolderContent)):
			if logFolderContent[i].startswith('SDRSharp_'):
				filename = path+logFolderContent[i]
				print("latest found file is: ", filename)
				break

	while True:
		if msvcrt.kbhit():
			c = msvcrt.getch().decode("utf-8")
			print(c)
			c = c.lower()

			if c == 'r':
				[fs, dataI, dataQ] = wav.readWaveFile(filename)
				print('Sampling freq = ', fs)
				print(dataI.shape, dataQ.shape)
				dataI = dataI[int(0.0e6):int(8.5e6)]
				dataQ = dataQ[int(0.0e6):int(8.5e6)]

				data = dataI+1j*dataQ
				fp.fftPlot(data, n=1, fs=fs)
				fp.specPlot(data, fs=fs)

			elif c == 'k':
				fileName_local = path + 'SDRSharp_20210612_204730Z_433200000Hz_IQ.wav'
				[fs, dataI, dataQ] = wav.readWaveFile(fileName_local)
				print('Sampling freq = ', fs)
				print(dataI.shape, dataQ.shape)

				dataI = dataI[int(3.0e6):int(5.0e6)]
				dataQ = dataQ[int(3.0e6):int(5.0e6)]

				data = dataI+1j*dataQ
				fp.fftPlot(data, n=1, fs=fs)
				fp.specPlot(data, fs=fs)

				Bw = 5.0e3
				[dataI, dataQ, fs] = somfy.SomfyFilterBank(dataI, dataQ, fs, Bw=Bw, fMix=-210.0e3, downSamplingRate=40)
				# np.save(path_out+'test4_somfy', [dataI, dataQ])
				somfy.SomfyDemod(dataI, dataQ, fs)

			elif c == 'n':
				fileName_local = path + '../hormann/SDRSharp_20210912_172204Z_868000000Hz_IQ.wav'
				[fs, dataI, dataQ] = wav.readWaveFile(fileName_local)
				print('Sampling freq = ', fs)
				print(dataI.shape, dataQ.shape)

				dataI = dataI[int(0.0e6):int(8.0e6)]
				dataQ = dataQ[int(0.0e6):int(8.0e6)]
				
				data = dataI+1j*dataQ
				fp.specPlot(data, fs=fs)

				[dataI, dataQ, fs] = somfy.SomfyFilterBank(dataI, dataQ, fs, Bw=20.0E3, fMix=0.080e6, downSamplingRate=1)
				data = somfy.HormannDemod(dataI, dataQ, fs)
				np.save(path_out + 'Hormann_data', data)

			elif c == 'm':
				data = np.load('Hormann_data.npy')
				print(f'data size = {data.size}')
				data = data[int(0e6):int(500e6):4]
				last_edge = 0
				bits = []
				packet = []
				for i in range(1, data.size):
					if data[i] != data[i-1]:
						len = i-last_edge
						print(f'state: {data[i-1]}, duration: {i-last_edge}')
						if (len >= 400) and (len <= 650):
							bits.append(data[i-1])
						elif (len >= 800) and (len <= 1300):
							bits.append(data[i-1])
							bits.append(data[i-1])
						elif len > 1400:
							packet.append(np.array(bits))
							bits = []
						last_edge = i
				
				for dd in packet:
					# print(f'packet# {i}: {packet[i]}')
					print(f'packet# : {dd}')
				
				# print('data extracted: ', [hex(i) for i in data_decoded])

				plt.plot(data)
				plt.grid()
				plt.show()
				
			elif c == 's':
				[fs, dataI, dataQ] = wav.readWaveFile(filename)
				fp.specPlot(dataI)

			elif c == 'h':
				[fs, dataI, dataQ] = wav.readWaveFile(filename)
				hp.histogram2jpeg(dataI)

			elif c == 'a':
				[fs, dataI, dataQ] = wav.readWaveFile(filename)
				# adcData = np.multiply(adcData, np.cos((np.arange(adcData.size)*2*np.pi*120.0e+6/240.0e+6)+0.06287))
				fp.fftPlot(dataI)

			elif c == 'x':
				break
			print('Press new command:')

	print('Exit')
