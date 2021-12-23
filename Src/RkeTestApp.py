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
	print('B: Remote Keylees Entry (BMW@868MHz)')
	print('V: Remote Keylees Entry 2 (BMW@868MHz)')
	
	print('')
	print('S: Spectogram of sampled data')
	print('H: Histogram to Jpeg of sampled data')
	print('A: FFT plot')
	print('X: Exit')
	print('>> ')

	path		= '../../traces/RKE/'
	path_out	= path + '../RKE_output/'
	filename	= path + 'SDRSharp_20210815_142820Z_868000266Hz_IQ.wav'
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


			elif c == 'b':
				# fileName_local = path + 'SDRSharp_20210704_200704Z_868000000Hz_IQ.wav'
				fileName_local = path + 'RKE/' + 'SDRSharp_20210815_142820Z_868000266Hz_IQ.wav'
				[fs, dataI, dataQ] = wav.readWaveFile(fileName_local)
				print('Sampling freq = ', fs)
				print(dataI.shape, dataQ.shape)
				
				dataI = dataI[int(0.7e6):int(5.4e6)]
				dataQ = dataQ[int(0.7e6):int(5.4e6)]

				data = dataI+1j*dataQ
				fp.fftPlot(data, n=1, fs=fs)
				fp.specPlot(data, fs=fs)

				Bw = 200.0e3
				[dataI, dataQ, fs] = somfy.SomfyFilterBank(dataI, dataQ, fs, Bw=Bw, fMix=-300.0e3, downSamplingRate=1)
				# np.save('test4_somfy', [dataI, dataQ])
				# SomfyDemod(dataI, dataQ, fs)x
				fp.fftPlot(dataI+1j*dataQ, n=1, fs=fs)
				# fp.specPlot(dataI+1j*dataQ, fs=fs)

				phase = np.arctan2(dataQ, dataI)
				freq = np.diff(phase)
				freq[freq>(1.0*np.pi)] -= 2*np.pi 
				freq[freq<(-1.0*np.pi)] += 2*np.pi 
				freq[freq>(0.5*np.pi)] -= np.pi 
				freq[freq<(-0.5*np.pi)] += np.pi 
				# plt.plot(phase)
				plt.plot(freq)
				plt.legend(['phase', 'freq'])
				plt.show()

				np.save(path_out + 'bmw_key_2', freq)

			elif c == 'v':
				
				data = np.load(path_out + 'bmw_key_4.npy')
				print('sample number =', data.size)

				# data = data[94000:]
				# data = data[1000000:1850000]
				# data = data[3550000:3900000]

				## period estimation :
				data_digi = cr.digitized(data)
				period = cr.period_calc(data_digi)
				file = open(path_out + "period.txt","w+")
				file.write(f'period array: {[v for v in period[0]]}')
				file.close
				print([v for v in period[0][:100]])
				
				## period
				period = 95

				bit_out, sampled_data = cr.Early_late(data, period = period, delta=5)

				file = open(path_out + "bmw_key_out_5.txt","w+")
				file.write(f'extracted data is: {[v for v in bit_out]}')
				file.close

				print([v for v in bit_out])
				plt.plot(data)
				plt.plot(sampled_data, '.')
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
