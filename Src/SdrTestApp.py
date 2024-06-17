import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import msvcrt
import os
import logging as log

import Spectrum.freqPlot as fp
import Filter.filterDesign as fd
from IOs import rawFile
import Common.ModemLib as myLib
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

	print('H: raw data')
	print('X: Exit')
	print('>> ')
	
	samples_path = 'D:/Documents/Dsp/Samples/'
	path = 'D:/Documents/Dsp/Samples/SDR/'
	logFolderContent = os.listdir(path)
	logFolderContent.reverse()
	for i in range(len(logFolderContent)):
		if logFolderContent[i].startswith('Wpc_adc1'):
			if logFolderContent[i][7] == '1':
				filename1 = path+logFolderContent[i]
				filename2 = filename1.replace('_adc1_', '_adc2_')
			else:
				filename2 = path+logFolderContent[i]
				filename1 = filename2.replace('_adc2_', '_adc1_')
			print("latest found file is (ch1): ", filename1)
			print("latest found file is (ch2): ", filename2)
			break

	# sample rate is set to 4MSPS
	fs = 4.0e6

	while True:
		if msvcrt.kbhit():
			c = msvcrt.getch().decode("utf-8")
			print(c)
			c = c.lower()

			if c == 'h':
				
				## sdr test
				filename_local = samples_path + 'SDR/test14.bin'
				adcData = rawFile.readRawFile(filename_local, 16)
				fs = 4.0e6

				I0 = adcData[0::2]
				Q0 = adcData[1::2]
				I1 = adcData[2::4]
				Q1 = adcData[3::4]
				plt.plot(I0[:100000])
				plt.plot(Q0[:100000])
				plt.grid()
				plt.show()
				
				plt.plot(I1[:10000])
				plt.plot(Q1[:10000])
				plt.grid()
				plt.show()
				
				
				print(f'ADC DataI Min/Max (in selected range): {I0.min()}, {I0.max()} , {type(I0[0])}')
				print(f'ADC DataQ Min/Max (in selected range): {Q0.min()}, {Q0.max()} , {type(Q0[0])}')
				data0 = I0 +1j*Q0
				data1 = I1 +1j*Q1

				fp.fftPlot(data0, fs=fs)
				fp.fftPlot(data1, fs=fs)

			elif c == 'x':
				break
			print()
			print('==================')
			print('Press new command:')

	print('Exit')
