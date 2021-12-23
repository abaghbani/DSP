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

from Wpc import WpcReceiver as demod_shift
from Wpc import WpcReceiverNew as demod
from Wpc import WpcHdlModel
from Wpc import WpcPacket

from Wpc import WpcBER

if __name__=="__main__":
	
	verbose = True
	if verbose:
		log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG)
		#log.info("Verbose output.")
		#log.info("===============")
	else:
		log.basicConfig(format="%(levelname)s: %(message)s")
	log.getLogger('matplotlib.font_manager').disabled = True

	print('G: process HDL debug data')
	print('V: process extract bit')
	print('D: process raw data (final model)')
	print('A: process raw data (shift model)')
	print('H: raw data')
	print('M: ASK Modulation Qi (BER test)')
	print('N: FSK Modulation Qi (BER test)')
	print('F: generate filters coef')
	print('X: Exit')
	print('>> ')
	
	samples_path = 'D:/Documents/Dsp/Samples/'
	path = 'D:/Documents/Dsp/Samples/Wpc/'
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

	# sample rate is set to 5MSPS
	fs = 5.0e6

	while True:
		if msvcrt.kbhit():
			c = msvcrt.getch().decode("utf-8")
			print(c)
			c = c.lower()

			if c == 'g':  ## process raw flt data
				#filename_local  = '../Samples/Wpc/' + 'debug_data_20210818_103530.bttraw'
				#rawData = np.fromfile(filename_local, dtype='int16')
				rawData = np.fromfile(filename2, dtype='int16')
				
				mag_ac = rawData[0::4]
				mag_avg = rawData[1::4].astype('uint16')
				freq_ac = rawData[2::4]
				freq_avg = rawData[3::4].astype('uint16')
				mag = mag_ac[np.nonzero(mag_ac & 2)]
				mag_avg = mag_avg[np.nonzero(mag_ac & 2)]
				freq = freq_ac[np.nonzero(freq_ac & 2)]
				freq_avg = freq_avg[np.nonzero(freq_ac & 2)]

				#ask_bit = (mag & 1)*1e6
				#fsk_bit = (freq & 1)*1e7
				#WpcHdlAskBitExtractor(mag)
				#WpcHdlFskBitExtractor(freq)
				
				plt.plot(mag)
				plt.plot(mag_avg)
				#plt.plot(ask_bit)
				plt.grid()
				plt.show()
				plt.plot(freq)
				plt.plot(freq_avg)
				#plt.plot(fsk_bit)
				plt.grid()
				plt.show()

			elif c == 'v':  ## process extract bit,rssi,freq
				#filename_local = '../Samples/Wpc/' + 'bit_data_20210818_103125.bttraw'
				#rawData = np.fromfile(filename_local, dtype='uint16')
				rawData = np.fromfile(filename1, dtype='uint16')
				
				demodData = rawData[0::4]
				#timestamp = rawData[1::4].astype('uint16')
				timestamp = rawData[1::4]
				freq = rawData[2::4].astype('uint16')
				rssi = rawData[3::4].astype('uint16')

				freq = freq[np.nonzero(demodData&0xf0 == 0x0)]
				rssi = rssi[np.nonzero(demodData&0xf0 == 0x0)]
				data = demodData[np.nonzero(demodData&0xf0 == 0x0)]

				bit_ask = data & 1
				bit_fsk = (data[np.nonzero(data & 8)] & 4)//4
				ask_index = np.arange(bit_ask.size)
				fsk_index = (ask_index[np.nonzero(data & 8)])

				#plt.plot(demodData)
				#plt.show()
				#plt.plot(timestamp)
				#plt.show()
				#plt.plot(freq)
				#plt.show()
				#plt.plot(rssi)
				#plt.show()

				plt.plot(freq)
				plt.plot(rssi)
				plt.plot(bit_ask, '.')
				plt.show()
				
				#plt.plot(freq)
				#plt.plot(fsk_index, bit_fsk, '.')
				plt.plot(bit_fsk, '.')
				plt.show()

				WpcPacket.WpcPacket(bit_ask, ask_index, bit_fsk, fsk_index, freq, rssi, '../Samples/Wpc_output/' + 'output.json')
			
			elif c == 'd': ## final methode
				filename_local = '../Samples/Wpc/' + 'raw_data_20210811_095818.bttraw'
				adcData = rawFile.readRawFile(filename_local, 13)
				#adcData = rawFile.readRawFile(filename1, 13)
				
				data = adcData[int(0.5e6):int(17.0e6)]
				print(f'data size : {data.size}')
				print(f'ADC Data Min/Max (in selected range): {data.min()}, {data.max()} , {type(data[0])}')

				type = 'int32'

				data_flt = demod.WpcFrontendFiltering(data, fs, 10, type=type)
				ask_data, rssi = demod.WpcAskDemodulator(data_flt, fs, type=type)
				fsk_data, fsk_index, period =  demod.WpcFskDemodulator(data_flt, fs, type=type)
				
				rxData_ask, rxIndex_ask = cr.Early_late(ask_data, 25, 2, plot_data=True)
				rxData_fsk, rxIndex_fsk = cr.Early_late(fsk_data, 8, 2, plot_data=True)
				
				WpcPacket.WpcPacket(rxData_ask, rxIndex_ask, rxData_fsk, fsk_index[rxIndex_fsk], period[::10], rssi, '../Samples/Wpc_output/' + 'output.json')

			elif c == 'a': ## shift model
				#filename_local = '../Samples/Wpc/' + 'Wpc_adc1_20210811_095818.bttraw'
				#adcData = rawFile.readRawFile(filename_local, 13)
				adcData = rawFile.readRawFile(filename1, 13)

				data = adcData[int(5.0e6):int(26.0e6)].astype('float')
				print(f'ADC Data Min/Max (in selected range): {data.min()}, {data.max()} , {type(data[0])}')

				data, freq_measured, fs_low = demod_shift.WpcFrontendFiltering(data, fs, 0, 10, mode='SubSampling')
				data_ask, rssi = demod_shift.WpcAskDemodulator(data, fs_low)
				data_fsk = demod_shift.WpcFskDemodulator(data, fs_low, freq_measured)

				rxData_ask, rxIndex_ask = cr.Early_late(data_ask, int(fs/(10*4e3)), 5)
				rxData_fsk, rxIndex_fsk = cr.Early_late(data_fsk, int(fs*256/(127.7e3 * 10)), 50)
				
				WpcPacket.WpcPacket(rxData_ask, rxIndex_ask, rxData_fsk, rxIndex_fsk, freq_measured, rssi, '../Samples/Wpc_output/' + 'output.json')

			elif c == 'h':
				
				## sdr test
				filename_local = samples_path + 'SDR/adc_raw_1.bin'
				adcData = rawFile.readRawFile(filename_local, 12)
				
				#plt.plot(adcData[0:10000:2])
				#plt.plot(adcData[1:10000:2])
				#plt.show()
				
				dataI = adcData[int(00e6):int(70.0e6):2].astype('float')
				dataQ = adcData[int(00e6)+1:int(70.0e6):2].astype('float')
				print(f'ADC DataI Min/Max (in selected range): {dataI.min()}, {dataI.max()} , {type(dataI[0])}')
				print(f'ADC DataI Min/Max (in selected range): {dataQ.min()}, {dataQ.max()} , {type(dataQ[0])}')
				fs = 30720000*2
				data = dataI #+1j*dataQ

				#filename_local = '../Samples/Wpc/' + 'Wpc_adc2_20211020_130755.bttraw'
				#adcData = rawFile.readRawFile(filename_local, 13)
				#adcData = rawFile.readRawFile(filename1, 13)
				#data = adcData[int(0.0e6):int(26.0e6)].astype('float')
				#print(f'ADC Data Min/Max (in selected range): {data.min()}, {data.max()} , {type(data[0])}')
				
				fp.fftPlot(data, fs=fs)
				plt.plot(data[1000:10000])
				#plt.plot(data)
				plt.show()

			elif c == 'm':
				n_bit = 4000
				n_loop = 5
				errorlist = []
				for snr_dB in range(5, 10, 1):
					error = 0
					run = 0
					while(run<n_loop):
						print(f'---------run:{run:2d}-----------')
						error_run = WpcBER.AskBERtest(n_bit//2, 127.0e3, snr_dB, model='new')
						if error_run != -1:
							error += error_run
							run += 1
						print('---------------------------')
					print(f'for SNR = {snr_dB}dB => BER = {np.log10(error/(n_loop*n_bit))}  -- error:{error} -- total bit:{n_loop*n_bit}')
					errorlist.append(np.log10(error/(n_loop*n_bit)))
				print(errorlist)
				
			elif c == 'n':
				n_bit = 4000
				n_loop = 5
				errorlist = []
				for snr_dB in range(-5, 5, 1):
					error = 0
					run = 0
					while(run<n_loop):
						print(f'---------run:{run:2d}-----------')
						error_run = WpcBER.FskBERtest(n_bit//2, 127.0e3, snr_dB, model='new')
						if error_run != -1:
							error += error_run
							run += 1
						print('---------------------------')
					print(f'for SNR = {snr_dB}dB => BER = {np.log10(error/(n_loop*n_bit))}  -- error:{error} -- total bit:{n_loop*n_bit}')
					errorlist.append(np.log10(error/(n_loop*n_bit)))
				print(errorlist)

			## generate filters coeff
			elif c == 'f':
				## front-end filter:
				fs = 5.0e6
				len = 81
				fc = 250.0e3
				b_low = fd.LowpassFilter(len, fc, fs)
				gain = 1.0/b_low[int((len-1)/2)]
				b_low *= 1.0/b_low[int((len-1)/2)]
				print(f'front-end filter with nTap = {len} f_cut = {fc} gain = {gain} :')
				print(np.array2string(b_low, separator=', ', formatter={'float':lambda x: "%1.8f" % x}, max_line_width=25000))
				myLib.ModemLib().freqResPlot(b_low, fs=fs)

				## filter_1
				down_rate = 10
				fs = 5.0e6 / down_rate
				len = 101
				fc = 1.5e3
				b_low = fd.LowpassFilter(len, fc, fs)
				gain = 1.0/b_low[int((len-1)/2)]
				b_low *= 1.0/b_low[int((len-1)/2)]
				print(f'amplitude signal filter with nTap = {len} f_cut = {fc} gain = {gain} :')
				print(np.array2string(b_low, separator=', ', formatter={'float':lambda x: "%1.8f" % x}, max_line_width=25000))
				myLib.ModemLib().freqResPlot(b_low, fs=fs)

				## filter_2
				down_rate = 10*5
				fs = 5.0e6 / down_rate
				len = 41
				fc = 1.0e3
				b_low = fd.LowpassFilter(len, fc, fs)
				gain = 1.0/b_low[int((len-1)/2)]
				b_low *= 1.0/b_low[int((len-1)/2)]
				print(f'phase signal filter with nTap = {len} f_cut = {fc} gain = {gain} :')
				print(np.array2string(b_low, separator=', ', formatter={'float':lambda x: "%1.8f" % x}, max_line_width=25000))
				myLib.ModemLib().freqResPlot(b_low, fs=fs)

			elif c == 'p': ## NFC
				filename_local = '../Samples/NFC/' + 'Wpc_adc2_20210817_152547.bttraw'
				adcData = rawFile.readRawFile(filename_local, 13)
				#adcData = rawFile.readRawFile(filename2, 13)
				
				data = adcData[int(0.0e6):int(20.0e6)].astype('float')
				print(f'ADC Data Min/Max (in selected range): {data.min()}, {data.max()} , {type(data[0])}')
								
				#plt.plot(data)
				#plt.show()
				fp.fftPlot(data, fs=fs)
				fp.specPlot(data, fs=fs)

				#data = np.abs(signal.hilbert(data))

				data = np.abs(data)

				ask_data = np.convolve(fd.LowpassFilter(301, 250000.0, fs), data, mode='same')
				plt.plot(ask_data)
				plt.show()

			elif c == 'x':
				break
			print()
			print('==================')
			print('Press new command:')

	print('Exit')
