import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import msvcrt
import os
import logging as log

import IOs
import Spectrum as fp
import Filter as fd
import Common as myLib
import ClockRecovery as cr
import Wpc

if __name__=="__main__":
	
	## level : debug - info - warning - error - critical - exception
	log.basicConfig(format="%(levelname)s: %(message)s", level=log.INFO)
	log.getLogger('matplotlib.font_manager').disabled = True
	log.getLogger('PIL').setLevel(log.INFO)

	print('G: process HDL debug data')
	print('V: process extract bit')
	print('D: process raw data (final model)')
	print('A: process raw data (shift model)')
	print('H: raw data')
	print('M: ASK Modulation Qi (BER test)')
	print('N: FSK Modulation Qi (BER test)')
	print('F: generate filters coef')
	print('E: generate new filters coef')
	print('Y: Hilber design')
	print('X: Exit')
	print('>> ')

	path		= 'D:/Documents/Samples/Wpc/'
	path_out	= 'D:/Documents/Samples/Wpc_output/'

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
				filename_local  = path + 'demod_debug.bin'
				rawData = np.fromfile(filename_local, dtype='int16')
				
				mag_ac = rawData[0::4]
				mag_avg = rawData[1::4]
				freq_ac = rawData[2::4]
				freq_avg = rawData[3::4]
				mag = mag_ac[np.nonzero(mag_ac & 2)]
				mag_avg = mag_avg[np.nonzero(mag_ac & 2)]
				freq = freq_ac[np.nonzero(freq_ac & 2)]
				freq_avg = freq_avg[np.nonzero(freq_ac & 2)]
			
				plt.plot(mag)
				plt.plot(mag_avg)
				plt.legend(['mag_ac', 'mag_avg'], loc='best')
				plt.grid()
				plt.show()
				plt.plot(freq)
				plt.plot(freq_avg)
				plt.legend(['freq_ac', 'freq_avg'], loc='best')
				plt.grid()
				plt.show()

			elif c == 'v':  ## process extract bit,rssi,freq
				filename_local = path + 'demod_data.bin'
				filename = filename_local
				rawData = np.fromfile(filename, dtype='uint16')
				selected_channel = 0	# channel could be 0 or 1

				demodData = rawData[0::4]
				mag = rawData[2::4].astype('uint16')
				freq = rawData[3::4].astype('uint16')

				channel_mask = 0x100
				channel_value = channel_mask if selected_channel == 1 else 0
				freq = freq[np.nonzero((demodData&channel_mask) == channel_value)]
				mag = mag[np.nonzero((demodData&channel_mask) == channel_value)]
				data = demodData[np.nonzero((demodData&channel_mask) == channel_value)]

				bit_ask = data[np.nonzero(data & 2)] & 1
				bit_fsk = (data[np.nonzero(data & 8)] & 4)//4
				ask_index = np.arange(bit_ask.size)
				#fsk_index = (ask_index[np.nonzero(data & 8)])

				#plt.show()
				#plt.show()
				plt.plot(mag)
				plt.plot(np.arange(bit_ask.size)*25, bit_ask*500+8000, '.')
				plt.legend(['freq', 'mag', 'ask'], loc='best')
				plt.show()
				
				#plt.plot(freq)
				#plt.plot(fsk_index, bit_fsk, '.')
				plt.plot(bit_fsk, '.')
				plt.show()

				Wpc.WpcPacket(bit_ask, ask_index, bit_fsk, fsk_index, freq, mag, path_out + 'output.json')

			elif c == 'b':  ## process debug data insetting ch0/1 (debug bitstream: debug mode enabled)
				filename_local = 'C:/Users/Akbar/Documents/FromGit/sw_source/Hardware/Wpe1/TestApp/TestApp/bin/Debug/net48/wpt_20220616_144638.wpe1dump'
				filename = filename_local
				rawData = np.fromfile(filename, dtype='uint16')

				ask_data_ac = rawData[0::4].astype('int16')
				ask_data_avg = rawData[1::4].astype('uint16')
				fsk_data_ac = rawData[2::4].astype('int16')
				fsk_data_avg = rawData[3::4].astype('uint16')

				plt.plot(ask_data_ac)
				plt.plot(ask_data_avg)
				plt.legend(['ac', 'avg'], loc='best')
				plt.show()
				
				plt.plot(fsk_data_ac)
				plt.plot(fsk_data_avg)
				plt.legend(['ac', 'avg'], loc='best')
				plt.show()
				
			elif c == 'w':  ## process extract bit from sw
				filename_fsk = 'C:/Users/Akbar/Documents/FromGit/sw_source/Hardware/Wpe1/TestApp/TestApp/bin/Debug/net48/' +'fsk_0.bin'
				bit_fsk = np.fromfile(filename_fsk, dtype='uint8')
				fsk_index = np.arange(bit_fsk.size)

				plt.plot(bit_fsk, '.')
				#plt.plot(bit_fsk)
				plt.show()

				Wpc.WpcPacket(np.zeros(bit_fsk.size), np.zeros(bit_fsk.size), bit_fsk, fsk_index, np.zeros(bit_fsk.size), np.zeros(bit_fsk.size), 'fsk_output.json')

			elif c == 'q':  ## process extract bit from sw
				filename = 'C:/Users/Akbar/Documents/FromGit/sw_source/Hardware/Wpe1/TestApp/TestApp/bin/Debug/net48/ask_0.bin'
				bit_ask = np.fromfile(filename, dtype='uint8')
				ask_index = np.arange(bit_ask.size)
				filename = 'C:/Users/Akbar/Documents/FromGit/sw_source/Hardware/Wpe1/TestApp/TestApp/bin/Debug/net48/sample_0.bin'
				data = np.fromfile(filename, dtype='uint16')

				magnitude = data[::2]
				period = data[1::2]

				#plt.plot(period)
				#plt.show()
				plt.plot(magnitude)
				plt.plot(np.arange(7, bit_ask.size+7)*5, bit_ask*500+6000, '.')
				#plt.plot(bit_ask)
				plt.show()

				Wpc.WpcPacket(bit_ask, ask_index, np.zeros(bit_ask.size), np.zeros(bit_ask.size), np.zeros(bit_ask.size), np.zeros(bit_ask.size), 'ask_output.json')
			
			elif c == 's':  ## process extract bit from sw
				filename_fsk = 'C:/Users/Akbar/Documents/FromGit/sw_source/Hardware/Wpe1/TestApp/TestApp/bin/Debug/net48/' +'sample_1.bin'
				data = np.fromfile(filename_fsk, dtype='uint16')
				
				magnitude = data[::2]
				period = data[1::2]

				plt.plot(magnitude)
				plt.show()
				plt.plot(period)
				plt.show()

				
			elif c == 'd': ## final methode
				#filename_local = path + 'adc_dump.bin'
				filename_local = 'C:/Users/Akbar/Documents/FromGit/sw_source/Hardware/Wpe1/TestApp/TestApp/bin/Debug/net48/' +'wpt_adc_20220504_110054.wpe1adcdump'
				adcData = IOs.rawFile.readRawFile(filename_local, 16)
				#adcData = IOs.rawFile.readRawFile(filename1, 13)
				selected_channel = 0	# channel could be 0 or 1

				data = adcData[selected_channel::2]
				print(f'data size : {data.size}')
				print(f'ADC Data Min/Max (in selected range): {data.min()}, {data.max()} , {type(data[0])}')
				#plt.plot(data[int(0e6):int(5e6)])
				#plt.show()
				#fp.fftPlot(data, fs=fs)
				
				type = 'int32'
				data_flt = Wpc.WpcFrontendFiltering(data, fs, type=type)
				ask_data, rssi = Wpc.WpcAskDemodulator(data_flt, fs, type=type)
				fsk_data, fsk_index, period =  Wpc.WpcFskDemodulator(data_flt, fs, 32, type=type)
				
				#rxData_ask, rxIndex_ask = cr.EarlyLate(ask_data, 25, delta=2, plot_data=True)
				#rxData_fsk, rxIndex_fsk = cr.EarlyLate(fsk_data, 8, delta=2, plot_data=True)
				rxData_ask, rxIndex_ask = cr.CrossZero(ask_data, 25)
				rxData_fsk, rxIndex_fsk = cr.CrossZero(fsk_data, 256/32)
				
				rxData_ask = np.where(rxData_ask>=0, 1, 0)
				rxData_fsk = np.where(rxData_fsk>=0, 1, 0)
				Wpc.WpcPacket(rxData_ask, rxIndex_ask, rxData_fsk, fsk_index[rxIndex_fsk], period[::10], rssi, path_out + 'output.json')

			elif c == 'a': ## shift model
				filename_local = path + 'adc_dump.bin'
				adcData = IOs.rawFile.readRawFile(filename_local, 13)
				#adcData = rawFile.readRawFile(filename1, 13)

				data = adcData[int(5.0e6):int(26.0e6)].astype('float')
				print(f'ADC Data Min/Max (in selected range): {data.min()}, {data.max()} , {type(data[0])}')

				data, freq_measured, fs_low = Wpc.WpcFrontendFiltering(data, fs, 0, 10, mode='SubSampling')
				data_ask, rssi = Wpc.WpcAskDemodulator(data, fs_low)
				data_fsk = Wpc.WpcFskDemodulator(data, fs_low, freq_measured)

				rxData_ask, rxIndex_ask = cr.Early_late(data_ask, int(fs/(10*4e3)), 5)
				rxData_fsk, rxIndex_fsk = cr.Early_late(data_fsk, int(fs*256/(127.7e3 * 10)), 50)
				
				Wpc.WpcPacket(rxData_ask, rxIndex_ask, rxData_fsk, rxIndex_fsk, freq_measured, rssi, path_out + 'output.json')

			elif c == 'h':
				filename_local = path + 'adc_dump.bin'
				adcData = IOs.rawFile.readRawFile(filename_local, 16)
				#adcData = rawFile.readRawFile(filename1, 13)
				
				##data = adcData[int(0.0e6):int(26.0e6)].astype('float')
				##data = adcData.astype('float')
				data = adcData[1+int(0e6):int(24e6):2]
				#data = adcData[1::2]

				print(f'ADC Data Min/Max (in selected range): {data.min()}, {data.max()} , {type(data[0])}')
				
				fp.fftPlot(data, fs=fs)
				#plt.plot(data[1000:1000000])
				plt.plot(data)
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
						error_run = Wpc.AskBERtest(n_bit//2, 127.0e3, snr_dB, model='new')
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
						error_run = Wpc.FskBERtest(n_bit//2, 127.0e3, snr_dB, model='new')
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

			elif c == 'e':
				## front-end filter:
				fs = 5.0e6
				len = 79
				fc = 500.0e3
				b_low = fd.LowpassFilter(len, fc, fs)
				gain = 1.0/b_low[int((len-1)/2)]
				b_low *= 1.0/b_low[int((len-1)/2)]
				print(f'front-end filter with nTap = {len} f_cut = {fc} gain = {gain} :')
				print(np.array2string(b_low, separator=', ', formatter={'float':lambda x: "%1.8f" % x}, max_line_width=25000))
				myLib.ModemLib().freqResPlot(b_low, fs=fs)

				## filter_1 - Hilber transfer
				down_rate = 5
				fs = 5.0e6 / down_rate
				len = 93
				fc_1 = 100.0e3
				fc_2 = 400.0e3
				f_trans = 20.0e3
				#b_low = signal.remez(len, np.array([0., (fc_1-f_trans)/fs, fc_1/fs, fc_2/fs, (fc_2+f_trans)/fs, 0.5]), [0, 1, 0], type='hilbert')
				b_low = signal.remez(len, np.array([0., 70e3/fs, 100e3/fs, 400.0e3/fs, 420.0e3/fs, 0.5]), [0, 1, 0], type='hilbert')
				gain = 1.0/b_low[int((len-1)/2)]
				#b_low *= 1.0/b_low[int((len-1)/2)]
				print(f'amplitude signal filter with nTap = {len} f_cut_1 = {fc_1} f_cut_2 = {fc_2} gain = {gain} :')
				print(np.array2string(b_low, separator=', ', formatter={'float':lambda x: "%1.8f" % x}, max_line_width=25000))
				myLib.ModemLib().freqResPlot(b_low, fs=fs)

				## filter_2 - low pass
				down_rate = 5
				fs = 5.0e6 / down_rate
				len = 81
				fc = 2.0e3
				b_low = fd.LowpassFilter(len, fc, fs)
				gain = 1.0/b_low[int((len-1)/2)]
				b_low *= 1.0/b_low[int((len-1)/2)]
				print(f'phase signal filter with nTap = {len} f_cut = {fc} gain = {gain} :')
				print(np.array2string(b_low, separator=', ', formatter={'float':lambda x: "%1.8f" % x}, max_line_width=25000))
				myLib.ModemLib().freqResPlot(b_low, fs=fs)

			elif c == 'y':
				#filename_local = path + 'adc_dump.bin'
				#adcData = IOs.readRawFile(filename_local, 16)
				#selected_channel = 0	# channel could be 0 or 1
				#data = adcData[selected_channel:selected_channel+int(25e6):2]
				#fp.fftPlot(data, fs=fs)
				
				filename_local = path + 'wpc_sample.bin' # 'adc_dump.bin'
				adcData = IOs.readRawFile(filename_local, 16)
				selected_channel = 0 #895000	# channel could be 0 or 1
				data = adcData[selected_channel:selected_channel+int(25e6)]
				#fp.fftPlot(data, fs=fs)
				
				data_input = np.convolve(fd.LowpassFilter(81, 500.0e3, fs), data, mode='same')
				## decimation
				downSamplingRate = 5
				data_flt = np.sum(np.reshape(data_input[:(data_input.size//downSamplingRate)*downSamplingRate], (-1, downSamplingRate)), axis=1)
				fs = fs/downSamplingRate
				#fp.fftPlot(data_flt, fs=fs)
				#data = (data/downSamplingRate).astype(type)

				n_tap = 93
				sig_I = data_flt
				sig_Q = np.convolve(signal.remez(n_tap, np.array([0., 70e3/fs, 100e3/fs, 400.0e3/fs, 420.0e3/fs, 0.5]), [0, 1, 0], type='hilbert'), sig_I, mode='same')
				delay = np.zeros(n_tap)
				delay[int(n_tap/2)] = 1.0
				sig_I = np.convolve(delay, sig_I, mode='same')
				#my_abs = np.power(np.abs(sig_I+1j*sig_Q),2)
				my_abs = sig_I*sig_I+sig_Q*sig_Q


				#data_flt = signal.hilbert(data)
				#data_abs = np.abs(data_flt)

				my_abs_flt = np.convolve(fd.LowpassFilter(81, 2.0e3, fs), my_abs, mode='same')
				#fp.fftPlot(data, fs=fs)
				#fp.fftPlot(data_flt, fs=fs)
				
				#plt.plot(data_flt.real, label='real')
				#plt.plot(data_flt.imag, label='imag')
				
				plt.plot(sig_I, label='my_real')
				plt.plot(sig_Q, label='my_imag')
				#plt.plot(data, label='sig')
				#plt.plot(data_abs, label='abs')
				plt.plot(my_abs, label='my_abs')
				plt.plot(my_abs_flt, label='my_abs_flt')
				plt.legend()
				plt.show()

				#fp.fftPlot(data_abs, fs=fs)

				downSamplingRate = 10
				ask_data = np.sum(np.reshape(my_abs_flt[:(my_abs_flt.size//downSamplingRate)*downSamplingRate], (-1, downSamplingRate)), axis=1)
				fs = fs/downSamplingRate

				ask_sample_number = 128  ## (2*fs/2.0e3) = 100, should be 100 but because of divider is changed to 128
				ask_data_avg = rssi = np.convolve(ask_data, np.ones(ask_sample_number), 'same')/ask_sample_number
				ask_data_ac = ask_data-ask_data_avg
				plt.plot(ask_data)
				plt.plot(ask_data_avg)
				plt.plot(ask_data_ac)
				plt.grid()
				plt.show()

				rxData_ask, rxIndex_ask = cr.CrossZero(ask_data_ac, 25)

				fsk_data = ask_data
				fsk_sample_number = 4*(200)
				fsk_data_avg = np.convolve(fsk_data, np.ones(fsk_sample_number), 'same')/fsk_sample_number
				fsk_data_ac = fsk_data-fsk_data_avg
				plt.plot(fsk_data)
				plt.plot(fsk_data_avg)
				plt.plot(fsk_data_ac)
				plt.grid()
				plt.show()

				rxData_fsk, rxIndex_fsk = cr.CrossZero(fsk_data_ac, 200)
				
				rxData_ask = np.where(rxData_ask>=0, 1, 0)
				rxData_fsk = np.where(rxData_fsk>=0, 1, 0)
				#Wpc.WpcPacket(rxData_ask, rxIndex_ask, rxData_fsk, fsk_index[rxIndex_fsk], period[::10], rssi, path_out + 'output.json')
				Wpc.WpcPacket(rxData_ask, rxIndex_ask, np.empty(0), np.empty(0), np.zeros(rssi.size), rssi, 'output.json')

			elif c == 'p': ## NFC
				filename_local = '../Samples/NFC/' + 'Wpc_adc2_20210817_152547.bttraw'
				adcData = IOs.rawFile.readRawFile(filename_local, 13)
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
