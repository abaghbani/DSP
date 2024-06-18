import numpy as np
import scipy.fftpack as fft
import scipy.signal as signal
import matplotlib.pyplot as plt
import msvcrt
import logging as log

import init
import IOs
import RfModel as rf
import ChannelFilter as cf
import Filter as fd
import Common as myLib

import Gfsk
import Dpsk
import Spectrum as sp

if __name__=="__main__":
	
	## level : debug - info - warning - error - critical - exception
	log.basicConfig(format="%(levelname)s: %(message)s", level=log.INFO)
	log.getLogger('matplotlib.font_manager').disabled = True
	log.getLogger('PIL').setLevel(log.INFO)
	
	print('F: Gfsk modem')
	print('B: Gfsk modem with RF model')
	print('R: Gfsk receiver')
	print('P: Dpsk modem')
	print('T: Dpsk receiver')
	print('G: Generate Gfsk in all channel')
	print('S: Specgram of sampled data')
	print('H: Histogram to Jpeg of sampled data')
	print('A: FFT plot')
	print('Q: Wpan modem')
	print('D: Debugging')
	print('C: Filter design')
	print('X: Exit')
	print('>> ')

	path = 'D:/Documents/Samples/Dpsk'
	#filename = path + '/../Bex2/lna000_att0_adc1_20221223_144209.bttraw' 
	filename = path + '/HR/HR4-2404-adc1-DH3.bin' 
	#filename = path + '/HDR/dpskData_6_.bin' 
	#filename = 'D:/upf67/LE1M2MIssue/test1.bin'

	while True:
		if msvcrt.kbhit():
			c = msvcrt.getch().decode("utf-8")
			print(c)
			c = c.lower()

			if c == 'f':

				## transmit
				payload_len = 4
				Fs = 15.0e6
				payload = np.array(np.random.rand(payload_len)*256, dtype=np.uint8)
				baseband = Gfsk.Modulation(payload, Fs)
				#sp.fftPlot(baseband, Fs)

				## Receiver
				payload_extracted = Gfsk.Demodulation(baseband, Fs)


			if c == 'b':
				Gfsk.GfskModem(20, 200, Gfsk.Constant.GfskModulationType.Gfsk2M, 25, cf.Constant.ChannelFilterType.Gfsk2M)
				
			elif c == 'r':
				adcData = IOs.readRawFile(filename)
				#adcData = adcData[240*96500:240*99000]
				adcData = adcData[:240*15000]
				(freq, rssi, data) = Gfsk.GfskReceiver(adcData, 2, cf.Constant.ChannelFilterType.Gfsk2M)
				plt.plot(freq)
				plt.plot(rssi)
				plt.plot(data)
				plt.grid()
				plt.show()
			
			elif c == 'p':
				plt.subplot(211)
				plt.plot(Dpsk.Constant.DpskSyncEdr)
				plt.plot(Dpsk.Constant.DpskSyncEdr, 'b.')
				plt.plot(np.arange(1, 6), Dpsk.Constant.DpskSyncEdr[1:6], '-.')
				plt.grid()
				#plt.show()
				plt.subplot(212)
				plt.plot(Dpsk.Constant.DpskSyncHr2)
				plt.plot(Dpsk.Constant.DpskSyncHr2, 'b.')
				plt.plot(np.arange(1, 1+10), Dpsk.Constant.DpskSyncHr2[1:1+10], 'r.')
				plt.plot(np.arange(2, 2+5), Dpsk.Constant.DpskSyncHr2[2:2+5], '-.')
				plt.grid()
				plt.show()
				plt.plot(Dpsk.Constant.DpskSyncHr4)
				plt.plot(Dpsk.Constant.DpskSyncHr4, 'b.')
				plt.plot(np.arange(42, 62),Dpsk.Constant.DpskSyncHr4[42:62], 'r.')
				plt.plot(np.arange(44, 49),Dpsk.Constant.DpskSyncHr4[44:49], '-.')
				plt.grid()
				plt.show()
				plt.plot(Dpsk.Constant.DpskSyncHr8)
				plt.plot(Dpsk.Constant.DpskSyncHr8, 'b.')
				plt.plot(np.arange(42, 82),Dpsk.Constant.DpskSyncHr8[42:82], 'r.')
				plt.plot(np.arange(48, 53),Dpsk.Constant.DpskSyncHr8[48:53], '-.')
				plt.grid()
				plt.show()
				Dpsk.DpskModem(2, 10, Dpsk.Constant.ModulationType.HR2, 50)
			
			elif c == 't':
				adcData = IOs.readRawFile(filename)
				(rssi, sync, valid, data) = Dpsk.DpskReceiver(adcData, 2, Dpsk.Constant.ModulationType.HR4)
				plt.plot(data)
				plt.plot(valid)
				plt.plot(sync)
				plt.grid()
				plt.show()
	
			elif c == 'g':
				Fs_BB = 15.0e6
				payload = np.array(np.random.rand(200)*256, dtype=np.uint8)
				txBaseband = Gfsk.Modulation(payload, Fs_BB)
				txData = np.zeros(0)
				Fs_RF = cf.Constant.AdcSamplingFrequency
				tx_upsampled = rf.UpSampling(txBaseband, Gfsk.Constant.GfskModulationType.Gfsk1M*1.0e6, int(Fs_RF/(Fs_BB*Gfsk.Constant.GfskModulationType.Gfsk1M)), Fs_RF)
				for i in range(40):
					tx_mixer = rf.Mixer(tx_upsampled, cf.Constant().IfMixerFrequency+(i*2.0e6), 0, Fs_RF)
					tx_sig = tx_mixer.real + rf.WhiteNoise(tx_mixer, 20)
					txData = np.append(txData, tx_sig)
				IOs.writeRawFile('gfskAllChannels.bttraw', txData.astype('int16'))

			elif c == 's':
				adcData = IOs.readRawFile(filename)
				sp.specPlot(adcData)
				
			elif c == 'h':
				adcData = IOs.readRawFile(filename)
				sp.histogram2jpeg(adcData)
				
			elif c == 'a':
				adcData = IOs.readRawFile(filename)
				adcData = np.multiply(adcData, np.cos((np.arange(adcData.size)*2*np.pi*120.0e+6/240.0e+6)+0.06287))
				sp.fftPlot(adcData, fs = 240)
				


			elif c == 'd':
				adcData = IOs.readRawFile(filename)
				adcData = adcData[int(36e5):int(37e5)]
				sp.specPlot(adcData)
				(data4M, data2M, data1M) = cf.ChannelDecimate(adcData)
				for i in range(40,50,2):
					(dataI, dataQ, fs) = cf.ChannelFilter(data4M, data2M, data1M, i, cf.Constant.ChannelFilterType.Gfsk2M)
					(freq, rssi, valid, data) = Gfsk.Demodulation(dataI, dataQ, fs)

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

			elif c == 'c':
				import scipy.signal as signal
				chFltGfsk1M_my_org = signal.firls(29, [0, 0.57-0.13, 0.57+0.13, 1.875/2], [1, 1.4, 0, 0], fs=1.875)
				chFltGfsk2M_my_org = signal.firls(29, [0, 1.06-0.22, 1.06+0.22, 3.75/2], [1, 1.4, 0, 0], fs=3.75)
				chFltGfsk2M_new = signal.firls(29, [0, 1.0-0.21, 1.0+0.21, 3.75/2], [1, 1.3, 0, 0], fs=3.75)
				chFltGfsk2M_new2 = signal.firls(29, [0, 0.98-0.22, 0.98+0.22, 3.75/2], [1, 1.3, 0, 0], fs=3.75)
				
				chFltGfsk1M_my_org /= chFltGfsk1M_my_org[29//2]
				#print(chFltGfsk1M_my_org)
				chFltGfsk2M_my_org /= chFltGfsk2M_my_org[29//2]
				#print(chFltGfsk2M_my_org)
				chFltGfsk2M_new /= chFltGfsk2M_new[29//2]
				chFltGfsk2M_new2 /= chFltGfsk2M_new2[29//2]
				
				print(f'{chFltGfsk2M_new=}')
				
				fs = 1.875e6
				#w, h = signal.freqz(cf.Constant.chFltGfsk1M, 1)
				#plt.plot((w*fs)/(2*np.pi), 20.0*np.log10(np.abs(h)), label='1M_encl')
				w, h = signal.freqz(cf.Constant.chFltGfsk2M, 1)
				plt.plot((w*2*fs)/(2*np.pi), 20.0*np.log10(np.abs(h)), label='2M_encl')
				
				#w, h = signal.freqz(chFltGfsk1M_my_org, 1)
				#plt.plot((w*fs)/(2*np.pi), 20.0*np.log10(np.abs(h)), label='1M_mine')
				w, h = signal.freqz(chFltGfsk2M_my_org, 1)
				plt.plot((w*2*fs)/(2*np.pi), 20.0*np.log10(np.abs(h)), label='2M_mine')
				
				w, h = signal.freqz(chFltGfsk2M_new, 1)
				plt.plot((w*2*fs)/(2*np.pi), 20.0*np.log10(np.abs(h)), label='2M_new')
				w, h = signal.freqz(chFltGfsk2M_new2, 1)
				plt.plot((w*2*fs)/(2*np.pi), 20.0*np.log10(np.abs(h)), label='2M_new2')
				
				plt.ylabel('Magnitude (db)')
				plt.xlabel(r'Frequency (Hz)')
				plt.title(r'Frequency response')
				plt.legend()
				plt.grid()
				plt.show()

			elif c== 'z':
				##################################
				## Gaussian filter
				##################################
				t = np.linspace(-1, 1, int(2*15+1))
				gaussianFlt = np.exp(-2/np.log(2) * (t*np.pi*Gfsk.Constant.GfskBT)**2)
				gaussianFlt /= np.sum(gaussianFlt)
				print(np.array2string(gaussianFlt, separator=', ', formatter={'float':lambda x: "%1.8f" % x}, max_line_width=100))
				plt.plot(gaussianFlt)
				#plt.show()

				t = np.linspace(-2, 2, int(4*7.5+1))
				gaussianFlt = np.exp(-2/np.log(2) * (t*np.pi*Gfsk.Constant.GfskBT)**2)
				gaussianFlt /= np.sum(gaussianFlt)
				print(np.array2string(gaussianFlt, separator=', ', formatter={'float':lambda x: "%1.8f" % x}, max_line_width=100))
				plt.plot(gaussianFlt)
				plt.show()

				##################################
				## Modulation
				##################################
				over_sample_rate = 15
				bit_frame = np.array(np.random.rand(20)<0.5, dtype=np.int8)
				frequency_sample = bit_frame*2-1
				frequency_sample = np.concatenate((np.zeros(4, dtype=np.int8), frequency_sample, np.zeros(4, dtype=np.int8) ))
				frequency_sample = frequency_sample.repeat(over_sample_rate)
				frequency_sig = np.convolve(gaussianFlt, frequency_sample, 'same')
				#phase_sig = signal.lfilter(np.array([0.5, 0.5]), np.array([1, -1]), frequency_sig*((C.GfskBleModulationIndex[0]+C.GfskBleModulationIndex[1])/4)*2*np.pi/over_sample_rate)
				#baseband_sig = np.exp(1j*phase_sig)
				plt.plot(frequency_sample)
				plt.plot(frequency_sig)
				#plt.plot(phase_sig)
				plt.show()

			elif c == 'x':
				break
			print('Press new command:')

	print('Exit')