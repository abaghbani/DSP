import numpy as np
import scipy.fftpack as fft
import matplotlib.pyplot as plt
import msvcrt
import logging as log

import IOs
import RfModel as rf
import ChannelFilter as cf

import Gfsk
import Dpsk
import Ofdm
import Oqpsk as Wpan
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
	print('E: Ofdm modem')	
	print('Z: Ofdm receiver')	
	print('D: Debugging')
	print('X: Exit')
	print('>> ')

	path = 'D:/Documents/Samples/Gfsk/'
	filename = path + 'test1.bin'

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
				(freq, rssi, data) = Gfsk.GfskReceiver(adcData, 58, cf.Constant.ChannelFilterType.Gfsk2M)
				plt.plot(freq)
				plt.plot(data)
				plt.grid()
				plt.show()
			
			elif c == 'p':
				#DpskModem(20, 200, DpskCons.DpskModulationType.Edr2, 5, ChFltCons.ChannelFilterType.Dpsk4M.ch1M )
				Dpsk.DpskModem(20, 2000, Dpsk.Constant.DpskModulationType.Edr3, 1, 15, cf.Constant.ChannelFilterType.Dpsk1M)
			
			elif c == 't':
				adcData = IOs.readRawFile(filename)
				(rssi, sync, valid, data) = Dpsk.DpskReceiver(adcData, 20, cf.Constant.ChannelFilterType.Dpsk1M)
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
				
			elif c == 'e':
				Ofdm.OfdmModem(20, 2000, 50)
			
			elif c == 'z':
				adcData = IOs.readRawFile(filename)
				lnaGain = 4000
				adcDataNoGain = adcData[67800+1034:80000]/lnaGain
				rxData = Ofdm.OfdmReceiver(adcDataNoGain, 20)
			
			elif c == 'd':
				adcData = IOs.readRawFile(filename)
				adcData = adcData[int(36e5):int(37e5)]
				sp.specPlot(adcData)
				(data4M, data2M, data1M) = cf.ChannelDecimate(adcData)
				for i in range(40,50,2):
					(dataI, dataQ, fs) = cf.ChannelFilter(data4M, data2M, data1M, i, C.ChannelFilterType.Gfsk2M)
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

			elif c == 'x':
				break
			print('Press new command:')

	print('Exit')