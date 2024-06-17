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
	
	print('B: filter bank with fft')
	print('C: read csv file')
	print('D: Debugging')
	print('======================================')
	print('S: Specgram of sampled data')
	print('H: Histogram to Jpeg of sampled data')
	print('A: FFT plot')
	print('X: Exit')
	print('>> ')

	path = 'D:/Documents/Samples/Gfsk/'
	filename = path + 'test1.bin'

	while True:
		if msvcrt.kbhit():
			c = msvcrt.getch().decode("utf-8")
			print(c)
			c = c.lower()

			if c == 'd':
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

			elif c == 'b':
				#path = 'U:/Events/Bluetooth Events/IOP 2022-04/'
				#filename = path + '20220407 - NXP - 2M Phy - 3 mode 0 and 79 mode 1 - with PN Seq 71764129 - Trace 1.bin'
				adcData = IOs.readRawFile(filename)
				#sp.specPlot(adcData[int(240e6*0e-6):int(240e6*1300.0e-6)])
				
				adcData = adcData[int(240e6*0e-6):int(240e6*600.0e-6)]
				adcData = adcData.reshape(-1, 80)
				data = np.empty(adcData.shape)
				for i in range(adcData.shape[0]):
					data[i,:] = fft.fft(adcData[i,:], n=data.shape[1])
					
				data[:,0:15] = 0
				data[:,30:] = 0
				data_flt = np.empty((adcData.shape[0], adcData.shape[1]//2))
				for i in range(adcData.shape[0]):
					#data_flt[i,:] = fft.ifft(np.concatenate((np.zeros(21), data[i,21:32], np.zeros(80-32))))
					data_flt[i,:40] = fft.ifft(data[i,:40])

				data_flt = data_flt.reshape(-1)
				sp.specPlot(data_flt, fs=1)
					
				#plt.plot(data)
				#plt.show()

			elif c == 'c':
				#my_data = np.genfromtxt('D:/Documents/Samples/trace1_80ch2M.csv', delimiter=',')
				#np.save('test_spec', my_data)
				my_data = np.load('test_spec.npy')

				nSamples = 10
				ch2 = my_data[:,2]
				ch3 = my_data[:,3]
				ch4 = my_data[:,4]
				ch2 = np.min(np.reshape(ch2[:(ch2.size//nSamples)*nSamples], (-1, nSamples)), axis=1)
				ch3 = np.min(np.reshape(ch3[:(ch3.size//nSamples)*nSamples], (-1, nSamples)), axis=1)
				ch4 = np.min(np.reshape(ch4[:(ch4.size//nSamples)*nSamples], (-1, nSamples)), axis=1)

				plt.plot(ch2, label='ch2')
				plt.plot(ch3, label='ch3')
				plt.plot(ch4, label='ch4')
				plt.legend(loc='best')
				plt.show()
				
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

			elif c == 'x':
				break
			print('Press new command:')

	print('Exit')