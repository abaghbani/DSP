import numpy as np
import matplotlib.pyplot as plt
import msvcrt

from Spectrum.freqPlot import fftPlot, specPlot
from Spectrum.Histogram2jpeg import histogram2jpeg
from IOs.wavFile import readWaveFile, writeWaveFile
from IOs.rawFile import readRawFile, writeRawFile

from ChannelFilter.Constant import Constant as ChFltConst
from ChannelFilter.ChannelDecimate import ChannelDecimate
from ChannelFilter.ChannelFilter import ChannelFilter
from RfModel.RfTransceiver import RfTransceiver
from Gfsk.Constant import Constant as GfskConst
from Gfsk.GfskModulation import GfskModulation
from Gfsk.GfskDemodulation import GfskDemodulation
from Gfsk.GfskModem import GfskModem, GfskReceiver
from Dpsk.DpskModem import DpskModem, DpskReceiver
from Dpsk.Constant import Constant as DpskConst
#from Oqpsk.WpanModem import WpanModem, WpanReceiver
#from Ofdm.OfdmModem import OfdmModem, OfdmBaseband, OfdmReceiver


if __name__=="__main__":
	
	print('F: Gfsk modem')
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

	filename = '../Samples/Gfsk/' + 'test1.bttraw'
	
	while True:
		if msvcrt.kbhit():
			c = msvcrt.getch().decode("utf-8")
			print(c)
			c = c.lower()
			if c == 'f':
				GfskModem(20, 200, GfskConst.GfskModulationType.Gfsk2M, 25, ChFltConst.ChannelFilterType.Gfsk2M)
				
			elif c == 'r':
				adcData = readRawFile(filename)
				(freq, rssi, valid, data) = GfskReceiver(adcData, 20, ChFltConst.ChannelFilterType.Gfsk2M)
				plt.plot(freq)
				plt.plot(valid)
				plt.grid()
				plt.show()
			
			elif c == 'p':
				#DpskModem(20, 200, DpskCons.DpskModulationType.Edr2, 5, ChFltCons.ChannelFilterType.Dpsk4M.ch1M )
				DpskModem(20, 2000, DpskConst.DpskModulationType.Edr3, 1, 15, ChFltConst.ChannelFilterType.Dpsk1M)
			
			elif c == 't':
				adcData = readRawFile(filename)
				(rssi, sync, valid, data) = DpskReceiver(adcData, 20, ChFltConst.ChannelFilterType.Dpsk1M)
				plt.plot(data)
				plt.plot(valid)
				plt.plot(sync)
				plt.grid()
				plt.show()
	
			elif c == 'g':
				AccessAddress = np.array(list(bin(0x71764129)[2:]), dtype='int')*2-1
				payload = np.concatenate((np.flip(AccessAddress), np.array((np.random.rand(1600) >= 0.5)*2-1)), axis=None)
				txBaseband = GfskModulation(payload)
				txData = np.zeros(0)
				for i in range(40):
					IfSig = RfTransceiver(txBaseband, i*2, C.SymboleRate.R1M, 20)
					txData = np.append(txData, IfSig)
				writeRawFile('gfskAllChannels.bttraw', txData.astype('int16'))

			elif c == 's':
				adcData = readRawFile(filename)
				specPlot(adcData)
				
			elif c == 'h':
				adcData = readRawFile(filename)
				histogram2jpeg(adcData)
				
			elif c == 'a':
				adcData = readRawFile(filename)
				adcData = np.multiply(adcData, np.cos((np.arange(adcData.size)*2*np.pi*120.0e+6/240.0e+6)+0.06287))
				fftPlot(adcData, fs = 240)
				
			elif c == 'q':
				WpanModem(15, 20, 50)
			
			elif c == 'e':
				OfdmModem(20, 2000, 50)
				#OfdmBaseband(20, 2000, 50)
			
			elif c == 'z':
				adcData = readRawFile(filename)
				lnaGain = 4000
				adcDataNoGain = adcData[67800+1034:80000]/lnaGain
				rxData = OfdmReceiver(adcDataNoGain, 20)
			
			elif c == 'd':
				adcData = readRawFile(filename)
				adcData = adcData[int(36e5):int(37e5)]
				specPlot(adcData)
				(data4M, data2M, data1M) = ChannelDecimate(adcData)
				for i in range(40,50,2):
					(dataI, dataQ, fs) = ChannelFilter(data4M, data2M, data1M, i, C.ChannelFilterType.Gfsk2M)
					(freq, rssi, valid, data) = GfskDemodulation(dataI, dataQ, fs)

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