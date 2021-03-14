import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import msvcrt
import sys

sys.path.insert(1, './Lib')
from Spectrum.ModemLib import ModemLib
from Spectrum.Constant import Constant
from Spectrum.Histogram2jpeg import histogram2jpeg
from IOs.RawFile import readRawFile, writeRawFile

from ChannelFilter.ChannelDecimate import ChannelDecimate
from ChannelFilter.ChannelFilter import ChannelFilter
from RfModel.RfTransceiver import RfTransceiver
from Gfsk.GfskModulation import GfskModulation
from Gfsk.GfskDemodulation import GfskDemodulation
from Gfsk.GfskModem import GfskModem, GfskReceiver

def debugging():
	
	adcData = readRawFile('data1.bttraw')
	adcData = adcData[int(36e5):int(37e5)]
	mLib.specPlot(adcData)
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

if __name__=="__main__":
	
	mLib = ModemLib(0)
	GfskParam = Constant

	print('F: Gfsk modem')
	print('R: Gfsk receiver')
	print('G: Generate Gfsk in all channel')
	print('S: Specgram of sampled data')
	print('H: Histogram to Jpeg of sampled data')
	print('A: FFT plot')
	print('d: debugging')
	print('X: Exit')
	print('>> ')

	while True:
		if msvcrt.kbhit():
			c = msvcrt.getch().decode("utf-8")
			print(c)
			c = c.lower()
			if c == 'f':
				GfskModem(20, 200, GfskParam.GfskModulationType.Gfsk2M, 25, GfskParam.ChannelFilterType.Gfsk2M)
			
			elif c == 'r':
				adcData = readRawFile('../Lib/Python/Samples/gfskData.bttraw')
				(freq, rssi, valid, data) = GfskReceiver(adcData, 20, GfskParam.ChannelFilterType.Gfsk2M)
				plt.plot(freq)
				plt.plot(valid)
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
				adcData = readRawFile('../Lib/Python/Samples/gfskData.bttraw')
				mLib.specPlot(adcData[:5000000])
				
			elif c == 'h':
				adcData = readRawFile('../Lib/Python/Samples/gfskData.bttraw')
				histogram2jpeg(adcData)
				
			elif c == 'a':
				adcData = readRawFile('../Lib/Python/Samples/gfskData.bttraw')
				adcData = np.multiply(adcData, np.cos((np.arange(adcData.size)*2*np.pi*120.0e+6/240.0e+6)+0.06287))
				mLib.fftPlot(adcData, fs = 240)
				
			elif c == 'd':
				debugging()
			
			elif c == 'x':
				break
			print('Press new command:')

	print('Exit')
