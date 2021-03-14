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
from Dpsk.DpskModem import DpskModem, DpskReceiver

if __name__=="__main__":
	
	mLib = ModemLib(0)
	DpskParam = Constant

	print('P: Dpsk modem')
	print('T: Dpsk receiver')
	print('S: Specgram of sampled data')
	print('H: Histogram to Jpeg of sampled data')
	print('A: FFT plot')
	print('X: Exit')
	print('>> ')

	while True:
		if msvcrt.kbhit():
			c = msvcrt.getch().decode("utf-8")
			print(c)
			c = c.lower()
			
			if c == 'p':
				#DpskModem(20, 200, C.DpskModulationType.Edr2, 5, C.ChannelFilterType.Dpsk4M.ch1M )
				DpskModem(20, 2000, DpskParam.DpskModulationType.Edr3, 15, DpskParam.ChannelFilterType.Dpsk1M)
			
			elif c == 't':
				adcData = readRawFile('../Lib/Python/Samples/dpskData.bttraw')
				(rssi, sync, valid, data) = DpskReceiver(adcData, 20, DpskParam.ChannelFilterType.Dpsk1M)
				plt.plot(data)
				plt.plot(valid)
				plt.plot(sync)
				plt.grid()
				plt.show()
			
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
				
			elif c == 'x':
				break
			print('Press new command:')

	print('Exit')
