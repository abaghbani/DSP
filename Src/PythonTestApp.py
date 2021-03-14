import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import msvcrt
import sys

sys.path.insert(1, './Lib')
from Spectrum.ModemLib import ModemLib
from Spectrum.Constant import Constant
from Spectrum.Histogram2jpeg import histogram2jpeg
from Sound.DistortionCalc import Thd, ThdN
from Sound.KarplusStrong import KarplusStrong
from IOs.WriteToWav import WriteToWav
from IOs.RawFile import readRawFile
from IOs.WavFile import readWaveFile
from IOs.MatFile import readMatFile

from ChannelFilter.ChannelDecimate import ChannelDecimate
from ChannelFilter.ChannelFilter import ChannelFilter
from RfModel.RfTransceiver import RfTransceiver
from Dpsk.DpskModem import DpskModem, DpskReceiver
from Oqpsk.WpanModem import WpanModem, WpanReceiver
from Ofdm.OfdmModem import OfdmModem, OfdmBaseband, OfdmReceiver
from LPWAN.LoRa.LoRaDemodulation import LoRaDemodulation, LoRaFilterBank
from LPWAN.LoRa.LoRaModulation import LoRaModulation


mLib = ModemLib(0)
C = Constant





if __name__=="__main__":
	
	print('G: Generate Gfsk in all channel')
	print('S: Specgram of sampled data')
	print('H: Histogram to Jpeg of sampled data')
	print('A: FFT plot')
	print('Q: Wpan modem')
	print('E: Ofdm modem')	
	print('Z: Ofdm receiver')	
	print('L: LoRa receiver')
	print('M: LoRa modem')
	print('N: read matfile')
	print('')

	while True:
		if msvcrt.kbhit():
			c = msvcrt.getch().decode("utf-8")
			print(c)
			c = c.lower()
			
	
			elif c == 'q':
				WpanModem(15, 20, 50)
			
			elif c == 'e':
				OfdmModem(20, 2000, 50)
				#OfdmBaseband(20, 2000, 50)
			
			elif c == 'z':
				adcData = readRawFile('wifi-6Mbps-ch3-1.bttraw')
				lnaGain = 4000
				adcDataNoGain = adcData[67800+1034:80000]/lnaGain
				rxData = OfdmReceiver(adcDataNoGain, 20)
			
			elif c == 'l':
				[fs, dataI, dataQ] = readWaveFile('../Lib/Python/Samples/LoRa/SDRSharp_20160109_195047Z_869600kHz_IQ_BW21CR48SF6PL64x00.wav')
				[dataI, dataQ, fs] = LoRaFilterBank(dataI, dataQ, fs, Bw=20800.0, fMix=251500.0, downSamplingRate=10)
				LoRaDemodulation(dataI, dataQ, fs, Bw=20800, SF=6)
			
			elif c == 'm':
				SF = 7
				Bw = 20800.0
				fs = (1*Bw)

				payload = np.random.rand(5)*(np.exp2(SF))
				payload = payload.astype('int')
				baseband = LoRaModulation(payload, fLow=-Bw/2, fHigh=Bw/2, SF=SF, Fs=fs)
				LoRaDemodulation(baseband.real, baseband.imag, fs, Bw, SF )
			
			elif c == 'n':
				[dataI, dataQ, fs] = readMatFile('../Lib/Python/Samples/LoRa/pluto-capture-5.mat', 'cf_ad9361_lpc_voltage0', 'cf_ad9361_lpc_voltage1')
				print(dataI.shape, dataQ.shape)	
				fs = 16.0e+6
				Bw = 125000.0

				#dataI = dataI[int(1.5e6):int(2.2e6)]
				#dataQ = dataQ[int(1.5e6):int(2.2e6)]
				#dataI = dataI[int(17.2e6):int(17.4e6)]
				#dataQ = dataQ[int(17.2e6):int(17.4e6)]
				#dataI = dataI[:int(2.6e5)]
				#dataQ = dataQ[:int(2.6e5)]
				data = dataI+1j*dataQ
				
				mLib.fftPlot(data, n=1, fs=fs)
				mLib.specPlot(data, fs=fs)

				[dataI, dataQ, fs] = LoRaFilterBank(dataI, dataQ, fs, Bw=Bw, fMix=.992078e6, downSamplingRate=10) #int(fs/Bw)//2)
				LoRaDemodulation(dataI, dataQ, fs, Bw=Bw, SF=7)
			
			elif c == 'x':
				break
			print('Press new command:')

	print('Exit')