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
			
			elif c == 'x':
				break
			print('Press new command:')

	print('Exit')