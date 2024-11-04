import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import msvcrt
import logging as log

import IOs
import Spectrum as sp
import RfModel as rf
import Oqpsk as Wpan

if __name__=="__main__":
	
	log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG)
	log.getLogger('matplotlib.font_manager').disabled = True
	log.getLogger('PIL').setLevel(log.INFO)

	print('A: process debug data')
	print('M: Modem')
	print('X: Exit')
	print('>> ')

	path = 'D:/Documents/Samples/Wpan/'
	filename = path + 'wpanData.bttraw'

	while True:
		if msvcrt.kbhit():
			c = msvcrt.getch().decode("utf-8")
			print(c)
			c = c.lower()

			if c == 'a':  ## transmit/receive
				
				## transmit
				bit_number = 4
				Fs = 30.0e6
				payload = np.array(np.random.rand(bit_number)*256, dtype=np.uint8)
				baseband = Wpan.Modulation(payload, Fs)

				## add noise and phase offset (some part of RF model)
				white_noise = rf.WhiteNoise(baseband, 50)
				baseband_rx = (baseband.real+white_noise)+1j*(baseband.imag+white_noise)
				baseband_rx = rf.PhaseOffset(baseband_rx, -4*np.pi/10)
	
				## receive
				Wpan.Demodulation(baseband_rx, Fs)

			if c == 'b':  ## transmit/receive with rf model
				bit_number = 4
				Fs = 100.0e6
				payload = np.array(np.random.rand(bit_number)*256, dtype=np.uint8)
				baseband = Wpan.Modulation(payload, Fs)

				Fch = 35.0e6
				rf_sig = rf.Mixer(baseband, Fch, 1*np.pi/8, Fs)
				tx_sig = rf_sig.real + rf.WhiteNoise(rf_sig, 50)
				#sp.fftPlot(tx_sig.real, tx_sig.imag, n=2, fs=Fs)

				rx_sig = rf.Mixer(tx_sig, -1*Fch, -1*np.pi/8, Fs)*2
				#sp.fftPlot(rx_sig.real, rx_sig.imag, n=2, fs=Fs)
	
				b = signal.remez(300+1, [0, 2.5e6, 3.0e6, Fs/2], [1,0], fs=Fs)
				baseband_flt = np.convolve(b, rx_sig, 'same')

				# down sampling
				baseband_rx = baseband_flt[4::10]
				Fs = 10.0e6

				Wpan.Demodulation(baseband_rx, Fs)

			elif c == 'c': # test crc
				import crcmod

				data = np.array([0x01,0x02], dtype=np.uint8)
				#data = np.array(np.random.rand(10)*256, dtype=np.uint8)
				crc16 = crcmod.mkCrcFun(0x11021, rev=False, initCrc=0x0000, xorOut=0x0000)
				print(hex(crc16(data)))

				result = Wpan.CrcCalculation(data)
				print(hex(result[0]) , hex(result[1]))

			elif c == 'm':
				Wpan.WpanModem(15, 10, 50)

			elif c == 'x':
				break
			print()
			print('==================')
			print('Press new command:')

	print('Exit')
