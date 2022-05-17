import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import msvcrt
import os
import logging as log

import Spectrum as sp
import Filter as fd
import IOs
import Common as myLib
import ClockRecovery as cr
import RfModel as rf

import QAM

if __name__=="__main__":
	
	log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG)
	log.getLogger('matplotlib.font_manager').disabled = True
	log.getLogger('PIL').setLevel(log.INFO)

	print('A: process debug data')
	print('M: QAM Modem')
	print('X: Exit')
	print('>> ')

	while True:
		if msvcrt.kbhit():
			c = msvcrt.getch().decode("utf-8")
			print(c)
			c = c.lower()

			if c == 'a':  ## Qam16 modem
				
				payload = (np.random.rand(100)*256).astype(np.uint8)
				txBaseband = QAM.modulation(payload, QAM.Constant.ModulationType.QAM16)
				#plt.plot(txBaseband.real, txBaseband.imag, 'bo')
				#plt.grid()
				#plt.show()

				tx_upsampled = rf.UpSampling(txBaseband, 5/100, 100)
				tx_mixer = rf.Mixer(tx_upsampled, 30/100, np.pi/4)
				tx_sig = tx_mixer.real + rf.WhiteNoise(tx_mixer, 50)

				#sp.fftPlot(txBaseband.real, txBaseband.imag, n=2)
				#sp.fftPlot(tx_upsampled.real, tx_upsampled.imag, n=2)
				#sp.fftPlot(tx_mixer.real, tx_mixer.imag, n=2)
				#sp.fftPlot(tx_sig.real)
				
				rx_mixer = rf.Mixer(tx_sig, -30/100, -1*np.pi/4)
				b = signal.remez(100+1, [0, .1, 0.2, 0.5], [1, 1e-4])
				baseband_flt = np.convolve(b, rx_mixer, 'same')
				sp.fftPlot(rx_mixer.real, rx_mixer.imag, n=2)
				sp.fftPlot(baseband_flt.real, baseband_flt.imag, n=2)
	
				#baseband_flt = baseband_flt[40::100]

				#plt.plot(baseband_flt[0:300].real, baseband_flt[0:300].imag, 'ro')
				plt.plot(baseband_flt[5::10].real, baseband_flt[5::10].imag, 'ro')
				plt.plot(baseband_flt[40::100].real, baseband_flt[40::100].imag, 'bo')
				plt.grid()
				plt.show()

			elif c == 'm':
				QAM.QamModem(30, 100, QAM.Constant.ModulationType.QAM16, 13)
			
			elif c == 'x':
				break
			print()
			print('==================')
			print('Press new command:')

	print('Exit')
