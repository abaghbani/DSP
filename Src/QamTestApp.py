import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import msvcrt
import os
import logging as log

import Spectrum as sp
import Filter as fd
import IOs
import Common as cm
import ClockRecovery as cr
import RfModel as rf

import QAM
import Ofdm

'''
########################
### Python float format:

3.1415926	{:.2f}	3.14	Format float 2 decimal places
3.1415926	{:+.2f}	+3.14	Format float 2 decimal places with sign
-1			{:+.2f}	-1.00	Format float 2 decimal places with sign
2.71828		{:.0f}	3		Format float with no decimal places
4			{:0>2d}	04		Pad number with zeros (left padding, width 2)
4			{:x<4d}	4xxx	Pad number with x (right padding, width 4)
10			{:x<4d}	10xx	Pad number with x (right padding, width 4)
1000000		{:,}	1,000,000	Number format with comma separator
0.35		{:.2%}	35.00%	Format percentage
1000000000	{:.2e}	1.00e+09Exponent notation
11			{:11d}	     11	Right-aligned (default, width 10)
11			{:<11d}	11		Left-aligned (width 10)
11			{:^11d}	  11	Center aligned (width 10)

########################
'''



if __name__=="__main__":
	
	log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG)
	log.getLogger('matplotlib.font_manager').disabled = True
	log.getLogger('PIL').setLevel(log.INFO)

	print('M: QAM Modem')
	print('O: QAM Modem baseband')	
	print('N: QAM Receiver')
	print('W: Ofdm modem')	
	print('Y: Ofdm receiver - SDR samples')	
	print('Z: Ofdm receiver - Bv1 samples')	
	print('S: Specgram of sampled data')
	print('H: Histogram to Jpeg of sampled data')
	print('A: FFT plot')
	print('U: Pulse shaping design')
	print('L: LTS sequences')
	print('D: Debugging')
	print('X: Exit')
	print('>> ')

	path = 'D:/Documents/Samples/QAM/'
	filename = 'QamData.bin'

	while True:
		if msvcrt.kbhit():
			c = msvcrt.getch().decode("utf-8")
			print(c)
			c = c.lower()

			if c == 'm':
				adcData = QAM.QamModem(30, 100, QAM.Constant.ModulationType.QAM16, 40)
				IOs.writeRawFile('./output/qamData.bin', adcData)
			
			elif c == 'o':
				payload = (np.random.rand(100)*256).astype(np.uint8)
				txBaseband, fs = QAM.modulation(payload, QAM.Constant.ModulationType.QAM16)
	
				tx_upsampled = rf.UpSampling(txBaseband, 15)
	
				freq_offset = (np.random.random(1)-0.5)/5	# offset -0.1 to +0.1 (-100KHz to +100KHz)
				phase_offset = (np.random.random(1)-0.5)*2*np.pi # offset -pi to +pi

				tx_mixer = rf.Mixer(tx_upsampled, freq_offset, phase_offset, 15*fs)
				noise = rf.WhiteNoise(tx_mixer, 30)
				tx_sig = tx_mixer + (noise+1j*noise)

				tx_sig *= (2**16)/np.abs(tx_sig).max()
				rx_sig = (tx_sig.real//2)+1j*(tx_sig.imag//2)
				print(f'transmitter: payload={payload.size=} bytes, freq_offset={float(freq_offset*1000):.3f} KHz, phase_offset={phase_offset*180/np.pi} Deg')
				QAM.Demodulation(tx_sig, 15*fs, QAM.Constant.ModulationType.QAM16)

			elif c == 'n':
				adcData = IOs.readRawFile(filename)
				adcData = adcData[600*240:1000*240]
				QAM.QamReceiver(adcData, 0, QAM.Constant.ModulationType.QPSK)
			
			elif c == 'w':
				adcData = Ofdm.OfdmModem(30, 100, Ofdm.Constant.ModulationType.QAM64, 55)
				#IOs.writeRawFile('./output/ofdmData.bin', adcData)
			
			elif c == 'y':
				adcData = IOs.readRawFile(filename)
				data_in = adcData[0::2]+1j*adcData[1::2]
				fs = 20.0
				
				sp.fftPlot(data_in, fs=fs)
				sp.specPlot(data_in, fs=fs)
				Ofdm.Demodulation(data_in, 20.0)

			elif c == 'z':
				adcData = IOs.readRawFile(filename)
				adcDataNoGain = adcData[240*10:240*(10+200)]*2
				Ofdm.OfdmReceiver(adcDataNoGain, 20)

			elif c == 's':
				adcData = IOs.readRawFile(filename)
				sp.specPlot(adcData)
				
			elif c == 'h':
				adcData = IOs.readRawFile(filename)
				sp.histogram2jpeg(adcData)
				
			elif c == 'a':
				adcData = IOs.readRawFile(filename)
				sp.fftPlot(adcData)				

			elif c == 'u':
				# pulse shaping
				fs = 16.0
				f_symb = 2.0
				flt_new = fd.rrc(65, 0.4, f_symb, fs)
				flt = fd.raised_root_cosine(64, fs/f_symb, 0.4)
				
				w, h = signal.freqz(flt)
				cm.ModemLib().plot_amp(w, h, fs, 'new', False)
				w, h = signal.freqz(flt_new*8)
				cm.ModemLib().plot_amp(w, h, fs, 'firls', True)

				print(f'{flt_new=}')
				print(f'{flt=}')
				
			elif c == 'l':
				# LTS sequences:
				for i in range(16):
					print(f'----------seq={i}-------------')
					print(np.array2string(QAM.Constant().preamble_lts[i].real, separator=', ', formatter={'float':lambda x: "%1.8f" % x}, max_line_width=1000))
					print(np.array2string(QAM.Constant().preamble_lts[i].imag, separator=', ', formatter={'float':lambda x: "%1.8f" % x}, max_line_width=1000))
				
				x = np.angle(QAM.Constant().preamble_lts[0]) * 1024/(2*np.pi)
				print(x)

			elif c == 'd':

				print('\n','='*30,'\n', 'end of debug.')

			elif c == 'x':
				break
			print('\n','='*30,'\n', 'Press new command:', sep='')


	print('Exit')
