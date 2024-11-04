import numpy as np
import scipy.signal as signal
import scipy.io as sio
import scipy.stats as stats
import matplotlib.pyplot as plt
import msvcrt
import logging as log
import datetime

import Spectrum as sp
import Filter as fd
import IOs
import Common as cm
import ClockRecovery as cr
import RfModel as rf
import ChannelFilter as cf

import QAM
import Ofdm

'''
########################
### string format:
### https://docs.python.org/3/library/string.html#format-examples

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
42			{0:x}	2a		hex format
42			{0:#x}	0x2a	hex format
42			{0:o}	52		oct format
42			{0:#o}	0o52	oct format
42			{0:b}	101010	bin format
42			{0:#b}	0b101010bin format

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
	print('U: Pulse shaping design')
	print('L: LTS sequences')
	print('D: Debugging')
	print('X: Exit')
	print('>> ')

	path = 'D:/Documents/Samples/QAM/'

	while True:
		if msvcrt.kbhit():
			c = msvcrt.getch().decode("utf-8")
			cm.prRedbgGreen('Command <<'+str(c)+'>> is running:')
			c = c.lower()

			if c == 'm':
				snr = float(input('snr value (50dB*) = ').strip() or "50.0")
				# adcData = QAM.QamModem(20, 60, 1, QAM.Constant.ModulationType.TEST_PSK4, snr)
				# adcData = QAM.QamModem(20, 60, 1, QAM.Constant.ModulationType.TEST_PSK8, snr)
				adcData = QAM.QamModem(20, 60, 1, QAM.Constant.ModulationType.TEST_QAM16, snr)
				# IOs.writeRawFile('./qam_modem_' + str(datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))+'.bin', adcData)

			elif c == 'o':
				snr = float(input('snr value (50dB*) = ').strip() or "50.0")
				adcData = QAM.qam_modem_baseband(60, 1, QAM.Constant.ModulationType.TEST_QAM16, snr)
	
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
				IOs.writeRawFile('./ofdm_modem_' + str(datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))+'.bin', adcData)
			
			elif c == 'y':
				filename = IOs.get_file_from_path(path+'Wifi/', def_file=0)
				adcData = IOs.readRawFile(filename)
				data_in = adcData[0::2]+1j*adcData[1::2]
				Ofdm.Demodulation(data_in, 20.0)

			elif c == 'z':
				filename = IOs.get_file_from_path(path+'Wifi/', def_file=0)
				adcData = IOs.readRawFile(filename)
				adcDataNoGain = adcData[240*10:240*(10+200)]*2
				Ofdm.OfdmReceiver(adcDataNoGain, 20)

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
