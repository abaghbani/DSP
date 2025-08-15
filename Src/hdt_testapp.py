import numpy as np
import scipy.signal as signal
import scipy.io as sio
import matplotlib.pyplot as plt
import logging as log
import datetime

import Spectrum as sp
import Filter as fd
import IOs
import Common as cm
import ClockRecovery as cr
import RfModel as rf
import ChannelFilter as cf

import HDT

if __name__=="__main__":
	
	log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG)
	log.getLogger('matplotlib.font_manager').disabled = True
	log.getLogger('PIL').setLevel(log.INFO)

	print('M: HDT Modem')
	print('O: HDT Modem baseband')	
	print('N: HDT Receiver')
	print('B: HDT bit-processing-En/Decode')
	print('C: HDT bit-processing-crccheck')
	print('S: HDT symb-processing-crccheck')
	print('U: Pulse shaping design(bq1)')
	print('V: Pulse shaping design(bv1)')
	print('L: LTS sequences')
	print('D: Debugging')
	print('X: Exit')
	print('>> ')

	path = 'D:/Documents/Samples/HDT'
	
	while True:
		c = IOs.get_console_key()
		cm.prRedbgGreen('Command <<'+str(c)+'>> is running:')

		if c == 'm':
			snr = float(input('snr value (50dB*) = ').strip() or "50.0")
			adcData = HDT.modem(39, 60, 1, HDT.Constant.ModulationType.PSK4, snr)
			# IOs.writeRawFile('./hdt_modem_' + str(datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))+'.bin', adcData)

		elif c == 'o':
			snr = float(input('snr value (50dB*) = ').strip() or "50.0")
			adcData = HDT.modem_baseband(60, 1, HDT.Constant.ModulationType.PSK4, snr)
			# IOs.writeRawFile('./hdt_modem_' + str(datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))+'.bin', adcData)
			# np.save('hdt_baseband', adcData)

		elif c == 'q':
			sig = np.load('hdt_baseband.npy')
			HDT.demodulation(sig, 30.0,  HDT.Constant.ModulationType.PSK4)

		elif c == 'n':
			# filename = IOs.get_file_from_path(path+'/realtek/', def_file=1)
			# adcData = IOs.readRawFile(filename, count=240*1300, offset=240*2*1400)
			# filename = IOs.get_file_from_path(path+'/upf75/apple/', def_file=0, extension='.bin')
			# adcData = IOs.readRawFile(filename, count=-1, offset=240*2*0)
			# adcData = adcData[int(240*(48.7e3+52e3)):int(240*(51e3+53e3))]
			# sp.specPlot(adcData)
			# ch = int(input('what is ch number (0..78) = '))


			filename = IOs.get_file_from_path(path+'/UPF76/', def_file=0, extension='bin')
			adcData = IOs.readRawFile(filename)
			# adcData = adcData[int(240*63e3):int(240*66e3)]
			# adcData = adcData[int(240*77300):int(240*77500)]
			adcData = adcData[int(240*6300):int(240*6700)]
			print('ADC Data Min/Max: ',adcData.min(),adcData.max(), type(adcData[0]))

			ch = 2
			HDT.receiver(adcData, ch, HDT.Constant.ModulationType.PSK4)
		
		elif c == 'b':
			input_data = np.array([ # sample payload data (50 bytes) + crc32 (4bytes at the end)
			0x44, 0xFF, 0xC1, 0xFB, 0xE8, 0x4C, 0x90, 0x72, 
			0x8B, 0xE7, 0xB3, 0x51, 0x89, 0x63, 0xAB, 0x23, 
			0x23, 0x02, 0x84, 0x18, 0x72, 0xAA, 0x61, 0x2F, 
			0x3B, 0x51, 0xA8, 0xE5, 0x37, 0x49, 0xFB, 0xC9, 
			0xCA, 0x0C, 0x18, 0x53, 0x2C, 0xFD, 0x45, 0xE3, 
			0x9A, 0xE6, 0xF1, 0x5D, 0xB0, 0xB6, 0x1B, 0xB4, 
			0xBE, 0x2A, 0x01, 0x5D, 0x67, 0x0A
			# 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x2B, 0x2C, 0x2D, 0xDF, 0x31, 0x7B, 0x7F
			], dtype=np.uint8)
			whitener_seed = 0x7A7F
			puncturing_rate = 1 # 1, 4/6, 3/4, 16/30

			bitstream = np.unpackbits(input_data, bitorder='little')
			whitened_bits = HDT.hdt_whitener(bitstream, whitener_seed)
			encoded_bits = HDT.hdt_encoder(whitened_bits)
			punctured_bits = HDT.hdt_puncturing(encoded_bits, puncturing_rate)
			# punctured_data = np.packbits((punctured_bits[:int(punctured_bits.size/8)*8].reshape(-1,8)), axis=1, bitorder='little').ravel()
			# print('Input data: \n', np.array2string(punctured_data, separator=', ', formatter={'int':lambda x: "0x%02X" % x}, max_line_width=1000))
			
			decoded_bits = HDT.hdt_decoder(punctured_bits, puncturing_rate)
			dewhitened_bits = HDT.hdt_whitener(decoded_bits, whitener_seed)
			extracted_data = np.packbits((dewhitened_bits[:int(dewhitened_bits.size/8)*8].reshape(-1,8)), axis=1, bitorder='little').ravel()
			crc_result = HDT.hdt_crc32(extracted_data)

			print('Extracted data length = ', extracted_data.size, input_data.size, bitstream.size, punctured_bits.size, decoded_bits.size, dewhitened_bits.size)
			print('Extracted data: \n', np.array2string(extracted_data, separator=', ', formatter={'int':lambda x: "0x%02X" % x}, max_line_width=1000))
			print('CRC result:', 'crc valid' if np.all(crc_result==0) else 'crc invalid', '--- crc calc output: ', np.array2string(crc_result, separator=', ', formatter={'int':lambda x: "0x%02X" % x}))

		elif c == 'c':
			filename = IOs.get_file_from_path(path+'/test/', def_file=0, extension='.txt')
			test = IOs.readtxtfile(filename, 0)
			test_number = int(input('test number (0*) = ').strip() or "0")
			data = test[test_number][64:]
			mod_type = 'QAM16' # 'PSK4', 'PSK8', 'QAM16'
			whitener_seed = 0x7A7F
			payload_length = 298

			phase = np.array([(dd>>0)&0x1f for dd in data], dtype='float')
			ampli = np.array([(dd>>5)&0x07 for dd in data], dtype='float')
			k = np.sqrt(10)/8
			ampli_bias = 4.5
			phase_bias = 0.5
			ampli_mod = k*(ampli+ampli_bias)
			phase_mod = (32+phase+phase_bias)%32 * np.pi/16

			psk4_centers = np.arange(8, 32, 8)
			psk4_table = np.array([0, 2, 3, 1], dtype=np.uint8)
			psk8_centers = np.arange(2, 32, 4)
			psk8_table = np.array([0, 4, 6, 2, 3, 7, 5, 1, 0], dtype=np.uint8)
			qam16_centers = np.arange(-2, 3, 2)
			qam16_table = np.array([0, 2, 3, 1], dtype=np.uint8)
					
			if mod_type == 'PSK4':
				data = np.where(np.arange(phase.size) % 2 == 0, phase, (32+phase-4)%32)
				phase_digitized = np.digitize(data, psk4_centers)
				bitstream = psk4_table[phase_digitized]
				bitstream = np.array([np.unpackbits(bits, bitorder='little')[:2] for bits in bitstream]).ravel()
				puncturing_rate = 1 # 1 or 4/6
			elif mod_type == 'PSK8':
				phase_digitized = np.digitize(phase, psk8_centers)
				bitstream = psk8_table[phase_digitized]
				bitstream = np.array([np.unpackbits(bits, bitorder='little')[:3] for bits in bitstream]).ravel()
				puncturing_rate = 3/4
			elif mod_type == 'QAM16':
				data_i = ampli_mod*np.cos(phase_mod)
				data_q = ampli_mod*np.sin(phase_mod)
				data_i_digitized = np.digitize(data_i, qam16_centers)
				data_q_digitized = np.digitize(data_q, qam16_centers)
				data_i_bits = qam16_table[data_i_digitized]
				data_q_bits = qam16_table[data_q_digitized]
				bitstream = data_i_bits+4*data_q_bits
				bitstream = np.array([np.unpackbits(bits, bitorder='little')[:4] for bits in bitstream]).ravel()
				puncturing_rate = 4/6 # 4/6 or 16/30

			decoded_bits = HDT.hdt_decoder(bitstream, puncturing_rate)
			dewhitened_bits = HDT.hdt_whitener(decoded_bits, whitener_seed)
			data_byte = np.packbits((dewhitened_bits[:int(dewhitened_bits.size/8)*8].reshape(-1,8)), axis=1, bitorder='little').ravel()
			extracted_byte = data_byte[:payload_length+4]
			crc_result = HDT.hdt_crc32(extracted_byte)
			
			print('Extracted data length = ', extracted_byte.size, data.size, bitstream.size, decoded_bits.size, dewhitened_bits.size)
			print('Extracted data:', np.array2string(extracted_byte, separator=', ', formatter={'int':lambda x: "0x%02X" % x}, max_line_width=100))
			print('CRC result:', 'crc valid' if np.all(crc_result==0) else 'crc invalid', '--- crc calc output: ', np.array2string(crc_result, separator=', ', formatter={'int':lambda x: "0x%02X" % x}))
			np.savetxt('./_output/hdt_extract_data_' + str(datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))+'.txt', extracted_byte.reshape(1, -1), fmt='0x%02X', delimiter=' ')

		elif c == 's':
			filename = IOs.get_file_from_path(path+'/test/', def_file=5, extension='.txt')
			test = IOs.readtxtfile(filename, 0)
			test_number = int(input('test number (0*) = ').strip() or "0")
			data = test[test_number]

			ampli, phase = HDT.raw_symbol_convert(data)
			# print('ampli = ', ampli[:64])
			print('phase = ', phase[:64])
			phase += np.pi/2
			print('phase = ', phase[:64])
			# phase_sync_control, error_history_control = HDT.demod_psk4_sync(phase)
			phase_sync_control, error_history_control = HDT.demod_psk4_sync(phase[:64])
			phase_sync, error_history = HDT.demod_qam_sync(phase[64:], ampli[64:], init_error=error_history_control[-1])
			
			# np.savetxt('./_output/phase_sync.txt', np.concatenate((phase_sync_control, phase_sync)).reshape(1, -1), fmt='%f', delimiter=', ')
			# np.savetxt('./_output/phase_error.txt', np.concatenate((error_history_control, error_history)).reshape(1, -1), fmt='%f', delimiter=', ')

			# plt.plot(phase_sync_control, 'b.')
			# plt.show()
			plt.plot(error_history_control)
			plt.show()

			# plt.plot(phase_sync, 'b.')
			# for i in range(8):
			# 	plt.plot((i*np.pi/4 - 7*np.pi/8)*np.ones(phase.size), '--')
			# plt.show()
			
			print('freq offset = ', error_history[-1]*1000/((error_history.size+64)*np.pi), 'kHz')
			plt.plot(np.hstack((error_history_control, error_history)))
			plt.show()
			
			whitener_seed = 0x7A7F
			payload_length = 46
			puncturing_rate = 4/6 ## 1, 3/4 4/6, 16/30

			# bitstream = HDT.demod_psk4(phase_sync_control[64:])
			bitstream = HDT.demod_qam16(phase_sync, ampli[64:])
			decoded_bits = HDT.hdt_decoder(bitstream, puncturing_rate)
			dewhitened_bits = HDT.hdt_whitener(decoded_bits, whitener_seed)
			data_byte = np.packbits((dewhitened_bits[:int(dewhitened_bits.size/8)*8].reshape(-1,8)), axis=1, bitorder='little').ravel()
			extracted_byte = data_byte[:payload_length+4]
			crc_result = HDT.hdt_crc32(extracted_byte)
			
			print('Extracted data length = ', extracted_byte.size, data.size, bitstream.size, decoded_bits.size, dewhitened_bits.size)
			print('Extracted data:', np.array2string(extracted_byte, separator=', ', formatter={'int':lambda x: "0x%02X" % x}, max_line_width=100))
			print('CRC result:', 'crc valid' if np.all(crc_result==0) else 'crc invalid', '--- crc calc output: ', np.array2string(crc_result, separator=', ', formatter={'int':lambda x: "0x%02X" % x}))
			np.savetxt('./_output/hdt_extract_data_' + str(datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))+'.txt', extracted_byte.reshape(1, -1), fmt='0x%02X', delimiter=' ')

		elif c == 't':
			filename = IOs.get_file_from_path(path+'/testvectors/', def_file=0, extension='.mat')
			readdata = sio.loadmat(filename)
			data = readdata['test_vect']
			test_vec = int(input('test vector 0..32 (0*) = ').strip() or "0")
			control_vector = data[0, test_vec]['b_ctrl_hdr'][0,0][0]
			payload_vector = data[0, test_vec]['b_in0'][0,0][0]
			control_encoded_vector = data[0, test_vec]['btv0'][0,0][0]
			payload_encoded_vector = data[0, test_vec]['btv2'][0,0][0]
			baseband_sample_vector = data[0, test_vec]['stv0'][0,0][0]

			with open(path+'/testvectors/test_vect_' + str(test_vec) + '.txt', "w") as f:

				data_byte = np.packbits((control_vector[:int(control_vector.size/8)*8].reshape(-1,8)), axis=1, bitorder='little').ravel()
				print('control data: ', np.array2string(data_byte, separator=', ', formatter={'int':lambda x: "0x%02X" % x}, max_line_width=100))
				f.write('\n\nControl data: \n')
				np.savetxt(f, data_byte.reshape(1, -1), fmt='0x%02X', delimiter=', ')
				data_byte = np.packbits((control_encoded_vector[:int(control_encoded_vector.size/8)*8].reshape(-1,8)), axis=1, bitorder='little').ravel()
				print('control encoded data: ', np.array2string(data_byte, separator=', ', formatter={'int':lambda x: "0x%02X" % x}, max_line_width=100))
				f.write('\n\nControl encoded data: \n')
				np.savetxt(f, data_byte.reshape(1, -1), fmt='0x%02X', delimiter=', ')
				data_byte = np.packbits((payload_vector[:int(payload_vector.size/8)*8].reshape(-1,8)), axis=1, bitorder='little').ravel()
				print('payload data: ', np.array2string(data_byte, separator=', ', formatter={'int':lambda x: "0x%02X" % x}, max_line_width=100))
				f.write('\n\nPayload encoded data: \n')
				np.savetxt(f, data_byte.reshape(1, -1), fmt='0x%02X', delimiter=', ')
				data_byte = np.packbits((payload_encoded_vector[:int(payload_encoded_vector.size/8)*8].reshape(-1,8)), axis=1, bitorder='little').ravel()
				print('payload encoded data: ', np.array2string(data_byte, separator=', ', formatter={'int':lambda x: "0x%02X" % x}, max_line_width=100))
				f.write('\n\nPayload encoded encoded data: \n')
				np.savetxt(f, data_byte.reshape(1, -1), fmt='0x%02X', delimiter=', ')

		elif c == 'u':
			# pulse shaping
			#generate channel filter for EBQ

			fs = 16.0
			f_symb = 2.0
			osr = int(fs/f_symb)
			span = 8
			flt = fd.rrc_filter(0.4, osr*span, osr)
			# flt2 = fd.rrc(65, 0.4, f_symb, fs)

			def GenFIRCoeff(nTap, Fcut_off, Ftrans, Fs):
				b = signal.firls(nTap+1, np.array([0., Fcut_off-Ftrans, Fcut_off+Ftrans, fs/2]), [1, 1.4, 0, 0], fs=Fs)
				return b
			
			hdt2m_2p4MHz = GenFIRCoeff(64, 1.2, .22, fs)
			w, h = signal.freqz(hdt2m_2p4MHz*7)
			cm.ModemLib().plot_amp(w, h, fs, 'hdt2m_BW2p4MHz', False)
			# hdt2m_2p4MHz = hdt2m_2p4MHz[:64]*7
			w, h = signal.freqz(hdt2m_2p4MHz[:64]*7)
			cm.ModemLib().plot_amp(w, h, fs, 'hdt2m_BW2p4MHz_mod', False)
			# print('hdt2m_BW2p4MHz = ' + np.array2string(hdt2m_2p4MHz, separator=', ', formatter={'float':lambda x: " %1.8e" % x}, max_line_width=1000))


			# w, h = signal.freqz(flt2)
			# cm.ModemLib().plot_amp(w, h, fs, 'new', False)
			w, h = signal.freqz(flt)
			cm.ModemLib().plot_amp(w, h, fs, 'org', True)
			print('flt= ' + np.array2string(flt, separator=', ', formatter={'float':lambda x: " %1.8e" % x}, max_line_width=1000))
	
		elif c == 'v':
			# pulse shaping
			#generate channel filter for analyzer

			fs = 1.875 * 2
			def GenFIRCoeff(nTap, Fcut_off, Ftrans, Fs):
				b = signal.firls(nTap+1, np.array([0., Fcut_off-Ftrans, Fcut_off+Ftrans, fs/2]), [1, 1.4, 0, 0], fs=Fs)
				return b
			
			hdt2m_2p4MHz = GenFIRCoeff(28, 1.2, .22, fs)
			hdt2m_2p4MHz /= hdt2m_2p4MHz[14]
			w, h = signal.freqz(hdt2m_2p4MHz)
			print('hdt2m_BW2p4MHz = ' + np.array2string(hdt2m_2p4MHz, separator=', ', formatter={'float':lambda x: " %1.8e" % x}, max_line_width=1000))
			cm.ModemLib().plot_amp(w, h, fs, 'hdt2m_BW2p4MHz', False)

			# rrc2m_2p4MHz = fd.rrc(48+1, 0.4, 2.4, fs)
			# rrc2m_2p4MHz /= rrc2m_2p4MHz[24]
			# w, h = signal.freqz(rrc2m_2p4MHz)
			# print('rrc2m_2p4MHz = ' + np.array2string(rrc2m_2p4MHz, separator=', ', formatter={'float':lambda x: " %1.8e" % x}, max_line_width=1000))
			# cm.ModemLib().plot_amp(w, h, fs, 'rrc2m_2p4MHz', False)
						
			# flt_2 = fd.rrc_filter(0.4, 29, fs/2)
			# flt_2 /= flt_2[28//2]
			# w, h = signal.freqz(flt_2)
			# print('flt2 = ' + np.array2string(flt_2, separator=', ', formatter={'float':lambda x: " %1.8e" % x}, max_line_width=1000))
			# cm.ModemLib().plot_amp(w, h, fs, 'new2', False)
			
			b_new = fd.redesign_filter(cf.Constant().chFltGfsk2M, 1.0, 1.42, 1.2, 28+1, fs)
			b_new /= b_new[b_new.size//2]
			print('hdd2m_Eq = ' + np.array2string(b_new, separator=', ', formatter={'float':lambda x: " %1.8e" % x}, max_line_width=1000))
			w, h = signal.freqz(b_new)
			cm.ModemLib().plot_amp(w, h, fs, 'hdd2m_Eq', False)
			

			w, h = signal.freqz(cf.Constant().chFltGfsk2M)
			# print('gfsk2M = ' + np.array2string(samsung_8Msps, separator=', ', formatter={'float':lambda x: " %1.8e" % x}, max_line_width=1000))
			cm.ModemLib().plot_amp(w, h, fs, 'Gfsk2M', True)

		elif c == 'l':
			# LTS sequences:
			for i in range(16):
				print(f'----------seq={i}-------------')
				print(np.array2string(HDT.Constant().preamble_lts[i].real, separator=', ', formatter={'float':lambda x: "%1.8f" % x}, max_line_width=1000))
				print(np.array2string(HDT.Constant().preamble_lts[i].imag, separator=', ', formatter={'float':lambda x: "%1.8f" % x}, max_line_width=1000))
			
			np.set_printoptions(precision=2, suppress=True, linewidth=500)
			for seq in range(16):
				print(f'{seq=:2d} => {(HDT.Constant().preamble_lts_phase[seq]*(2**10)/(2*np.pi))}')
			
			# LTS integer table (for hdl)
			print(HDT.Constant().lts_int)

			lts = np.array([int(np.round(HDT.Constant().preamble_lts_phase[seq][14]*(2**10)/(2*np.pi))) for seq in range(16)])
			print(np.array2string(lts, separator=', ', formatter={'int':lambda x: "%d" % x}, max_line_width=1000))

		elif c == 'f':

			# payload = (np.random.rand(100)*256).astype(np.uint8)
			# txBaseband, fs = HDT.modulation(payload, 1, HDT.Constant.ModulationType.QAM16, 0)
			# np.save('hdt_baseband', txBaseband)

			txBaseband = np.load('hdt_baseband.npy')

			fs = 2.0
			osr = 8

			fs *= osr
			data_upsampled = txBaseband.repeat(osr)
			b = fd.rrc_filter(0.4, osr*7, osr)
			
			# b /= np.sqrt(np.sum(b**2))
			# b1 = signal.firls(141, np.array([0., 1.2-0.2, 1.2+0.2, fs/2]), [10, 10, 1e-2, 1e-2], fs=fs)
			# b1 /= np.sqrt(np.sum(b1**2))
			# b2 = np.array([4.0, 2, -3, -10, -10, 2, 23, 47, 63, 61,44,19,-1,-10,-8,-2,2,3])
			# b2 /= np.sqrt(np.sum(b2**2))
			# w, h = signal.freqz(b2)
			# cm.ModemLib().plot_amp(w, h, fs, 'samsung_8Msps', False)
			# w, h = signal.freqz(b)
			# cm.ModemLib().plot_amp(w, h, fs, 'mine', False)

			sig_tx = np.convolve(b, data_upsampled, 'same')

			w, h = signal.freqz(b)
			# h_new = np.array([1.0/np.abs(i) if np.abs(i) > 2.0 else np.abs(i) for i in h])
			i = int(np.argwhere((w/np.pi) > (1.1*2/fs))[0])
			# f_new = np.concatenate((w[:i]/np.pi, [1.4*2/fs, 1]))
			# h_new = np.concatenate((1.0/np.abs(h[:i]), [1e-4, 1e-5]))

			f_new = np.array([0., 1.2-0.2, 1.2+0.2, fs/2])*(2/fs)
			h_new = np.array([1.0, 1.3, 1e-4, 1e-5])
			b_rx = signal.firls(141, f_new, h_new)
			# w, h = signal.freqz(b_rx)
			# cm.ModemLib().plot_amp(w, h, fs, 'mine', True)

			sig = np.convolve(b_rx, sig_tx, 'same')
			# sig = sig_tx # np.convolve(b1, sig_tx, 'same')
			
			# f, pow = signal.welch(data_upsampled, fs=fs, nperseg=2**12, scaling='spectrum')
			# plt.plot(f, 10.0*np.log10(pow/data_upsampled.size))
			# f, pow = signal.welch(sig_tx, fs=fs, nperseg=2**12, scaling='spectrum')
			# plt.plot(f, 10.0*np.log10(pow/sig_tx.size))
			# f, pow = signal.welch(sig, fs=fs, nperseg=2**12, scaling='spectrum')
			# plt.plot(f, 10.0*np.log10(pow/sig.size))

			# plt.legend(['data_upsampled', 'sig_tx', 'sig'])
			# plt.grid()
			# plt.show()

			sym_width = osr
			data = sig[(16+16+4*9+4+2*17)*sym_width:-24*sym_width]/8.0
			phase = np.angle(data)
			ampli = np.abs(data)

			phase_recovered = (phase[sym_width//2::sym_width] + phase[sym_width//2+1::sym_width])/2
			ampli_recovered = (ampli[sym_width//2::sym_width] + ampli[sym_width//2+1::sym_width])/2
			
			# index = np.arange(i, phase.size, step=sym_width)
			# plt.plot(phase, label='phase')
			# plt.plot(index, phase_recovered, 'b.')
			# plt.plot(ampli, label='ampli')
			# plt.plot(index, ampli_recovered, 'r.')
			# plt.legend()
			# plt.grid()
			# plt.show()

			plt.figure(figsize=(6, 6))
			plt.plot(ampli_recovered*np.cos(phase_recovered), ampli_recovered*np.sin(phase_recovered), 'r.')
			plt.grid()
			plt.show()

		elif c == 'd':

			fir_rx = np.array([	35,		10,		-43,	53,		-13,	-65,	121,	-87,
				-48,	204,	-244,	81,		222,	-451,	376,	50,
				-587,	817,	-438,	-439,	1247,	-1285,	253,	1361,
				-2411,	1774,	721,	-3716,	4745,	-1730,	-5604,	22798,
				22798,	-5604,	-1730,	4745,	-3716,	721,	1774,	-2411,
				1361,	253,	-1285,	1247,	-439,	-438,	817,	-587,
				50,		376,	-451,	222,	81,		-244,	204,	-48,
				-87,	121,	-65,	-13,	53,		-43,	10,		35])

			fir_tx = np.array([22,		-17,	-48,	42,		-26,	-49,	103,	-90,
				-21,	163,	-219,	97,		161,	-379,	348,	-2,
				-469,	704,	-422,	-315,	1035,	-1124,	289,	1090,
				-2041,	1574,	510,	-3097,	4064,	-1625,	-4374,	20471,
				20471,	-4374,	-1625,	4064,	-3097,	510,	1574,	-2041,
				1090,	289,	-1124,	1035,	-315,	-422,	704,	-469,
				-2,		348,	-379,	161,	97,		-219,	163,	-21,
				90,		103,	-49,	-26,	42,		-48,	-17,	22])
			fs = 4.0
			w, h = signal.freqz(fir_rx)
			cm.ModemLib().plot_amp(w, h, fs, 'rx', False)
			w, h = signal.freqz(fir_tx)
			cm.ModemLib().plot_amp(w, h, fs, 'tx', True)

			fs = 1.875 * 2
			
			# w, h = signal.freqz(cf.Constant().chFltGfsk2M)
			# cm.ModemLib().plot_amp(w, h, fs, 'gfsk_2M', False)
			# w, h = signal.freqz(cf.Constant().chFltHdt2M_old)
			# cm.ModemLib().plot_amp(w, h, fs, 'hdt_old_2M', False)
			w, h = signal.freqz(cf.Constant().hbFlt0)
			cm.ModemLib().plot_amp(w, h, 240, 'hb0', False)
			hb0_gen=cf.GenHalfBandFilterCoeff(18, 40.8, 240.0)
			w, h = signal.freqz(hb0_gen)
			cm.ModemLib().plot_amp(w, h, 240, 'hb0_gen', True)
			print('hb0 = ' + np.array2string(cf.Constant().hbFlt0, separator=', ', formatter={'float':lambda x: " %1.8e" % x}, max_line_width=1000))
			print('hb0_gen = ' + np.array2string(hb0_gen, separator=', ', formatter={'float':lambda x: " %1.8e" % x}, max_line_width=1000))
			
			'''
			w, h = signal.freqz(cf.Constant().chFltGfsk2M)
			i = int(np.argwhere((w*fs/(2*np.pi) > 1.1))[0])
			if i%2 != 0:
				i-=1
			print(i, w[i]*fs/2/np.pi, h[i])
			f_new = w[:i]*fs/2/np.pi * 0.7
			plt.plot(w[:i]*fs/2/np.pi, np.abs(h[:i]))
			plt.plot(f_new, np.abs(h[:i]))
			plt.show()
			b = signal.firls(28+1, np.concatenate((w[:i]*fs/2/np.pi, [1.277, fs/2])), np.concatenate((np.abs(h[:i]), [0, 0])), fs=fs)
			b /= b[14]
			b_new = signal.firls(28+1, np.concatenate((f_new, [1.277*0.7, fs/2])), np.concatenate((np.abs(h[:i]), [0, 0])), fs=fs)
			b_new /= b_new[14]
			'''
			# b_new = fd.redesign_filter(cf.Constant().chFltGfsk2M, 1.1, 1.277, 0.65, fs)
			# b_new /= b_new[14]

			# print('rrc2m_2p4MHz = ' + np.array2string(b_new, separator=', ', formatter={'float':lambda x: " %1.8e" % x}, max_line_width=1000))

			# w, h = signal.freqz(cf.Constant().chFltGfsk2M)
			# cm.ModemLib().plot_amp(w, h, fs, 'gfsk2m', False)
			# w, h = signal.freqz(b_new)
			# cm.ModemLib().plot_amp(w, h, 2*fs, 'design_hdt4m', True)
			

			b_new = fd.redesign_filter(cf.Constant().chFltGfsk2M, 1.0, 1.5, 1.3, fs)
			b_new /= b_new[14]

			print('rrc2m_2p4MHz = ' + np.array2string(b_new, separator=', ', formatter={'float':lambda x: " %1.8e" % x}, max_line_width=1000))

			w, h = signal.freqz(cf.Constant().chFltGfsk2M)
			cm.ModemLib().plot_amp(w, h, fs, 'gfsk2m', False)
			w, h = signal.freqz(b_new)
			cm.ModemLib().plot_amp(w, h, fs, 'design_hdt4m', True)
			

			# b = signal.firls(28+1, np.array([0., 1.2-0.22, 1.42, fs/2]), [1, 1.4, 0, 0], fs=fs)
			# b /= b[14]
			# w, h = signal.freqz(b)
			# cm.ModemLib().plot_amp(w, h, fs, 'design_hdt', False)
			# w, h = signal.freqz(cf.Constant().chFltHdt2M)
			# cm.ModemLib().plot_amp(w, h, fs, 'hdt_2M', False)
			# b = signal.firls(28+1, np.array([0., 1.0, 1.4, fs/2]), [1, 1.4, 1e-4, 1e-5], fs=fs)
			# b /= b[14]
			# w, h = signal.freqz(b)
			# cm.ModemLib().plot_amp(w, h, fs, 'hdt_4M', True)
			
			def rrc_filter(N, alpha, Ts, L):
				"""
				Generate root raised cosine filter coefficients.

				N: Number of taps
				alpha: Roll-off factor
				Ts: Symbol period
				L: Upsampling rate
				"""
				t = np.arange(-N/2, N/2) * Ts / L
				h = np.zeros_like(t)

				for i in range(len(t)):
					if t[i] == 0:
						h[i] = 1 - alpha + (4 * alpha / np.pi)
					elif alpha != 0 and t[i] == (Ts / (4 * alpha)):
						h[i] = (alpha / np.sqrt(2)) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha)))
					elif alpha != 0 and t[i] == -(Ts / (4 * alpha)):
						h[i] = (alpha / np.sqrt(2)) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha)))
					else:
						h[i] = (np.sin(np.pi * t[i] * (1 - alpha) / Ts) + 4 * alpha * t[i] / Ts * np.cos(np.pi * t[i] * (1 + alpha) / Ts)) / (np.pi * t[i] * (1 - (4 * alpha * t[i] / Ts)**2) / Ts)

				return h

			# Parameters
			Rs = 2e6  # Symbol rate
			L = 8     # Upsampling rate
			Fs = Rs * L  # Sampling rate
			alpha = 0.35  # Roll-off factor
			N = 101  # Number of taps
			Ts = 1 / Rs  # Symbol period

			# Generate RRC filter coefficients
			h = rrc_filter(N, alpha, Ts, L)
			h_2 = rrc_filter(N, alpha, 1.0, L)
			b = fd.rrc_filter(alpha, N, L)

			# Plot the filter coefficients
			plt.plot(h)
			plt.plot(h_2)
			plt.plot(b)
			plt.title('Root Raised Cosine Filter Coefficients')
			plt.xlabel('Tap Index')
			plt.ylabel('Amplitude')
			plt.grid(True)
			plt.show()

			# Print the filter coefficients
			print(h)
			print(b)

			def upsample(data, rate):
				data_upsampled = data.repeat(rate)
				# b = signal.firls(141, np.array([0., 4.0-0.5, 4.0+0.5, fs_RF/2]), [1, 1.4, 0, 0], fs=fs_RF)
				# b = fd.rrc(141, 0.4, 2.4, 16.0)
				b = fd.rrc_filter(0.4, 64, rate)
				upsampled = np.convolve(b, data_upsampled, mode='same')

				return upsampled

			payload = (np.random.rand(100)*256).astype(np.uint8)
			lts_seq = int(np.random.random(1)*16)

			tx1, fs = HDT.modulation(payload, 1, HDT.Constant().ModulationType.PSK4, lts_seq)
			tx2, fs = HDT.modulation(payload, 1, HDT.Constant().ModulationType.PSK8, lts_seq)
			tx3, fs = HDT.modulation(payload, 1, HDT.Constant().ModulationType.QAM16, lts_seq)
			
			up_rate = 8
			tx1 = upsample(tx1, up_rate)
			tx2 = upsample(tx2, up_rate)
			tx3 = upsample(tx3, up_rate)
			fs *= up_rate

			nperseg=2**12
			f, pow = signal.welch(tx1, fs=fs, nperseg=nperseg, scaling='spectrum')
			plt.plot(f, 10.0*np.log10(pow/tx1.size), label='PSK4')
			f, pow = signal.welch(tx2, fs=fs, nperseg=nperseg, scaling='spectrum')
			plt.plot(f, 10.0*np.log10(pow/tx2.size), label='PSK8')
			f, pow = signal.welch(tx3, fs=fs, nperseg=nperseg, scaling='spectrum')
			plt.plot(f, 10.0*np.log10(pow/tx2.size), label='QAM16')
			plt.legend()
			plt.grid()
			plt.show()

			"""
			fs = 7.5
			flt_1 = cf.Constant().hbFlt5_2M
			w, h = signal.freqz(flt_1)
			cm.ModemLib().plot_amp(w, h, fs, 'flt_1', False)
			
			flt_2 = cf.GenHalfBandFilterCoeff(30, 1.3, fs)
			w, h = signal.freqz(flt_2)
			cm.ModemLib().plot_amp(w, h, fs, 'flt_2', True)
			print('rrc2m_2p4MHz = ' + np.array2string(flt_2, separator=', ', formatter={'float':lambda x: " %1.8e" % x}, max_line_width=1000))


			def rrc_filter(beta, symbol_rate, sampling_rate, span):
				oversampling = sampling_rate / symbol_rate
				num_taps = int(span * oversampling)
				time_index = np.arange(-span/2, span/2 + 1/oversampling, 1/oversampling)

				# Root raised cosine formula
				h_rrc = np.zeros_like(time_index)
				for i, t in enumerate(time_index):
					if t == 0.0:
						h_rrc[i] = 1.0 - beta + (4 * beta / np.pi)
					elif abs(t) == 1 / (4 * beta):
						h_rrc[i] = (beta / np.sqrt(2)) * (
							((1 + 2 / np.pi) * (np.sin(np.pi / (4 * beta)))) +
							((1 - 2 / np.pi) * (np.cos(np.pi / (4 * beta))))
						)
					else:
						numerator = np.sin(np.pi * t * (1 - beta)) + 4 * beta * t * np.cos(np.pi * t * (1 + beta))
						denominator = np.pi * t * (1 - (4 * beta * t)**2)
						h_rrc[i] = numerator / denominator

				# Normalize filter to unit energy
				# h_rrc /= np.sqrt(np.sum(h_rrc**2))
				h_rrc /= h_rrc[np.where(time_index == 0)[0][0]]
				print(time_index)
				print(h_rrc)

				return h_rrc

			# Parameters
			beta = 0.4
			sampling_rate = 1.0 #3.75e6
			symbol_rate = sampling_rate / 4  # Oversampling factor = 4
			span = 7  # Filter spans 6 symbols

			# Design the RRC filter
			# h_rrc = rrc_filter(beta, symbol_rate, sampling_rate, span)
			h_rrc = fd.rrc_filter(beta, 28+1, 4)
			# Normalize filter to unit energy
			# h_rrc /= np.sqrt(np.sum(h_rrc**2))
			# Normalize filter to have center value of 1
			h_rrc /= h_rrc[np.where(time_index == 0)[0][0]]

			w, h = signal.freqz(h_rrc)
			cm.ModemLib().plot_amp(w, h, sampling_rate, 'design', False)
			
			samsung_8Msps = np.array([4.0, 2, -3, -10, -10, 2, 23, 47, 63, 61,44,19,-1,-10,-8,-2,2,3])
			w, h = signal.freqz(samsung_8Msps/100)
			cm.ModemLib().plot_amp(w, h, 1.0, 'samsung_8Msps', True)

			# Plot the filter response
			plt.figure(figsize=(10, 6))
			plt.plot(h_rrc, label="RRC Filter Impulse Response")
			plt.title("Root Raised Cosine (RRC) Filter Impulse Response")
			plt.xlabel("Sample Index")
			plt.ylabel("Amplitude")
			plt.grid()
			plt.legend()
			plt.show()
			"""

			print('\n','='*30,'\n', 'end of debug.')

		elif c == 'x':
			break

	print('Exit')
