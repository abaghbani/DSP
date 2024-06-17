import numpy as np
import scipy as scipy
import scipy.signal as signal
import matplotlib.pyplot as plt

import Common as cm
from .OfdmSynchronization import *
from .Constant import *
C = Constant()

def downsampling_demod(data, sample_rate):
	if sample_rate == 240:
		downSampleData = (data[6::12]+data[7::12])/2
	elif sample_rate == 20:
		downSampleData = data[1:]
	elif sample_rate == 40:
		downSampleData = (data[2:4500:2]+data[3:4500:2])/2
	else:
		downSampleData = 0
		print('Error: unknown sample rate!!')
	return downSampleData

def ofdm_demod(data):
	ofdmData_time = data[:(data.size//80)*80].reshape([-1, 80])
	ofdmData_freq = np.array([np.fft.fftshift(np.fft.fft(dd[16:80])) for dd in ofdmData_time])
	
	return ofdmData_freq

def channelEstimate_demod(data):
	allCarriers = np.arange(64)
	ret_val = []
	index = 0

	for dd in data:
		Hest_at_pilots = dd[C.pilot_index] / (C.pilot_symbol*C.pilot_polarity[index%C.pilot_polarity.size])

		Hest_abs = scipy.interpolate.interp1d(C.pilot_index, abs(Hest_at_pilots), fill_value ='extrapolate')(allCarriers)
		Hest_phase = scipy.interpolate.interp1d(C.pilot_index, np.angle(Hest_at_pilots), fill_value ='extrapolate')(allCarriers)
		
		Hest = Hest_abs * np.exp(1j*Hest_phase)
		data_est = dd/Hest
		
		#plt.plot(allCarriers, abs(dd), label='input data')
		#plt.plot(allCarriers, abs(data_est), label='correct data')
		#plt.stem(C.pilot_index, abs(Hest_at_pilots), label='Pilot estimates')
		#plt.stem(allCarriers, abs(Hest), label='Estimated channel via interpolation')
		#plt.legend(fontsize=10)
		#plt.show()

		#plt.plot(allCarriers, np.angle(dd), label='input data')
		#plt.plot(allCarriers, np.angle(data_est), label='Correct data')
		#plt.stem(C.pilot_index, np.angle(Hest_at_pilots), label='Pilot estimates')
		#plt.plot(allCarriers, np.angle(Hest), label='Estimated channel via interpolation')
		#plt.legend(fontsize=10)
		#plt.show()
		
		ret_val = np.append(ret_val, np.hstack([data_est[6:11], data_est[12:25], data_est[26:32], data_est[33:39], data_est[40:53], data_est[54:59]]))
		index += 1
	
	return ret_val


def demap_demod(data):
	data_I_demapped = np.array([-7 if val<-6 else -5 if val<-4 else -3 if val<-2 else -1 if val<0 else 1 if val<2 else 3 if val<4 else 5 if val<6 else 7 for val in data.real])
	data_Q_demapped = np.array([-7 if val<-6 else -5 if val<-4 else -3 if val<-2 else -1 if val<0 else 1 if val<2 else 3 if val<4 else 5 if val<6 else 7 for val in data.imag])

	plt.plot(data.real, data.imag, 'bo')
	plt.plot(data_I_demapped, data_Q_demapped, 'ro')
	#plt.xlim(-2,2)
	#plt.ylim(-2,2)
	plt.grid()
	plt.show()

	return data_I_demapped+1j*data_Q_demapped

def bpsk_demod(data):
	bit_stream_I = np.array([list(C.BPSK_QPSK_table.keys())[list(C.BPSK_QPSK_table.values()).index(val)] for val in data.real])

	return np.hstack([bits for bits in bit_stream_I])

def qpsk_demod(data):
	bit_stream_I = np.array([list(C.BPSK_QPSK_table.keys())[list(C.BPSK_QPSK_table.values()).index(val)] for val in data.real])
	bit_stream_Q = np.array([list(C.BPSK_QPSK_table.keys())[list(C.BPSK_QPSK_table.values()).index(val)] for val in data.imag])
	bit_stream = np.array([np.hstack([bit_I, bit_Q]) for bit_I, bit_Q in zip(bit_stream_I, bit_stream_Q)])

	return np.hstack([bits for bits in bit_stream])

def qam16_demod(data):
	bit_stream_I = np.array([list(C.QAM16_2bits_table.keys())[list(C.QAM16_2bits_table.values()).index(val)] for val in data.real])
	bit_stream_Q = np.array([list(C.QAM16_2bits_table.keys())[list(C.QAM16_2bits_table.values()).index(val)] for val in data.imag])
	bit_stream = np.hstack((bit_stream_I, bit_stream_Q))

	return np.hstack([bits for bits in bit_stream])

def qam64_demod(data):
	bit_stream_I = np.array([list(C.QAM64_3bits_table.keys())[list(C.QAM64_3bits_table.values()).index(val)] for val in data.real])
	bit_stream_Q = np.array([list(C.QAM64_3bits_table.keys())[list(C.QAM64_3bits_table.values()).index(val)] for val in data.imag])
	bit_stream = np.hstack((bit_stream_I, bit_stream_Q))

	return np.hstack([bits for bits in bit_stream])

def DataExtraction(bit_sequence):
	data_byte = np.hstack(np.packbits((bit_sequence[:int(bit_sequence.size/8)*8].reshape(-1,8)), axis=1, bitorder='little'))
	#print(data_byte)
	if(np.any(data_byte[0:5] != [0xa0, 0xa0, 0xa0, 0xa0, 0xa7])):
		print('Data Extraction Error: preamble/SFD is not extracted correctly.')
		return 0,0,0
	else:
		data_len_extracted = int(data_byte[5])
		data_extracted = data_byte[6:6+data_len_extracted]
		crc_result = CrcCalculation(data_byte[6:6+data_len_extracted+2])
		return data_extracted, data_len_extracted, np.all(crc_result == 0)

def Demodulation(data, sample_rate, modulation_type=None):
	
	######################
	## synchronization:
	######################
	data_sync = StsCarrierSync(data, sample_rate)
	data_aligned = LtsCarrierSync(data_sync, sample_rate)
	
	np.save('test201', data_aligned)
	#data_aligned=np.load('test171.npy')
	#data_aligned_2=np.load('test172.npy')

	######################
	## demodulation:
	######################
	data_downsample = downsampling_demod(data_aligned, sample_rate)
	data_freq = ofdm_demod(data_downsample)
	data_demod = channelEstimate_demod(data_freq)

	#for i in range(10):
	#	plt.plot(data_demod[i*48:(i+1)*48].real, data_demod[i*48:(i+1)*48].imag, 'bo')
	#	plt.xlim(-2,2)
	#	plt.ylim(-2,2)
	#	plt.grid()
	#	plt.show()

	if modulation_type != None:
		if modulation_type == C.ModulationType.BPSK:
			data_demapped = demap_demod(data_demod[:]/C.ModulationFactor.BPSK)
			bit_stream = bpsk_demod(data_demapped)
		elif modulation_type == C.ModulationType.QPSK:
			data_demapped = demap_demod(data_demod[:]/C.ModulationFactor.QPSK)
			bit_stream = qpsk_demod(data_demapped)
		elif modulation_type == C.ModulationType.QAM16:
			data_demapped = demap_demod(data_demod[:]/C.ModulationFactor.QAM16)
			bit_stream = qam16_demod(data_demapped)
		elif modulation_type == C.ModulationType.QAM64:
			data_demapped = demap_demod(data_demod[:]/C.ModulationFactor.QAM64)
			bit_stream = qam64_demod(data_demapped)
		else:
			bit_stream = 0

		## extract data
		data_extract, data_len, crc = DataExtraction(bit_stream)
		
		if data_len:
			if crc:
				print(f'Extracted data: Crc is valid, {data_len=}')
			else:
				print(f'Extracted data: Crc is not valid, {data_len=}')
		else:
			print(f'Extracted data: no data is axtracted')
		return data_extract, data_len, crc
	else:
		# non-HT L-sig
		data_demapped = demap_demod(data_demod[:48]/C.ModulationFactor.BPSK)
		bit_stream = bpsk_demod(data_demapped)
		bit_interleaved = [bit_stream[k] for k in interleaving_transmitter(48, 1)]
		
		# A = 1 + x^2 + x^3 + x^5 + x^6 b1101101 = 109
		# B = 1 + x^1 + x^2 + x^3 + x^6 b1001111 = 79
		conv = cm.ConvolutionalCode((109, 79))
		signal_bits, _, corrected_errors = conv.decode(bit_interleaved)
		print(f'{signal_bits=} {corrected_errors=}')
		signal_rate = C.signal_rate_value[tuple(signal_bits[:4])]
		signal_length = int(np.packbits(signal_bits[5:17], bitorder='little')[0])
		signal_bits = np.array(signal_bits)
		signal_parity = True if (signal_bits[signal_bits != 0]).size % 2 == 0 else False
		print(f'{signal_rate=}, {signal_length=}, {signal_parity=}\n\n')

		symb_length = int(np.ceil( (signal_length*8+22)/(signal_rate*4) ))

		# HT and VHT detection
		if signal_rate == 6:
			data_demapped = demap_demod(data_demod[48:3*48]/C.ModulationFactor.BPSK)
			bit_stream = bpsk_demod(data_demapped*(-1j))
			bit_interleaved_1 = [bit_stream[k] for k in interleaving_transmitter(48, 1)]
			bit_interleaved_2 = [bit_stream[48+k] for k in interleaving_transmitter(48, 1)]
			ht_signal_1, _, corrected_errors_1 = conv.decode(bit_interleaved_1)
			ht_signal_2, _, corrected_errors_2 = conv.decode(bit_interleaved_2)
			ht_sig_crc = CrcCalculation_HT_sig(np.concatenate((ht_signal_1[:], ht_signal_2[:10])))

			non_ht = True
			if (ht_sig_crc == np.packbits(ht_signal_2[10:18])) or (corrected_errors_1 < 4 and corrected_errors_2 < 4):
				print(f'{ht_signal_1=} {corrected_errors_1=}')
				print(f'{ht_signal_2=} {corrected_errors_2=}')
			
				ht_signal_mcs = int(np.packbits(ht_signal_1[0:7], bitorder='little')[0])
				ht_signal_40M = ht_signal_1[7]
				ht_signal_length = int(np.packbits(ht_signal_1[8:24], bitorder='little')[0])
				#signal_bits = np.array(signal_bits)
				#signal_parity = True if (signal_bits[signal_bits != 0]).size % 2 == 0 else False
				print(f'{ht_signal_mcs=}, {ht_signal_length=}, {ht_signal_40M=}, calc crc = {hex(ht_sig_crc)}, rx crc = {hex(int(np.packbits(ht_signal_2[10:18])))}')
				print('='*40,'\n\r', f'STBC = {ht_signal_2[4:6]}, short GI = {ht_signal_2[7]}, Ness = {ht_signal_2[8:10]}\n\r', '='*40, '\n\n', sep='')
				
				non_ht = False
				#data_demapped = demap_demod(data_demod[5*48:6*48]/C.ModulationFactor.BPSK)

		elif (signal_rate == 6 and non_ht) or signal_rate == 9:
			data_demapped = demap_demod(data_demod[48:48*(symb_length+1)]/C.ModulationFactor.BPSK)
			bit_stream = bpsk_demod(data_demapped)
		elif signal_rate == 12 or signal_rate == 18:
			data_demapped = demap_demod(data_demod[48:48*(symb_length+1)]/C.ModulationFactor.QPSK)
			bit_stream = qpsk_demod(data_demapped)
		elif signal_rate == 24 or signal_rate == 36:
			data_demapped = demap_demod(data_demod[48:48*(symb_length+1)]/C.ModulationFactor.QAM16)
			bit_stream = qam16_demod(data_demapped)
		elif signal_rate == 48 or signal_rate == 54:
			data_demapped = demap_demod(data_demod[48:48*(symb_length+1)]/C.ModulationFactor.QAM64)
			bit_stream = qam64_demod(data_demapped)

		return data_demod
