import numpy as np
import scipy as scipy
import scipy.signal as signal
import matplotlib.pyplot as plt

import ClockRecovery as cr

from .Constant import *
C = Constant()

def DataExtraction(bit_sequence):
	data_xcorr = np.array([i for i in range(bit_sequence.size-24) if np.all(bit_sequence[i:i+24] == np.hstack(np.unpackbits(np.hstack((C.preamble[:2], C.sfd)), bitorder = 'little')))])
	if data_xcorr.size > 0:
		index_first_bit = int(data_xcorr[0])
		data_byte = np.hstack(np.packbits(np.reshape(bit_sequence[index_first_bit:index_first_bit+int((bit_sequence.size-index_first_bit)/8)*8],(-1,8)), axis=1, bitorder='little'))
		data_len_extracted = int(data_byte[3])
		data_extracted = data_byte[4:4+data_len_extracted]
		crc_result = CrcCalculation(data_byte[4:4+data_len_extracted+2])
	else:
		print('Data Extraction Error: preamble/SFD is not found in extracted data')
		data_len_extracted = 0
		data_extracted = 0
		crc_result = -1
		
	return data_extracted, data_len_extracted, np.all(crc_result == 0)

def Demodulation(data, over_sample_rate):
	
	#plt.plot(data.real, data.imag, 'ro')
	#plt.plot(data[over_sample_rate//2::over_sample_rate].real, data[over_sample_rate//2::over_sample_rate].imag, 'bo')
	#plt.plot(data.real[:200])
	#plt.plot(data.imag[:200])
	#plt.grid()
	#plt.show()

	## carrier recovery: phase offset detection
	preamble_index = 0
	n_slice = 8
	for index in range(1, data.size-n_slice*over_sample_rate):
		data_slice = data[index:index+n_slice*over_sample_rate:over_sample_rate]

		## Methode-1
		xcorr_I = np.abs(np.mean(data_slice.real))
		xcorr_Q = np.abs(np.mean(data_slice.imag))
		print(f'{index=:=>4}-----{xcorr_I=}, {xcorr_Q=}, {np.abs(data_slice.real).min()}, {np.abs(data_slice.imag).min()}')
		if xcorr_I < 0.1*np.abs(data_slice.real).min() and xcorr_Q < 0.1*np.abs(data_slice.imag).min():
			preamble_index = index
			break

		## Methode-2
		#error_det = np.abs(data_slice[1:8]+data_slice[0:7])
		#print(f'{index=:=>4}-----error data: {error_det.round(1)}, {xcorr_I=}, {xcorr_Q=}')
		##print(f'{index=:=>4}-----error data:', ['{:.1f}'.format(i) for i in error_det])
		#if np.all(error_det < 2440):
		#	preamble_index = index
		#	break
		
	if preamble_index:
		phase_offset = np.pi/4 - np.arctan2(data.imag[index+over_sample_rate], data.real[index+over_sample_rate])
		magnitude_scale = np.mean(np.abs(data[index:index+3*over_sample_rate]))
		print(f'{phase_offset=},  {magnitude_scale=}')
	else:
		print(f'phase offset cancelation: no preamble is detected')
		return 0

	## phase offset cancelation
	data_offset = data * np.exp(1j*phase_offset)
	plt.plot(data_offset.real, data_offset.imag, '.')
	plt.show()
	
	## clock/data recovery
	data_I_extract, index = cr.EarlyLate(data_offset.real, over_sample_rate, magnitude_scale*0.1, over_sample_rate//10, False)
	#data_Q_extract, index = cr.EarlyLate(data_offset.imag, over_sample_rate, magnitude_scale*0.1, over_sample_rate//10, True)
	data_I_extract = np.array([np.mean(data_offset.real[i-1:i+2]) for i in index])
	data_Q_extract = np.array([np.mean(data_offset.imag[i-1:i+2]) for i in index])

	## scaled I/Q data to range [-3, -1, 1, 3]
	data_I_scaled = np.array([-3 if val <-0.5*magnitude_scale else -1 if val < 0 else 1 if val < 0.5*magnitude_scale else 3 for val in data_I_extract])
	data_Q_scaled = np.array([-3 if val <-0.5*magnitude_scale else -1 if val < 0 else 1 if val < 0.5*magnitude_scale else 3 for val in data_Q_extract])
	
	## convert phase (-3-3j,-3-1j, ...3+3j) to bit
	bit4_sequence = np.array([(C.QAM16_bit[int((val_i+3)/2 +(val_q+3)*2)]) for val_i, val_q in zip(data_I_scaled, data_Q_scaled)], dtype=np.uint8)
	bit4_sequence = bit4_sequence[:int((bit4_sequence.size//2)*2)]
	bit_sequence = np.unpackbits(bit4_sequence[0::2]+bit4_sequence[1::2]*16, bitorder='little')
	
	## extract data
	data_extract, data_len, crc = DataExtraction(bit_sequence)

	if data_len:
		print(f'{[hex(val)[2:] for val in data_extract]}, {data_len=}, {crc=}')
	else:
		print('no data extracted')
	
	## plot data:
	#plt.plot(data_offset.real)
	#plt.plot(data_offset.imag)
	#plt.show()
	
	#plt.plot(data_I_scaled)
	#plt.plot(data_Q_scaled)
	#plt.show()

	plt.plot(data_offset.real)
	plt.plot(data_offset.imag)
	plt.plot(index, data_I_extract, 'b.')
	plt.plot(index, data_Q_extract, 'r.')
	plt.legend(['real', 'imag'])
	plt.grid()
	plt.show()


	return 1
