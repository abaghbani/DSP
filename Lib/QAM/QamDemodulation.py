import numpy as np
import matplotlib.pyplot as plt

import RfModel as rf
import ClockRecovery as cr
import Common

from .QamSynchronization import *
from .Constant import *
C = Constant()

def DataExtraction(bit_sequence):
	data_byte = np.hstack(np.packbits((bit_sequence[:int(bit_sequence.size/8)*8].reshape(-1,8)), axis=1, bitorder='little'))
	#print(data_byte)
	if(np.any(data_byte[0:5] != [0xa0, 0xa0, 0xa0, 0xa0, 0xa7])):
		return 0, 0, False
	data_len_extracted = int(data_byte[5])
	data_extracted = data_byte[6:6+data_len_extracted]
	crc_result = CrcCalculation(data_byte[6:6+data_len_extracted+2])
		
	return data_extracted, data_len_extracted, np.all(crc_result == 0)

def demapping(data, sample_rate, type):
	basic_sample_rate = 2 # HDT sample rate is 2MSymb/s
	period = int(sample_rate/basic_sample_rate)
	data_scaled = data
	
	## clock/data recovery
	data_I_extract, index_I = cr.EarlyLate(data_scaled.real, period, gap = 0.1, delta = period//10)
	data_Q_extract, index_Q = cr.EarlyLate(data_scaled.imag, period, gap = 0.1, delta = period//10)
	
	## avraging samples
	#index_I = index_I[:np.min(index_I.size, index_Q.size)]
	#index_Q = index_Q[:np.min(index_I.size, index_Q.size)]
	data_I_extract = np.array([np.mean(data_scaled.real[int((i+j)/2)-1:int((i+j)/2)+2]) for i, j in zip(index_I, index_Q)])
	data_Q_extract = np.array([np.mean(data_scaled.imag[int((i+j)/2)-1:int((i+j)/2)+2]) for i, j in zip(index_I, index_Q)])

	if type != C.ModulationType.PSK8:
		data_I_scaled = np.array([-7 if val<-6 else -5 if val<-4 else -3 if val<-2 else -1 if val<0 else 1 if val<2 else 3 if val<4 else 5 if val<6 else 7 for val in data_I_extract])
		data_Q_scaled = np.array([-7 if val<-6 else -5 if val<-4 else -3 if val<-2 else -1 if val<0 else 1 if val<2 else 3 if val<4 else 5 if val<6 else 7 for val in data_Q_extract])
	else:
		data_I_scaled = np.array([-4 if val<-3.5 or val>3.5 else -3 if val<-2.5 else -2 if val<-1.5 else -1 if val<-0.5 else 0 if val<0.5 else 1 if val<1.5 else 2 if val<2.5 else 3 for val in np.angle(data_I_extract+1j*data_Q_extract)*4/np.pi])
		data_Q_scaled = np.zeros(data_I_scaled.size)

	plt.plot(data_I_extract, data_Q_extract, 'ro')
	plt.grid()
	plt.show()

	plt.plot(data_scaled.real)
	plt.plot(data_scaled.imag)
	plt.plot(index_I[:data_I_extract.size], data_I_extract, 'b.')
	plt.plot(index_Q[:data_Q_extract.size], data_Q_extract, 'r.')
	#plt.plot(index, data_I_scaled, 'bo')
	#plt.plot(index, data_Q_scaled, 'ro')
	plt.legend(['real', 'imag'])
	plt.grid()
	plt.show()

	return data_I_scaled+1j*data_Q_scaled

def demodulationBPSK(data):
	bit_stream_I = np.array([list(C.BPSK_QPSK_table.keys())[list(C.BPSK_QPSK_table.values()).index(val)] for val in data.real])

	return np.hstack([bits for bits in bit_stream_I])

def demodulationQPSK(data):
	bit_stream_I = np.array([list(C.BPSK_QPSK_table.keys())[list(C.BPSK_QPSK_table.values()).index(val)] for val in data.real])
	bit_stream_Q = np.array([list(C.BPSK_QPSK_table.keys())[list(C.BPSK_QPSK_table.values()).index(val)] for val in data.imag])
	bit_stream = np.array([np.hstack([bit_I, bit_Q]) for bit_I, bit_Q in zip(bit_stream_I, bit_stream_Q)])

	return np.hstack([bits for bits in bit_stream])

def demodulation8PSK(data):
	bit_stream = np.array([list(C.PSK8_3bits_table.keys())[list(C.PSK8_3bits_table.values()).index(val)] for val in data.real])

	return np.hstack([bits for bits in bit_stream])

def demodulationQAM16(data):
	if np.all(np.abs(data.real) <=3) and np.all(np.abs(data.imag) <=3):
		bit_stream_I = np.array([list(C.QAM16_2bits_table.keys())[list(C.QAM16_2bits_table.values()).index(val)] for val in data.real])
		bit_stream_Q = np.array([list(C.QAM16_2bits_table.keys())[list(C.QAM16_2bits_table.values()).index(val)] for val in data.imag])
		bit_stream = np.hstack((bit_stream_I, bit_stream_Q))
		return np.hstack([bits for bits in bit_stream])
	else:
		return 0

def demodulationQAM64(data):
	bit_stream_I = np.array([list(C.QAM64_3bits_table.keys())[list(C.QAM64_3bits_table.values()).index(val)] for val in data.real])
	bit_stream_Q = np.array([list(C.QAM64_3bits_table.keys())[list(C.QAM64_3bits_table.values()).index(val)] for val in data.imag])
	bit_stream = np.hstack((bit_stream_I, bit_stream_Q))

	return np.hstack([bits for bits in bit_stream])

def Demodulation(data, sample_rate, type):
	
	data_sync = ts_sync_hdl(data, sample_rate)
	#data_sync = sync().ts_sync(data, sample_rate)

	data_mapped = demapping(data_sync, sample_rate, type)

	if type == C.ModulationType.BPSK:
		bit_stream = demodulationBPSK(data_mapped)
	elif type == C.ModulationType.QPSK:
		bit_stream = demodulationQPSK(data_mapped)
	elif type == C.ModulationType.PSK8:
		bit_stream = demodulation8PSK(data_mapped)
	elif type == C.ModulationType.QAM16:
		bit_stream = demodulationQAM16(data_mapped)
	elif type == C.ModulationType.QAM64:
		bit_stream = demodulationQAM64(data_mapped)
	else:
		bit_stream = 0
	
	## extract data
	if hasattr(bit_stream, "__len__") and bit_stream.size != 0:
		data_extract, data_len, crc = DataExtraction(bit_stream)
		
		if crc and data_len!=0:
			print(f'Extracted data: Crc is valid, {data_len=}')
		elif not crc and data_len!=0:
			print(f'Extracted data: Crc is not valid, {data_len=}')
		elif not crc and data_len==0:
			print(f'Extracted data: preamble/SFD is not extracted correctly.')
			#print(f'{[hex(val)[2:] for val in payload[np.nonzero(data_extract != payload)]]}, {data_len=}')
			#print(f'{[hex(val)[2:] for val in data_extract[np.nonzero(data_extract != payload)]]}, {data_len=}')
	else:
		print(f'Demodulation is failed.')
		data_extract = 0
		data_len = 0
		crc = False

	return data_extract, data_len, crc
