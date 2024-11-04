import numpy as np
import matplotlib.pyplot as plt

import RfModel as rf
import ClockRecovery as cr
import Common

from .QamSynchronization import *
from .Constant import *
C = Constant()

def clock_recovery(phase, ampli, fs):
	fsymb = 2 # HDT symbol rate is 2MSymb/s
	period = fs/fsymb

	## clock/data recovery
	phase_extract, index_phase = cr.EarlyLate(phase, period, delta=0)

	phase_scaled = phase_extract
	ampli_scaled = ampli[index_phase]
	
	# plt.plot(ampli[index_phase]*np.cos(phase[index_phase]), ampli[index_phase]*np.sin(phase[index_phase]), 'ro')
	plt.figure(figsize=(6, 6))
	plt.plot(ampli_scaled*np.cos(phase_scaled), ampli_scaled*np.sin(phase_scaled), 'ro')
	plt.grid()
	plt.show()
	
	plt.plot(phase, label='phase')
	plt.plot(index_phase, phase_scaled, 'b.')
	plt.plot(ampli, label='ampli')
	plt.plot(index_phase, ampli_scaled, 'r.')
	plt.legend()
	plt.grid()
	plt.show()
	
	# plt.subplot(211)
	# plt.plot(index_phase, phase_scaled, 'b.')
	# plt.subplot(212)
	# plt.plot(index_phase, ampli_scaled, 'g.')
	# plt.grid()
	# plt.show()

	return phase_scaled, ampli_scaled

def test_packet_extraction(bit_sequence):
	data_byte = np.hstack(np.packbits((bit_sequence[:int(bit_sequence.size/8)*8].reshape(-1,8)), axis=1, bitorder='little'))
	if(np.any(data_byte[:5] != [0xa0, 0xa0, 0xa0, 0xa0, 0xa7])):
		print(data_byte[:10])
		print(f'Extracted data: preamble/SFD is not extracted correctly.')
		return
	data_len_extracted = int(data_byte[5])
	crc_result = CrcCalculation(data_byte[6:6+data_len_extracted+2])
	# print(f'{crc_result=}')
	# print(data_byte[:20])
	crc = np.all(crc_result == 0)

	if crc:
		print(f'Extracted data: Crc is valid, {data_len_extracted=}')
	else:
		print(f'Extracted data: Crc is not valid, {data_len_extracted=}')

	return

def demod_phase_psk4(ph, phase_slice = np.pi/8.0):
	bits =	3 if ph<-4*phase_slice else \
			1 if ph< 0*phase_slice else \
			0 if ph< 4*phase_slice else \
			2
	return bits

def demod_phase_psk8(ph, phase_slice = np.pi/8.0):
	bits =	3 if ph<-7*phase_slice else \
			7 if ph<-5*phase_slice else \
			5 if ph<-3*phase_slice else \
			1 if ph<-1*phase_slice else \
			0  if ph<1*phase_slice else \
			4 if ph<3*phase_slice else \
			6 if ph<5*phase_slice else \
			2 if ph<7*phase_slice else 3
	return bits

def demod_phase_qam16(amp, ph):
	
	def qam_16(data):
		ret_value = 0 if data<-2 else 2 if data<0 else 3 if data<2 else 1
		return ret_value
	
	bit01 = qam_16(amp*np.cos(ph))
	bit23 = qam_16(amp*np.sin(ph))
	# print(f'ph={int(ph*180/np.pi)}, amp={amp:.2f} => {bit01} - {bit23}')
	return bit23*4+bit01

def demod_phase_p4psk4(phase, phase_slice = np.pi/8.0):
	data_out = np.zeros(phase.size, dtype=np.uint8)
	for i in range(phase.size):
		if (i%2) == 0:
			data_out[i] = 3 if phase[i]<-4*phase_slice else 1 if phase[i]< 0*phase_slice else 0 if phase[i]< 4*phase_slice else 2
		else:
			data_out[i] = 2 if phase[i]<-6*phase_slice else 3 if phase[i]< -2*phase_slice else 1 if phase[i]< 2*phase_slice else 0 if phase[i]< 6*phase_slice else 2

	return data_out

def Demodulation(data, fs, mod_type):
	
	phase_sync, ampli_sync = ts_sync(data, fs)
	if phase_sync.size == 0: print('sync is failed.'); return

	phase_recovered, ampli_recovered = clock_recovery(phase_sync, ampli_sync, fs)

	if mod_type == C.ModulationType.TEST_PSK4:
		bit_stream = np.array([demod_phase_psk4(ph) for ph in phase_recovered], dtype=np.uint8)
		bit_stream = np.array(np.hstack([np.unpackbits(bits, bitorder='little')[:2] for bits in bit_stream]))
		test_packet_extraction(bit_stream)
	elif mod_type == C.ModulationType.TEST_PSK8:
		bit_stream = np.array([demod_phase_psk8(ph) for ph in phase_recovered], dtype=np.uint8)
		bit_stream = np.array(np.hstack([np.unpackbits(bits, bitorder='little')[:3] for bits in bit_stream]))
		test_packet_extraction(bit_stream)
	elif mod_type == C.ModulationType.TEST_QAM16:
		bit_stream = np.array([demod_phase_qam16(amp, ph)  for amp, ph in zip(ampli_recovered, phase_recovered)], dtype=np.uint8)
		bit_stream = np.array(np.hstack([np.unpackbits(bits, bitorder='little')[:4] for bits in bit_stream]))
		test_packet_extraction(bit_stream)
	elif mod_type == C.ModulationType.PSK4:
		bit_stream = demod_phase_p4psk4(phase_recovered)
		bit_stream = np.array(np.hstack([np.unpackbits(bits, bitorder='little')[:2] for bits in bit_stream]))
		np.savetxt('p4psk4_bitstream.txt', bit_stream, fmt='%1u', delimiter=',')
	elif mod_type == C.ModulationType.PSK8:
		bit_stream = demod_phase_p4psk4(phase_recovered)
		bit_stream = np.array(np.hstack([np.unpackbits(bits, bitorder='little')[:2] for bits in bit_stream]))
		np.savetxt('psk8_bitstream.txt', bit_stream, fmt='%1u', delimiter=',')
	else:
		print(f'this case should be complete...')
		print(f'Demodulation is failed.')

	return

class demod_iq:
	# unused func
	def demodulationBPSK(data):
		bit_stream_I = np.array([list(C.BPSK_QPSK_table.keys())[list(C.BPSK_QPSK_table.values()).index(val)] for val in data.real])

		return np.hstack([bits for bits in bit_stream_I])

	def demodulationPSK4(data):
		bit_stream_I = np.array([list(C.BPSK_QPSK_table.keys())[list(C.BPSK_QPSK_table.values()).index(val)] for val in data.real])
		bit_stream_Q = np.array([list(C.BPSK_QPSK_table.keys())[list(C.BPSK_QPSK_table.values()).index(val)] for val in data.imag])
		bit_stream = np.array([np.hstack([bit_Q, bit_I]) for bit_I, bit_Q in zip(bit_stream_I, bit_stream_Q)])

		return np.hstack([bits for bits in bit_stream])

	def demodulationPSK8(data):
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

	def demapping_IQ(data, sample_rate, mod_type):
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

		if mod_type != C.ModulationType.PSK8:
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

		data_mapped = data_I_scaled+1j*data_Q_scaled

		if mod_type == C.ModulationType.BPSK:
			bit_stream = demodulationBPSK(data_mapped)
		elif mod_type == C.ModulationType.PSK4:
			bit_stream = demodulationPSK4(data_mapped)
		elif mod_type == C.ModulationType.PSK8:
			bit_stream = demodulationPSK8(data_mapped)
		elif mod_type == C.ModulationType.QAM16:
			bit_stream = demodulationQAM16(data_mapped)
		elif mod_type == C.ModulationType.QAM64:
			bit_stream = demodulationQAM64(data_mapped)
		else:
			bit_stream = 0
		return bit_stream

	def clock_recovery_IQ(data, fs):
		fsymb = 2 # HDT symbol rate is 2MSymb/s
		period = fs/fsymb
		
		## clock/data recovery
		data_I_extract, index_I = cr.EarlyLate(data.real, period, delta = 0)
		data_Q_extract, index_Q = cr.EarlyLate(data.imag, period, delta = 0)
		
		data_I_extract = np.array([np.mean(data.real[int((i+j)/2)-1:int((i+j)/2)+2]) for i, j in zip(index_I, index_Q)])
		data_Q_extract = np.array([np.mean(data.imag[int((i+j)/2)-1:int((i+j)/2)+2]) for i, j in zip(index_I, index_Q)])

		plt.plot(data_I_extract, data_Q_extract, 'r.')
		plt.grid()
		plt.show()

		# plt.plot(data.real)
		# plt.plot(data.imag)
		# plt.plot(index_I[:data_I_extract.size], data_I_extract, 'b.')
		# plt.plot(index_Q[:data_Q_extract.size], data_Q_extract, 'r.')
		# plt.legend(['real', 'imag'])
		# plt.grid()
		# plt.show()

		return data_I_extract, data_Q_extract

