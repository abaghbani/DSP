import numpy as np
import matplotlib.pyplot as plt

import RfModel as rf
import ClockRecovery as cr
import Common

from .synchronization import *
from .sts_detection import *
from .lts_detection import *
from .constant import *
C = Constant()

def clock_recovery(phase, ampli, fs, plot_enable=False):
	fsymb = 2 # HDT symbol rate is 2MSymb/s
	period = fs/fsymb

	## clock/data recovery
	phase_extract, index_phase = cr.EarlyLate(phase, period, delta=0)

	phase_scaled = phase_extract
	ampli_scaled = ampli[index_phase]
	
	if plot_enable:
		plt.plot(phase, label='phase')
		plt.plot(index_phase, phase_scaled, 'b.')
		plt.plot(ampli, label='ampli')
		plt.plot(index_phase, ampli_scaled, 'r.')
		plt.legend()
		plt.grid()
		plt.show()
	
	return phase_scaled, ampli_scaled

def test_packet_extraction(bit_sequence):
	data_byte = np.packbits((bit_sequence[:int(bit_sequence.size/8)*8].reshape(-1,8)), axis=1, bitorder='little').ravel()
	if sum(1 for x, y in zip(data_byte[:5], np.array([0xa0, 0xa0, 0xa0, 0xa0, 0xa7])) if x == y) < 4:
		print(data_byte[:10])
		print(f'Extracted data: preamble/SFD is not extracted correctly.')
		return
	data_len_extracted = int(data_byte[5])
	crc_result = crc_calc(data_byte[6:6+data_len_extracted+2])
	crc = np.all(crc_result == 0)

	if crc:
		print(f'Extracted data: Crc is valid, {data_len_extracted=}')
	else:
		print(f'Extracted data: Crc is not valid, {data_len_extracted=}')

	return

def demod_psk4(data):
	# [3 if ph<(-1*np.pi/2) else 1 if ph<(0*np.pi/2) else 0 if ph<(1*np.pi/2) else 2]
	phase_centers = np.arange(-1, 2) * (np.pi/2)
	demodulated_table = np.array([3, 1, 0, 2], dtype=np.uint8)
	phase_digitized = np.digitize(data, phase_centers)
	bit_stream = demodulated_table[phase_digitized]
	bit_stream = np.array([np.unpackbits(bits, bitorder='little')[:2] for bits in bit_stream]).ravel()

	return bit_stream

def demod_psk8(data):
	# [3 if ph<(-7*np.pi/8) else 7 if ph<(-5*np.pi/8) else 5 if ph<(-3*np.pi/8) else 1 ...]
	phase_centers = np.arange(-7, 8, 2) * (np.pi/8)
	demodulated_psk8_table = np.array([3, 7, 5, 1, 0, 4, 6, 2, 3], dtype=np.uint8)
	phase_digitized = np.digitize(data, phase_centers)
	bit_stream = demodulated_psk8_table[phase_digitized]
	bit_stream = np.array([np.unpackbits(bits, bitorder='little')[:3] for bits in bit_stream]).ravel()
	return bit_stream

def demod_qam16(phase, ampli):

	def qam_16(data):
		# [0 if data<-2 else 2 if data<0 else 3 if data<2 else 1]
		data_centers = np.arange(-2, 3, 2)
		demodulated_table = np.array([0, 2, 3, 1], dtype=np.uint8)
		data_digitized = np.digitize(data, data_centers)
		ret_value = demodulated_table[data_digitized]
		return ret_value
	
	bit01 = qam_16(ampli*np.cos(phase))
	bit23 = qam_16(ampli*np.sin(phase))
	bit_stream = bit23*4+bit01
	bit_stream = np.array([np.unpackbits(bits, bitorder='little')[:4] for bits in bit_stream]).ravel()
	return bit_stream

def demod_psk4_sync(data, init_error = 0.0, sum_limit=0.85*np.pi/4, eta = 0.1):
	data = np.where(np.arange(data.size) % 2 == 0, data, phase_correction(data-(np.pi/4)))
	
	data_out = np.zeros(data.size, dtype=type(data[0]))
	error_history = np.zeros(data.size, dtype='float')
	offset = init_error
	
	phase_centers = np.arange(-1, 2)*(np.pi/2)
	for i, dd in enumerate(data):
		d1=phase_correction(dd-offset)
		error = d1-(2*np.digitize(d1, phase_centers)-3)*(np.pi/4)
		if abs(error) < sum_limit:
			offset += eta*error
		error_history[i] = offset
		data_out[i] = d1

	return data_out, error_history

def demod_psk8_sync(data, init_error = 0.0, error_limit=0.85*np.pi/8, eta = 0.2):
	data_out = np.zeros(data.size, dtype=type(data[0]))
	error_history = np.zeros(data.size, dtype='float')
	offset = init_error
	
	phase_centers = np.arange(-7, 8, 2) * (np.pi/8)
	for i, dd in enumerate(data):
		d1=phase_correction(dd-offset)
		error = d1-(np.digitize(d1, phase_centers)-4)*(np.pi/4)
		if abs(error) < error_limit:
			offset += eta*error
		error_history[i] = offset
		data_out[i] = d1

	return data_out, error_history

def demod_qam_sync(phase, ampli, init_error = 0.0, sum_limit=0.85*np.pi/4, eta = 0.2):
	data_out = np.zeros(phase.size, dtype=type(phase[0]))
	error_history = np.zeros(phase.size, dtype='float')
	offset = init_error
	
	phase_centers = np.arange(-1, 2)*(np.pi/2)
	for i, dd in enumerate(phase):
		d1=phase_correction(dd-offset)
		if (ampli[i] <2.0 or ampli[i]>4.01):
			error = phase_correction(d1-(2*np.digitize(d1, phase_centers)-3)*(np.pi/4))
			if abs(error) < sum_limit:
				offset += eta*error
		error_history[i] = offset
		data_out[i] = d1

	return data_out, error_history


def demodulation(data, fs, mod_type, save_enable=False):

	# phase, ampli = ts_sync(data, fs)
	# if phase.size == 0: print('sync is failed.'); return
	# phase_recovered, ampli_recovered = clock_recovery(phase, ampli, fs, plot_enable=True)

	phase_sts, ampli_sts, data_sts  = sts_detection(data, fs, plot_enable=False)
	phase_recovered, ampli_recovered  = lts_detection(phase_sts, ampli_sts, data_sts, plot_enable=True)

	plt.figure(figsize=(6, 6))
	plt.plot(ampli_recovered*np.cos(phase_recovered), ampli_recovered*np.sin(phase_recovered), 'r.')
	plt.grid()
	plt.show()

	def ploting_result(ph, ampli, test, type):
		plt.figure(figsize=(6, 6))
		plt.plot(ampli*np.cos(ph), ampli*np.sin(ph), 'r.')
		if type == C.ModulationType.PSK4:
			plt.plot(1.4*np.cos(np.arange(np.pi/4, 2*np.pi, np.pi/2)), 1.4*np.sin(np.arange(np.pi/4, 2*np.pi, np.pi/2)), 'bo')
		elif type == C.ModulationType.PSK8:
			plt.plot(np.cos(np.arange(0, 2*np.pi, np.pi/4)), np.sin(np.arange(0, 2*np.pi, np.pi/4)), 'bo')
		elif type == C.ModulationType.QAM16:
			plt.plot(*np.meshgrid(np.arange(-3, 4, 2), np.arange(-3, 4, 2)), 'bo')
		plt.grid()
		plt.show()
		plt.plot(test)
		plt.show()

	if mod_type == C.ModulationType.PSK4:
		phase_recovered, test = demod_psk4_sync(phase_recovered)
		ploting_result(phase_recovered, ampli_recovered, test, mod_type)
		bit_stream = demod_psk4(phase_recovered)
		test_packet_extraction(bit_stream)
	elif mod_type == C.ModulationType.PSK8:
		phase_recovered, test = demod_psk8_sync(phase_recovered)
		ploting_result(phase_recovered, ampli_recovered, test, mod_type)
		bit_stream = demod_psk8(phase_recovered)
		test_packet_extraction(bit_stream)
	elif mod_type == C.ModulationType.QAM16:
		phase_recovered, test = demod_qam_sync(phase_recovered, ampli_recovered)
		ploting_result(phase_recovered, ampli_recovered, test, mod_type)
		bit_stream = demod_qam16(phase_recovered, ampli_recovered)
		test_packet_extraction(bit_stream)
	else:
		print(f'modulation type is unknown...')
		bit_stream = []

	if save_enable:
		np.savetxt('received_bitstream.txt', bit_stream, fmt='%1u', delimiter=',')

	return
