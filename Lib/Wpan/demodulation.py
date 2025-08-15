import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

import ClockRecovery as cr
import Spectrum as sp

from .constant import *
C = Constant()

def data_extractor(chip_bit):
	## detecting preamble (first chip_sequence should be PN[0])
	index_first_bit = int([i for i in range(chip_bit.size) if np.all(chip_bit[i:i+32] == C.WpanPN[0])][0])
	chip_sequence = np.reshape(chip_bit[index_first_bit:index_first_bit+int((chip_bit.size-index_first_bit)/32)*32],(-1,32))
	data_4bits = np.array([i for chip in chip_sequence for i in range(16) if np.all(chip == C.WpanPN[i])], dtype=np.uint8)

	return data_4bits

def packet_extractor(data):
	index_first_4bits = int([i for i in range(data.size-4) if np.all(data[i:i+4] == np.array([0, 0, C.sfd[0]&0x0f, (C.sfd[0]>>4)&0x0f]) )][0])
	data_byte = np.array([(d_first+16*d_second) for d_first, d_second in zip(data[index_first_4bits::2], data[index_first_4bits+1::2])], dtype=np.uint8)
	start_index = np.argwhere(data_byte==C.sfd)
	if start_index.size:
		start_index = int(start_index[0])
		packet_len = int(data_byte[start_index+1])
		packet_extracted = data_byte[(start_index+2):(start_index+2+packet_len)]
		crc_result = crc_calculation(data_byte[(start_index+2):(start_index+2+packet_len+2)])
	else:
		print('SFD Error: delimiter is not found in extracted data')
		packet_len = 0
		packet_extracted = 0
		crc_result = -1
		
	return packet_extracted, packet_len, np.all(crc_result == 0)

## non coherent demodulation
def demodulation_nco(data, Fs):
	
	#sp.fftPlot(data.real, data.imag, n=2, fs=Fs)
	#plt.plot(data.real, data.imag, '.')
	#plt.show()
	
	basic_sample_rate = 1 # baseband sample rate is 2MS/s or 1Mchip/s
	period = Fs/basic_sample_rate
	chip_duration = int(len(C.WpanPN[0]) * period / 2)
	pn_complex = np.zeros((16, chip_duration), dtype=np.complex)
	for i in range(16):
		pn_complex[i] = np.repeat((np.array(C.WpanPN[i][0::2])+1j*np.array(C.WpanPN[i][1::2]))*2-1, period)
	
	## preamble detection
	xcorr = signal.correlate(data, pn_complex[0], mode='valid')
	xcorr_abs = np.abs(xcorr)

	sync_point, _ = signal.find_peaks(xcorr_abs, prominence=6.8e7, width=int(.5*period), distance=chip_duration-period)
	print(f'{sync_point[:8]}')
	#plt.plot(xcorr.real)
	#plt.plot(xcorr.imag)
	#plt.plot(xcorr_abs)
	#plt.plot(sync_point, xcorr_abs[sync_point], 'x')
	#plt.legend(['real', 'imag', 'ampli'])
	#plt.show()
	
	data_sync = data[sync_point[0]:]
	data_sync = data_sync[:(data_sync.size//chip_duration)*chip_duration].reshape((-1,chip_duration))

	## demodulation
	xcorr_abs = np.zeros(16, dtype=np.complex)
	data_extracted = np.zeros(data_sync.shape[0], dtype=np.uint8)
	index = 0
	for dd in data_sync:
		xcorr_abs = np.array([np.abs(signal.correlate(dd, pn, mode='valid')) for pn in pn_complex])
		data_extracted[index] = np.argmax(xcorr_abs)
		index += 1
		#plt.plot(xcorr_abs)
		#plt.show()
	#print([hex(dd) for dd in data_extracted])

	## packet extraction
	packet_extract, packet_len, crc = packet_extractor(data_extracted)
	if packet_len != 0:
		print([hex(dd) for dd in packet_extract], packet_len, crc)

	return packet_extract


def demodulation(data, Fs):
	
	# sp.fftPlot(data.real, data.imag, n=2, fs=Fs)
	plt.plot(data.real, data.imag, '.')
	plt.show()

	## carrier recovery: phase offset detection
	nSample_per_chip = int(Fs*C.ChipDuration)
	preamble_data = C.WpanPN0Phase[:7]
	preamble_data = np.repeat(preamble_data, nSample_per_chip)
	phase_offset = 0

	for index in range(data.size//nSample_per_chip - 8):
		corr_value = np.zeros(20)
		for i in range(corr_value.size):
			data_fix = data[index*nSample_per_chip:(index+7)*nSample_per_chip]*np.exp(1j*(i-10)*np.pi/20)
			dataI = data_fix.real
			dataQ = np.roll(data_fix.imag, -nSample_per_chip//2)
			phase = np.arctan2(dataQ, dataI)*4/np.pi
			corr_value[i] = np.sum(np.multiply(preamble_data, phase))
		## check if correlation is greater than 90% of max value index and offset is acceptable
		if np.max(corr_value) > (0.825 * np.sum(preamble_data*preamble_data)):
			print('phase offset is detected: ', corr_value, np.sum(preamble_data*preamble_data), np.argmax(corr_value), index)
			phase_offset = (np.argmax(corr_value)-10)*np.pi/20
			break
	
	## original phase
	#dataI = data.real
	#dataQ = np.roll(data.imag, -nSample_per_chip//2)
	#phase_raw = np.arctan2(dataQ, dataI)*4/np.pi

	## phase offset cancelation
	data_offset = data * np.exp(1j*phase_offset)
	plt.plot(data_offset.real, data_offset.imag, '.')
	plt.show()


	dataI = data_offset.real
	dataQ = np.roll(data_offset.imag, -nSample_per_chip//2)
	phase = np.arctan2(dataQ, dataI)*4/np.pi

	## clock/data recovery
	phase_extract, index = cr.EarlyLate(phase, nSample_per_chip, 1.0, 5, True)
	phase_extract = np.floor(phase_extract+0.5)
	
	## convert phase (-3,-1,1,3) to chip bit
	chip_bit = np.hstack([(C.phase_to_bit[int((ph+3)/2)]) for ph in phase_extract])

	## convert chip_bit to 4 bits data
	data_4bits = data_extractor(chip_bit)
	
	## extract packet
	data_extract, data_len, crc = packet_extractor(data_4bits)

	print(data_extract, data_len, crc)

	return data_extract
