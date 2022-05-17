import numpy as np
import matplotlib.pyplot as plt

import ClockRecovery as cr
import Spectrum as sp

from .Constant import Constant as C
from .Constant import *


def ChipConvert(chip_bit):
	## detecting preamble (first chip_sequence should be PN[0])
	index_first_bit = int([i for i in range(chip_bit.size) if np.all(chip_bit[i:i+32] == C.WpanPN[0])][0])
	return np.reshape(chip_bit[index_first_bit:index_first_bit+int((chip_bit.size-index_first_bit)/32)*32],(-1,32))

def DataExtraction(chip_sequence):
	data = np.array([i for chip in chip_sequence for i in range(16) if np.all(chip == C.WpanPN[i])], dtype=np.uint8)
	data_byte = np.array([(d_first+16*d_second) for d_first, d_second in zip(data[0::2], data[1::2])], dtype=np.uint8)
	start_index = np.argwhere(data_byte==C.sfd)
	if start_index.size:
		start_index = int(start_index[0])
		data_len_extracted = int(data_byte[start_index+1])
		data_extracted = data_byte[(start_index+2):(start_index+2+data_len_extracted)], data_len_extracted
		crc_result = CrcCalculation(data_byte[(start_index+2):(start_index+2+data_len_extracted+2)])
	else:
		print('SFD Error: delimiter is not found in extracted data')
		data_len_extracted = 0
		data_extracted = 0
		crc_result = -1
		
	return data_extracted, data_len_extracted, np.all(crc_result == 0)

def Demodulation(data, Fs):
	
	#sp.fftPlot(data.real, data.imag, n=2, fs=Fs)
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

	## convert chip_bit to chip_sequence
	chip_sequence = ChipConvert(chip_bit)
	
	## extract data
	data_extract, data_len, crc = DataExtraction(chip_sequence)

	print(data_extract, data_len, crc)

	return data_extract
