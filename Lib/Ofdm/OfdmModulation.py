import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

from .Constant import *
C = Constant()

def training_sequence_mod():
	sts_freq = np.hstack([np.zeros(6), C.preamble_sts, np.zeros(5)])
	sts_time = np.fft.ifft(np.fft.fftshift(sts_freq))

	lts_freq = np.hstack([np.zeros(6), C.preamble_lts, np.zeros(5)])
	lts_time = np.fft.ifft(np.fft.fftshift(lts_freq))

	return np.hstack([np.tile(sts_time[:16], 10), lts_time[32:], np.tile(lts_time, 2)])

def bpsk_mod(inputData):
	return np.array([(C.BPSK_QPSK_table[(bits)])*C.ModulationFactor.BPSK for bits in inputData])

def qpsk_mod(inputData):
	chipStream = inputData[:(inputData.size//2)*2].reshape((-1,2))
	return np.array([(C.BPSK_QPSK_table[(bits[0])]+1j*C.BPSK_QPSK_table[(bits[1])])*C.ModulationFactor.QPSK for bits in chipStream])

def qam16_mod(inputData):
	chipStream = inputData[:(inputData.size//4)*4].reshape((-1,4))
	return np.array([(C.QAM16_2bits_table[tuple(bits[0:2])]+1j*C.QAM16_2bits_table[tuple(bits[2:4])])*C.ModulationFactor.QAM16 for bits in chipStream])

def qam64_mod(inputData):
	chipStream = inputData[:(inputData.size//6)*6].reshape((-1,6))
	return np.array([(C.QAM64_3bits_table[tuple(bits[0:3])]+1j*C.QAM64_3bits_table[tuple(bits[3:6])])*C.ModulationFactor.QAM64 for bits in chipStream])

def ofdm_mod(dataStream):
	dataStream_48 = dataStream[:(dataStream.size//48)*48].reshape([-1, 48])
	dataStream_64 = np.array([np.hstack([np.zeros(6), b[0:5], 0, b[5:18], 0, b[18:24], 0, b[24:30], 0, b[30:43], 0, b[43:48], np.zeros(5)]) for b in dataStream_48])
	index=0
	for dd in dataStream_64:
		dd[C.pilot_index] = C.pilot_symbol*C.pilot_polarity[index%C.pilot_polarity.size]
		index += 1

	dataStream_time = np.array([np.fft.ifft(np.fft.fftshift(b)) for b in dataStream_64])

	return dataStream_time

def cp_mod(data):
	dataout = np.array([np.hstack([d[-16:], d]) for d in data])

	return dataout.reshape(-1)

def Modulation(payload, modulation_type):
	crc = CrcCalculation(payload)
	payload_frame = np.concatenate((np.array([0xa0, 0xa0, 0xa0, 0xa0, 0xa7], dtype=np.uint8), np.array([payload.size], dtype=np.uint8), payload, crc, np.zeros(18, dtype=np.uint8)))
	payload_bit = np.unpackbits(payload_frame, bitorder='little')
	
	if modulation_type == C.ModulationType.BPSK:
		modulatedData = bpsk_mod(payload_bit)
	elif modulation_type == C.ModulationType.QPSK:
		modulatedData = qpsk_mod(payload_bit)
	elif modulation_type == C.ModulationType.QAM16:
		modulatedData = qam16_mod(payload_bit)
	elif modulation_type == C.ModulationType.QAM64:
		modulatedData = qam64_mod(payload_bit)
	else:
		modulatedData = 0

	baseband = cp_mod(ofdm_mod(modulatedData))
	fs = 20.0 # 64 samples per 3.2 us , after adding CP 80 samples per 4.0 us

	return np.hstack([np.zeros(100), training_sequence_mod(), baseband]), fs
