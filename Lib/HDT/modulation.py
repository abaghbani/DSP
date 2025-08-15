import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

from .constant import *
C = Constant()

##################################
## Modulation
##################################
def carrier(phase, num):
	return np.repeat(np.exp(1j*phase), num)

def training_sequence(nSts:int, nLts:int, lts_seq:int):
	symbol = np.hstack([np.tile(C.preamble_sts, nSts), C.preamble_lts[lts_seq][-4:], np.tile(C.preamble_lts[lts_seq], nLts)])
	return symbol

def phy_int_training_sequence(nPits:int):
	symbol = np.hstack([C.preamble_pits[:nPits]])
	return symbol

def modulationBPSK(bit_frame):
	symbol = np.hstack([C.BPSK_QPSK_table[bits] for bits in bit_frame])
	return symbol

def modulationPSK4(bit_frame):
	packet_frame_2bits = bit_frame[:(bit_frame.size//2)*2].reshape((-1,2))
	symbol = np.hstack([C.BPSK_QPSK_table[bits[1]]+1j*C.BPSK_QPSK_table[bits[0]] for bits in packet_frame_2bits])
	symbol = np.array([sym*np.exp(1j*np.pi/4) if i%2==1 else sym for i, sym in enumerate(symbol)])
	return symbol

def modulation8PSK(bit_frame):
	packet_frame_3bits = bit_frame[:(bit_frame.size//3)*3].reshape((-1,3))
	symbol = np.hstack([np.exp(1j*np.pi/4*C.PSK8_3bits_table[tuple(bits)]) for bits in packet_frame_3bits])
	return symbol

def modulationQAM16(bit_frame):
	packet_frame_4bits = bit_frame[:(bit_frame.size//4)*4].reshape((-1,4))
	symbol = np.hstack([C.QAM16_2bits_table[tuple(bits[0:2])]+1j*C.QAM16_2bits_table[tuple(bits[2:4])] for bits in packet_frame_4bits])
	return symbol

def modulationQAM64(bit_frame):
	packet_frame_6bits = bit_frame[:(bit_frame.size//6)*6].reshape((-1,6))
	symbol = np.hstack([C.QAM64_3bits_table[tuple(bits[0:3])]+1j*C.QAM64_3bits_table[tuple(bits[3:6])] for bits in packet_frame_6bits])
	return symbol

def modulation(payload, block_number, type, lts_seq=0):
	crc = crc_calc(payload)
	payload_frame = np.concatenate((np.array([0xa0, 0xa0, 0xa0, 0xa0, 0xa7], dtype=np.uint8), np.array([payload.size], dtype=np.uint8), payload, crc))
	payload_bit = np.unpackbits(payload_frame, bitorder='little')
	fs = 2.0		## baseband sample rate is 2MS/s

	if type == C.ModulationType.PSK4:
		modulatedData = modulationPSK4(payload_bit)
	elif type == C.ModulationType.PSK8:
		modulatedData = modulation8PSK(payload_bit)
	elif type == C.ModulationType.QAM16:
		modulatedData = modulationQAM16(payload_bit)
	else:
		print(f'this case should be complete...')
		print(f'Demodulation is failed.')
		modulatedData = 0

	ret_val = np.hstack([np.zeros(16), carrier(np.pi/4, 16), training_sequence(9, 2, lts_seq), modulatedData])
	if block_number > 1:
		for i in range(block_number-1):
			ret_val = np.hstack([ret_val, phy_int_training_sequence(6), modulatedData])

	ret_val = np.hstack([ret_val, carrier(np.pi/4, 8), np.zeros(16)])
	return ret_val, fs
