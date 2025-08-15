import numpy as np

from .constant import *
C = Constant()

def modulation_qpsk(bit_frame):
	packet_frame_2bits = bit_frame[:(bit_frame.size//2)*2].reshape((-1,2))
	symbol = np.hstack([C.BPSK_QPSK_table[bits[0]]+1j*C.BPSK_QPSK_table[bits[1]] for bits in packet_frame_2bits])
	return symbol

def modulation(payload):
	
	###################################
	## Wpan packet - generate bit_stream
	## preamble(4B) + SFD(1B) + Len(1B) + payload(0-125B) + CRC(2B)
	###################################
	crc = crc_calculation(payload)
	packet_frame = np.concatenate((C.preamble, C.sfd, np.array([payload.size], dtype=np.uint8), payload, crc))
	packet_frame_4bits = np.hstack([(data & 0x0f, (data>>4) & 0x0f) for data in packet_frame])
	bit_stream = np.hstack([(C.WpanPN[data]) for data in packet_frame_4bits])

	##################################
	## Modulation
	##################################
	baseband = modulation_qpsk(bit_stream)
	fs = 2.0

	# nSample_per_chip = 16
	# half_sin = np.sin(np.pi*np.arange(nSample_per_chip)/nSample_per_chip)

	# dataI = np.hstack([(even_bit*2-1)*half_sin for even_bit in chip_sequence[0::2]])
	# dataQ = np.hstack([(odd_bit*2-1)*half_sin for odd_bit in chip_sequence[1::2]])
	# dataI = np.concatenate((np.zeros(10*nSample_per_chip), dataI, np.zeros(10*nSample_per_chip)))
	# dataQ = np.concatenate((np.zeros(10*nSample_per_chip), dataQ, np.zeros(10*nSample_per_chip)))

	# ## imply Offset-QPSK
	# baseband = dataI+1j*np.roll(dataQ, nSample_per_chip//2)
	# fs = nSample_per_chip * 2.0 / 2		# bit rate = 2.0Mbps, every two bits are modulated in one chip
	
	return baseband, fs
