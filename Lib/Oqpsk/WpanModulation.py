import numpy as np

from .Constant import *
C = Constant()

def Modulation(payload, Fs):
	
	###################################
	## Wpan packet
	## preamble(4B) + SFD(1B) + Len(1B) + payload(0-125B) + CRC(2B)
	###################################
	crc = CrcCalculation(payload)
	packet_frame = np.concatenate((C.preamble, C.sfd, np.array([payload.size], dtype=np.uint8), payload, crc))
	packet_frame_4bits = np.hstack([(data & 0x0f, (data>>4) & 0x0f) for data in packet_frame])
	chip_sequence = np.hstack([(C.WpanPN[data]) for data in packet_frame_4bits])

	##################################
	## Modulation
	##################################
	nSample_per_chip = int(Fs*C.ChipDuration)
	half_sin = np.sin(np.pi*np.arange(nSample_per_chip)/nSample_per_chip)

	dataI = np.hstack([(even_bit*2-1)*half_sin for even_bit in chip_sequence[0::2]])
	dataQ = np.hstack([(odd_bit*2-1)*half_sin for odd_bit in chip_sequence[1::2]])
	dataI = np.concatenate((np.zeros(10*nSample_per_chip), dataI, np.zeros(10*nSample_per_chip)))
	dataQ = np.concatenate((np.zeros(10*nSample_per_chip), dataQ, np.zeros(10*nSample_per_chip)))

	## imply Offset-QPSK
	baseband = dataI+1j*np.roll(dataQ, nSample_per_chip//2)
	
	return baseband
