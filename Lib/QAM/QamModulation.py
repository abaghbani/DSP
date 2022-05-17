import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

from .Constant import *
C = Constant()

##################################
## Modulation
##################################
def modulationQAM16(payload):
	###################################
	## packet format:
	## preamble(4B) + SFD(1B) + Len(1B) + payload(0-125B) + CRC(2B)
	###################################

	crc = CrcCalculation(payload)
	packet_frame = np.concatenate((np.zeros(4, dtype=np.uint8), C.preamble, C.sfd, np.array([payload.size], dtype=np.uint8), payload, crc, np.zeros(4, dtype=np.uint8)))
	packet_frame_4bits = np.hstack([(data & 0x0f, (data>>4) & 0x0f) for data in packet_frame])
	symbole = np.hstack([(C.QAM16_symbole[data]) for data in packet_frame_4bits])
	return symbole

def modulationQAM64(payload):
	crc = CrcCalculation(payload)
	packet_frame = np.concatenate((C.preamble, C.sfd, np.array([payload.size], dtype=np.uint8), payload, crc, np.zeros(4, dtype=np.uint8)))
	bit_frame = np.unpackbits(packet_frame, bitorder='little')
	packet_frame_6bits = np.packbits(bit_frame[:(bit_frame.size//6)*6].reshape((-1,6)), axis=1, bitorder='little')
	symbole = np.hstack([(C.QAM64_symbole[data]) for data in packet_frame_6bits])
	return symbole

def modulation(payload, type):
	if type == C.ModulationType.QAM16:
		modulatedData = modulationQAM16(payload)
	elif type == C.ModulationType.QAM64:
		modulatedData = modulationQAM64(payload)
	else:
		modulatedData = 0

	return modulatedData
