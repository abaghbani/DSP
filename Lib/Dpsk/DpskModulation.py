import numpy as np

from .Constant import *
C = Constant()

def DpskSyncGenerator():

	sync_mapper = np.array(C.DpskSyncHr_mapper_IQ)
	sync = np.array(sync_mapper[C.DpskSyncHr2_IQ])

	phase = np.angle(sync)
	phase_diff = np.diff(phase)
	phase_diff[phase_diff>np.pi] += -2*np.pi
	phase_diff[phase_diff<-np.pi] += 2*np.pi
	phase_diff *= 4/np.pi

	print(list(phase_diff.astype(np.int)))
	return phase_diff

def modulationEdr2(bit_frame):
	packet_frame_2bits = bit_frame[:(bit_frame.size//2)*2].reshape((-1,2))
	symbol = np.hstack([C.EDR2_mapper_table[np.packbits(bits, bitorder='little')[0]] for bits in packet_frame_2bits])
	return symbol

def modulationEdr3(bit_frame):
	packet_frame_3bits = bit_frame[:(bit_frame.size//3)*3].reshape((-1,3))
	symbol = np.hstack([C.EDR3_mapper_table[np.packbits(bits, bitorder='little')[0]] for bits in packet_frame_3bits])
	return symbol

def DpskModulation(payload, type):
	crc = CrcCalculation(payload)
	payload_frame = np.concatenate((np.array([0xa0, 0xa0, 0xa0, 0xa0, 0xa7], dtype=np.uint8), np.array([payload.size], dtype=np.uint8), payload, crc))
	payload_bit = np.unpackbits(payload_frame, bitorder='little')

	if type == C.ModulationType.EDR2:
		modulatedData = modulationEdr2(payload_bit)
		symbolStream = np.concatenate((np.zeros(10), C.DpskSync, modulatedData, np.zeros(10)), axis=None)
	elif type == C.ModulationType.EDR3:
		modulatedData = modulationEdr3(payload_bit)
		symbolStream = np.concatenate((np.zeros(10), C.DpskSync, modulatedData, np.zeros(10)), axis=None)
	elif type == C.ModulationType.HDR4:
		modulatedData = modulationEdr2(payload_bit)
		symbolStream = np.concatenate((np.zeros(10), C.DpskSyncHdr4, modulatedData, np.zeros(10)), axis=None)
	elif type == C.ModulationType.HDR8:
		modulatedData = modulationEdr2(payload_bit)
		symbolStream = np.concatenate((np.zeros(10), C.DpskSyncHdr8, modulatedData, np.zeros(10)), axis=None)
	elif type == C.ModulationType.HR2:
		modulatedData = modulationEdr3(payload_bit)
		symbolStream = np.concatenate((np.zeros(10), C.DpskSyncHr2, modulatedData, np.zeros(10)), axis=None)
	elif type == C.ModulationType.HR4:
		modulatedData = modulationEdr3(payload_bit)
		symbolStream = np.concatenate((np.zeros(10), C.DpskSyncHr4, modulatedData, np.zeros(10)), axis=None)
	elif type == C.ModulationType.HR8:
		modulatedData = modulationEdr3(payload_bit)
		symbolStream = np.concatenate((np.zeros(10), C.DpskSyncHr8, modulatedData, np.zeros(10)), axis=None)
	
	phase = np.cumsum(symbolStream*np.pi/4)
	baseband = np.exp(1j*phase)
	
	return baseband
