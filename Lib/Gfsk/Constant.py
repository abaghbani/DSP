import numpy as np

class Constant:
	
	bit_rate = 1.0
	GfskBT = 0.5
	GfskBrModulationIndex = [0.28, 0.35]
	GfskBleModulationIndex = [0.45, 0.55]
	GfskPreamble = np.array([0x55, 0x55], dtype=np.uint8)
	GfskAccessAddress_Adv = np.array([0xD6, 0xBE, 0x89, 0x8E], dtype=np.uint8)
	GfskAccessAddress_Test = ~GfskAccessAddress_Adv

	class GfskModulationType:
		Gfsk1M = 1
		Gfsk2M = 2
	
	## Gfsk demodulator
	ModeRateThreshold = 11 # fix me! dependancy of sample rate (this is for 15Msps)
	FrequencyAvrageMaximum = 10
	
def CrcCalculation(payload):
	# crc polinomial : x^16 + x^12 + x^5 + 1
	crc_polinomial = np.uint16(0x1021)
	crc_init = np.uint16(0x0000)

	crc = crc_init
	for data in payload:
		crc ^= data << 8
		for _ in range(8):
			if (crc & 0x8000):
				crc = (crc << 1) ^ crc_polinomial
			else:
				crc = (crc << 1)
	
	return np.array([(crc >> 8) & 0xff, crc & 0xff], dtype=np.uint8)
