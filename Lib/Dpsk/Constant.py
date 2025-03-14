import numpy as np

class Constant:
	
	## from calculation by: "DpskSyncGenerator"
	DpskSync = [3,-3,3,-3,3,-3,-3,3,3,3]
	DpskSyncEdr = [3,-3,3,-3,3,-3,-3,3,3,3]
	DpskSyncHdr4 = [-1,1,1,3,-1,-3,-1,3,3,1,1,1,-3,-1,1,-1,1,-3,1,-3,-3,3,-3,-3,-3,1,-3,1,-3,-3,1,1,-3,1,1,3,-1,-3,-3,-1,1,-3,1,-1,3,-1,1,1,1,1,1,1,3,3,1,1,3,3,1]
	DpskSyncHdr8 = [-1,3,-1,1,3,-3,-3,-1,-1,1,-3,-3,1,-3,1,1,-1,-1,3,-3,3,-3,3,-3,-3,3,-3,3,-1,-1,1,1,3,-1,-1,3,1,-1,3,3,-1,-3,-1,-1,-1,-3,-3,3,3,-1,-1,-3,3,-1,-3,
				 -1,1,-3,1,3,3,1,3,-3,-1,-1,1,3,3,1,3,3,-3,-3,-3,1,3,-1,3,1,1,1,-1,1,3,-3,1,-1,3,1,-3,-3,1,3,3,1,-3,1,-1,3,-1,-1,-1,1,1,-3,-3,3,3,3,-1,-1,-1,-3,1,1,3,-1,1]
	
	DpskSyncHr2 = [-1,1,3,3,1,-3,-3,-1,-1,-1,-1,-3,-1,1,1,-3,1,3,3,3,-1,3,-1,3,3,-3,-3,-1,-3]
	DpskSyncHr4 = [-1,1,3,3,1,-3,-3,-1,-1,-1,-1,-3,-1,1,1,-3,1,3,3,3,-1,3,-1,3,3,-3,-3,-1,-3,
				   -3,1,-3,3,-3,3,-3,1,-1,-1,-1,3,3,1,3,1,-1,-3,3,1,3,-1,3,-3,1,-3,-3,1,3,-1,
				   -3,1,1,-1,1,-1,-3,-1,-1,-3,-3,3,-1,-3,-3,-1,1,3,-1,-1,3,-1,-1,-3,3,-1,-1,
				   -1,1,1,-1,3,-3,3,-1,3,1,-1,3,-1,1,1,3,-3,3,1,-1,1,-3,-1,1,-1,1,1,1,-1,-3,1,3,-3
	]

	DpskSyncHr8 = [
		-1,-1,3,-1,1,-1,-3,1,-3,3,-3,1,1,1,-3,1,-3,1,-1,-3,-1,-1,3,3,1,3,-3,-1,1,-3,3,3,3,1,-1,-3,3,1,3,-1,-3,
		-3,3,-3,-3,1,-3,-3,-1,1,-1,1,-1,-3,-1,1,-1,-1,3,1,-3,3,-1,3,-3,-1,3,-1,-1,-3,-3,-1,-3,1,1,3,1,1,-1,-3,3,-1,
		-1,3,-1,-1,3,1,3,-1,3,1,-1,-1,-1,1,1,3,1,-1,3,1,1,-3,-1,-1,1,1,-3,-3,3,1,3,1,1,1,1,-1,-3,-3,3,3,1,-3,-1,
		-1,-1,-3,1,-1,1,1,-1,-1,-1,-1,-3,-1,3,-3,3,-3,-1,3,1,3,1,-1,-3,-3,-3,-1,-3,-1,-3,-1,3,3,-1,1,-3,1,1,-1,-1,1,
		3,1,-3,1,-3,-1,3,1,-3,-3,3,-1,-1,-3,3,3,1,3,3,3,-3,-1,1,3,-1,-1,1,3,-1,1,-3,-1,-3,3,-3,3,3,-3,3,-1,3,3,3,
		-1,-3,-3,-3,1,1,3,-1,-3,3,-1,1,-1,3,-3,1,1,-1,1,-3,-1,3,-1,1,1,1,-3,-1,1,-3
	]

	#############################
	## from specification
	#############################
	DpskSyncHdr4_IQ = [
		1.0000 + 0.0000j,	 0.7071 - 0.7071j,	 1.0000 + 0.0000j,	 0.7071 + 0.7071j,	
		-1.0000 + 0.0000j,	 -0.7071 + 0.7071j,	 1.0000 + 0.0000j,	 0.7071 - 0.7071j,	
		0.0000 + 1.0000j,	 -0.7071 - 0.7071j,	 -0.0000 - 1.0000j,	 0.7071 - 0.7071j,	
		1.0000 + 0.0000j,	 -0.7071 - 0.7071j,	 -1.0000 + 0.0000j,	 -0.7071 - 0.7071j,	
		-1.0000 + 0.0000j,	 -0.7071 - 0.7071j,	 0.0000 + 1.0000j,	 -0.7071 + 0.7071j,	
		1.0000 + 0.0000j,	 -0.7071 - 0.7071j,	 1.0000 + 0.0000j,	 -0.7071 - 0.7071j,	
		0.0000 + 1.0000j,	 0.7071 - 0.7071j,	 1.0000 + 0.0000j,	 -0.7071 - 0.7071j,	
		-0.0000 - 1.0000j,	 -0.7071 + 0.7071j,	 1.0000 + 0.0000j,	 0.7071 + 0.7071j,	
		0.0000 + 1.0000j,	 0.7071 - 0.7071j,	 1.0000 + 0.0000j,	 0.7071 + 0.7071j,	
		-1.0000 + 0.0000j,	 -0.7071 + 0.7071j,	 1.0000 + 0.0000j,	 -0.7071 - 0.7071j,	
		-1.0000 + 0.0000j,	 -0.7071 - 0.7071j,	 0.0000 + 1.0000j,	 -0.7071 + 0.7071j,	
		0.0000 + 1.0000j,	 -0.7071 - 0.7071j,	 -1.0000 + 0.0000j,	 -0.7071 - 0.7071j,	
		-0.0000 - 1.0000j,	 0.7071 - 0.7071j,	 1.0000 + 0.0000j,	 0.7071 + 0.7071j,	
		0.0000 + 1.0000j,	 -0.7071 - 0.7071j,	 1.0000 + 0.0000j,	 0.7071 + 0.7071j,	
		0.0000 + 1.0000j,	 -0.7071 - 0.7071j,	 1.0000 + 0.0000j,	 0.7071 + 0.7071j
	]

	DpskSyncHdr8_IQ = [
		1.0000 + 0.0000j,	 0.7071 - 0.7071j,	 0.0000 + 1.0000j,	 0.7071 + 0.7071j,	
		0.0000 + 1.0000j,	 -0.7071 - 0.7071j,	 0.0000 + 1.0000j,	 0.7071 - 0.7071j,	
		-0.0000 - 1.0000j,	 -0.7071 - 0.7071j,	 -0.0000 - 1.0000j,	 -0.7071 + 0.7071j,	
		1.0000 + 0.0000j,	 0.7071 + 0.7071j,	 -0.0000 - 1.0000j,	 0.7071 - 0.7071j,	
		1.0000 + 0.0000j,	 0.7071 - 0.7071j,	 -0.0000 - 1.0000j,	 0.7071 + 0.7071j,	
		-0.0000 - 1.0000j,	 0.7071 + 0.7071j,	 -0.0000 - 1.0000j,	 0.7071 + 0.7071j,	
		-0.0000 - 1.0000j,	 -0.7071 + 0.7071j,	 -0.0000 - 1.0000j,	 -0.7071 + 0.7071j,	
		-0.0000 - 1.0000j,	 -0.7071 - 0.7071j,	 -1.0000 + 0.0000j,	 -0.7071 - 0.7071j,	
		-0.0000 - 1.0000j,	 0.7071 + 0.7071j,	 1.0000 + 0.0000j,	 0.7071 - 0.7071j,	
		0.0000 + 1.0000j,	 -0.7071 + 0.7071j,	 0.0000 + 1.0000j,	 -0.7071 - 0.7071j,	
		1.0000 + 0.0000j,	 0.7071 - 0.7071j,	 -1.0000 + 0.0000j,	 -0.7071 + 0.7071j,	
		0.0000 + 1.0000j,	 0.7071 + 0.7071j,	 -0.0000 - 1.0000j,	 -0.7071 + 0.7071j,	
		-0.0000 - 1.0000j,	 0.7071 + 0.7071j,	 1.0000 + 0.0000j,	 0.7071 - 0.7071j,	
		-1.0000 + 0.0000j,	 0.7071 - 0.7071j,	 -0.0000 - 1.0000j,	 -0.7071 + 0.7071j,	
		0.0000 + 1.0000j,	 -0.7071 + 0.7071j,	 1.0000 + 0.0000j,	 0.7071 + 0.7071j,	
		-1.0000 + 0.0000j,	 0.7071 - 0.7071j,	 1.0000 + 0.0000j,	 -0.7071 + 0.7071j,	
		1.0000 + 0.0000j,	 0.7071 - 0.7071j,	 -0.0000 - 1.0000j,	 0.7071 - 0.7071j,	
		0.0000 + 1.0000j,	 -0.7071 - 0.7071j,	 -0.0000 - 1.0000j,	 0.7071 + 0.7071j,	
		-1.0000 + 0.0000j,	 0.7071 + 0.7071j,	 -0.0000 - 1.0000j,	 -0.7071 + 0.7071j,	
		-1.0000 + 0.0000j,	 0.7071 - 0.7071j,	 -0.0000 - 1.0000j,	 0.7071 + 0.7071j,	
		0.0000 + 1.0000j,	 -0.7071 + 0.7071j,	 -1.0000 + 0.0000j,	 -0.7071 + 0.7071j,	
		-1.0000 + 0.0000j,	 0.7071 - 0.7071j,	 -1.0000 + 0.0000j,	 -0.7071 - 0.7071j,	
		-1.0000 + 0.0000j,	 0.7071 - 0.7071j,	 1.0000 + 0.0000j,	 -0.7071 - 0.7071j,	
		0.0000 + 1.0000j,	 -0.7071 + 0.7071j,	 -0.0000 - 1.0000j,	 0.7071 + 0.7071j,	
		0.0000 + 1.0000j,	 0.7071 - 0.7071j,	 1.0000 + 0.0000j,	 0.7071 - 0.7071j,	
		0.0000 + 1.0000j,	 0.7071 + 0.7071j,	 1.0000 + 0.0000j,	 0.7071 - 0.7071j,	
		1.0000 + 0.0000j,	 0.7071 + 0.7071j,	 -0.0000 - 1.0000j,	 -0.7071 + 0.7071j,	
		-0.0000 - 1.0000j,	 0.7071 + 0.7071j,	 -1.0000 + 0.0000j,	 -0.7071 + 0.7071j,	
		0.0000 + 1.0000j,	 0.7071 + 0.7071j,	 -0.0000 - 1.0000j,	 0.7071 - 0.7071j,	
		1.0000 + 0.0000j,	 -0.7071 + 0.7071j,	 0.0000 + 1.0000j,	 -0.7071 + 0.7071j
	]

	DpskSyncHr_mapper_IQ = [1.0, 0.7071+0.7071j, 1.0j, -0.7071+0.7071j, -1.0, -0.7071-0.7071j, -1.0j, 0.7071-0.7071j ]
	DpskSyncHr2_IQ = [0,7,0,3,6,7,4,1,0,7,6,5,2,1,2,3,0,1,4,7,2,1,4,3,6,1,6,3,2,7]
	DpskSyncHr4_IQ = [
		0,7,0,3,6,7,4,1,0,7,6,5,2,1,2,3,0,1,4,7,2,1,4,3,6,1,6,3,2,7,
		4,5,2,5,2,5,2,3,2,1,0,3,6,7,2,3,2,7,2,3,6,5,0,5,6,3,0,1,4,3,
		0,1,2,1,2,1,6,5,4,1,6,1,0,5,2,1,2,5,4,3,6,5,4,1,4,3,2,1,2,3,
		2,5,2,5,4,7,0,7,2,1,2,3,6,3,6,7,6,7,4,3,4,3,4,5,6,5,2,3,6,3
		]
	DpskSyncHr8_IQ = [
		0,7,6,1,0,1,0,5,6,3,6,3,4,5,6,3,4,1,2,1,6,5,4,7,2,3,6,3,2,3,
		0,3,6,1,2,1,6,1,2,5,4,1,6,1,6,3,4,1,6,5,6,5,6,5,2,1,2,1,0,3,
		4,1,4,3,6,3,2,5,4,3,0,5,4,1,2,3,6,7,0,7,4,7,6,5,0,7,6,1,2,5,
		4,7,0,7,6,5,6,7,2,3,2,5,6,7,4,3,2,3,4,1,6,1,2,5,6,7,0,1,0,5,
		2,5,0,1,6,5,4,3,0,1,0,1,2,1,0,7,6,3,2,5,2,5,2,1,4,5,0,1,0,5,
		2,7,6,3,2,7,6,1,4,3,4,1,2,3,2,1,2,5,6,3,4,1,0,3,4,1,6,1,0,7,
		4,7,2,3,6,1,4,1,0,1,4,3,2,3,6,5,6,3,2,7,2,7,2,5,2,5,4,7,2,5,
		4,1,6,3,4,5,0,7,4,7,6,7,6,1,6,7,0,7,0,5,4,7,6,7,0,1,6,5,6,3
		]
	#############################

	class ModulationType:
		EDR2 = 1
		EDR3 = 2
		HDR4 = 3
		HDR8 = 4
		HR2  = 5
		HR4  = 6
		HR8  = 7

	## Edr2 symbol = [-3, -1, 1, 3] * pi/4
	EDR2_mapper_table = [-3, -1, 1, 3]
	## Edr3 symbol = [-4, -3, -2, -1, 0, 1, 2, 3] * pi/4
	EDR3_mapper_table = [-4, -3, -2, -1, 0, 1, 2, 3]

	## Dpsk demodulator
	TableRxPhase4DQPSK = [100, 1, 1, 100, 100, 3, 3, 100, 100, -3, -3, 100, 100, -1, -1, 100]
	TableRxPhase8DQPSK = [0, 1, 1, 2, 2, 3, 3, 4, 4, -3, -3, -2, -2, -1, -1, 0]

def SymbolRate(modulation_type):
	if modulation_type == Constant.ModulationType.EDR2:
		return 1.0
	elif modulation_type == Constant.ModulationType.EDR3:
		return 1.0
	elif modulation_type == Constant.ModulationType.HDR4:
		return 2.0
	elif modulation_type == Constant.ModulationType.HDR8:
		return 4.0
	elif modulation_type == Constant.ModulationType.HR2:
		return 1.0
	elif modulation_type == Constant.ModulationType.HR4:
		return 2.0
	elif modulation_type == Constant.ModulationType.HR8:
		return 4.0

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
