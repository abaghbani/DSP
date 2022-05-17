import numpy as np

class Constant:
	
	def QAMConstellation(M):
		D = np.sqrt(M).astype(int)

		n = np.arange(M)
		a = np.array([x^(x>>1) for x in n]) #convert linear addresses to Gray 
		a = np.reshape(a,(D,D))
		a[1::2] = np.flip(a[1::2], axis=1)
		a = np.reshape(a,(M))
				
		(x,y) = np.divmod(n,D)
		(x,y) = (x*2+1-D , y*2+1-D)
		constellation = x+1j*y
		constellation = constellation[np.argsort(a)]

		return constellation


	## QAM modulator
	QAM16_table = {
		(0,0,0,0) : -3-3j,
		(0,0,0,1) : -3-1j,
		(0,0,1,0) : -3+3j,
		(0,0,1,1) : -3+1j,
		(0,1,0,0) : -1-3j,
		(0,1,0,1) : -1-1j,
		(0,1,1,0) : -1+3j,
		(0,1,1,1) : -1+1j,
		(1,0,0,0) :  3-3j,
		(1,0,0,1) :  3-1j,
		(1,0,1,0) :  3+3j,
		(1,0,1,1) :  3+1j,
		(1,1,0,0) :  1-3j,
		(1,1,0,1) :  1-1j,
		(1,1,1,0) :  1+3j,
		(1,1,1,1) :  1+1j
	}

	QAM16_symbole = QAMConstellation(16)
	QAM64_symbole = QAMConstellation(64)
	QAM16_bit = np.argsort((QAM16_symbole.real+3)/2 + (QAM16_symbole.imag+3)*2)
	preamble = np.array([0xA0, 0xA0, 0xA0, 0xA0], dtype=np.uint8)
	sfd = np.array([0xA7], dtype=np.uint8)
	class ModulationType:
		QAM16 = 1
		QAM64 = 2

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
