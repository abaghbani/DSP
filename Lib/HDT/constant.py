import numpy as np

class Constant:

	BPSK_QPSK_table = {
		(0) : +1,
		(1) : -1
	}

	QAM16_2bits_table = {
		(0,0) : -3,
		(0,1) : -1,
		(1,0) : +3,
		(1,1) : +1
	}

	QAM64_3bits_table = {
		(0,0,0) : -7,
		(0,0,1) : -5,
		(0,1,1) : -3,
		(0,1,0) : -1,
		(1,1,0) : +1,
		(1,1,1) : +3,
		(1,0,1) : +5,
		(1,0,0) : +7
	}
	
	PSK8_3bits_table = {
		(0,0,0) : 0,
		(0,0,1) : +1,
		(0,1,1) : +2,
		(0,1,0) : +3,
		(1,1,0) : -4,
		(1,1,1) : -3,
		(1,0,1) : -2,
		(1,0,0) : -1
	}
	
	preamble_sts = np.array([-1, -1j, 1j, 1])
	preamble_pits = np.array([-1, 1, 1, 1, -1, 1])
	lts_sequence = np.array([2, 7, 12, 12, 4, 13, 8, 14, 15, 13, 12, 10, 5, 15, 14, 0])

	def __init__(self):
		ZC_lts = lambda u, L: [np.exp((1j*np.pi*(-(u+1)*n*(n+1) + 2*self.lts_sequence[u]))/L) for n in range(L)]
		ZC_lts_seq = lambda L: [ZC_lts(u, L) for u in range(self.lts_sequence.size)]
		self.preamble_lts = np.array(ZC_lts_seq(17))
		self.preamble_lts_phase = np.angle(self.preamble_lts)
		lts = np.array([[(-(u+1)*n*(n+1)+2*self.lts_sequence[u])%34 for n in range(17)] for u in range(16)]) # 2pi/(pi/17) = 34
		lts[lts > 17] -= 34
		self.lts_int = lts//2

	class ModulationType:
		Unknown = 0
		PSK4 = 1
		PSK8 = 2
		QAM16 = 3

class Const_unused:

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
	
	QAM16_symbole = QAMConstellation(16)
	QAM64_symbole = QAMConstellation(64)
	QAM16_bit = np.argsort((QAM16_symbole.real+3)/2 + (QAM16_symbole.imag+3)*2)

def crc_calc(payload):
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

def nco(data, freq_current, freq_off, phase_off=0):
	# NCO: mixer with compensation of frequency and phase
	freq = freq_current + freq_off
	mixer = np.exp((freq + phase_off)*1j)
	data_out = data * mixer
	
	return data_out, freq

def sts_xcorrolate_1(data, symb_width, training_sequence):
	data_xcorr = np.zeros(data.size, dtype='complex')
	for i in range(4*symb_width, data.size):
		data_xcorr[i] = np.sum(np.arry([np.sum(data[:symb_width]), np.sum(data[symb_width:2*symb_width]), np.sum(data[2*symb_width:3*symb_width]), np.sum(data[3*symb_width:])])*np.conjugate(training_sequence))
	
	return data_xcorr

def ts_xcorrolate_2(data, symb_width, training_sequence):
	return np.sum(np.arry([np.sum(data[:symb_width]), np.sum(data[symb_width:2*symb_width]), np.sum(data[2*symb_width:3*symb_width]), np.sum(data[3*symb_width:])])*np.conjugate(training_sequence))
	
def ts_xcorrolate(data, training_sequence):
	return np.sum(data*np.conjugate(training_sequence))
	
def linear_regression(y):
	
	n = y.size
	x = np.arange(y.size)
	a = (n*np.sum(x*y) - np.sum(x)*np.sum(y))/np.sqrt((n*np.sum(x*x) - np.sum(x)*np.sum(x))*(n*np.sum(y*y) - np.sum(y)*np.sum(y)))
	b = np.mean(y-a*x)

	return a, b

def magnitude_estimator(data):
	## Mag ~=Alpha * max(|I|, |Q|) + Beta * min(|I|, |Q|) @ alpha = 1.0 and Beta = 0.5
	mag = np.maximum(np.abs(data.real), np.abs(data.imag))+0.5*np.minimum(np.abs(data.real), np.abs(data.imag))

	return mag

def phase_correction(phase):
	return (phase+np.pi)%(2*np.pi)-np.pi

def phase_correction_2(angle):
	return np.arctan2(np.sin(angle), np.cos(angle))

def raw_symbol_convert(data):
	phase = np.array([(dd>>0)&0x1f for dd in data], dtype='float')
	ampli = np.array([(dd>>5)&0x07 for dd in data], dtype='float')
	ampli = (np.sqrt(10)/8)*(ampli+4.5)
	phase[phase>=16] -= 32
	phase = (phase+0.5)*(np.pi/16)
	return ampli, phase

