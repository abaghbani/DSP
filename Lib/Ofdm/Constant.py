import numpy as np
import math

class Constant:
	
	BPSK_QPSK_table = {
		(0) : -1,
		(1) : +1
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

	preamble_sts = np.sqrt(13/6) * np.array([0,0,1+1j,0,0,0,-1-1j,0,0,0,1+1j,0,0,0,-1-1j,0,0,0,-1-1j,0,0,0,1+1j,0,0,0,0,0,0,0,-1-1j,0,0,0,-1-1j,0,0,0,1+1j,0,0,0,1+1j,0,0,0,1+1j,0,0,0,1+1j,0,0])
	preamble_sts_time_calc = np.fft.ifft(np.fft.fftshift(np.hstack([np.zeros(6), preamble_sts, np.zeros(5)])))[:16]
	preamble_lts = np.array([1,1,-1,-1,1,1,-1,1,-1,1,1,1,1,1,1,-1,-1,1,1,-1,1,-1,1,1,1,1,0,1,-1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,1,1,1])
	preamble_lts_time_calc = np.fft.ifft(np.fft.fftshift(np.hstack([np.zeros(6), preamble_lts, np.zeros(5)])))
	
	preamble_sts_time = np.array([
		0.04599875+0.04599875j, -0.13244372+0.00233959j,
       -0.01347272-0.07852479j,  0.14275529-0.01265117j,
        0.09199751+0.j        ,  0.14275529-0.01265117j,
       -0.01347272-0.07852479j, -0.13244372+0.00233959j,
        0.04599875+0.04599875j,  0.00233959-0.13244372j,
       -0.07852479-0.01347272j, -0.01265117+0.14275529j,
        0.        +0.09199751j, -0.01265117+0.14275529j,
       -0.07852479-0.01347272j,  0.00233959-0.13244372j
	])

	preamble_lts_time = np.array([ 		
		0.15625   +0.j        , -0.00512125-0.12032513j,
        0.0397497 -0.11115794j,  0.09683188+0.08279791j,
        0.02111177+0.02788592j,  0.05982384-0.08770676j,
       -0.11513121-0.0551805j , -0.03831597-0.10617091j,
        0.09754126-0.02588835j,  0.05333773+0.00407633j,
        0.00098898-0.11500464j, -0.13680488-0.04737981j,
        0.02447585-0.0585318j ,  0.05866877-0.014939j  ,
       -0.02248321+0.16065733j,  0.11923909-0.00409559j,
        0.0625    -0.0625j    ,  0.03691794+0.09834415j,
       -0.05720635+0.03929859j, -0.13126261+0.06522723j,
        0.08221832+0.09235655j,  0.06955685+0.01412196j,
       -0.0603101 +0.08128612j, -0.05645513-0.02180392j,
       -0.03504126-0.15088835j, -0.12188701-0.01656622j,
       -0.12732436-0.02050138j,  0.0750737 -0.07404042j,
       -0.00280594+0.05377427j, -0.09188756+0.11512871j,
        0.09171655+0.10587166j,  0.01228459+0.09759955j,
       -0.15625   +0.j        ,  0.01228459-0.09759955j,
        0.09171655-0.10587166j, -0.09188756-0.11512871j,
       -0.00280594-0.05377427j,  0.0750737 +0.07404042j,
       -0.12732436+0.02050138j, -0.12188701+0.01656622j,
       -0.03504126+0.15088835j, -0.05645513+0.02180392j,
       -0.0603101 -0.08128612j,  0.06955685-0.01412196j,
        0.08221832-0.09235655j, -0.13126261-0.06522723j,
       -0.05720635-0.03929859j,  0.03691794-0.09834415j,
        0.0625    +0.0625j    ,  0.11923909+0.00409559j,
       -0.02248321-0.16065733j,  0.05866877+0.014939j  ,
        0.02447585+0.0585318j , -0.13680488+0.04737981j,
        0.00098898+0.11500464j,  0.05333773-0.00407633j,
        0.09754126+0.02588835j, -0.03831597+0.10617091j,
       -0.11513121+0.0551805j ,  0.05982384+0.08770676j,
        0.02111177-0.02788592j,  0.09683188-0.08279791j,
        0.0397497 +0.11115794j, -0.00512125+0.12032513j
	])

	pilot_symbol = np.array([1,1,1,-1])
	pilot_index = np.array([-21, -7, 7, 21])+32
	pilot_polarity = np.array([
		1,1,1,1,-1,-1,-1,1,-1,-1,-1,-1,1,1,-1,1,-1,-1,1,1,-1,1,1,-1,1,1,1,1,1,1,-1,1,
		1,1,-1,1,1,-1,-1,1,1,1,-1,1,-1,-1,-1,1,-1,1,-1,-1,1,-1,-1,1,1,1,1,1,-1,-1,1,1,
		-1,-1,1,-1,1,-1,1,1,-1,-1,-1,1, 1,-1,-1,-1,-1,1,-1,-1,1,-1,1,1,1,1,-1,1,-1,1,-1,1,
		-1,-1,-1,-1,-1,1,-1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,-1,-1
		])

	class Signal_field:
		rate_index = range(0,4)
		reserved_index = 4
		length_index = range(5, 17)
		parity_index = 17
		tail_index = range(18, 24)
	
	signal_rate_value = {
		(1,1,0,1) : 6,
		(1,1,1,1) : 9,
		(0,1,0,1) : 12,
		(0,1,1,1) : 18,
		(1,0,0,1) : 24,
		(1,0,1,1) : 36,
		(0,0,0,1) : 48,
		(0,0,1,1) : 54
	}

	class ModulationFactor:
		BPSK = 1
		QPSK = 1/np.sqrt(2)
		QAM16 = 1/np.sqrt(10)
		QAM64 = 1/np.sqrt(42)

	class ModulationType:
		BPSK = 1
		QPSK = 2
		QAM16 = 3
		QAM64 = 4

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

def CrcCalculation_HT_sig(bits):
	# crc polinomial : x^8 + x^2 + x^1 + 1
	crc_polinomial = np.uint8(0x07)
	crc_init = np.uint8(0xff)

	crc = crc_init
	for bit in bits:
		crc ^= (bit << 7)
		if (crc & 0x80):
			crc = (crc << 1) ^ crc_polinomial
		else:
			crc = (crc << 1)

	return (crc & 0xFF) ^ 0xFF

def DataScrambler(payload):
	# LFSR polinomial : x^7 + x^4 + 1
	lfsr_polinomial = np.uint8(0x91)
	#lfsr_init = np.uint8(payload_bit[0:7])
	return 0

def convolutional_encoder(data, seed):
	# Convolutional K = 7
	# A = 1 + x^2 + x^3 + x^5 + x^6 
	# B = 1 + x^1 + x^2 + x^3 + x^6 
	A_polinomial = np.array([1,0,1,1,0,1,1], dtype=np.uint8)
	B_polinomial = np.array([1,1,1,1,0,0,1], dtype=np.uint8)
	data_new = np.zeros(7, dtype=np.uint8)
	data_new[1:] = seed
	ret_A = []
	ret_B = []
	ret_val = []
	for dd in data:
		data_new[0] = dd
		data_A = np.zeros(1, dtype=np.uint8)
		data_B = np.zeros(1, dtype=np.uint8)
		for i in range(7):
			data_A = data_A ^ (data_new[i]*A_polinomial[i])
			data_B = data_B ^ (data_new[i]*B_polinomial[i])
		
		data_new[1:] = data_new[:-1]
		ret_A = np.append(ret_A, data_A)
		ret_B = np.append(ret_B, data_B)
		ret_val.extend(list(data_A))
		ret_val.extend(list(data_B))
	return ret_val, ret_A, ret_B, data_new[1:]

def convolutional_decoder(data_A, data_B, seed):
	
	return 0

def interleaving_transmitter(NCBPS = 48, NBPSC = 1):
	#NBPSC : number of bit per sub-carrier
	#NCBPS : number of coded-bit per symbol
	
	i_val = [(NCBPS/16)*(k % 16) + math.floor(k/16) for k in range(NCBPS)]
	s = max(NBPSC/2,1)
	j = [int(s*math.floor(i/s) + (i + NCBPS + math.floor(16*i/NCBPS))%s) for i in i_val]
	
	return j

def interleaving_receiver(NCBPS = 48, NBPSC = 1):
	#NBPSC : number of bit per sub-carrier
	#NCBPS : number of coded-bit per symbol
	
	s = max(NBPSC/2,1)
	i_val = [s*math.floor(j/s) + (j + math.floor(16*j/NCBPS))%s for j in range(NCBPS)]
	k = [16*i - (NCBPS-1)*math.floor(16*i/NCBPS) for i in i_val]
	
	return k
