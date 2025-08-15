import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

def rrc_2(n, b, sps):
	def rrc_calc(f):
		if f<(1-b)/sps:
			return 1.0
		elif f <= (1+b)/sps and f >= (1-b)/sps:
			return np.sqrt(0.5*(1-np.sin(np.pi*(f*sps-1)/(2*b)))) 
		else:
			return 0.0

	freq = np.linspace(0, 1, int(np.ceil(50*sps/2.0)*2))
	rrc_filter = np.array([rrc_calc(f) for f in freq])
	fir_filter = signal.firls(n, freq, rrc_filter)

	return fir_filter

def rrc(n, b, f_symb, fs):
	T = 1/f_symb

	def rrc_calc(f):
		if np.abs(f)<(1-b)/(2*T):
			return 1.0
		elif np.abs(f) <= (1+b)/(2*T) and np.abs(f) >= (1-b)/(2*T):
			return np.sqrt(0.5*(1-np.sin(np.pi*(2*np.abs(f)*T-1)/(2*b)))) 
		else:
			return 0.0

	freq = np.linspace(0, fs/2, int(np.ceil(100*fs*T/2.0)*2))
	rrc_filter = np.array([rrc_calc(f) for f in freq])
	fir_filter = signal.firls(n, freq, rrc_filter, fs=fs)

	return fir_filter

def raised_root_cosine(N, upsample, alpha):
	"""
	Root raised cosine (RRC) filter (FIR) impulse response.

	N: filter order and 'should be even number'
	upsample: number of samples per symbol
	alpha: roll-off factor
	"""

	t = (np.arange(N) - N / 2) / upsample

	# result vector
	h_rrc = np.zeros(t.size, dtype=np.float64)

	# index for special cases
	sample_i = np.zeros(t.size, dtype=np.bool_)

	# deal with special cases
	subi = t == 0
	sample_i = np.bitwise_or(sample_i, subi)
	h_rrc[subi] = 1.0 - alpha + (4 * alpha / np.pi)

	subi = np.abs(t) == 1 / (4 * alpha)
	sample_i = np.bitwise_or(sample_i, subi)
	h_rrc[subi] = (alpha / np.sqrt(2)) \
				* (((1 + 2 / np.pi) * (np.sin(np.pi / (4 * alpha))))
				+ ((1 - 2 / np.pi) * (np.cos(np.pi / (4 * alpha)))))

	# base case
	sample_i = np.bitwise_not(sample_i)
	ti = t[sample_i]
	h_rrc[sample_i] = np.sin(np.pi * ti * (1 - alpha)) \
					+ 4 * alpha * ti * np.cos(np.pi * ti * (1 + alpha))
	h_rrc[sample_i] /= (np.pi * ti * (1 - (4 * alpha * ti) ** 2))
	
	return h_rrc				

def rrc_filter(beta, n_tap, oversampling):
	if n_tap % 2 == 0:
		time_index = np.linspace(-n_tap/2, n_tap/2, n_tap, endpoint=False)/oversampling
	else:
		time_index = np.linspace(-(n_tap-1)/2, (n_tap-1)/2, n_tap)/oversampling

	# Root raised cosine formula
	h_rrc = np.zeros_like(time_index)
	for i, t in enumerate(time_index):
		if t == 0.0:
			h_rrc[i] = 1.0 - beta + (4 * beta / np.pi)
		elif abs(t) == 1 / (4 * beta):
			h_rrc[i] = (beta / np.sqrt(2)) * (
				((1 + 2 / np.pi) * (np.sin(np.pi / (4 * beta)))) +
				((1 - 2 / np.pi) * (np.cos(np.pi / (4 * beta))))
			)
		else:
			numerator = np.sin(np.pi * t * (1 - beta)) + 4 * beta * t * np.cos(np.pi * t * (1 + beta))
			denominator = np.pi * t * (1 - (4 * beta * t)**2)
			h_rrc[i] = numerator / denominator

	return h_rrc				
