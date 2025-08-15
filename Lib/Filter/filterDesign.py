import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

def LowpassFilter(n, f_cutoff, fs):
	return signal.firwin(n, cutoff = f_cutoff/(fs/2), window = "hamming")

def HighpassFilter(n, f_cutoff, fs):
	a = - signal.firwin(n, cutoff = f_cutoff/(fs/2), window = "hanning")
	a[n//2] = a[n//2] + 1
	return a

def MidpassFilter(n, f_low, f_high, fs):
	a = signal.firwin(n, cutoff = f_low/(fs/2), window = 'blackmanharris')
	b = - signal.firwin(n, cutoff = f_high/(fs/2), window = 'blackmanharris')
	b[n//2] = b[n//2] + 1
	d = -(a+b)
	d[n//2] = d[n//2] + 1
	return d

def RemezFilter(n, f_cut1, f_cut2, fs):
	return signal.remez(n, np.array([0., f_cut1/(fs/2), f_cut2/(fs/2), 0.5]), [1, 1e-4])

def FirHalfBandFilter(n, f_cut, fs):
	b = signal.remez(n+1, np.array([0., f_cut/fs, 0.5-f_cut/fs, 0.5]), [1,0])
	b[abs(b) <= 1e-4] = 0.0		# force all even coef to be exact zero (they are close to zero)
	b[int(n/2)] = 0.5			# force center coef to be exact 0.5 (it is close to 0.5)
	return b

def MidpassCalc(b_lowpass, b_highpass):
	assert b_lowpass.size == b_highpass.size, "Error: input parameters should have same size"
	n = b_lowpass.size
	temp = 1.0*b_highpass
	temp[n//2] += 1
	d = -(b_lowpass+temp)
	d[n//2] += 1
	return d

def redesign_filter(filter_coef, f_band, f_unband, r, num_taps, fs):
	w, h = signal.freqz(filter_coef)
	freq = w*fs/(2*np.pi)
	i = int(np.argwhere(freq > f_band)[0])
	i = i if i%2 == 0 else i-1
	filter_new = signal.firls(num_taps, np.concatenate((freq[:i]*r, [f_unband, fs/2])), np.concatenate((np.abs(h[:i]), [0, 0])), fs=fs)
	
	return filter_new