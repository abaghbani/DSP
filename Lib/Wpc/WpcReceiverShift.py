import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

import Spectrum as sp
import Filter as fd
import Common as myLib

def WpcEstimateFrequency(data, fs, nperseg):
	# Estimate frequency by counting zero crossings
	
	data_resized = np.reshape(data[:(data.size//nperseg)*nperseg], (-1, nperseg))
	freq = np.empty(0, dtype='int')
	for sig in data_resized:
		# Find all indices right before a rising-edge zero crossing
		indices = np.nonzero((sig[1:] >= 0) & (sig[:-1] < 0))[0]
	
		# More accurate, using linear interpolation to find intersample
		#crossings = [i - sig[i] / (sig[i+1] - sig[i]) for i in indices]
		crossings = indices - (sig[indices]/(sig[indices+1]-sig[indices]))
		period = np.mean(np.diff(crossings))
		freq = np.append(freq, 0 if period == 0 or np.isnan(period) else int(fs/period))
	freq = np.repeat(freq, nperseg)
	freq = np.append(freq, np.repeat(freq[-1], data.size-freq.size))

	return freq

def WpcFrontendFiltering(data, fs, fop, downSamplingRate, mode='Averaging', type='float'):
	# frequency measuring (measuring over 5000 sample @5Msps, so resolution is 1ms)
	freq_slice = int(fs/1000)
	if fop == 0:
		freq = WpcEstimateFrequency(data, fs, freq_slice)
	else:
		freq = np.repeat(fop, data.size)

	# mixer
	f_mix = -20.0e3 * (freq//20e3)
	
	#f_mix = ((freq-16)//32)*32
	#f_mix = np.zeros(data.size, dtype='float')
	#for i in range(freq.size):
	#	step = (freq[i]-freq[i-1])/freq_slice
	#	for j in range(freq_slice):
	#		f_mix[j+i*freq_slice] = freq[i-1]+ j*step

	#plt.plot(np.repeat(freq, freq_slice))
	
	#freq = np.repeat(130e3, data.size)
	#f_mix = np.repeat(-130e3, data.size)
	#plt.plot(freq)
	#plt.plot(f_mix)
	#plt.show()

	#f_mix -= 10000

	n = np.arange(data.size)
	cos_mix = np.cos(n*2*np.pi*f_mix/fs)
	sin_mix = np.sin(n*2*np.pi*f_mix/fs)
	#cos_sin = np.array([myLib().cordic_xy(i*2*np.pi*f_mix[i]/fs) for i in n])
	#cos_mix = cos_sin[:,0]
	#sin_mix = cos_sin[:,1]
	data_mix_I = np.multiply(data, cos_mix)
	data_mix_Q = np.multiply(data, sin_mix)
	
	## filtering
	b_low = fd.LowpassFilter(101, 20.0e3, fs)
	data_flt_I = np.convolve(b_low, data_mix_I, mode='same')
	data_flt_Q = np.convolve(b_low, data_mix_Q, mode='same')


	## down sampling
	if mode == 'SubSampling':
		data_low_I = data_flt_I[::downSamplingRate]
		data_low_Q = data_flt_Q[::downSamplingRate]
	elif mode == 'Averaging':
		data_low_I = np.sum(np.reshape(data_flt_I[:(data_flt_I.size//downSamplingRate)*downSamplingRate], (-1, downSamplingRate)), axis=1)
		data_low_Q = np.sum(np.reshape(data_flt_Q[:(data_flt_Q.size//downSamplingRate)*downSamplingRate], (-1, downSamplingRate)), axis=1)
	elif mode == 'Gaussian':
		data_low_I = 0
		data_low_Q = 0
	freq = freq[::downSamplingRate]
	fs_low = fs/downSamplingRate

	#fftPlot(data_mix_I+1j*data_mix_Q, fs=fs)
	#fftPlot(data_flt_I+1j*data_flt_Q, fs=fs)
	#fftPlot(data_low_I+1j*data_low_Q, fs=fs_low)

	return data_low_I+1j*data_low_Q, freq, fs_low

def WpcDemodulation(data, fs):
	
	demod_data = np.array([myLib().cordic_rp(data.real[i], data.imag[i]) for i in range(data.size)])
	r = demod_data[:,0]
	beta = demod_data[:,1]

	r = np.convolve(fd.LowpassFilter(61, 3000.0, fs), r, mode='same')
	
	freq = np.diff(beta)
	freq[freq>(1.0*np.pi)] -= 2*np.pi 
	freq[freq<(-1.0*np.pi)] += 2*np.pi 
	freq = np.convolve(fd.LowpassFilter(21, 650.0, fs), freq, mode='same')
	
	#plt.plot(r)
	plt.plot(freq)
	plt.show()

	return r, freq


def WpcAskDemodulator(data, fs, type='float'):

	mag = data.real*data.real + data.imag*data.imag
	data_ask = np.convolve(fd.LowpassFilter(61, 4000.0, fs), mag, mode='same')
	
	## ask data rate = 2Ksps and fsk data rate = fc/512 cycles of main sin
	ask_sample_number = int(2*fs/2.0e3)
	data_ask_avg = np.convolve(data_ask, np.ones(ask_sample_number), 'same')/ask_sample_number
	data_ask_ac = data_ask-data_ask_avg
	rssi = (data_ask_avg/1000).astype('int16')

	#plt.plot(data_ask)
	#plt.plot(data_ask_ac)
	#plt.legend(['raw', 'ac'])
	#plt.grid()
	#plt.show()

	return data_ask_ac, rssi

def WpcFskDemodulator(data, fs, fop, type='float'):

	freq = np.diff(np.arctan2(data.imag, data.real))
	freq[freq>(1.0*np.pi)] -= 2*np.pi 
	freq[freq<(-1.0*np.pi)] += 2*np.pi 

	data_fsk = np.convolve(fd.LowpassFilter(101, 650.0, fs), freq, mode='same')

	## ask data rate = 2Ksps and fsk data rate = fc/512 cycles of main sin
	fsk_sample_number = int(np.mean(4*fs*512/fop))
	data_fsk_avg = np.convolve(data_fsk, np.ones(fsk_sample_number), 'same')/fsk_sample_number
	
	data_fsk_ac = data_fsk-data_fsk_avg

	#plt.plot(data_fsk)
	#plt.plot(data_fsk_ac)
	#plt.legend(['raw', 'ac'])
	#plt.grid()
	#plt.show()

	return data_fsk_ac
