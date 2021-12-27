import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

import Spectrum.freqPlot as fp
import Filter.filterDesign as fd
import Common.ModemLib as myLib

def WpcFrequencyCalc(data, fs, nSamples):
	data_resized = np.reshape(data[:(data.size//nSamples)*nSamples], (-1, nSamples))
	freq = [(np.count_nonzero((sig[1:] >= 0) != (sig[:-1] >= 0))/2) for sig in data_resized]
	freq = np.repeat(freq, nSamples)
	freq = np.append(freq, np.repeat(freq[-1], data.size-freq.size))
	plt.plot(freq)
	plt.show()

	return freq

def WpcDecimateRate(data, fs):
	# fsk_bit_rate = Fop / 256
	# downsampling_rate for reach to 64 samples for symbol : 64 = (fs/10*k) / (Fop / 256) => k' = 1024*k = 1024 * 256/(64*10) (fs * T) = 409.6 ~= 819/2
	indices = np.nonzero((data[1:] >= 0)  != (data[:-1] >= 0))[0]
	crossings = indices - (data[indices]/(data[indices+1]-data[indices]))
	decimateRate = np.array([(crossings[i]-crossings[i-64]) for i in range(64, crossings.size, 64)])
	#decimateRate = np.hstack([[val]*val for val in decimateRate])
	#plt.plot(decimateRate)
	#plt.show()

	return decimateRate

def WpcFrontendFiltering(data, fs, downSamplingRate, mode='Averaging', type='float'):

	## filtering
	filter_gain = 2**17
	len = 81
	fc = 250.0e3
	b_low = fd.LowpassFilter(len, fc, fs)
	#b_low *= 1.0/b_low[int((len-1)/2)]
	b_low = (b_low*filter_gain).astype(type)
	data_flt = np.convolve(b_low, data, mode='same')
	data_flt = (data_flt/filter_gain).astype(type)

	#fp.fftPlot(data, fs=fs, nperseg = 2**16)
	#fp.fftPlot(data_flt, fs=fs, nperseg = 2**16)
				
	#plt.plot(data)
	#plt.plot(data_flt)
	#plt.show()

	return data_flt

def WpcAskDemodulator(data_input, fs_input, type='float'):
	
	## decimation
	downSamplingRate = 10
	data = np.sum(np.reshape(data_input[:(data_input.size//downSamplingRate)*downSamplingRate], (-1, downSamplingRate)), axis=1)
	fs = fs_input/downSamplingRate
	#fp.fftPlot(data, fs=fs, nperseg = 2**16)

	## demodulation
	#data = np.array([-dd if dd<0 else dd for dd in data])
	data = np.abs(data)

	## ASK extraction :
	filter_gain = 2**17
	len = 101
	fc = 1.5e3
	b_low = fd.LowpassFilter(len, fc, fs)
	#b_low *= 1.0/b_low[int((len-1)/2)]
	b_low = (b_low*filter_gain).astype(type)
	data = np.convolve(b_low, data, mode='same')
	data = (data/filter_gain).astype(type)
	
	## decimation
	downSamplingRate = 5
	data = np.sum(np.reshape(data[:(data.size//downSamplingRate)*downSamplingRate], (-1, downSamplingRate)), axis=1)
	fs = fs/downSamplingRate
	
	len = 41
	fc = 1.0e3
	b_low = fd.LowpassFilter(len, fc, fs)
	#b_low *= 1.0/b_low[int((len-1)/2)]
	b_low = (b_low*filter_gain).astype(type)
	ask_data = np.convolve(b_low, data, mode='same')
	ask_data = (ask_data/filter_gain).astype(type)
	
	ask_sample_number = 128  ## (2*fs/2.0e3) = 100, should be 100 but because of divider is changed to 128
	ask_data_avg = rssi = np.convolve(ask_data, np.ones(ask_sample_number), 'same')/ask_sample_number
	ask_data_ac = ask_data-ask_data_avg
	
	## dc removal by a high pas filter (IIR)
	#ask_data_ac = signal.lfilter([1, -1], [1, -0.993], ask_data[::16])

	#plt.plot(ask_data)
	#plt.plot(ask_data_avg)
	#plt.plot(ask_data_ac)
	#plt.grid()
	#plt.show()

	return ask_data_ac, rssi

def WpcFskDemodulator(data, fs, type='float'):
	
	## demodulation
	indices = np.nonzero((data[1:] >= 0)  != (data[:-1] >= 0))[0]

	#new_level_detected = 0
	#limit_hi = 500 ##0.5*data.max()
	#limit_lo = -500 ##0.5*data.min()
	#indices = np.empty(0, dtype='int')
	#for i in range(data.size):
	#	if data[i] > limit_hi and new_level_detected == 0:
	#		new_level_detected = 1
	#		indices = np.append(indices, i)
	#	elif data[i] < limit_lo and new_level_detected == 1:
	#		new_level_detected = 0
	#		indices = np.append(indices, i)

	crossings = indices - (data[indices]/(data[indices+1]-data[indices]))
	fsk_data = np.array([(crossings[i]-crossings[i-64]) for i in range(64, crossings.size, 64)])
	fsk_data[fsk_data>10000] = 10000

	## dc removal
	fsk_sample_number = 4*8
	fsk_data_avg = np.convolve(fsk_data, np.ones(fsk_sample_number), 'same')/fsk_sample_number
	fsk_data_ac = fsk_data-fsk_data_avg
	
	#plt.plot(fsk_data)
	#plt.plot(fsk_data_avg)
	#plt.plot(fsk_data_ac)
	#plt.grid()
	#plt.show()
	
	fsk_index = indices[64::64]//(10 * 5) # to be sync with ASK data index, ask data has two step decimation, 10 and 5
	period_value = np.hstack([[val]*int(val) for val in fsk_data])
	period_value = np.append(period_value, np.repeat(period_value[-1], data.size-period_value.size))
	
	return fsk_data_ac, fsk_index, period_value