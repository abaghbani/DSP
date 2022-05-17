import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

import Filter

def WpcHdlEstimateFrequency(data, fs, nperseg):
	# fsk_bit_rate = Fop / 256
	# downsampling_rate for reach to 64 samples for symbol : 64 = (fs/10*k) / (Fop / 256) => period = k*1024/fs = 1024 * 256/(64*10) = 409.6 ~= 819/2
	cycle = 0
	last_index = 0
	period = np.empty(0, dtype='int')
	for i in range(data.size):
		if (data[i] >= 0) != (data[i-1] >= 0):
			cycle += 1
			if cycle == 819:
				period = np.append(period, np.repeat(i-last_index, i-last_index))
				last_index = i
				cycle = 0
	period = (period-512)//1024
	period = np.append(period, np.repeat(period[-1], data.size-period.size))
	#plt.plot(period)

	data_resized = np.reshape(data[:(data.size//nperseg)*nperseg], (-1, nperseg))
	freq = np.empty(0, dtype='int')
	for sig in data_resized:
		freq = np.append(freq, int(np.count_nonzero((sig[1:] >= 0) != (sig[:-1] >= 0))/2) )
	freq = np.repeat(freq, nperseg)
	freq = np.append(freq, np.repeat(freq[-1], data.size-freq.size))
	#plt.plot(freq)
	#plt.show()

	return freq, period

def WpcHdlFrontendFiltering(data, fs, fop, downSamplingRate, mode='Averaging'):
	## frequency measuring (measuring over 5000 sample @5Msps, so resolution is 1ms)
	freq_gain = 2**14
	if fop == 0:
		freq, period = WpcHdlEstimateFrequency(data, fs, freq_gain)
	else:
		freq = np.repeat((fop*freq_gain)//fs, data.size)
		period = np.repeat(int((fs*0.4)/fop), data.size)

	## mixer
	f_mix = ((freq-16)//32)*32
	plt.plot(freq)
	plt.plot(f_mix)
	plt.show()
	
	print(f'freq measured = {freq[1]}, {freq[10]}, {freq[-2]}, {freq[-1]}')
	print(f'freq_mixer    = {f_mix[1]}, {f_mix[10]}, {f_mix[-2]}, {f_mix[-1]}')

	n = np.arange(data.size)
	mixer_gain = 2**17 ## 1+17
	cos_mix = (np.cos(n*2*np.pi*f_mix/freq_gain)*mixer_gain).astype('int32')
	sin_mix = (np.sin(n*2*np.pi*f_mix/freq_gain)*mixer_gain).astype('int32')
	#cos_sin = np.array([myLib().cordic_xy(i*2*np.pi*f_mix[i]) for i in n])
	#cos_mix = (cos_sin[:,0]*mixer_gain).astype('int32')
	#sin_mix = (cos_sin[:,1]*mixer_gain).astype('int32')
	data_mix_I = np.multiply(data, cos_mix)
	data_mix_Q = np.multiply(data, sin_mix)
	
	print(f'mixer data min = {data_mix_I.min()}, {data_mix_Q.min()} -- data max = {data_mix_I.max()} , {data_mix_Q.max()}')

	## filtering
	filter_gain = 2**17
	len = 101
	fc = 20.0e3
	b_low = Filter.LowpassFilter(len, fc, fs)
	gain = 1.0/b_low[int((len-1)/2)]
	b_low *= 1.0/b_low[int((len-1)/2)]
	b_low = (b_low*filter_gain).astype('int64')
	data_flt_I = np.convolve(b_low, data_mix_I, mode='same')
	data_flt_Q = np.convolve(b_low, data_mix_Q, mode='same')

	data_flt_I = (data_flt_I/(2**25)).astype('int32')
	data_flt_Q = (data_flt_Q/(2**25)).astype('int32')

	print(f'filter data type = {type(data_flt_I[0])} : data min = {data_flt_I.min()}, {data_flt_Q.min()} -- data max = {data_flt_I.max()} , {data_flt_Q.max()}')

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
	period = period[::downSamplingRate]
	fs_low = fs/downSamplingRate
	
	print(f'data type = {type(data_low_I[0])} : data min = {data_low_I.min()}, {data_low_Q.min()} -- data max = {data_low_I.max()} , {data_low_Q.max()}')
	
	#fftPlot(data, fs=fs)
	#fftPlot(data_mix_I+1j*data_mix_Q, fs=fs)
	#fftPlot(data_flt_I+1j*data_flt_Q, fs=fs)
	#fftPlot(data_low_I+1j*data_low_Q, fs=fs_low)

	return data_low_I, data_low_Q, freq, period, fs_low

def hdl_cordic_rp(x, y, phase_width, N=7):
	arctan_table_c = np.array([
		2.5000000000E-01, 1.4758361765E-01, 7.7979130377E-02, 3.9583424161E-02,
		1.9868524306E-02, 9.9439478236E-03, 4.9731872790E-03, 2.4867453937E-03,
		1.2433916687E-03, 6.2169820592E-04, 3.1084939941E-04, 1.5542473676E-04,
		7.7712373013E-05, 3.8856187085E-05, 1.9428093615E-05, 9.7140468166E-06,
		4.8570234094E-06, 2.4285117048E-06, 1.2142558524E-06, 6.0712792622E-07,
		3.0356396311E-07, 1.5178198156E-07, 7.5890990778E-08, 3.7945495389E-08,
		1.8972747694E-08, 9.4863738472E-09, 4.7431869236E-09, 2.3715934618E-09,
		1.1857967309E-09, 5.9289836545E-10, 2.9644918273E-10, 1.4822459136E-10
	])
	arctan_gain_c = (2**(phase_width-1))-1
	arctan_table_c = (arctan_table_c*arctan_gain_c).astype('int32')
	beta = 0.0

	#if x == 0:
	#	return (y, int(arctan_gain_c/2)) if y >= 0 else (-y, -1*int((arctan_gain_c+1)/2))
	#if y == 0:
	#	return (x, 0.0) if x >= 0 else (-x, arctan_gain_c)

	if x < 0:
		beta = -arctan_gain_c if y<0 else arctan_gain_c
		(x, y) = (-x, -y)
		
	for i in range(N):
		d = 1 if y < 0 else -1
		(x,y) = (x - (d*(2**(-i))*y), (d*(2**(-i))*x) + y)
		beta = beta - (d*arctan_table_c[i])
	return x, beta


def WpcHdlDemodulation(dataI, dataQ, fs, period_mr):
	
	phase_width = 16
	demod_data = np.array([hdl_cordic_rp(x, y, phase_width, 16) for x,y in zip(dataI, dataQ)])
	mag = demod_data[:,0].astype('int32')
	beta = demod_data[:,1]
	print(f'mag data type = {type(mag[0])} : data min = {mag.min()} -- data max = {mag.max()}')
	print(f'beta data type = {type(beta[0])} : data min = {beta.min()} -- data max = {beta.max()}')

	# ASK demod :
	len = 41
	fc = 3.0e3
	b_low = Filter.LowpassFilter(len, fc, fs)
	gain = 1.0/b_low[int((len-1)/2)]
	b_low *= 1.0/b_low[int((len-1)/2)]
	filter_gain = 2**17
	b_low = (b_low*filter_gain).astype('int64')
	mag_flt = np.convolve(b_low, mag, mode='same')

	ask_sample_number = 512
	mag_flt_avg = np.convolve(mag_flt, np.ones(ask_sample_number), 'same')//ask_sample_number
	mag_flt_ac = mag_flt-mag_flt_avg
	
	plt.plot(mag_flt)
	plt.plot(mag_flt_ac)
	plt.show()

	## FSK demod :
	phase_gain = 2**(phase_width-1)
	freq = np.diff(beta)
	freq[freq>int(phase_gain)] -= int(2*phase_gain)
	freq[freq<int(-phase_gain)] += int(2*phase_gain)
	freq = np.convolve(freq, np.ones(4), 'same')
	print(f'freq data type = {type(freq[0])} : data min = {freq.min()} -- data max = {freq.max()}')

	len = 21
	fc = 450.0
	b_low = Filter.LowpassFilter(len, fc, fs)
	gain = 1.0/b_low[int((len-1)/2)]
	b_low *= 1.0/b_low[int((len-1)/2)]
	filter_gain = 2**17
	b_low = (b_low*filter_gain).astype('int64')
	freq_flt_high = np.convolve(b_low, freq, mode='same')//(2**17)

	period_mr = np.where(period_mr > 4, period_mr, 100)
	index = np.zeros(1, 'int32')
	while index[-1] < freq_flt_high.size:
		index = np.append(index, index[-1]+period_mr[index[-1]])
	freq_flt = freq_flt_high[index[:-1]]
	
	fsk_sample_number = 256
	freq_flt_avg = np.convolve(freq_flt, np.ones(fsk_sample_number), 'same')//fsk_sample_number
	freq_flt_ac = freq_flt - freq_flt_avg

	plt.plot(freq_flt)
	plt.plot(freq_flt_ac)
	plt.show()

	return mag_flt_ac, freq_flt_ac
