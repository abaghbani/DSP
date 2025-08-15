import numpy as np
import scipy.signal as signal
import scipy.stats as stats
import matplotlib.pyplot as plt

import Common as cm

from .constant import *
C = Constant()

def max_detect(data, symb_width):
	interval = round(symb_width*4)
	mag_1 = data[:(data.size//interval)*interval].reshape((-1, interval))
	peak_1 = np.argmax(mag_1, axis=1)
	peak_1 += interval*np.arange(peak_1.size)
	mag_1 = data[interval//2:interval//2+(data.size//interval - 1)*interval].reshape((-1, interval))
	peak_2 = np.argmax(mag_1, axis=1)
	peak_2 += interval//2
	peak_2 += interval*np.arange(peak_2.size)

	peaks = np.unique(np.hstack((peak_1[:peak_2.size], peak_2)))
	return peaks

def peak_detect(data, symb_width, plot_enable=True):
	symb_width_by2 = round(symb_width*2)
	symb_width_by3 = round(symb_width*3)
	symb_width_by4 = round(symb_width*4)

	data_xcorr = np.array([magnitude_estimator(ts_xcorrolate([d3, d2, d1, d0], C.preamble_sts)) for d3, d2, d1, d0 in zip(np.roll(data, symb_width_by3), np.roll(data, symb_width_by2), np.roll(data, int(symb_width)), data)])	
	peaks = max_detect(data_xcorr, symb_width)

	sts_index = 0
	peak_count = 0
	
	for i, pk_diff in enumerate(np.diff(peaks)):
		if np.abs(pk_diff-symb_width_by4) < 6:
			peak_count += 1
			if peak_count >= 8:
				sts_index = peaks[i+1]
				break
		else:
			peak_count = 0
		
		if peak_count > 3:
			cm.prRed(f'STS peak detected = {peaks[i+1]} (peak_count={peak_count}, pk_diff={pk_diff})')
	
	if sts_index != 0:
		cm.prGreen(f'STS peak detected = {sts_index}')
		normalize_index = sts_index
	else:
		cm.prRed('Error: STS detection is failed (sts peaks are less than 8).')
		normalize_index = 500

	if plot_enable:
		plt.plot(data_xcorr, label='mag')
		plt.plot(peaks, data_xcorr[peaks], 'r.')
		plt.plot(sts_index, data_xcorr[sts_index], 'g.')
		# plt.plot(data_acorr_sum/data_ampli_sum[normalize_index], label='acorr_5')
		plt.legend()
		plt.grid()
		plt.show()

	return sts_index

def sts_detection(data, fs, plot_enable=True):

	fsymb = 2.0 # HDT symbol rate is 2MSymb/s
	symb_width = fs/fsymb

	plt.plot(np.angle(data), label='phase')
	plt.plot(np.abs(data), label='ampli')
	plt.legend()
	plt.grid()
	plt.show()

	sts_index = peak_detect(data, symb_width, True)
	
	if sts_index == 0:
		return np.empty((3, 0))
	else:

		data_sts_1 = data[sts_index-round((8*4-0.5)*symb_width) : sts_index-round((0*4-0.5)*symb_width)+1]
		data_sts_2 = data[sts_index-round((9*4-0.5)*symb_width) : sts_index-round((1*4-0.5)*symb_width)+1]
		acorr_data = data_sts_1*np.conjugate(data_sts_2)
		freq_offset = np.angle(np.sum(acorr_data[round(2*symb_width):-round(2*symb_width)]))/round(4*symb_width)

		phase = np.angle(data)
		phase_sync_1 = phase_correction(phase - freq_offset*np.arange(phase.size))
		phase_offset = np.mean(phase_sync_1[sts_index-round(4*4*symb_width): sts_index+1: round(4*symb_width)])
		phase_sync_2 = phase_correction(phase_sync_1 - phase_offset)
		
		data_sync = data*np.exp(-1j*(freq_offset*np.arange(data.size)+phase_offset))

		ampli = np.abs(data)
		ampli_sync = ampli/ampli[sts_index]
		
		print(f'Sts detection(2):  index={sts_index}, freq_off = {freq_offset*1000*fs/(2*np.pi):.3f} KHz, {phase_offset*180/np.pi:.3f} Deg')
		
		if plot_enable:
			plt.plot(np.angle(acorr_data), label='acorr')
			plt.legend()
			plt.grid()
			plt.show()

		index_range = (np.arange(sts_index, phase_sync_2.size, symb_width)).astype('int')
		return phase_sync_2[index_range], ampli_sync[index_range], data_sync[index_range]
		