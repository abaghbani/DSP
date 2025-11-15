import numpy as np
import scipy.signal as signal
import scipy.stats as stats
import matplotlib.pyplot as plt

import Common as cm

from .constant import *
C = Constant()

def peak_detect(data, symb_width, plot_enable=True):
	symb_width_by2 = round(symb_width*2)
	symb_width_by3 = round(symb_width*3)
	symb_width_by4 = round(symb_width*4)

	data_xcorr = np.array([magnitude_estimator(ts_xcorrolate([d3, d2, d1, d0], C.preamble_sts)) for d3, d2, d1, d0 in zip(np.roll(data, symb_width_by3), np.roll(data, symb_width_by2), np.roll(data, int(symb_width)), data)])	
	data_peak = np.zeros(9*round(4*symb_width), dtype='int')
	data_peak[round(4*symb_width)-1::round(4*symb_width)] = 1
	xcorr_peak = np.zeros(data_xcorr.size)
	for i in range(data_peak.size, data.size):
		xcorr_peak[i-1] = np.sum(data_xcorr[i-data_peak.size:i]*data_peak)
	
	peak_index, _ = signal.find_peaks(xcorr_peak, distance=round(4*symb_width)*8)

	cm.prGreen(f'STS peak detected (new) = {peak_index[0]}')
	if plot_enable:
		plt.plot(xcorr_peak, label='xcorr_peak')
		plt.plot(peak_index, xcorr_peak[peak_index], 'r.')
		plt.legend()
		plt.grid()
		plt.show()

	return peak_index[0]

def sts_detection(data, fs, plot_enable=True):

	fsymb = 2.0 # HDT symbol rate is 2MSymb/s
	symb_width = fs/fsymb

	sts_index = peak_detect(data, symb_width, plot_enable)
	
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
		