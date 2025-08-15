import numpy as np
import scipy.signal as signal
import scipy.stats as stats
import matplotlib.pyplot as plt

import Common as cm

from .constant import *
C = Constant()

def lts_seq_detection(phase, plot_enable=False):
	lts_int_det = np.round(phase[5:21+1]*17/(2*np.pi)).astype('int')
	
	for k in range(16):
		num_matched = np.size(np.nonzero(np.abs(lts_int_det - C.lts_int[k])<2))
		# print(f'{lts_int_det}')
		# print(f'{C.lts_int[k]}')
		# print(f'lts detect for index:{k} is value: {num_matched}')
		if num_matched >= 8:
			lts_seq = k
			cm.prCyan(f'LTS sequence detection result: {lts_seq=} {num_matched=}')
			return lts_seq
	
	cm.prRed(f'LTS sequence detection is failed.')

	if plot_enable:
		plt.plot(lts_int_det, label = 'phase_int')
		plt.plot(C.lts_int[0], label = 'lts_int')
		plt.plot(phase[5:21+1], label = 'phase')
		plt.plot(C.preamble_lts_phase[0], label = 'lts')
		plt.legend()
		plt.grid()
		plt.show()
		
	return -1
	
def lts_detection(phase, ampli, data, plot_enable=False):
	fs = 2.0
	lts_seq = lts_seq_detection(phase, plot_enable)
	
	## auto correlation phase samples
	lts_1_range = np.arange(5, 5+15)
	lts_2_range = np.arange(5+17, 5+17+15)
	freq = phase_correction(phase[lts_2_range]-phase[lts_1_range])/15
	freq_offset = np.mean(freq)
	phase_sync_1 = phase_correction(phase - freq_offset*np.arange(phase.size))
	phase_offset = np.mean(phase_correction(phase_sync_1[lts_2_range]-C.preamble_lts_phase[lts_seq][:-2]))
	phase_sync_2 = phase_correction(phase_sync_1 - phase_offset)
	
	print(f'Lts detection(phase): freq_off = {freq_offset*1000*fs/(2*np.pi):.3f} KHz, phase_off = {phase_offset*180/np.pi:.3f} Deg')
	
	## auto correlation IQ samples
	acorr_data = data[lts_2_range]*np.conjugate(data[lts_1_range])
	freq_offset = np.angle(np.sum(acorr_data))/15
	# data_sync_3 = data*np.exp(-1j*freq_offset*np.arange(data.size))
	# phase_offset = np.angle(np.sum(data_sync_3[lts_2_range]*np.conjugate(C.preamble_lts[lts_seq][:-2])))
	# data_sync = data_sync_3*np.exp(-1j*phase_offset)

	print(f'Lts detection(new): freq_off = {freq_offset*1000*fs/(2*np.pi):.3f} KHz, phase_off = {phase_offset*180/np.pi:.3f} Deg')
	
	## cross correlation IQ samples
	xcorr_lts_1 = data[lts_1_range]*np.conjugate(C.preamble_lts[lts_seq][:-2])
	xcorr_lts_2 = data[lts_2_range]*np.conjugate(C.preamble_lts[lts_seq][:-2])
	freq_offset = np.angle(np.sum(xcorr_lts_2)*np.conjugate(np.sum(xcorr_lts_1)))/15
	
	print(f'Lts detection(new2): freq_off = {freq_offset*1000*fs/(2*np.pi):.3f} KHz')

	if plot_enable:
		# plt.plot(freq*1e6*fs/(2*np.pi), label='freq')
		# plt.plot((np.angle(acorr_data)/15)*1e6*fs/(2*np.pi), label='acorr')
		plt.plot(phase_sync_2, label='phase_sync')
		plt.legend()
		plt.grid()
		plt.show()

	return phase_sync_2[39:], ampli[39:]
	