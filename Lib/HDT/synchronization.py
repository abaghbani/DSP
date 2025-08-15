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
	data_acorr = data*np.conjugate(np.roll(data, symb_width_by4))
	data_acorr_sum = np.abs(np.correlate(data_acorr, np.ones(5*symb_width_by4), 'full'))/(5*symb_width_by4)
	data_ampli_sum = np.correlate(np.abs(data), np.ones(2*symb_width_by4), 'full')/(2*symb_width_by4)
	data_acorr_diff = np.abs(np.correlate(np.diff(data_acorr_sum), np.ones(1*symb_width_by4), 'full'))/(1*symb_width_by4)
	
	peaks = max_detect(data_xcorr, symb_width)

	sts_index = 0
	# for pk in peaks[1:]:
	# 	if (np.any(pk-symb_width_by4 == peaks) or np.any(pk-symb_width_by4-1 == peaks) or np.any(pk-symb_width_by4+1 == peaks)):
	# 		if data_acorr_sum[pk]/(data_ampli_sum[pk]**2) > 0.7 and (np.max(data_acorr_sum[pk-symb_width_by4:pk+1])-np.min(data_acorr_sum[pk-symb_width_by4:pk+1]))/(data_ampli_sum[pk]**2) < 0.05:
	# 			print(pk, (np.max(data_acorr_sum[pk-symb_width_by4:pk+1])-np.min(data_acorr_sum[pk-symb_width_by4:pk+1]))/(data_ampli_sum[pk]**2), data_acorr_sum[pk]/(data_ampli_sum[pk]**2))
	# 			sts_index = pk
	
	for i, pk_diff in enumerate(np.diff(peaks)):
		if np.abs(pk_diff-symb_width_by4) < 4:
			pk = peaks[i+1]
			# and data_acorr_diff[pk] < 0.001: # \
			if	data_acorr_sum[pk]/(data_ampli_sum[pk]**2) > 0.8 \
			and (np.max(data_acorr_sum[pk-symb_width_by4:pk+1])-np.min(data_acorr_sum[pk-symb_width_by4:pk+1]))/(data_ampli_sum[pk]**2) < 0.05:
				
				print(pk, (np.max(data_acorr_sum[pk-symb_width_by4:pk+1])-np.min(data_acorr_sum[pk-symb_width_by4:pk+1]))/(data_ampli_sum[pk]**2), data_acorr_sum[pk]/(data_ampli_sum[pk]**2))
				sts_index = pk
	
	if sts_index != 0:
		cm.prGreen(f'STS peak detected = {sts_index}')
		normalize_index = sts_index
	else:
		cm.prRed('Error: STS detection is failed (sts peaks are less than 8).')
		normalize_index = 500

	if plot_enable:
		plt.plot(data_xcorr, label='mag')
		plt.plot(peaks, data_xcorr[peaks], 'r.')
		plt.plot(data_acorr_sum/data_ampli_sum[normalize_index], label='acorr_5')
		plt.legend()
		plt.grid()
		plt.show()

	return sts_index

def sts_sync(data, symb_width, fs):

	sts_index = peak_detect(data, symb_width, False)
	
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
		
		if False:
			plt.plot(np.angle(acorr_data), label='acorr')
			plt.legend()
			plt.grid()
			plt.show()
			
		return phase_sync_2[sts_index:], ampli_sync[sts_index:], data_sync[sts_index:]
	
def lts_seq_detection(phase, symb_width):
	index_range = np.arange(5*symb_width, 21*symb_width+1, symb_width)
	lts_int_det = np.round(phase[index_range.astype(int)]*17/(2*np.pi)).astype('int')
	
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

	if False:
		plt.plot(np.repeat(lts_int_det, symb_width), label = 'phase_int')
		plt.plot(np.repeat(C.lts_int[0], symb_width), label = 'lts_int')
		plt.plot(np.repeat(phase[5*symb_width:22*symb_width+1:symb_width], symb_width), label = 'phase')
		plt.plot(np.repeat(C.preamble_lts_phase[0], symb_width), label = 'lts')
		plt.legend()
		plt.grid()
		plt.show()
		
	return -1
	
def lts_sync(phase, data, symb_width, fs):
	
	lts_seq = lts_seq_detection(phase, symb_width)

	lts_1_range = np.arange((4+1)*symb_width, (4+17-2)*symb_width+1).astype('int')
	lts_2_range = np.arange((4+18)*symb_width, (4+34-2)*symb_width+1).astype('int')
	lts_1_range_center = np.arange((4+1)*symb_width, (4+17-2)*symb_width+1 ,symb_width).astype('int')
	lts_2_range_center = np.arange((4+18)*symb_width, (4+34-2)*symb_width+1 ,symb_width).astype('int')
	
	## auto correlation phase samples
	freq = phase_correction(phase[lts_2_range]-phase[lts_1_range])/(17*symb_width)
	freq_offset = np.mean(freq)
	phase_sync_1 = phase_correction(phase - freq_offset*np.arange(phase.size))
	phase_offset = np.mean(phase_correction(phase_sync_1[lts_2_range_center]-C.preamble_lts_phase[lts_seq][:-2]))
	phase_sync_2 = phase_correction(phase_sync_1 - phase_offset)
	
	print(f'Lts detection(phase): freq_off = {freq_offset*1000*fs/(2*np.pi):.3f} KHz, phase_off = {phase_offset*180/np.pi:.3f} Deg')
	
	## auto correlation IQ samples
	acorr_data = data[lts_2_range]*np.conjugate(data[lts_1_range])
	freq_offset = np.angle(np.sum(acorr_data))/(17*symb_width)
	data_sync_3 = data*np.exp(-1j*freq_offset*np.arange(data.size))
	phase_offset = np.angle(np.sum(data_sync_3[lts_2_range_center]*np.conjugate(C.preamble_lts[lts_seq][:-2])))
	data_sync = data_sync_3*np.exp(-1j*phase_offset)

	print(f'Lts detection(new): freq_off = {freq_offset*1000*fs/(2*np.pi):.3f} KHz, phase_off = {phase_offset*180/np.pi:.3f} Deg')
	
	## cross correlation IQ samples
	if int(symb_width) == symb_width:
		lts_1_range = np.arange((4+1-0.5)*symb_width, (4+17-2+0.5)*symb_width).astype('int')
		lts_2_range = np.arange((4+18-0.5)*symb_width, (4+34-2+0.5)*symb_width).astype('int')
		xcorr_lts_1 = data[lts_1_range]*np.conjugate(np.repeat(C.preamble_lts[lts_seq][:-2], int(symb_width)))
		xcorr_lts_2 = data[lts_2_range]*np.conjugate(np.repeat(C.preamble_lts[lts_seq][:-2], int(symb_width)))
		freq_offset = np.angle(np.sum(xcorr_lts_2)*np.conjugate(np.sum(xcorr_lts_1)))/(17*symb_width)
		
		print(f'Lts detection(new2): freq_off = {freq_offset*1000*fs/(2*np.pi):.3f} KHz')

	if False:
		plt.plot(freq*1e6*fs/(2*np.pi), label='freq')
		plt.plot((np.angle(acorr_data)/(17*symb_width))*1e6*fs/(2*np.pi), label='acorr')
		plt.legend()
		plt.grid()
		plt.show()

	return phase_sync_2[round(38*symb_width):], data_sync[round(38*symb_width):]

def ts_sync(data, fs):
	fsymb = 2.0 # HDT symbol rate is 2MSymb/s
	symb_width = fs/fsymb

	phase_sts, ampli, data_sts  = sts_sync(data, symb_width, fs)
	phase_lts, data_lts = lts_sync(phase_sts, data_sts, symb_width, fs) if phase_sts.size > int(38*symb_width) else np.empty(0)
	
	return phase_lts, ampli[round(38*symb_width):]
	