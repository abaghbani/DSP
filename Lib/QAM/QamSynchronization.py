import numpy as np
import scipy.signal as signal
import scipy.stats as stats
import matplotlib.pyplot as plt

import Common as cm
from .Constant import *
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

	peaks = np.unique(np.hstack(np.column_stack((peak_1[:peak_2.size], peak_2))))
	return peaks

# def max_detect_2(data, symb_width):
# 	prominence = 0.15*4*np.max(data.real)
# 	width = round(symb_width)-2
# 	peaks, properties = signal.find_peaks(data, height=0, distance=symb_width_by4-3, prominence=prominence, width=width)
# 	print(f'{prominence=}, {width=}')
# 	print(properties["prominences"][:10])
# 	print(properties["widths"][:10])


def peak_detect(data, symb_width, plot_enable=True):
	symb_width_by2 = round(symb_width*2)
	symb_width_by3 = round(symb_width*3)
	symb_width_by4 = round(symb_width*4)

	data_xcorr = np.array([magnitude_estimator(ts_xcorrolate([d3, d2, d1, d0], C.preamble_sts)) for d3, d2, d1, d0 in zip(np.roll(data, symb_width_by3), np.roll(data, symb_width_by2), np.roll(data, int(symb_width)), data)])
	data_acorr = data*np.conjugate(np.roll(data, symb_width_by4))
	data_acorr_sum = np.abs(np.correlate(data_acorr, np.ones(5*symb_width_by4), 'full'))/(5*symb_width_by4)
	data_ampli_sum = np.abs(np.correlate(np.abs(data), np.ones(2*symb_width_by4), 'full'))/(2*symb_width_by4)
	
	peaks = max_detect(data_xcorr, symb_width)

	sts_index = 0
	for pk in peaks[1:]:
		if (np.any(pk-symb_width_by4 == peaks) or np.any(pk-symb_width_by4-1 == peaks) or np.any(pk-symb_width_by4+1 == peaks)):
			if data_acorr_sum[pk]/(data_ampli_sum[pk]**2) > 0.7 and (np.max(data_acorr_sum[pk-symb_width_by4:pk+1])-np.min(data_acorr_sum[pk-symb_width_by4:pk+1]))/(data_ampli_sum[pk]**2) < 0.05:
				print(pk, (np.max(data_acorr_sum[pk-symb_width_by4:pk+1])-np.min(data_acorr_sum[pk-symb_width_by4:pk+1]))/(data_ampli_sum[pk]**2), data_acorr_sum[pk]/(data_ampli_sum[pk]**2))
				sts_index = pk
	
	if sts_index != 0:
		cm.prGreen(f'STS peak detected = {sts_index}')
		normalize_index = sts_index
	else:
		cm.prRed('Error: STS detection is failed (sts peaks are less than 8).')
		normalize_index = 500

	if plot_enable:
		# plt.plot(data_xcorr, label='mag')
		# plt.plot(peaks, data_xcorr[peaks], 'r.')
		
		plt.plot(data_xcorr, label='mag')
		plt.plot(peaks, data_xcorr[peaks], 'r.')
		plt.plot(data_acorr_sum/data_ampli_sum[normalize_index], label='acorr_5')
		# plt.plot(peaks_new, (data_xcorr[peaks_new]/4)**2, 'g.')
		# plt.plot(np.correlate(data_acorr, np.ones(8*symb_width_by4), 'same'), label='acorr*8')
		# plt.plot(np.correlate(data_acorr, np.ones(7*symb_width_by4), 'same'), label='acorr*7')
		# plt.plot(np.abs(data_acorr)/1e2, label='acorr')
		# plt.plot(np.abs(np.correlate(data_acorr, np.ones(8*symb_width_by4), 'full'))/1e4, label='acorr*8')
		# plt.plot((data_acorr_sum/np.roll(data_acorr_sum, symb_width_by4)-1)*1e3, label='acorr_rate')
		# plt.plot(peaks[sts_index[0]], data_xcorr[peaks[sts_index[0]]], 'go')
		plt.legend()
		plt.grid()
		plt.show()

	return sts_index

	# peaks_diff = np.diff(peaks)
	# sts_index = np.array([i+1 for i in range(7, peaks_diff.size) if np.all(abs(peaks_diff[i-7:i+1]-symb_width_by4) < 3)])
	# sts_index_num = 9
	# if sts_index.size == 0:
	# 	sts_index = np.array([i+1 for i in range(6, peaks_diff.size) if np.all(abs(peaks_diff[i-6:i+1]-symb_width_by4) < 3)])
	# 	sts_index_num = 8
	
	# if plot_enable:
	# 	print(f'{prominence=}, {width=}')
	# 	print(properties["prominences"][:10])
	# 	print(properties["widths"][:10])
	# 	plt.plot(data_xcorr, label='mag')
	# 	plt.plot(peaks, data_xcorr[peaks], 'r.')
	# 	# plt.plot(np.correlate(data_acorr, np.ones(8*symb_width_by4), 'same'), label='acorr*8')
	# 	# plt.plot(np.correlate(data_acorr, np.ones(7*symb_width_by4), 'same'), label='acorr*7')
	# 	# plt.plot(np.abs(data_acorr)/1e2, label='acorr')
	# 	# plt.plot(np.abs(np.correlate(data_acorr, np.ones(8*symb_width_by4), 'full'))/1e4, label='acorr*8')
	# 	plt.plot(data_acorr_sum/1e2, label='acorr_5')
	# 	# plt.plot(peaks[sts_index[0]], data_xcorr[peaks[sts_index[0]]], 'go')
	# 	plt.legend()
	# 	plt.grid()
	# 	plt.show()
	
	# if sts_index.size != 0:
	# 	cm.prGreen(f'number of sts peak detected = {sts_index_num}, last sts index = {peaks[sts_index[0]]}')
	# 	return peaks[sts_index[0]]
	
	# cm.prRed('Error: STS detection is failed (sts peaks are less than 8).')
	# return 0

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
		# ampli_avg = np.array([np.mean(ampli_sync[i-round(2*symb_width):i+1]) for i in range(round(2*symb_width), ampli_sync.size)])
		# index_end_packet = np.nonzero(ampli_avg<0.1)[0]
		# index_end_packet = index_end_packet[0] if index_end_packet.size != 0 else ampli_sync.size
		
		print(f'Sts detection(2):  index={sts_index}, freq_off = {freq_offset*1000*fs/(2*np.pi):.3f} KHz, {phase_offset*180/np.pi:.3f} Deg')
		# print(f'Sts detection:  index={sts_index}, freq_off = {freq_offset_old*1000*fs/(2*np.pi):.3f} KHz, {phase_offset*180/np.pi:.3f} Deg, end_index={sts_index+index_end_packet}')
		
		if False:
			plt.plot(np.angle(acorr_data), label='acorr')
			# plt.plot(ampli_avg, label='ampli_avg')
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
	# plt.plot(np.repeat(lts_int_det, symb_width), label = 'phase_int')
	# plt.plot(np.repeat(C.lts_int[0], symb_width), label = 'lts_int')
	# plt.plot(np.repeat(phase[5*symb_width:22*symb_width+1:symb_width], symb_width), label = 'phase')
	# plt.plot(np.repeat(C.preamble_lts_phase[0], symb_width), label = 'lts')
	# plt.legend()
	# plt.grid()
	# plt.show()
	return -1
	
def lts_sync(phase, data, symb_width, fs):
	
	lts_seq = lts_seq_detection(phase, symb_width)

	lts_range = np.arange((4+17+1)*symb_width, (4+17+17)*symb_width+1).astype('int')
	lts_range_phase = np.arange((4+17+1)*symb_width, (4+17+17)*symb_width+1 ,symb_width).astype(int)
	
	freq = phase_correction(phase[lts_range]-phase[lts_range-round(17*symb_width)])/(17*symb_width)
	freq_offset = np.mean(freq)
	
	phase_sync_1 = phase_correction(phase - freq_offset*np.arange(phase.size))
	phase_offset = np.mean(phase_correction(phase_sync_1[lts_range_phase]-C.preamble_lts_phase[lts_seq]))
	phase_sync_2 = phase_correction(phase_sync_1 - phase_offset)
	
	print(f'Lts detection(phase): freq_off = {freq_offset*1000*fs/(2*np.pi):.3f} KHz, phase_off = {phase_offset*180/np.pi:.3f} Deg')
	
	acorr_data = data[lts_range]*np.conjugate(data[lts_range-round(17*symb_width)])
	freq_offset = np.angle(np.sum(acorr_data))/(17*symb_width)
	
	data_sync_3 = data*np.exp(-1j*freq_offset*np.arange(data.size))
	# lts_seq = lts_seq if lts_seq != -1 else 0
	phase_offset = np.angle(np.sum(data_sync_3[lts_range_phase]*np.conjugate(C.preamble_lts[lts_seq])))
	data_sync = data_sync_3*np.exp(-1j*phase_offset)

	print(f'Lts detection(new): freq_off = {freq_offset*1000*fs/(2*np.pi):.3f} KHz, phase_off = {phase_offset*180/np.pi:.3f} Deg')
	
	## another methode to calc freq offset in LTS, cross correlation of rx_sig and lts symboles
	xcorr_data = data[lts_range_phase]*np.conjugate(C.preamble_lts[lts_seq])
	freq_offset = np.mean(np.diff(np.angle(xcorr_data)))/symb_width
	
	print(f'Lts detection(new2): freq_off = {freq_offset*1000*fs/(2*np.pi):.3f} KHz, phase_off = {phase_offset*180/np.pi:.3f} Deg')
	plt.plot(np.angle(xcorr_data))
	plt.show()


	if False:
		plt.plot(freq, label='freq')
		plt.plot(np.angle(acorr_data)/(17*symb_width), label='acorr')
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
		
def ts_sync_hdl(data, sample_rate):
	baseband_symb_rate = 2 # HDT symbol rate is 2MSymb/s
	symb_width = int(sample_rate/baseband_symb_rate)

	ampli = np.zeros(data.size, dtype='float')
	phase = np.zeros(data.size, dtype='float')
	phase_diff = np.zeros(data.size, dtype='float')
	mag = np.zeros(data.size, dtype='float')
	mag_diff = np.zeros(data.size, dtype='float')
	mag_temp = np.zeros(data.size, dtype='float')
	mag_max = 0
	mag_max_index = 0
	mag_peak = np.empty(0, dtype='int')
	sts_detection = 0
	lts_detection = 0
	pits_detection = 0
	sts_freq_detection=0
	sts_index = 0
	lts_index = 0
	lts_symb_count = np.zeros(16, dtype='int')
	agc_rate = 1
	lts_seq = -1
	LTS_detection_type = "XCP" ## "XC" (cross correlation), "AC" (auto correlation), "XCP" (cross correlation phase)

	freq_off = 0 #-0.015/(2*np.pi)
	phase_off = 0 # -80*(np.pi/180)
	freq = 0
	data_sync = np.zeros(data.size, dtype=data.dtype)
	acorr = np.zeros(data.size, dtype=data.dtype)
	acorr_sum = np.zeros(data.size, dtype=data.dtype)
	
	n_sts_freq = 7
	n_sts_phase = 9
	for i, iq in enumerate(data):
		## mixer
		data_sync[i], freq = nco(iq, freq, -1*freq_off, -1*phase_off)
		## phase calculation
		phase[i] = np.angle(data_sync[i])
		ampli[i] = np.abs(data_sync[i])

		## STS detection
		if sts_detection==0 and i > 4*symb_width:
			phase_diff[i] = phase_correction(phase[i]-phase[i-4*symb_width])/(4*symb_width)
			mag[i] = magnitude_estimator(ts_xcorrolate(data_sync[i-3*symb_width:i+symb_width:symb_width], C.preamble_sts))
			if mag[i]> mag_max:
				mag_max = mag[i]
				mag_max_index = i
			if i%(4*symb_width) == 0:
				mag_peak = np.append(mag_peak, mag_max_index)
				mag_max = 0
		
			if  mag_peak.size>=n_sts_freq and np.all(np.abs(np.diff(mag_peak)[-n_sts_freq+1:] - 4*symb_width) < 4) and np.all(mag[mag_peak[-n_sts_freq+1:-1]]/mag[mag_peak[-n_sts_freq:-2]]<1.2) and sts_freq_detection==0:
				sts_index = mag_peak[-1]
				freq_off += np.mean(phase_diff[sts_index-int((n_sts_freq-1)*4*symb_width)+symb_width:sts_index-symb_width])
				sts_freq_detection=1
				print(f'Sts detection(freq):  index={sts_index}, freq_off = {freq_off*1000*sample_rate/(2*np.pi):.3f} KHz')

			if mag_peak.size>=n_sts_phase and np.all(np.abs(np.diff(mag_peak)[-n_sts_phase+1:] - 4*symb_width) < 4) and np.all(mag[mag_peak[-n_sts_phase+1:-1]]/mag[mag_peak[-n_sts_phase:-2]]<1.2):
				sts_index = mag_peak[-1]
				phase_off += phase[sts_index]
				sts_detection=1
				mag_max = 0
				mag_max_index = 0
				mag /= 1.0e6
				print(f'Sts detection(phase): index={sts_index}, phase_off = {phase_off*180/np.pi:.3f} Deg')

		## LTS detection
		elif sts_detection==1 and lts_detection==0:
			phase_diff[i] = phase_correction(phase[i]-phase[i-17*symb_width])/(17*symb_width)
			
			## LTS sequence detection (during first 17 lts symbols)
			if lts_seq == -1 and i<=(sts_index + 21*symb_width):
				if i> (sts_index+4*symb_width) and (i-sts_index)%symb_width == 0:
					lts_int_det = np.round(phase[i]*17/(2*np.pi)).astype('int')
					for k in range(16):
						if lts_int_det == C.lts_int[k][int((i-sts_index)/symb_width)-5]:
							lts_symb_count[k] += 1
							if lts_symb_count[k] >= 8:
								lts_seq = k
								print(f'LTS sequence detection result: {lts_seq=}')
					# print(f'counters = {lts_symb_count}')
				elif i == (sts_index + 21*symb_width):
					print(f'LTS sequence detection is failed.')
					# lts_detection = 1
				
			## LTS detection XCP
			if LTS_detection_type=="XCP":

				if i == (sts_index+int((4+17+15)*symb_width)):
					freq_off_avg = np.mean(phase_diff[i-14*symb_width:i+1])
					freq_off += freq_off_avg
					agc_rate = np.mean(ampli[i-5*symb_width:i+1:symb_width])
					print(f'Lts detection (freq): index={i}, freq_off = {freq_off*1000*sample_rate/(2*np.pi):.3f} KHz, {agc_rate=}, lts_freq_off = {freq_off_avg*1000*sample_rate/(2*np.pi):.3f} KHz')
				
				elif i == (sts_index+int((4+17+16)*symb_width)):
					if lts_seq != -1:
						phase_off_avg = np.mean(phase[i-3*symb_width:i+1:symb_width]-C.preamble_lts_phase[lts_seq][-5:-1])
						phase_off += phase_off_avg
						print(f'Lts detection (phase): index={i}, lts_phase_off = {phase_off_avg*180/np.pi:.3f} Deg')
					lts_index = i+symb_width
					lts_detection = 1
					
				# mag[i] = np.sum(phase[i-14*symb_width:i+symb_width:symb_width] * C.lts_int[lts_seq][1:-1])
				# mag_diff[i] = mag[i]-mag[i-1]
				
				# if i > (sts_index+int((4+17+15.8)*symb_width)) and i < (sts_index+int((4+17+16.5)*symb_width)):
				# 	if mag_diff[i]*mag_diff[i-1]<=0:
				# 		agc_rate = np.mean(ampli[i-5*symb_width:i+1:symb_width])
				# 		freq_off += np.mean(phase_diff[i-7*symb_width:i+1])
				# 		phase_off += phase[i]-C.preamble_lts_phase[lts_seq][-2]
				# 		print(f'Lts detection({LTS_detection_type}): index={i}, freq_off = {freq_off*1000*sample_rate/(2*np.pi):.3f} KHz, phase_off = {phase_off*180/np.pi:.3f} Deg, {agc_rate=}')
				# 		print(f'Lts detection({LTS_detection_type}): lts_freq_off = {np.mean(phase_diff[i-7*symb_width:i+1])*1000*sample_rate/(2*np.pi):.3f} KHz, lts_phase_off = {(phase[i]-C.preamble_lts_phase[lts_seq][-2])*180/np.pi:.3f} Deg')
				# 		lts_index = i+symb_width-1
				# 		lts_detection = 1

			## LTS detection XC
			elif LTS_detection_type=="XC":
				mag[i] = magnitude_estimator(ts_xcorrolate(data_sync[i-16*symb_width:i+symb_width:symb_width], C.preamble_lts[lts_seq]))
				if mag[i]> mag_max:
					mag_max = mag[i]
					mag_max_index = i
				if (i-12*symb_width)%(17*symb_width) == 0:
					mag_peak = np.append(mag_peak, mag_max_index)
					mag_max = 0

				if (mag_peak[-1]-sts_index)>2*17*symb_width and np.abs(np.diff(mag_peak)[-1] - 17*symb_width) < 5:
					lts_index = mag_peak[-1]
					agc_rate = np.mean(mag[lts_index-17*symb_width:lts_index+1:17*symb_width])/17
					lts_detection = 1
					freq_off += np.mean(phase_diff[lts_index-17*symb_width:lts_index+1])
					phase_off += phase[lts_index]-C.preamble_lts_phase[lts_seq][-2]
					print(f'Lts detection({LTS_detection_type}): index={lts_index}, freq_off = {freq_off*1000*sample_rate/(2*np.pi):.3f} KHz, phase_off = {phase_off*180/np.pi:.3f} Deg, {agc_rate=}')
				
			## LTS detection AC
			elif LTS_detection_type=="AC":
				# mag[i] = magnitude_estimator(ts_xcorrolate(data_sync[i-16*symb_width:i+symb_width:symb_width], data_sync[i-(16+17)*symb_width:i-16*symb_width:symb_width]))
				acorr[i] = data_sync[i]*np.conjugate(data_sync[i-17*symb_width])
				acorr_sum[i] = acorr_sum[i-1] + acorr[i] - acorr[i-17*symb_width]
				mag_temp[i] = magnitude_estimator(acorr_sum[i])
				mag[i] = np.mean(mag_temp[i-4*symb_width:i+1])
				
				if mag[i]> mag_max:
					mag_max = mag[i]
					mag_max_index = i
				if i%(4*symb_width) == 0:
					mag_peak = np.append(mag_peak, mag_max_index)
					mag_max = 0

				if i > sts_index + (2*17+4+4)*symb_width:
					lts_index = mag_peak[-1]-5
					agc_rate = np.sqrt(mag[lts_index]/(17*symb_width))*1.1  # this 1.1 is correction of magnitude estimation
					lts_detection = 1
					freq_off += np.mean(phase_diff[lts_index-17*symb_width:lts_index+1])
					phase_off += phase[lts_index]-C.preamble_lts_phase[lts_seq][-2]
					print(f'Lts detection({LTS_detection_type}): index={lts_index}, freq_off = {freq_off*1000*sample_rate/(2*np.pi):.3f} KHz, phase_off = {phase_off*180/np.pi:.3f} Deg, {agc_rate=}')
		
		## PITS detection
		elif pits_detection==0 and lts_detection==1:
			pits_limit = np.pi/10
			if (i-sts_index)%symb_width == 0:
				if np.abs(phase[i-3*symb_width]) < pits_limit and np.abs(phase[i-2*symb_width]) < pits_limit and np.abs(phase[i-1*symb_width]) > 9*pits_limit and np.abs(phase[i]) < pits_limit :
					phase_off += np.mean([phase[i-3*symb_width], phase[i-2*symb_width], phase[i]])
					print(f'PITS detection: index={i} phase={phase[i-3*symb_width:i+1:symb_width]} phase_off = {phase_off*180/np.pi:.3f} deg')
	
	#plt.plot(phase_diff*1000*sample_rate/(2*np.pi), label = 'diff_sts')
	#plt.plot(phase_diff*1000*sample_rate/(2*np.pi), label = 'diff_lts')
	#plt.plot(phase, label = 'phase')

	# plt.plot(ampli*10, label = 'ampli')
	# plt.plot(mag, label = 'mag')
	# plt.plot(mag_diff, label = 'mag_diff')
	# plt.plot(mag_peak, mag[mag_peak], 'ro')
	
	plt.plot(phase_diff*1.0e2, label = 'freq')
	plt.plot(phase, label = 'phase')
	# plt.plot(np.hstack([np.zeros(sts_index+4*symb_width+8), np.tile(np.repeat(C.preamble_lts_phase[lts_seq], symb_width), 2)]), label = 'lts_int')
	
	plt.legend()
	plt.grid()
	plt.show()
	
	#plt.plot(data_sync[1350:].real, data_sync[1350:].imag, 'ro')
	#plt.show()

	return 	phase[lts_index:], ampli[lts_index:]/agc_rate

class sync:

	def ts_sync(self, data, sample_rate):
		## new design to implement on HDL
		data_sts_sync = self.sts_sync(data, sample_rate)
		return self.lts_sync(data_sts_sync, sample_rate)

		## first design of QAM:
		#data_sync, data_gain = self.StsCarrierSync(data, sample_rate)
		#return self.LtsCarrierSync(data_sync, sample_rate)/data_gain

	def peak_detection(self, data, delta, margin, margin_2):
		signal = (data > margin*np.roll(data, delta)) & (data > margin*np.roll(data, -delta))
		peak_index = signal.nonzero()[0]
		peak_index_ret = np.empty(0, dtype='int')
		for i in range(1, peak_index.size-1):
			if data[peak_index[i]]>margin_2*data[peak_index[i-1]] and data[peak_index[i]]>margin_2*data[peak_index[i+1]] and \
				(peak_index_ret.size == 0 or data[peak_index[i]] > 0.5*data[peak_index_ret[-1]]):
				peak_index_ret = np.append(peak_index_ret, peak_index[i])

		return peak_index_ret

	def periodic_peak_detection(self, index, period_number, period_width):
		for i in range(period_number-1, index.size):
			flag = True
			for j in range(period_number-1):
				if np.abs((index[i-j]-index[i-j-1]) - period_width) < 0.1*period_width:
					continue
				else:
					flag = False
					break
			if flag:
				index_ret = index[i-period_number+1:i+1]
				return index_ret

		print('>>> Error: no periodic peak is detected.')
		return 0


	def sts_detection(self, data, symb_width):
	
		data_xcorr_2 = np.zeros(data.size, dtype='complex')
		for i in range(4*symb_width, data.size):
			data_xcorr_2[i] = ts_xcorrolate(data[i-3*symb_width:i+symb_width:symb_width], C.preamble_sts)
		#mag = np.abs(data_xcorr_2)
		mag = magnitude_estimator(data_xcorr_2)
		peak_index = self.peak_detection(mag, 1, 1, 1.5)
		sts_index = self.periodic_peak_detection(peak_index, 9, 4*symb_width)
	
		#plt.plot(mag)
		#plt.plot(peak_index, mag[peak_index], '.')
		#plt.plot(sts_index, mag[sts_index], 'X')
		#plt.show()

		## return exactly point in the middle of symbol
		return sts_index

	def lts_detection(self, data, symb_width):
	
		data_xcorr_2 = np.zeros(data.size, dtype='complex')
		for i in range(17*symb_width, data.size):
			data_xcorr_2[i] = ts_xcorrolate(data[i-16*symb_width:i+symb_width:symb_width], C.preamble_lts)
		#mag = np.abs(data_xcorr_2)
		mag = magnitude_estimator(data_xcorr_2)
		peak_index = self.peak_detection(mag, 1, 1, 2.7)
		lts_index = self.periodic_peak_detection(peak_index, 2, 17*symb_width)

		#plt.plot(mag)
		#plt.plot(peak_index, mag[peak_index], '.')
		#plt.plot(lts_index, mag[lts_index], 'X')
		#plt.show()

		## return exactly point in the middle of symbol
		return lts_index, mag[lts_index]/17

	def sts_sync(self, data, sample_rate):
	
		baseband_symb_rate = 2 # HDT symbol rate is 2MSymb/s
		symb_width = int(sample_rate/baseband_symb_rate)

		index = self.sts_detection(data, symb_width)
		print(f'\n<<Sts detection info:>>')
		print(f'sts index: {index}')
		print(f'sts index diff: {np.diff(index)}')
	
		phase = np.angle(data)
		phase_diff = np.zeros(phase.size, dtype='float')
		for i in range(4*symb_width, phase.size):
			phase_diff[i] = phase[i]-phase[i-4*symb_width]

		phase_diff[phase_diff>np.pi] -= 2*np.pi
		phase_diff[phase_diff<-np.pi] += 2*np.pi
		phase_diff /= 4*symb_width

		index[0] += 1*symb_width
		index[-1] += -1*symb_width
		freq_off = np.mean(phase_diff[index[0]:index[-1]])
	
		freq_off_sample = np.mean(phase_diff[index])
		freq_off_mean = np.zeros(index.size)
		for i in range(index.size):
			freq_off_mean[i] = np.mean(phase_diff[index[i]-5:index[i]+6])
	
		print(phase_diff[index]*1000*sample_rate/(2*np.pi))
		print(f'freq_off={freq_off*1000*sample_rate/(2*np.pi):.3f} KHz')
		print(f'freq_off_sample={freq_off_sample*1000*sample_rate/(2*np.pi):.3f} KHz ')
		print(f'freq_off_mean={np.mean(freq_off_mean)*1000*sample_rate/(2*np.pi):.3f} KHz ')
	
		plt.plot(phase_diff, label = 'diff')
		plt.plot(index, phase_diff[index], 'ro', label = 'index')
		plt.plot(phase, label = 'phase')
		plt.legend()
		plt.show()

		## phase at first seymbol of STS is -pi:
		freq = -(phase[index[0]-int(4.5*symb_width)]-np.pi)
		data_sync = np.zeros(data.size, dtype='complex')
		for i in range(index[0]-int(4.5*symb_width), data.size):
			data_sync[i], freq = nco(data[i], freq, -1*freq_off)
	
		#plt.plot(np.angle(data), label = 'phase')
		#plt.plot(np.angle(data_sync), label = 'phase_sync')
		#plt.legend()
		#plt.show()
	
		plt.plot(data_sync.real, label='real')
		plt.plot(data_sync.imag, label='imag')
		plt.legend()
		plt.show()

		return data_sync

	def lts_sync(self, data, sample_rate):
	
		baseband_symb_rate = 2 # HDT symbol rate is 2MSymb/s
		symb_width = int(sample_rate/baseband_symb_rate)

		index, gain = self.lts_detection(data, symb_width)
		print(f'\n<<Lts detection info:>>')
		print(f'Lts index: {index}')
		print(f'Lts index diff: {np.diff(index)}')

		phase = np.angle(data)
		phase_diff = np.zeros(phase.size, dtype='float')
		for i in range(17*symb_width, phase.size):
			phase_diff[i] = phase[i]-phase[i-17*symb_width]

		phase_diff[phase_diff>np.pi] -= 2*np.pi
		phase_diff[phase_diff<-np.pi] += 2*np.pi
		phase_diff /= 17*symb_width

		index[0] += 1*symb_width
		index[-1] += -1*symb_width
		freq_off = np.mean(phase_diff[index[0]:index[-1]])
		freq_off_sample = np.mean(phase_diff[index])
	
		print(phase_diff[index]*1000*sample_rate/(2*np.pi))
		print(f'freq_off={freq_off*1000*sample_rate/(2*np.pi):.3f} KHz')
		print(f'freq_off_sample={freq_off_sample*1000*sample_rate/(2*np.pi):.3f} KHz ')
	
		plt.plot(phase_diff, label = 'diff')
		plt.plot(index, phase_diff[index], 'ro', label = 'index')
		#plt.plot(phase, label = 'phase')
		plt.legend()
		plt.show()

		## phase at first seymbol of LTS is 0:
		freq = -(phase[index[0]-int(17.5*symb_width)]+0)
		data_sync = np.zeros(data.size, dtype='complex')
		for i in range(index[0]-int(17.5*symb_width), data.size):
			data_sync[i], freq = nco(data[i], freq, -1*freq_off)

		#plt.plot(np.angle(data), label = 'phase')
		#plt.plot(np.angle(data_sync), label = 'phase_sync')
		#plt.legend()
		#plt.show()
	
		plt.plot(data_sync.real, label='real')
		plt.plot(data_sync.imag, label='imag')
		plt.legend()
		plt.show()
		
		plt.plot(data_sync.real, data_sync.imag, 'ro')
		plt.show()

		return data_sync[index[-1]+int(1.0*symb_width):]/np.mean(gain)
	
	def StsCarrierSync(self, data, sample_rate):
		baseband_symb_rate = 2 # HDT symbol rate is 2MSymb/s
		n_sample_per_symb = int(sample_rate/baseband_symb_rate)
		sts_samples = np.repeat(C.preamble_sts, n_sample_per_symb)
		peak_level_threshold = 2000000*int(C.preamble_sts.size*n_sample_per_symb)

		#######################
		## first path
		#######################

		xcorr = signal.correlate(data, sts_samples, mode='full')
		teta = np.angle(xcorr)
		amp_cc = np.absolute(xcorr)
		peaks, _ = signal.find_peaks(amp_cc, prominence=peak_level_threshold, width=int(.5*n_sample_per_symb), distance=int((C.preamble_sts.size-0.5)*n_sample_per_symb))
		#plt.plot(xcorr.real)
		#plt.plot(xcorr.imag)
		#plt.plot(amp_cc)
		#plt.plot(peaks, amp_cc[peaks], 'x')
	
		plt.plot(teta)
		plt.plot(peaks, teta[peaks], 'x')
		plt.legend(['real', 'imag', 'phase', 'ampli'])
		plt.show()

		# manupulate phase value to calculate linear regression
		peaks = peaks[0:9]
		xcorr_phase = np.zeros(peaks.size, dtype='float')
		xcorr_phase[0] = teta[peaks[0]]
		for i in range(1, peaks.size):
			xcorr_phase[i] = teta[peaks[i]] if np.abs(teta[peaks[i]] - xcorr_phase[i-1]) < 4 else teta[peaks[i]]+2*np.pi*(np.abs(teta[peaks[i]] - xcorr_phase[i-1])//(2*np.pi) + 1)*np.sign(xcorr_phase[i-1]-teta[peaks[i]])
		res = stats.linregress(peaks, xcorr_phase)
		freq_offset = res.slope/(2*np.pi)
		phase_offset = res.intercept 
	
		print(f'stage 1 : freq_offset= {freq_offset*sample_rate*1000:.3f} KHz -- {phase_offset=}')

		plt.plot(teta)
		plt.plot(peaks, teta[peaks], 'x')
		plt.plot(peaks, xcorr_phase, 'o')
		plt.plot(peaks, res.intercept+res.slope*peaks)
		plt.legend(['real', 'imag', 'phase', 'ampli'])
		plt.grid()
		plt.show()
		
		import RfModel as rf
		data_mixer_1 = rf.Mixer(data, -1*np.mean(freq_offset), -1*phase_offset)

		#######################
		## Second path
		#######################

		xcorr = signal.correlate(data_mixer_1, sts_samples, mode='full')
		teta = np.angle(xcorr)
		amp_cc = np.absolute(xcorr)
		peaks, _ = signal.find_peaks(amp_cc, prominence=peak_level_threshold, width=int(.5*n_sample_per_symb), distance=int((C.preamble_sts.size-0.5)*n_sample_per_symb))
		#plt.plot(xcorr.real)
		#plt.plot(xcorr.imag)
		#plt.plot(amp_cc)
		#plt.plot(peaks, amp_cc[peaks], 'x')
		plt.plot(teta)
		plt.plot(peaks, teta[peaks], 'x')
		plt.show()
				
		peaks = peaks[:9]
		phase_offset = np.mean(teta[peaks])
		gain = (np.mean(amp_cc[peaks])*1.25)/int(C.preamble_sts.size*n_sample_per_symb)
		#plt.plot(teta[peaks])
		#plt.show()
		print(f'stage 2 : {phase_offset=} , {gain=}')
		data_mixer_2 = rf.Mixer(data_mixer_1, 0, -1*phase_offset)


		#xcorr = signal.correlate(data_3, sts_samples, mode='full')
		#teta = np.angle(xcorr)
		#amp_cc = np.absolute(xcorr)
		#plt.plot(xcorr.real)
		#plt.plot(xcorr.imag)
		#plt.plot(teta)
		#plt.plot(amp_cc)
		#plt.show()

		#plt.plot(data.real)
		#plt.plot(data_2.real)
		#plt.plot(data_3.real)
		#plt.legend(['rx_sig', 'freq_sync', 'phase_sync'])
		#plt.grid()
		#plt.show()
		#plt.plot(data.imag)
		#plt.plot(data_2.imag)
		#plt.plot(data_3.imag)
		#plt.legend(['rx_sig', 'freq_sync', 'phase_sync'])
		#plt.grid()
		#plt.show()

		return data_mixer_2[peaks[-1]:], gain

	def LtsCarrierSync(self, data, sample_rate):
		baseband_symb_rate = 2 # HDT symbol rate is 2MSymb/s
		n_sample_per_symb = int(sample_rate/baseband_symb_rate)
		lts_samples = np.repeat(C.preamble_lts, n_sample_per_symb)
		peak_level_threshold = 2000000*int(C.preamble_lts.size*n_sample_per_symb)

		xcorr = signal.correlate(data, lts_samples, mode='full')
		amp_cc = np.absolute(xcorr)
		peaks, _ = signal.find_peaks(amp_cc, height =peak_level_threshold) #, width=int(.5*n_sample_per_symb), distance=int((C.preamble_lts.size-0.5)*n_sample_per_symb))
	
		plt.plot(amp_cc)
		plt.plot(peaks, amp_cc[peaks], 'x')
		plt.show()

		return data[peaks[1]-int(.5*n_sample_per_symb):]

	def StsCarrierSync_acf(data, sample_rate):
		## Auto-Corrolation on sample data , that is complicated for HDL (not used)
		baseband_sample_rate = 2
		n_sample_per_symb = int(sample_rate/baseband_sample_rate)
		n_sample_sts = C.preamble_sts.size*n_sample_per_symb
		corr_delay = 4*n_sample_sts

		xcorr = np.zeros(data.size, dtype='complex')
		for i in range(2*corr_delay, data.size):
			#xcorr[i] = signal.correlate(data[i-n_sample_sts:i], data[i-corr_delay-n_sample_sts:i-corr_delay], mode='valid')
			xcorr[i] = np.sum(data[i-n_sample_sts:i]*np.conjugate(data[i-corr_delay-n_sample_sts:i-corr_delay]))

		amp_cc = np.absolute(xcorr)
		teta = np.angle(xcorr)
	
		teta_sts = teta[amp_cc>4e8]
		freq_offset = np.mean(teta_sts[:corr_delay])/(2*np.pi*corr_delay)
		print(freq_offset)
		import RfModel as rf
		data_2 = rf.Mixer(data, -1*freq_offset, -1*np.pi)
	
		plt.plot(xcorr.real)
		plt.plot(xcorr.imag)
		plt.plot(teta*1e9)
		plt.plot(amp_cc)
		plt.legend(['real', 'imag', 'phase', 'ampli'])
		plt.show()

		data_set = np.zeros(n_sample_sts, dtype='complex')
		xcorr = np.zeros(data.size, dtype='complex')
		sts_set = np.repeat(C.preamble_sts, n_sample_per_symb)
		phase_det = np.empty(0)
		phase_ind = np.empty(0)
		for i in range(data.size):
			data_set[1:] = data_set[:-1]
			data_set[0] = data_2[i]
			xcorr[i] = signal.correlate(data_set, sts_set, mode='valid')
			if np.abs(xcorr[i-2])>np.abs(xcorr[i]) and np.abs(xcorr[i-2])>np.abs(xcorr[i-1]) and np.abs(xcorr[i-2])>np.abs(xcorr[i-3]) and np.abs(xcorr[i-2])>np.abs(xcorr[i-4]) and np.abs(xcorr[i-2])>100000 :
				print('peak detected at = ', i-2, 'phase = ', np.angle(xcorr[i-2]))
				phase_det = np.append(phase_det, np.angle(xcorr[i-2]))
				phase_ind = np.append(phase_ind, i-2)
	
		phase_offset = np.mean(phase_det[:8])
		print(phase_offset)
		data_3 = rf.Mixer(data_2, 0, -1*phase_offset)

		#plt.plot(xcorr.real)
		#plt.plot(xcorr.imag)
		plt.plot(np.angle(xcorr))
		#plt.plot(np.abs(xcorr))
		plt.plot(phase_ind, phase_det, 'x')
		plt.legend(['real', 'imag', 'phase', 'ampli'])
		plt.show()

		return 0
	
	def MseCarrierSync(data, delay): # mean square error (MSE)
		mse = [np.square(np.absolute(data[i:i+delay]-data[i+delay:i+2*delay])).mean() for i in range(data.size-2*delay)]

		plt.plot(mse)
		plt.show()

		return mse

	def AcfCarrierSync(data, delay):
		acorr = [signal.correlate(data[i:i+delay], data[i+delay:i+2*delay], mode='valid')[0] for i in range(data.size-2*delay)]
	
		freq_offset = np.angle(acorr)/(2*np.pi*delay)
		plt.plot(freq_offset)
		#plt.plot(np.absolute(acorr))
		plt.show()

		return freq_offset

