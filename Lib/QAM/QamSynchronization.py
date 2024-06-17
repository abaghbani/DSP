import numpy as np
import scipy.signal as signal
import scipy.stats as stats
import matplotlib.pyplot as plt

import RfModel as rf

from .Constant import *
C = Constant()

'''

ISI : Inter Symbol Interference
ICI : Inter Carrier Interference

STO : Symbol Timing Offset
CFO : Carrier Frequency Offset
CPO : Carrier Phase Offset

LTF : Long Training Field
STF : Short Training Field

CFS : Coarse Frequency Synchronization
FFS : Fine Frequency Synchronization

CPS : Coarse Phase Synchronization
FPS : Fine Phase Synchronization

CTS : Coarse Time Synchronization
FTS : Fine Time Synchronization

'''

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
	sts_freq_detection=0
	lts_detection = 0
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
		
			if mag_peak.size>=7 and np.all(np.abs(np.diff(mag_peak)[-6:] - 4*symb_width) < 4) and np.all(mag[mag_peak[-6:-1]]/mag[mag_peak[-7:-2]]<1.2) and sts_freq_detection==0:
				sts_index = mag_peak[-1]
				freq_off += np.mean(phase_diff[sts_index-int(6*4*symb_width)+symb_width:sts_index-symb_width])
				sts_freq_detection=1
				print(f'Sts detection(freq):  index={sts_index}, freq_off = {freq_off*1000*sample_rate/(2*np.pi):.3f} KHz')

			if mag_peak.size>=9 and np.all(np.abs(np.diff(mag_peak)[-8:] - 4*symb_width) < 4) and np.all(mag[mag_peak[-8:-1]]/mag[mag_peak[-9:-2]]<1.2):
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
			if i> (sts_index+4*symb_width) and (i-sts_index)%symb_width == 0 and lts_seq==-1:
				lts_int_det = np.round(phase[i]*17/(2*np.pi)).astype('int')
				for k in range(16):
					if lts_int_det == C.lts_int[k][int((i-sts_index)/symb_width)-5]:
						lts_symb_count[k] += 1
						if lts_symb_count[k] >= 10:
							lts_seq = k
							print(f'LTS sequence detection result: {lts_seq=}')
				print(f'counters = {lts_symb_count}')

			## LTS detection XCP
			if LTS_detection_type=="XCP":

				if i == (sts_index+int((4+17+15)*symb_width)):
					freq_off_avg = np.mean(phase_diff[i-14*symb_width:i+1])
					freq_off += freq_off_avg
					agc_rate = np.mean(ampli[i-5*symb_width:i+1:symb_width])
					print(f'Lts detection (freq): index={i}, freq_off = {freq_off*1000*sample_rate/(2*np.pi):.3f} KHz, {agc_rate=}, lts_freq_off = {freq_off_avg*1000*sample_rate/(2*np.pi):.3f} KHz')
				
				elif i == (sts_index+int((4+17+16)*symb_width)):
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
				
	
	#plt.plot(phase_diff*1000*sample_rate/(2*np.pi), label = 'diff_sts')
	#plt.plot(phase_diff*1000*sample_rate/(2*np.pi), label = 'diff_lts')
	#plt.plot(phase, label = 'phase')

	# plt.plot(mag, label = 'mag')
	# plt.plot(mag_diff, label = 'mag_diff')
	# # plt.plot(mag_peak, mag[mag_peak], 'ro')
	# # plt.plot(phase_diff*1.0e2, label = 'freq')
	# plt.plot(phase, label = 'phase')
	# plt.plot(np.hstack([np.zeros(sts_index+4*symb_width+8), np.tile(np.repeat(C.preamble_lts_phase[lts_seq], symb_width), 2)]), label = 'lts_int')
	# plt.legend()
	# plt.grid()
	# plt.show()
	
	#plt.plot(data_sync[1350:].real, data_sync[1350:].imag, 'ro')
	#plt.show()

	return 	data_sync[lts_index:]/agc_rate

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

