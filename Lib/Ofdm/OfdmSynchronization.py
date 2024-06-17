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

def StsCarrierSync(data, sample_rate):
	baseband_symb_rate = 20 # symbol rate is 20 MSymb/s
	n_sample_per_symb = int(sample_rate/baseband_symb_rate)
	sts_samples = np.repeat(C.preamble_sts_time, n_sample_per_symb)
	sts_cc_amplitude = float(np.abs(signal.correlate(sts_samples, sts_samples, mode='valid')))
	

	#xcorr = np.zeros(data.size, dtype=np.complex)
	#for i in range(data.size-2*16*n_sample_per_symb):
	#	xcorr[i] = signal.correlate(data[i+16*n_sample_per_symb:i+2*16*n_sample_per_symb], data[i:i+16*n_sample_per_symb], mode='valid')
	#amp_cc = np.absolute(xcorr)
	#teta = np.angle(xcorr)
	#peaks = signal.find_peaks_cwt(amp_cc,  widths=int(8*16*n_sample_per_symb))#, height =500e6 , prominence=11e6, width=int(0.5*n_sample_per_symb))#, distance=int((C.preamble_sts_time.size-0.5)*n_sample_per_symb))
	#plt.plot(amp_cc)
	#plt.plot(peaks, amp_cc[peaks], 'x')
	#plt.plot(teta)
	#plt.show()

	#######################
	## first path
	#######################
	xcorr = signal.correlate(data, sts_samples, mode='full')
	teta = np.angle(xcorr)
	amp_cc = np.absolute(xcorr)
	peaks, _ = signal.find_peaks(amp_cc, prominence=2000*sts_cc_amplitude, width=int(.5*n_sample_per_symb), distance=int((C.preamble_sts_time.size-0.5)*n_sample_per_symb))
	#plt.plot(xcorr.real)
	#plt.plot(xcorr.imag)
	#plt.plot(amp_cc)
	#plt.plot(teta)
	#plt.plot(peaks, amp_cc[peaks], 'x')
	#plt.plot(peaks, teta[peaks], 'x')
	#plt.legend(['real', 'imag', 'ampli', 'phase'])
	#plt.show()

	peaks = peaks[0:10]
	xcorr_phase = np.zeros(peaks.size, dtype=float)
	xcorr_phase[0] = teta[peaks[0]]
	for i in range(1, peaks.size):
		xcorr_phase[i] = teta[peaks[i]] if np.abs(teta[peaks[i]] - xcorr_phase[i-1]) < 4 else teta[peaks[i]]+2*np.pi*(np.abs(teta[peaks[i]] - xcorr_phase[i-1])//(2*np.pi) + 1)*np.sign(xcorr_phase[i-1]-teta[peaks[i]])
	res = stats.linregress(peaks, xcorr_phase)
	freq_offset = res.slope/(2*np.pi)
	phase_offset = res.intercept 
	gain_1 = np.mean(amp_cc[peaks])/sts_cc_amplitude
	print(f'stage 1 : freq_offset= {freq_offset*sample_rate*1000:.3f} KHz -- {phase_offset=:.2f}')

	plt.plot(teta, label='phase')
	plt.plot(peaks, teta[peaks], 'x')
	plt.plot(peaks, xcorr_phase, 'o')
	plt.plot(peaks, res.intercept+res.slope*peaks, label='reg')
	plt.legend()
	plt.grid()
	plt.show()
	
	data_mixer_1 = rf.Mixer(data, -freq_offset, -phase_offset)

	#######################
	## Second path
	#######################

	xcorr = signal.correlate(data_mixer_1, sts_samples, mode='full')
	teta = np.angle(xcorr)
	amp_cc = np.absolute(xcorr)
	peaks, _ = signal.find_peaks(amp_cc, prominence=0.9*gain_1*sts_cc_amplitude, width=int(.5*n_sample_per_symb), distance=int((C.preamble_sts_time.size-0.5)*n_sample_per_symb))
				
	peaks = peaks[:10]
	phase_offset = np.mean(teta[peaks])
	gain = np.mean(amp_cc[peaks])/sts_cc_amplitude
	print(f'stage 2 : {phase_offset=:.2f} , {gain=:.2f}, {peaks.size}')
	data_mixer_2 = rf.Mixer(data_mixer_1, 0, -phase_offset)/gain

	#plt.plot(xcorr.real)
	#plt.plot(xcorr.imag)
	#plt.plot(amp_cc)
	#plt.plot(peaks, amp_cc[peaks], 'x')
	plt.plot(teta)
	plt.plot(peaks, teta[peaks], 'x')
	plt.plot(peaks, phase_offset+0*peaks)
	plt.show()

	return data_mixer_2[peaks[-1]:]

def LtsCarrierSync(data, sample_rate):
	baseband_symb_rate = 20 # symbol rate is 20 MSymb/s
	n_sample_per_symb = int(sample_rate/baseband_symb_rate)

	#xcorr = np.zeros(data.size, dtype=np.complex)
	#for i in range(data.size-2*64*n_sample_per_symb):
	#	xcorr[i] = signal.correlate(data[i+64*n_sample_per_symb:i+2*64*n_sample_per_symb], data[i:i+64*n_sample_per_symb], mode='valid')
	#amp_cc = np.absolute(xcorr)
	#teta = np.angle(xcorr)
	#plt.plot(amp_cc)
	#plt.plot(teta)
	#plt.show()
	
	freq_offset  = float(np.angle(signal.correlate(data[32+64*n_sample_per_symb:32+2*64*n_sample_per_symb], data[32:32+64*n_sample_per_symb], mode='valid'))/(2*np.pi*64*n_sample_per_symb))
	data_mixer = rf.Mixer(data, -freq_offset, 0)
	print(f'stage 3 : freq_offset= {freq_offset*sample_rate*1000:.3f} KHz\n\n')

	return data_mixer[160*n_sample_per_symb:]

def LtsCarrierSync_xcorr(data, sample_rate):
	baseband_symb_rate = 20 # symbol rate is 20 MSymb/s
	n_sample_per_symb = int(sample_rate/baseband_symb_rate)
	lts_samples = np.repeat(C.preamble_lts_time, n_sample_per_symb)
	lts_cc_amplitude = float(np.abs(signal.correlate(lts_samples, lts_samples, mode='valid')))

	xcorr = signal.correlate(data, lts_samples, mode='full')
	amp_cc = np.absolute(xcorr)
	teta = np.angle(xcorr)
	peaks, _ = signal.find_peaks(amp_cc, height=0.9*lts_cc_amplitude, distance=int(lts_samples.size*0.8))
	
	#res = stats.linregress(peaks[0:3], teta[peaks[0:3]])
	#freq_offset = res.slope/(2*np.pi)
	#phase_offset = res.intercept 
	freq_offset  = (teta[peaks[1]]-teta[peaks[0]])/(2*np.pi*lts_samples.size)
	phase_offset = (teta[peaks[0]]+teta[peaks[1]])/2
	
	data_mixer_1 = rf.Mixer(data, -freq_offset/2, -phase_offset)
	
	print(f'stage 3 : freq_offset= {freq_offset*sample_rate*1000:.3f} KHz -- {phase_offset=:.2f}\n\n')

	plt.plot(amp_cc)
	plt.plot(peaks, amp_cc[peaks], 'x')
	plt.plot(teta)
	plt.plot(peaks, teta[peaks], 'x')
	plt.plot(peaks[:2], phase_offset+freq_offset*2*np.pi*peaks[:2], label='reg')
	plt.show()

	return data[peaks[1]:], data_mixer_1[peaks[1]:]
