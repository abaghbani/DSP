import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

def MuellerMullerClockRecovery(data):
	index = 700
	mu = 0.5
	w = 100.0
	mu_gain = 0.175
	w_gain = 0.25 * mu_gain * mu_gain
	mm_val = 0.0
	dataIn = data
	out = np.zeros(2, dtype='float')
	mm_error = np.zeros(0, dtype='float')
	mu_error = np.zeros(0, dtype='float')
	valid = np.zeros(dataIn.size, dtype='int')
	while index < dataIn.size:
		out = np.append(out, dataIn[index])
		valid[index] = 1
		#mm_val = Clip(Sign(out[-2])*out[-1]-Sign(out[-1])*out[-2], 10)
		mm_val = (Sign(out[-2])*out[-1]-Sign(out[-1])*out[-2])*0.0005
		mm_error = np.append(mm_error, mm_val)

		w += w_gain*mm_val
		w = 100 + Clip(w-100, 10)
		mu += w  + mu_gain*mm_val
		mu_error = np.append(mu_error, mu)

		index += int(np.floor(mu))
		mu -= np.floor(mu)

	print([int(dd>=0) for dd in out])

	return out

def MuellerMullerClockRecovery_v2(samples):
	mu = 0 # initial estimate of phase of sample
	out = np.zeros(len(samples) + 10, dtype=np.complex)
	out_rail = np.zeros(len(samples) + 10, dtype=np.complex) # stores values, each iteration we need the previous 2 values plus current value
	i_in = 0 # input samples index
	i_out = 2 # output index (let first two outputs be 0)
	while i_out < len(samples) and i_in < len(samples):
		out[i_out] = samples[i_in + int(mu)] # grab what we think is the "best" sample
		out_rail[i_out] = int(np.real(out[i_out]) > 0) + 1j*int(np.imag(out[i_out]) > 0)
		x = (out_rail[i_out] - out_rail[i_out-2]) * np.conj(out[i_out-1])
		y = (out[i_out] - out[i_out-2]) * np.conj(out_rail[i_out-1])
		mm_val = np.real(y - x)
		mu += sps + 0.3*mm_val
		i_in += int(np.floor(mu)) # round down to nearest int since we are using it as an index
		mu = mu - np.floor(mu) # remove the integer part of mu
		i_out += 1 # increment output index
	out = out[2:i_out] # remove the first two, and anything after i_out (that was never filled out)

	return out
