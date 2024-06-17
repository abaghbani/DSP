import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

def CostasLoop(data, alpha = 0.132, beta = 0.00932, fs = 1):
	"""
    Create fractional delay filter
    
    Parameters
    ----------
	data: input samples
	alpha, beta: to adjust, to make the feedback loop faster or slower (which impacts stability)
   
    Returns
    -------
    h : filter coefficient
    """
	
	N = len(data)
	phase = 0
	freq = 0
	# These next two params is what to adjust, to make the feedback loop faster or slower (which impacts stability)
	alpha = 0.132
	beta = 0.00932
	out = np.zeros(N, dtype=np.complex)
	freq_log = []
	for i in range(N):
		out[i] = data[i] * np.exp(-1j*phase) # adjust the input sample by the inverse of the estimated phase offset
		error = np.real(out[i]) * np.imag(out[i]) # This is the error formula for 2nd order Costas Loop (e.g. for BPSK)

		# Advance the loop (recalc phase and freq offset)
		freq += (beta * error)
		freq_log.append(freq * fs / (2*np.pi)) # convert from angular velocity to Hz for logging
		phase += freq + (alpha * error)

		# Optional: Adjust phase so its always between 0 and 2pi, recall that phase wraps around every 2pi
		while phase >= 2*np.pi:
			phase -= 2*np.pi
		while phase < 0:
			phase += 2*np.pi

	return out