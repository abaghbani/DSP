import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

def RemezFilter(n, f_cut1, f_cut2, fs):
	return signal.remez(n, np.array([0., f_cut1/(fs/2), f_cut2/(fs/2), 0.5]), [1, 1e-4])
