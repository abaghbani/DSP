import numpy as np
import matplotlib.pyplot as plt

def DelayFilter(n, delay):
	"""
    Create fractional delay filter
    
    Parameters
    ----------
	delay: fractional delay, in samples
	n: number of taps
   
    Returns
    -------
    h : filter coefficient
    """       
	
	h = np.sinc(np.arange(-n//2, n//2) - delay)	# calc filter coef
	h *= np.hamming(n)							# window the filter to make sure it decays to 0 on both sides
	h /= np.sum(h)								# normalize to get unity gain
	
	return h
