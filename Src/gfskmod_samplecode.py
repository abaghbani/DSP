import numpy as np
import scipy.fftpack as fft
import scipy.signal as signal
import matplotlib.pyplot as plt


packet_frame = np.array([0xaa, 0xaa, 0xD6, 0xBE, 0x89, 0x8E]*4, dtype=np.uint8)
bit_frame = (np.unpackbits(packet_frame, bitorder='little').astype(np.int8))-0.5
over_sample_rate=16
frequency_sample = bit_frame.repeat(over_sample_rate)

##################################
## Gaussian filter
##################################
def gaussianFlt(over_sample_rate, GfskBT=0.5):
    t = np.linspace(-1, 1, 6*int(over_sample_rate)+1)
    alpha = np.sqrt(np.log(2)/2)/GfskBT
    gaussianFlt = (np.sqrt(np.pi)/alpha)*np.exp(-(t*np.pi/alpha)**2)
    gaussianFlt /= np.sum(gaussianFlt)
    return gaussianFlt

##################################
## Modulation
##################################
frequency_sig = np.convolve(gaussianFlt(over_sample_rate), frequency_sample, 'same')
phase_sig = signal.lfilter(1, np.array([1, -1]), frequency_sig)

plt.plot(gaussianFlt(over_sample_rate))
plt.show()
plt.plot(frequency_sample)
plt.plot(frequency_sig)
plt.show()

freq = np.diff(phase_sig)
plt.plot(freq)
plt.grid()
plt.show()

