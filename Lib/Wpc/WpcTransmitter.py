import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

from Spectrum.freqPlot import fftPlot, specPlot
from Filter.filterDesign import LowpassFilter, HighpassFilter, MidpassFilter
from Wpc import WpcCommon as wpc


def WpcPacketGenerator(payload_size):
	raw_payload = (np.random.rand(payload_size) >= 0.5).astype('int')
	payload = wpc.ManchesterEncoder(np.concatenate((np.ones(16), np.zeros(1), raw_payload, np.ones(2)), axis=None))
	payload = np.concatenate((np.zeros(5), payload, np.zeros(50)), axis=None)

	return payload


def WpcModulation(data, bit_rate, modulation_type, modulation_index, amplitude, freq, fs):
	data_up = np.repeat(data, fs/bit_rate)
	n = np.arange(data_up.size)
	data_up[data_up == 0] = modulation_index
	if modulation_type == 'ask':
		sig = amplitude*np.multiply(data_up, np.sin(2*np.pi*n*freq/fs+np.pi/4))
	elif modulation_type == 'fsk':
		sig = amplitude*np.sin(2*np.pi*np.multiply(n, data_up*freq)/fs+np.pi/4)
	else:
		sig = 0

	return sig

