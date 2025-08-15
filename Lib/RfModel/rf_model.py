import numpy as np
import scipy.signal as signal

def UpSampling(data, rate):
	data_upsampled = data.repeat(rate)
	b = signal.firwin(141, 1/rate, fs=1)
	return np.convolve(b, data_upsampled, mode='same')

def Mixer(data, Fmix, Phase_offset, fs=1):
	return data*np.exp( (np.arange(data.size)*2j*np.pi*(Fmix)/fs) + (1j*Phase_offset) )

def FrequencyOffset(data, offset, drift, fs=1):
	frequency_Offset = offset+drift*np.linspace(0, 1, data.size)
	return data*np.exp(1j*2*np.pi*np.cumsum(frequency_Offset/fs))

def PhaseOffset(data, offset):
	return data*np.exp(1j*offset)

def WhiteNoise(data, SNRdB):
	signal_power = np.mean(np.abs(data)**2)
	noise_power = signal_power * 10**(-SNRdB/10)
	noise = np.random.normal(0, np.sqrt(noise_power/2), data.size) + 1j * np.random.normal(0, np.sqrt(noise_power/2), data.size)

	return noise

def ChannelResponce(data):
	# the impulse response of the wireless channel
	channelResponse = np.array([1, 0, 0.3+0.3j])
	return np.convolve(data, channelResponse)
