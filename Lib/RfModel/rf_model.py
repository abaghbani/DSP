import numpy as np
import scipy.signal as signal

def UpSampling(data, bw, rate, fs=1):
	data_upsampled = (rate)*data.repeat(rate)
	b = signal.firwin(141, bw/(fs/2), window = "hamming")
	return np.convolve(b, data_upsampled, mode='same')

def Mixer(data, Fmix, Phase_offset, fs=1):
	return data*np.exp( (np.arange(data.size)*2j*np.pi*(Fmix)/fs) + (1j*Phase_offset) )

def FrequencyOffset(data, offset, drift, fs=1):
	frequency_Offset = offset+drift*np.linspace(0, 1, data.size)
	return data*np.exp(1j*2*np.pi*np.cumsum(frequency_Offset/fs))

def PhaseOffset(data, offset):
	return data*np.exp(1j*offset)

def WhiteNoise(data, SNRdB):
	signal_power = np.mean(abs(data**2))
	# calculate noise power based on signal power and SNR
	sigma2 = signal_power * 10**(-SNRdB/10)
	#print ("RX Signal power: %.4f, Noise power: %.4f" % (signal_power, sigma2))
	return np.sqrt(sigma2/2) * (np.random.randn(data.size)-0.5)

def ChannelResponce(data):
	# the impulse response of the wireless channel
	channelResponse = np.array([1, 0, 0.3+0.3j])
	return np.convolve(data, channelResponse)
