import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

def fftPlot(data1, data2=None, n=1, fs=1, index=1, nperseg=2**12):
	fig = plt.figure()
	if n == 1:
		f, pow = signal.welch(data1, fs=fs, nperseg=nperseg, scaling='spectrum')
		plt.plot(f, 10.0*np.log10(pow/data1.size))
		plt.xlabel('frequency [Hz]')
		plt.ylabel('PSD [dBFS]')
		plt.grid()
	if n == 2:
		f, pow = signal.welch(data1, fs=fs, nperseg=nperseg, scaling='spectrum')
		plt.subplot(211)
		plt.plot(f, 10.0*np.log10(pow/data1.size))
		plt.xlabel('frequency [Hz]')
		plt.ylabel('PSD [dBFS]')
		plt.grid()
		f, pow = signal.welch(data2, fs=fs, nperseg=nperseg, scaling='spectrum')
		plt.subplot(212)
		plt.plot(f, 10.0*np.log10(pow/data1.size))
		plt.xlabel('frequency [Hz]')
		plt.ylabel('PSD [dBFS]')
		plt.grid()
	plt.show()
	# fig.savefig('out'+str(index)+'.png', dpi=fig.dpi)

# plot specgram of sampled data with rate 240Msps of second nyquist
def specPlot(sampleData, fs=240.0e+6):
	fc = fs
	if fs==240.0e+6 :
		dataMix = np.multiply(sampleData, np.cos((np.arange(sampleData.size)*2*np.pi*120.0e+6/240.0e+6)+0.06287))
		fc = 2380.0e+6
	else:
		dataMix = sampleData
		fc = 0
	plt.specgram(dataMix, NFFT=1024, Fc=fc, Fs=fs)
	plt.show()

