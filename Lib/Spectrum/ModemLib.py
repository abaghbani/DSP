import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

class ModemLib:
	def __init__(self,coeff):
		self.mylocalcoeff = coeff
	
	def doSomeProcessing(self,data):
		data = data * self.mylocalcoeff
		return data

	def fftPlot(self, data1, data2=None, n=1, fs=1,index=1):
		fig = plt.figure()
		if n == 1:
			f, pow = signal.welch(data1, fs=fs, nperseg=2**12, window='hanning', scaling='spectrum')
			plt.plot(f, 10.0*np.log10(pow/data1.size))
			plt.xlabel('frequency [Hz]')
			plt.ylabel('PSD [dBFS]')
			plt.grid()
		if n == 2:
			f, pow = signal.welch(data1, fs=fs, nperseg=2**12, window='hanning', scaling='spectrum')
			plt.subplot(211)
			plt.plot(f, 10.0*np.log10(pow/data1.size))
			plt.xlabel('frequency [Hz]')
			plt.ylabel('PSD [dBFS]')
			plt.grid()
			f, pow = signal.welch(data2, fs=fs, nperseg=2**12, window='hanning', scaling='spectrum')
			plt.subplot(212)
			plt.plot(f, 10.0*np.log10(pow/data1.size))
			plt.xlabel('frequency [Hz]')
			plt.ylabel('PSD [dBFS]')
			plt.grid()
		plt.show()
		# fig.savefig('out'+str(index)+'.png', dpi=fig.dpi)

	#Plot frequency and phase response
	def freqResPlot(self, b,a=1,fs=1):
		w, h = signal.freqz(b,a)
		plt.subplot(211)
		plt.plot((w*fs)/(2*np.pi), 20.0*np.log10(np.abs(h)))
		plt.ylabel('Magnitude (db)')
		plt.xlabel(r'Frequency (Hz)')
		plt.title(r'Frequency response')
		plt.grid()
		plt.subplot(212)
		phase = np.unwrap(np.arctan2(np.imag(h),np.real(h)))
		plt.plot(w*fs/(2*np.pi), phase)
		plt.ylabel('Phase (radians)')
		plt.xlabel(r'Frequency (Hz)')
		plt.title(r'Phase response')
		plt.grid()
		plt.subplots_adjust(hspace=0.5)
		plt.show()

	#Plot step and impulse response
	def impz(self, b,a=1):
		l = len(b)
		impulse = np.repeat(0.,l); impulse[0] =1.
		x = np.arange(0,l)
		response = signal.lfilter(b,a,impulse)
		plt.subplot(211)
		plt.stem(x, response)
		plt.ylabel('Amplitude')
		plt.xlabel(r'n (samples)')
		plt.title(r'Impulse response')
		plt.subplot(212)
		step = np.cumsum(response)
		plt.stem(x, step)
		plt.ylabel('Amplitude')
		plt.xlabel(r'n (samples)')
		plt.title(r'Step response')
		plt.subplots_adjust(hspace=0.5)

	# plot specgram of sampled data with rate 240Msps of second nyquist
	def specPlot(self, sampleData, fs=240.0e+6):
		fc = fs
		if fs==240.0e+6 :
			dataMix = np.multiply(sampleData, np.cos((np.arange(sampleData.size)*2*np.pi*120.0e+6/240.0e+6)+0.06287))
			fc = 2380.0e+6
		else:
			dataMix = sampleData
			fc = 0
		plt.specgram(dataMix, NFFT=1024, Fc=fc, Fs=fs)
		plt.show()

	# CIC filter
	def cicFilter(self, inputStream):
		x = np.array([0]*3)
		y = np.array([0]*3)
		inSig = inputStream
		midSig = np.array([0]*(4*inSig.size))
		outSig = np.array([0]*(4*inSig.size))

		for i in range(inSig.size):
			midSig[i*4]=inSig[i]-x[0]-x[1]-x[2]
			x[2]=inSig[i]-x[0]-x[1]
			x[1]=inSig[i]-x[0]
			x[0]=inSig[i]

		for i in range(4*inSig.size):
			outSig[i]=midSig[i]+y[0]+y[1]+y[2]
			y[2]=midSig[i]+y[0]+y[1]+y[2]
			y[1]=midSig[i]+y[0]+y[1]
			y[0]=midSig[i]+y[0]
		
		return outSig

	# CORDIC (phase extraction from IQ data)
	def cordic2(self, in_x, in_y, N=7):
		# def K_vals(n):
		#	  K = []
		#	  acc = 1.0
		#	  for i in range(0, n):
		#		  acc = acc * (1.0/np.sqrt(1 + 2.0**(-2*i)))
		#		  K.append(acc)
		#	  return K
		#K = K_vals(N)
		# K = 0.6072529350088812561694
		K = 1
		
		x = in_x
		y = in_y
		beta = 0.0
		if x == 0:
			if y >= 0:
				return (y, np.pi/2.0)
			else:
				return (-y, -np.pi/2.0)
		if y == 0:
			if x >= 0:
				return (x, 0.0)
			else:
				return (-x, np.pi)
				
		if x < 0 and y < 0:
			(x, y) = (-x, -y)
			beta = -np.pi
		elif x < 0 and y > 0:
			(x, y) = (y, -x)
			beta = np.pi/2.0
		
		for i in range(0,N):
			d = 1 if y < 0 else -1
			(x,y) = (x - (d*(2**(-i))*y), (d*(2**(-i))*x) + y)
			beta = beta - (d*np.arctan(2**(-i)))
		return (x*K, beta)
