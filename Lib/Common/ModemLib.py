import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

class ModemLib:
	#def __init__(self, inVar):
	#	self.mylocalcoeff = inVar
	
	def doSomeProcessing(self,data):
		data = data * self.mylocalcoeff
		return data

	def plot_amp_pha(self, w, h, fs, label_text, show_flag = False):
		plt.subplot(211)
		plt.plot((w*fs)/(2*np.pi), 20.0*np.log10(np.abs(h)), label=label_text)
		plt.subplot(212)
		plt.plot((w*fs)/(2*np.pi), np.angle(h), label=label_text)
		if show_flag:
			plt.subplot(211)
			plt.ylabel('Magnitude (db)')
			plt.xlabel(r'Frequency (MHz)')
			plt.subplot(212)
			plt.ylabel('phase (rad)')
			plt.xlabel(r'Frequency (MHz)')
			plt.title(r'Frequency response')
			plt.legend()
			plt.grid()
			plt.show()

	def plot_amp(self, w, h, fs, label_text, show_flag = False):
		plt.plot((w*fs)/(2*np.pi), 20.0*np.log10(np.abs(h)), label=label_text)
		if show_flag:
			plt.ylabel('Magnitude (db)')
			plt.xlabel(r'Frequency (MHz)')
			plt.title(r'Frequency response')
			plt.legend()
			plt.grid()
			plt.show()

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

	# CIC filter
	def cicFilter(self, inSig):
		x = np.zeros(3)
		y = np.zeros(3)
		midSig = np.zeros(4*inSig.size)
		outSig = np.zeros(4*inSig.size)

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

	def Downsampling_CIC_n3(self, inSig, down_rate=4):
		x = np.zeros(3, dtype=inSig.dtype)
		y = np.zeros(3, dtype=inSig.dtype)
		midSig = np.zeros(inSig.size, dtype=inSig.dtype)
		outSig = np.zeros(inSig.size//down_rate, dtype=inSig.dtype)

		for i in range(inSig.size):
			midSig[i]=inSig[i]+x[0]+x[1]+x[2]
			x[2]=inSig[i]+x[0]+x[1]+x[2]
			x[1]=inSig[i]+x[0]+x[1]
			x[0]=inSig[i]+x[0]
		
		for i in range(inSig.size//down_rate):
			outSig[i]=midSig[i*down_rate]-y[0]-y[1]-y[2]
			y[2]=midSig[i*down_rate]-y[0]-y[1]
			y[1]=midSig[i*down_rate]-y[0]
			y[0]=midSig[i*down_rate]
		
		return outSig

	def Downsampling_CIC_n4(self, inSig, down_rate):
		# N: number of stage (for both decimator and integrator), by Xilinx Vivado between 3 and 6 (in this function = 4)
		# R: decimation rate, between 4 and 8192 (int this function = down_rate)
		# M: differential delay, between 1 and 2 (usually is 1, in this function = 1)
		# filter gain = (MR)^N = (down_rate)^4
		x = np.zeros(4, dtype=inSig.dtype)
		y = np.zeros(4, dtype=inSig.dtype)
		midSig = np.zeros(inSig.size, dtype=inSig.dtype)
		outSig = np.zeros(inSig.size//down_rate, dtype=inSig.dtype)

		for i in range(inSig.size):
			midSig[i]=inSig[i]+x[0]+x[1]+x[2]+x[3]
			x[3]=inSig[i]+x[0]+x[1]+x[2]+x[3]
			x[2]=inSig[i]+x[0]+x[1]+x[2]
			x[1]=inSig[i]+x[0]+x[1]
			x[0]=inSig[i]+x[0]
		
		for i in range(inSig.size//down_rate):
			outSig[i]=midSig[i*down_rate]-(y[0]+y[1]+y[2]+y[3])
			y[3]=midSig[i*down_rate]-(y[0]+y[1]+y[2])
			y[2]=midSig[i*down_rate]-(y[0]+y[1])
			y[1]=midSig[i*down_rate]-y[0]
			y[0]=midSig[i*down_rate]
		
		return outSig

	# CORDIC (phase extraction from IQ data)
	def cordic_rp(self, in_x, in_y, N=7):
		# def cordic_gain(n):
		#	  K = []
		#	  acc = 1.0
		#	  for i in range(0, n):
		#		  acc = acc * (1.0/np.sqrt(1 + 2.0**(-2*i)))
		#		  K.append(acc)
		#	  return K
		
		# K = cordic_gain(N)
		#K = 0.6072529350088812561694
		K = 1
		
		x = float(in_x)
		y = float(in_y)
		beta = 0.0
		if x == 0:
			return (y, np.pi/2.0) if y >= 0 else (-y, -np.pi/2.0)
		if y == 0:
			return (x, 0.0) if x >= 0 else (-x, np.pi)
		if x < 0:
			beta = -np.pi if y<0 else np.pi
			(x, y) = (-x, -y)
		
		for i in range(0,N):
			d = 1.0 if y < 0 else -1.0
			(x,y) = (x - (d*(2**(-i))*y), (d*(2**(-i))*x) + y)
			beta = beta - (d*np.arctan(2**(-i)))
		return x, beta

	def cordic_xy(self, phase, N=16):
		
		beta = 1.0 * phase
		x = 1.
		y = 0.
		k = 0.6072529350088812561694
		a_x = 1
		a_y = 1
		beta %= 2*np.pi
		if beta > np.pi:
			beta -= 2*np.pi
		if beta < -np.pi:
			beta += 2*np.pi

		if beta > np.pi/2:
			beta = np.pi - beta
			a_x = -1.
		elif beta < -np.pi/2:
			beta = -np.pi - beta
			a_x = -1.

		for i in range(0,N):
			d = 1.0 if beta > 0 else -1.0
			(x,y) = (x - (d*(2**(-i))*y), (y + d*(2**(-i))*x))
			beta = beta - (d*np.arctan(2**(-i)))
			#print(f'iter={i}-  : d={d:2.0f} -- beta={beta:.4f} -- x={x} -- y={y}')

		return x*a_x*k, y*a_y*k

