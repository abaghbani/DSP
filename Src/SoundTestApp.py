import numpy as np
import scipy.signal as signal
import scipy.fft as fft
import matplotlib.pyplot as plt
import msvcrt
import os
import logging as log

import Spectrum as sp
import Filter as fd
import IOs
import Common as myLib

import Sound as sd
import sounddevice as sdev

if __name__=="__main__":
	
	print('S: Specgram of sampled data')
	print('H: Histogram to Jpeg of sampled data')
	print('A: FFT plot')
	print('C: THD test')
	print('W: Wav file generator')
	print('X: Exit')
	print('>> ')

	filename = '../Samples/Sound/' + 'test1.wav'
	
	while True:
		if msvcrt.kbhit():
			c = msvcrt.getch().decode("utf-8")
			print(c)
			c = c.lower()
			if c == 's':
				[fs, dataI, dataQ] = IOs.readWaveFile(filename)
				sp.specPlot(dataI+1j*dataQ, fs=fs)
				
			elif c == 'a':
				[fs, dataI, dataQ] = IOs.readWaveFile(filename)
				sp.fftPlot(dataI+1j*dataQ, fs=fs)
				
			elif c == 'c':
				fs = 1000
				t = np.arange(0,1, 1/fs)
				x = 2*np.cos(2*np.pi*100*t)+0.01*np.cos(2*np.pi*200*t)+0.005*np.cos(2*np.pi*300*t)

				#plt.plot(x)
				#plt.grid()
				#plt.show()

				#mLib.fftPlot(x, fs=1000)
				sd.Thd(x, fs)
				sd.ThdN(x, fs)
			
			elif c == 'w':
				sRate = 44100.0
				t = np.arange(0, 2.0, 1/sRate)
				samples = np.sin(2*np.pi*440.0*t)
				IOs.writeWaveFile('test1.wav', samples, int(sRate))

				sampleRate = 44100
				samples = sd.KarplusStrong(1*sampleRate, sampleRate, 391)
				IOs.writeWaveFile('test2.wav', samples, int(sampleRate))


				samples -= np.mean(samples)
				windowed = samples * signal.blackmanharris(len(samples))

				f = np.fft.rfft(windowed)
				i = np.argmax(np.abs(f))
				ThdValue = sd.rms_harmony(f, i) / np.abs(f[i])

				print('freq = %f and thd = %f' %(i, ThdValue))

				#Thd(samples, sampleRate)
				#ThdN(samples, sampleRate)

				plt.plot(np.abs(f))
				#plt.legend(['test1', 'test2'], loc='best')
				plt.grid()
				plt.show()
			elif c == 't':
				#fs, dataI, dataQ = IOs.readWaveFile('D:/Documents/Samples/piano/UprightPianoKW-small-SFZ-20190703/samples/A5vH.wav')
				#fs, dataI, dataQ = IOs.readWaveFile('D:/Documents/Samples/piano/Piano.A1.wav')
				#print('sample rate : ', fs)

				fs = 44100.0
				freq = sd.ChromaticTone(48)
				data = sd.KarplusStrong(int(2*fs), fs, freq)
				#data = np.sin(2*np.pi*freq/fs*np.arange(2*fs))

				#print(f'freq of single tune = {freq}')
				#plt.plot(data)
				#plt.grid()
				#plt.show()

				#sdev.play(data, fs)
				#status = sdev.wait()

				#b = fd.RemezFilter(301, 250, 400, fs)
				#data_flt = np.convolve(b, dataI, 'same')

				#data_shift = data*np.sin(np.arange(data.size)*2*np.pi*420.0/fs)
				#b = fd.RemezFilter(301, 50, 200, fs)
				#data_shift_flt = np.convolve(b, data_shift, 'same')
				
				#plt.plot(dataI)
				#plt.plot(data_shift_flt)
				#plt.grid()
				#plt.show()
				n_point = 2**16
				data_ff = fft.dct(data[:int((data.size//n_point)*n_point)].reshape(-1, n_point))
				#data_ff = np.fft.fft(data)
				print(data_ff.shape, type(data_ff[0]))
				for i in range(data_ff.shape[0]):
					plt.plot((np.arange(n_point)/n_point) * fs/2, np.abs(data_ff[i]))
					plt.show()

				sp.fftPlot(data, fs=fs, nperseg=2**16)
				#sp.fftPlot(data_shift_flt[4::10], fs=fs/10, nperseg=2**13)

			elif c == 'x':
				break
			print('Press new command:')

	print('Exit')