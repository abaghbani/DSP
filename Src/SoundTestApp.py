import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import msvcrt
import sys

sys.path.insert(1, './Lib')
from Sound.DistortionCalc import Thd, ThdN, rms_harmony
from Sound.KarplusStrong import KarplusStrong
from Spectrum.freqPlot import fftPlot, specPlot
from IOs.WavFile import readWaveFile, writeWaveFile

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
				[fs, dataI, dataQ] = readWaveFile(filename)
				specPlot(dataI+1j*dataQ, fs=fs)
			
			elif c == 'a':
				[fs, dataI, dataQ] = readWaveFile(filename)
				fftPlot(dataI+1j*dataQ, fs=fs)
				
			elif c == 'c':
				fs = 1000
				t = np.arange(0,1, 1/fs)
				x = 2*np.cos(2*np.pi*100*t)+0.01*np.cos(2*np.pi*200*t)+0.005*np.cos(2*np.pi*300*t)

				#plt.plot(x)
				#plt.grid()
				#plt.show()

				#mLib.fftPlot(x, fs=1000)
				Thd(x, fs)
				ThdN(x, fs)
			
			elif c == 'w':
				sRate = 44100.0
				t = np.arange(0, 2.0, 1/sRate)
				samples = np.sin(2*np.pi*440.0*t)
				writeWaveFile('test1.wav', samples, int(sRate))

				sampleRate = 44100
				samples = KarplusStrong(1*sampleRate, sampleRate, 391)
				writeWaveFile('test2.wav', samples, int(sampleRate))


				samples -= np.mean(samples)
				windowed = samples * signal.blackmanharris(len(samples))

				f = np.fft.rfft(windowed)
				i = np.argmax(np.abs(f))
				ThdValue = rms_harmony(f, i) / np.abs(f[i])

				print('freq = %f and thd = %f' %(i, ThdValue))

				#Thd(samples, sampleRate)
				#ThdN(samples, sampleRate)

				plt.plot(np.abs(f))
				#plt.legend(['test1', 'test2'], loc='best')
				plt.grid()
				plt.show()
			
			elif c == 'x':
				break
			print('Press new command:')

	print('Exit')