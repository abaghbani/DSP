import numpy as np
import matplotlib.pyplot as plt
import msvcrt

import Spectrum as sp
import Pluto as pl

if __name__=="__main__":
	mPluto = pl.Pluto(0)
	mPluto.info()
	print('------------------------')

	
	print('R: Receiving')
	print('T: Transmitting')
	print('W: Receiving/Transmitting')
	print('d: debugging')
	print('X: Exit')
	print('>> ')
	
	while True:
		if msvcrt.kbhit():
			c = msvcrt.getch().decode("utf-8")
			print(c)
			c = c.lower()

			if c == 'r':
				[samples, fs] = mPluto.Read(5.0e6, 5.0e6, 868.0e6, 50.0, int(2e6))
				print('sample freq: ', fs, 'sample size: ', samples.size, 'sample min/max: ', samples.min(), samples.max())
				sp.fftPlot(samples.real, n=1, fs=fs)
				np.save('testPluto', samples)

			elif c == 't':
				t = np.arange(10000)/10.0e6
				samples = 0.5*np.exp(2.0j*np.pi*100e3*t)
				samples *= 2**14 # The PlutoSDR expects samples to be between -2^14 and +2^14, not -1 and +1 like some SDRs

				fs = mPluto.Write(10.0e6, 10.0e6, 868.0e6, -30.0, samples, 3)
				sp.fftPlot(samples.real, n=1, fs=fs)

			elif c == 'w':
				t = np.arange(10000)/3.0e6
				samples = 0.99*np.exp(2.0j*np.pi*120e3*t)
				#samples = np.array((np.random.rand(2000) >= 0.5)*0.5)
				#samples = samples.repeat(200)
				samples = np.concatenate((np.zeros(5000), samples, np.zeros(5000)), axis=None)
				
				samples *= 2**14 # The PlutoSDR expects samples to be between -2^14 and +2^14, not -1 and +1 like some SDRs

				[rxSamples, fs] = mPluto.ReadWrite(3.0e6, 3.0e6, 868.0e6, -30.0, 50.0, samples, int(2e6))
				sp.fftPlot(rxSamples.real, n=1, fs=fs)
				sp.specPlot(rxSamples, fs=fs)
				#writeWaveFile(rxSamples, int(fs), 'test1.wav')
				np.save('testPluto', rxSamples)

			elif c == 'd':
				rxSamples = np.load('testPluto.npy')
				print('data size = ', rxSamples.shape)
				fs = 3.0e6
				rxSamples = np.multiply(rxSamples, np.exp((np.arange(rxSamples.size)*(-2j)*np.pi*120.0e3/fs)+0.06287))

				sp.fftPlot(rxSamples.real, n=1, fs=fs)
				sp.specPlot(rxSamples, fs=fs)
				plt.plot(rxSamples[:10000])
				plt.show()

			elif c == 's':
				mPluto.DDS()
			elif c == 'p':
				mPluto.DDS_stop()

			elif c == 'x':
				break
			print('Press new command:')
	print('Exit from Pluto command.')
