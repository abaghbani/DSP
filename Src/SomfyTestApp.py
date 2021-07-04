import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import msvcrt
import sys

sys.path.insert(1, './Lib')
import Spectrum.ModemLib as modemLib
import Spectrum.Histogram2jpeg as hist
import IOs.WavFile as wav
import Somfy.SomfyDemodulation as somfy
import ClockRecovery.ClockRecovery as cr

if __name__=="__main__":

	mLib = modemLib.ModemLib(0)
	fileName =  '../Samples/Somfy/' + 'SDRSharp_20210407_200432Z_433200000Hz_IQ.wav'

	print('R: Read sample file')
	print('B: Remote Keylees Entry (BMW@868MHz)')
	print('K: Somfy demod')
	print('T: Somfy extract data')
	print('S: Specgram of sampled data')
	print('N: Hormann sampled data')
	print('H: Histogram to Jpeg of sampled data')
	print('A: FFT plot')
	print('X: Exit')
	print('>> ')

	while True:
		if msvcrt.kbhit():
			c = msvcrt.getch().decode("utf-8")
			print(c)
			c = c.lower()

			if c == 'r':
				[fs, dataI, dataQ] = wav.readWaveFile(fileName)
				print('Sampling freq = ', fs)
				print(dataI.shape, dataQ.shape)
				dataI = dataI[int(4.0e6):int(8.5e6)]
				dataQ = dataQ[int(4.0e6):int(8.5e6)]

				data = dataI+1j*dataQ
				mLib.fftPlot(data, n=1, fs=fs)
				mLib.specPlot(data, fs=fs)


			elif c == 'b':
				fileName_local = '../Samples/RKE_BMW/' + 'SDRSharp_20210501_131211Z_868000000Hz_IQ.wav'
				[fs, dataI, dataQ] = wav.readWaveFile(fileName_local)
				print('Sampling freq = ', fs)
				print(dataI.shape, dataQ.shape)
				
				dataI = dataI[int(4.2e6):int(8.5e6)]
				dataQ = dataQ[int(4.2e6):int(8.5e6)]

				data = dataI+1j*dataQ
				# mLib.fftPlot(data, n=1, fs=fs)
				mLib.specPlot(data, fs=fs)

				Bw = 100.0e3
				[dataI, dataQ, fs] = somfy.SomfyFilterBank(dataI, dataQ, fs, Bw=Bw, fMix=-400.0e3, downSamplingRate=1)
				# np.save('test4_somfy', [dataI, dataQ])
				# SomfyDemod(dataI, dataQ, fs)
				# mLib.fftPlot(dataI+1j*dataQ, n=1, fs=fs)
				# mLib.specPlot(dataI+1j*dataQ, fs=fs)

				phase = np.arctan2(dataQ, dataI)
				freq = np.diff(phase)
				freq[freq>(1.0*np.pi)] -= 2*np.pi 
				freq[freq<(-1.0*np.pi)] += 2*np.pi 
				freq[freq>(0.5*np.pi)] -= np.pi 
				freq[freq<(-0.5*np.pi)] += np.pi 
				# plt.plot(phase)
				plt.plot(freq)
				plt.legend(['phase', 'freq'])
				plt.show()

				np.save('bmw_key_2', freq)
			elif c == 'v':
				
				data = np.load('bmw_key_2.npy')

				# data = data[94000:]
				data = data[550000:850000]
				# data = data[3550000:3900000]

				## period estimation :
				data_digi = cr.digitized(data)
				period = cr.period_calc(data_digi)
				file = open("period.txt","w+")
				file.write(f'period array: {[v for v in period[0]]}')
				file.close
				print([v for v in period[0][:100]])
				
				## period
				period = 95

				bit_out, sampled_data = cr.Early_late(data, period = period, delta=5)

				file = open("bmw_key_out.txt","w+")
				file.write(f'extracted data is: {[v for v in bit_out]}')
				file.close

				print([v for v in bit_out])
				plt.plot(data)
				plt.plot(sampled_data, '.')
				plt.show()



			elif c == 'k':
				fileName_local = '../Samples/Somfy/' + 'SDRSharp_20210612_204730Z_433200000Hz_IQ.wav'
				[fs, dataI, dataQ] = wav.readWaveFile(fileName_local)
				print('Sampling freq = ', fs)
				print(dataI.shape, dataQ.shape)

				dataI = dataI[int(3.0e6):int(5.0e6)]
				dataQ = dataQ[int(3.0e6):int(5.0e6)]

				data = dataI+1j*dataQ
				mLib.fftPlot(data, n=1, fs=fs)
				mLib.specPlot(data, fs=fs)

				Bw = 5.0e3
				# [dataI, dataQ, fs] = SomfyFilterBank(dataI, dataQ, fs, Bw=Bw, fMix=-270.0e3, downSamplingRate=40)
				[dataI, dataQ, fs] = somfy.SomfyFilterBank(dataI, dataQ, fs, Bw=Bw, fMix=-210.0e3, downSamplingRate=40)
				np.save('test4_somfy', [dataI, dataQ])
				somfy.SomfyDemod(dataI, dataQ, fs)

			elif c == 't':
				[dataI, dataQ] = np.load('test4_somfy.npy')
				fs = 1.0e5
				somfy.SomfyDemod(dataI, dataQ, fs)

			elif c == 'n':
				# fileName_local = '../Samples/Hormann/' + 'SDRSharp_20210330_201642Z_867000000Hz_IQ_fromHormannTx.wav'
				fileName_local = '../Samples/Hormann/' + 'SDRSharp_20210612_153401Z_868000000Hz_IQ.wav'

				
				[fs, dataI, dataQ] = wav.readWaveFile(fileName_local)
				print('Sampling freq = ', fs)
				print(dataI.shape, dataQ.shape)

				# dataI = dataI[int(8.0e6):int(12.0e6)]
				# dataQ = dataQ[int(8.0e6):int(12.0e6)]
				dataI = dataI[int(4.0e6):int(8.0e6)]
				dataQ = dataQ[int(4.0e6):int(8.0e6)]
				
				# data = dataI+1j*dataQ
				# mLib.fftPlot(data, n=1, fs=fs)
				# mLib.specPlot(data, fs=fs)

				# [dataI, dataQ, fs] = SomfyFilterBank(dataI, dataQ, fs, Bw=20.0E3, fMix=-1.305e6, downSamplingRate=4)
				[dataI, dataQ, fs] = somfy.SomfyFilterBank(dataI, dataQ, fs, Bw=20.0E3, fMix=-0.305e6, downSamplingRate=1)
				data = somfy.HormannDemod(dataI, dataQ, fs)
				np.save('Hormann_data_byMain', data)

			elif c == 'h':
				# fileName_local = '../Samples/Hormann/' + 'SDRSharp_20210415_170122Z_868000000Hz_IQ_fromCC1101.wav'
				fileName_local = '../Samples/Hormann/' + 'SDRSharp_20210419_202554Z_868000000Hz_IQ.wav'
				[fs, dataI, dataQ] = wav.readWaveFile(fileName_local)
				print('Sampling freq = ', fs)
				print(dataI.shape, dataQ.shape)

				dataI = dataI[int(0.0e6):int(7.0e6)]
				dataQ = dataQ[int(0.0e6):int(7.0e6)]

				# data = dataI+1j*dataQ
				# mLib.fftPlot(data, n=1, fs=fs)
				# mLib.specPlot(data, fs=fs)

				[dataI, dataQ, fs] = somfy.SomfyFilterBank(dataI, dataQ, fs, Bw=20.0e3, fMix=-0.3e6, downSamplingRate=1)
				data = somfy.HormannDemod(dataI, dataQ, fs)
				np.save('Hormann_data_byCC1101', data)

			elif c == 'm':
				data = np.load('Hormann_data_byMain.npy')
				# data = np.load('Hormann_data_byCC1101.npy')
				print(f'data size = {data.size}')
				data = data[int(0e6):int(500e6):4]
				last_edge = 0
				bits = []
				packet = []
				for i in range(1, data.size):
					if data[i] != data[i-1]:
						len = i-last_edge
						print(f'state: {data[i-1]}, duration: {i-last_edge}')
						if (len >= 400) and (len <= 650):
							bits.append(data[i-1])
						elif (len >= 800) and (len <= 1300):
							bits.append(data[i-1])
							bits.append(data[i-1])
						elif len > 1400:
							packet.append(np.array(bits))
							bits = []
						last_edge = i
				
				for dd in packet:
					# print(f'packet# {i}: {packet[i]}')
					print(f'packet# : {dd}')
				
				# print('data extracted: ', [hex(i) for i in data_decoded])

				plt.plot(data)
				plt.grid()
				plt.show()
				
			elif c == 's':
				[fs, dataI, dataQ] = wav.readWaveFile(fileName)
				mLib.specPlot(dataI)

			elif c == 'h':
				[fs, dataI, dataQ] = wav.readWaveFile(fileName)
				hist.histogram2jpeg(dataI)

			elif c == 'a':
				[fs, dataI, dataQ] = wav.readWaveFile(fileName)
				# adcData = np.multiply(adcData, np.cos((np.arange(adcData.size)*2*np.pi*120.0e+6/240.0e+6)+0.06287))
				mLib.fftPlot(dataI)

			elif c == 'x':
				break
			print('Press new command:')

	print('Exit')
