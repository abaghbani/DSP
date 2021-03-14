import numpy as np
import scipy as scipy
import scipy.signal as signal
import matplotlib.pyplot as plt

from Spectrum.Constant import Constant
from Spectrum.ModemLib import ModemLib

C = Constant
myLib = ModemLib(0)

def OfdmDemodulation(baseband, fs=20.0, DemodType = 'Direct'):
	
	#print(baseband.size)
	allCarriers = np.arange(64)
	pilotCarriers = np.array([-21, -7, 7, 21])+32
	nullCarriers = np.array([-32, -31, -30, -29, -28, -27, 0, 27, 28, 29, 30, 31])+32
	dataCarriers = np.delete(allCarriers, np.hstack([pilotCarriers, nullCarriers]))
	shortCarriers = np.array([-24, -20, -16, -12, -8, -4, 4, 8, 12, 16, 20, 24])+32
	longCarriers = np.delete(allCarriers, nullCarriers)
	CP = 16
	pilotValue = 3+3j
	demapping_table = {v : k for k, v in C.QAM16_table.items()}

	def channelEstimate(OFDM_demod):
		pilots = OFDM_demod[pilotCarriers]  # extract the pilot values from the RX signal
		Hest_at_pilots = pilots / pilotValue # divide by the transmitted pilot values
		
		# Perform interpolation between the pilot carriers to get an estimate
		# of the channel in the data carriers. Here, we interpolate absolute value and phase 
		# separately
		Hest_abs = scipy.interpolate.interp1d(pilotCarriers, abs(Hest_at_pilots), kind='linear')(allCarriers)
		Hest_phase = scipy.interpolate.interp1d(pilotCarriers, np.angle(Hest_at_pilots), kind='linear')(allCarriers)
		Hest = Hest_abs * np.exp(1j*Hest_phase)
		
		# plt.plot(allCarriers, abs(H_exact), label='Correct Channel')
		# plt.stem(pilotCarriers, abs(Hest_at_pilots), label='Pilot estimates')
		# plt.plot(allCarriers, abs(Hest), label='Estimated channel via interpolation')
		# plt.grid(True); 
		# plt.xlabel('Carrier index'); 
		# plt.ylabel('$|H(f)|$'); 
		# plt.legend(fontsize=10)
		# plt.ylim(0,2)
		# plt.show()

		return Hest
	def Demapping(QAM):
		constellation = np.array([x for x in demapping_table.keys()])
		dists = abs(QAM.reshape((-1,1)) - constellation.reshape((1,-1)))
		const_index = dists.argmin(axis=1)
		hardDecision = constellation[const_index]
		return np.vstack([demapping_table[C] for C in hardDecision]), hardDecision

	if DemodType == 'test':
		# test = baseband[:64*12*2]
		# ofdmNoCp = test.reshape((-1, 64*12))
		# for i in range(2):
		# 	for j in range(12):
		# 		ofdm = np.fft.fft(ofdmNoCp[i][j::12])
		# 		ofdm = np.hstack([ofdm[ofdm.size/2:], ofdm[:ofdm.size/2]])
		# 		data = ofdm[longCarriers]
		# 		print i, j
		# 		plt.plot(data.real, data.imag, 'bo')
		# 		plt.grid()
		# 		plt.show()

		# test = baseband[12*(20*8+int(20*1.5)):12*(20*8+int(20*1.5))+64*12*2]
		# ofdmNoCp = test.reshape((-1, 64*12))
		for i in range(2407,2407+12*64*2,12*64):
			# for j in range(12):
			ofdmNoCp = baseband[i:i+64*12:12]
			ofdm = np.fft.fft(ofdmNoCp)
			# ofdm = np.fft.fft(ofdmNoCp[i][j::12])
			ofdm = np.hstack([ofdm[ofdm.size//2:], ofdm[:ofdm.size//2]])
			data = ofdm[longCarriers]
			print( i, ofdm.size)
			plt.plot(data.real, data.imag, 'bo')
			plt.grid()
			plt.show()
	
		for i in range(3943,3943+12*64*2,12):
			# for j in range(12):
			ofdmNoCp = baseband[i:i+64*12:12]
			ofdm = np.fft.fft(ofdmNoCp)
			# ofdm = np.fft.fft(ofdmNoCp[i][j::12])
			ofdm = np.hstack([ofdm[ofdm.size//2:], ofdm[:ofdm.size//2]])
			# data = ofdm[longCarriers]
			print( i, ofdm.size)
			pilot = ofdm[pilotCarriers]
			print(pilot)
			data = ofdm[dataCarriers]
			plt.plot(data.real, data.imag, 'bo')
			plt.plot(pilot.real, pilot.imag, 'ro')
			plt.grid()
			plt.show()
	
		test = baseband[3175+12*64:3175+12*64+80*12*4]
		ofdmNoCp = test.reshape((-1, 80*12))
		for i in range(4):
			ofdm = np.fft.fft(ofdmNoCp[i][CP::12])
			ofdm = np.hstack([ofdm[ofdm.size//2:], ofdm[:ofdm.size//2]])/10000.0
			pilot = ofdm[pilotCarriers]
			print(pilot)
			data = ofdm[dataCarriers]
			# plt.plot(data.real, data.imag, 'bo')
			# plt.plot(pilot.real, pilot.imag, 'ro')
			# plt.grid()
			# plt.show()
			PS_est, hardDecision = Demapping(data)
			for qam, hard in zip(data, hardDecision):
				plt.plot([qam.real, hard.real], [qam.imag, hard.imag], 'b-o')
				plt.plot(hardDecision.real, hardDecision.imag, 'ro')
			plt.grid()
			plt.show()

	
	
	
	elif DemodType == 'Preamble':
		
		dataStream = baseband.reshape((-1, 80))
		#short preamble
		for i in range(2):
			ofdmNoCp = dataStream[i][:-CP]
			ofdm = np.fft.fft(ofdmNoCp)
			ofdm = np.hstack([ofdm[ofdm.size//2:], ofdm[:ofdm.size//2]])
			preData = ofdm[shortCarriers]
			data = ofdm[dataCarriers]
			plt.plot(data.real, data.imag, 'bo')
			plt.plot(preData.real, preData.imag, 'ro')
			plt.grid()
			plt.show()
		# long preamble
		ofdmNoCp = np.hstack([dataStream[2][2*CP:],dataStream[3]])
		ofdmNoCp = ofdmNoCp.reshape(-1,64)
		# ofdmNoCp = ofdmNoCp.real/10000.0+1j*(ofdmNoCp.imag/25000.0)
		for i in range(2):
			ofdm = np.fft.fft(ofdmNoCp[i])
			ofdm = np.hstack([ofdm[ofdm.size//2:], ofdm[:ofdm.size//2]])
			data = ofdm[longCarriers]
			plt.plot(data.real, data.imag, 'bo')
			plt.grid()
			plt.show()

		for i in range(4,10):
			ofdmNoCp = dataStream[i][CP:]
			ofdm = np.fft.fft(ofdmNoCp)
			ofdm = np.hstack([ofdm[ofdm.size//2:], ofdm[:ofdm.size//2]])
			pilot = ofdm[pilotCarriers]
			print(pilot)
			data = ofdm[dataCarriers]
			# plt.plot(data.real, data.imag, 'bo')
			# plt.plot(pilot.real, pilot.imag, 'ro')
			# plt.grid()
			# plt.show()
			PS_est, hardDecision = Demapping(data)
			for qam, hard in zip(data, hardDecision):
				plt.plot([qam.real, hard.real], [qam.imag, hard.imag], 'b-o')
				plt.plot(hardDecision.real, hardDecision.imag, 'ro')
			plt.grid()
			plt.show()

	elif DemodType == 'Direct':
		dataStream = baseband[:(baseband.size//960)*960].reshape((-1, 960))
		for i in range(dataStream.shape[0]):
			ofdmNoCp = dataStream[i][CP*12:]
			ofdm = np.fft.fft(ofdmNoCp)
			ofdm = np.hstack([ofdm[ofdm.size//2:], ofdm[:ofdm.size//2]])
			plt.plot(ofdm.real, 'bo')
			plt.plot(ofdm.imag, 'ro')
			plt.show()
			ofdm = ofdm[ofdm.size//2-32:ofdm.size//2+32]/12
			# Hest = channelEstimate(ofdm)
			# ofdm /= Hest
			pilot = ofdm[pilotCarriers]
			print(pilot)
			data = ofdm[dataCarriers]
			# plt.plot(data.real, data.imag, 'bo')
			# plt.plot(pilot.real, pilot.imag, 'ro')
			# plt.grid()
			# plt.show()
			PS_est, hardDecision = Demapping(data)
			for qam, hard in zip(data, hardDecision):
				plt.plot([qam.real, hard.real], [qam.imag, hard.imag], 'b-o')
				plt.plot(hardDecision.real, hardDecision.imag, 'ro')
			plt.grid()
			plt.show()

	elif DemodType == 'DownSampling':

		downSamplingRate = int(fs/(64/3.2))  # baseband sampling rate = 80 samples / 4us = 64 samples / 3.2us = 20 MS/s
		corrGuard = np.zeros(baseband.size)
		for i in range(80*downSamplingRate, baseband.size):
			data = baseband[i-80*downSamplingRate:i:downSamplingRate]
			corrGuard[i] = np.correlate(data.real[:16], data.real[64:])
				
		plt.plot(corrGuard.real)
		plt.plot(baseband.real)
		plt.show()

		syncIndex = 0
		syncData = baseband[syncIndex::downSamplingRate]
		#plt.plot(baseband.real[syncIndex: syncIndex+960*3])
		#plt.show()

		# raw modulated data
		dataStream = syncData[:(syncData.size//80)*80].reshape((-1, 80))
		for i in range(dataStream.shape[0]):
			ofdmNoCp = dataStream[i][CP:]
			ofdm = np.fft.fft(ofdmNoCp)
			ofdm = np.hstack([ofdm[ofdm.size//2:], ofdm[:ofdm.size//2]])
			# Hest = channelEstimate(ofdm)
			# ofdm /= Hest
			pilot = ofdm[pilotCarriers]
			print(pilot)
			data = ofdm[dataCarriers]
			# plt.plot(data.real, data.imag, 'bo')
			# plt.plot(pilot.real, pilot.imag, 'ro')
			# plt.grid()
			# plt.show()
			PS_est, hardDecision = Demapping(data)
			for qam, hard in zip(data, hardDecision):
				plt.plot([qam.real, hard.real], [qam.imag, hard.imag], 'b-o')
				plt.plot(hardDecision.real, hardDecision.imag, 'ro')
			plt.grid()
			plt.show()


	# plt.plot(angle)
	# plt.plot(angleMid)
	# plt.plot(phase)
	# plt.legend(['angle', 'mid', 'phase'], loc='best')
	# plt.legend(['phase', 'phase2'], loc='best')
	# plt.grid()
	# plt.show()
	
	# ##################################
	# # sync detection and offset canselation
	# ##################################
	
	
	return PS_est
