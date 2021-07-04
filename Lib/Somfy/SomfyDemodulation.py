import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

from Spectrum.ModemLib import ModemLib
myLib = ModemLib(0)

######################################
# symb-time = (2**SF)/Bw
# symb-rate = Bw/(2**SF)
# bit-rate = SF*Bw/(2**SF)*CR
######################################

def SomfyFilterBank(dataI, dataQ, fs, Bw, fMix, downSamplingRate):
	n=np.arange(dataI.size)
	cosMix = np.cos((n*2*np.pi*fMix/fs)+0.06287)
	sinMix = np.sin((n*2*np.pi*fMix/fs)+0.06287)
	dataMixI = np.multiply(dataI, cosMix) - np.multiply(dataQ, sinMix)
	dataMixQ = np.multiply(dataQ, cosMix) + np.multiply(dataI, sinMix)
	# myLib.fftPlot(dataMixI+1j*dataMixQ, n=1, fs=fs)

	d = signal.firwin(301, cutoff = (1.2*Bw)/(fs/2.), window = 'blackmanharris')
	dataFltI = np.convolve(d, dataMixI)
	dataFltQ = np.convolve(d, dataMixQ)
	dataFltI = dataFltI[::downSamplingRate]
	dataFltQ = dataFltQ[::downSamplingRate]
	fs /= downSamplingRate

	# myLib.fftPlot(dataFltI+1j*dataFltQ, n=1, fs=fs)
	# myLib.specPlot(dataFltI+1j*dataFltQ, fs=fs)
	
	return dataFltI, dataFltQ, fs

def decod_manchester(data, type1 = False):
	data_out = np.empty(0, dtype='int')
	firstEdge = 0
	for i in range(data.size):
		if data[i] != data[i-1]:
			if firstEdge == 0:
				firstEdge = 1
				period = 0.0
				last_edge = i
				data_out = np.append(data_out, (int(data[i]) if type1 else int(data[i-1])) )
			elif period == 0 or ((i-last_edge)>(0.8*period) and (i-last_edge)<(1.2*period)):
				period =i-last_edge
				last_edge = i
				data_out = np.append(data_out, (int(data[i]) if type1 else int(data[i-1])))
			elif (i-last_edge)>(1.5*period):
				period =i-last_edge
				data_out = np.empty(0, dtype='int')
				print('reset detection = ',i)
				i = 0
	return data_out

def string_to_byte(bits):
	index = 0
	data = np.zeros(bits.size//8, dtype='uint8')
	for i in range((bits.size//8)*8):
		data[i//8] += (bits[i]*np.power(2, 7-index))
		index = 0 if index >= 7 else index+1

	return data

def encryption(data):
	retVal = data.astype('uint8')
	for i in range(1, data.size):
		retVal[i] = data[i] ^ retVal[i-1]
	return data

def decryption(data):
	retVal = data.astype('uint8')
	for i in range(1, retVal.size):
		retVal[i] = retVal[i] ^ data[i-1]
	return retVal

def crcCalc(data):
	checksum = 0
	for i in range(data.size):
		checksum = checksum ^ (data[i] & 0x0f) ^ ((data[i] >> 4) & 0x0f)
	
	return checksum & 0x0f

def SomfyDemodulation(dataI, dataQ, fs, payload_index, payload_end):
	
	# dI = dataI.astype('int64')
	# dQ = dataQ.astype('int64')
	dI = dataI
	dQ = dataQ
	magSample = (dI*dI)+(dQ*dQ)
	
	plt.plot(magSample)
	plt.show()

	data = magSample[payload_index:payload_end]
	print('max data is: ', data.max())
	high_threshold = int(0.5*data.max())
	data[data<high_threshold] = 0
	data[data!=0] = 1
	data_extracted = decod_manchester(data, True)
	print('data extract is: ', data_extracted)
	plt.plot(data)
	plt.show()
	
	data_in_byte = string_to_byte(data_extracted)
	data_decoded = decryption(data_in_byte)
	# print('data extract is: ', hex(data_in_byte))
	print('data extract is: ', [hex(i) for i in data_in_byte])
	print('data decoded: ', [hex(i) for i in data_decoded])
	
	crc = crcCalc(data_decoded)
	print('crc: ', hex(crc))
	# plt.legend(['raw', 'avg'], loc='best')
	
	# plt.plot(dataout)
	# plt.show()

def SomfyDemod(dataI, dataQ, fs):
	# digitizer
	# magSample = (dataI*dataI)+(dataQ*dataQ)
	data = (dataI*dataI)+(dataQ*dataQ)
	
	high_threshold = int(0.1*data.max())
	data[data<high_threshold] = 0
	data[data!=0] = 1
	
	# plt.plot(magSample)
	plt.plot(data)
	plt.grid()
	plt.show()

	# packet extraction
	slice_len = np.zeros(5)
	last_edge = 0
	detected = 0
	data_out = np.empty(0)
	firstEdge = 0
	packet_number = 0
	for i in range(1, data.size):
		if detected == 0:
			if data[i] != data[i-1]:
				slice_len =np.roll(slice_len, 1)
				slice_len[0] = i - last_edge
				# print('edge detect: ',i , i-last_edge)
				last_edge = i
				if slice_len[0]<500 and slice_len[0]>400 and slice_len[1]<300 and slice_len[1]>200 and slice_len[2]<300 and slice_len[2]>200 and slice_len[3]<300 and slice_len[3]>200 :
					detected = 1
					firstEdge = 0
					period = 0
					print('sync detected at index: ', i, 'sync detected: ', slice_len[0], slice_len[1], slice_len[2], slice_len[3])
		else:
			if data[i] != data[i-1] or (period != 0 and i-last_edge > 10*period):
				if firstEdge == 0:
					firstEdge = 1
					period = 0
					last_edge = i
					data_out = np.append(data_out, data[i])
				elif period == 0 or ((i-last_edge)>(0.8*period) and (i-last_edge)<(1.2*period)):
					period =i-last_edge
					last_edge = i
					data_out = np.append(data_out, data[i])
				elif (i-last_edge)>(1.5*period):
					detected = 0
					print('detected packet no: ', packet_number, 'with period: ', period, '   ***   << period should be between 120 to 130 (1200 to 1300 us)>>')
					print(data_out)

					data_in_byte = string_to_byte(data_out)
					data_decoded = decryption(data_in_byte)
					print('data extract is: ', [hex(i) for i in data_in_byte])
					print('data decoded: ', [hex(i) for i in data_decoded])
					crc = crcCalc(data_decoded)
					print('crc: ', hex(crc))

					packet_number += 1
					data_out = np.empty(0)
					slice_len = np.zeros(slice_len.size)

def HormannDemod(dataI, dataQ, fs):
	# digitizer
	# magSample = (dataI*dataI)+(dataQ*dataQ)
	data = (dataI*dataI)+(dataQ*dataQ)
	plt.plot(data)
	plt.grid()
	plt.show()
	
	high_threshold = data.max()/10
	data[data<high_threshold] = 0
	data[data!=0] = 1
		
	plt.plot(data)
	plt.grid()
	plt.show()

	return data

