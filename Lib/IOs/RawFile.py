import numpy as np

def readRawFile(fileName, bitvalid_number=12):
	#adcData = np.memmap(fileName, mode='r', dtype=np.dtype('<h'))
	adcData = np.fromfile(fileName, dtype='int16')
	adcData = (adcData<<(16-bitvalid_number))>>(16-bitvalid_number)
	print('ADC Data: {} samples'.format(adcData.size))
	print('ADC Data Min/Max: ',adcData.min(),adcData.max(), type(adcData[0]))
	return adcData

def writeRawFile(fileName, data):
	fp = np.memmap(fileName, mode='w+', dtype=np.dtype('<h'), shape=(1, data.size))
	fp[:] = data[:]

def readAdcDUMP(fileName, count=-1):
	data = np.fromfile(fileName, dtype='uint8', count=count)
	data = data[1:1+int(((data.size-1)//3)*3)]
	fst_uint8, mid_uint8, lst_uint8 = np.reshape(data, (data.shape[0] // 3, 3)).astype(np.int16).T
	fst_uint12 = (fst_uint8) + ((mid_uint8 % 16) << 8)
	snd_uint12 = (mid_uint8 >> 4)  + (lst_uint8 << 4)
	adcData = np.reshape(np.concatenate((fst_uint12[:, None], snd_uint12[:, None]), axis=1), 2 * fst_uint12.shape[0])
	adcData = (adcData<<(16-12))>>(16-12)
	print('ADC Data Min/Max: ',adcData.min(),adcData.max(), type(adcData[0]))
	return adcData


def rfdump_to_bin(filename, start_add, end_add, correction=0):
	data_rd = np.fromfile(filename, dtype='uint8', count=end_add-start_add + correction, offset=start_add)
	data_rd = data_rd[correction:]
	len = data_rd.size
	n_slice = len//3e6
	fp = np.memmap(filename[:-9]+'bin', mode='w+', dtype=np.dtype('<h'), shape=(1, int(n_slice*2e6)))
				
	for i in range(int(n_slice)):
		data = data_rd[int(i*3e6):int((i+1)*3e6)]
		fst_uint8, mid_uint8, lst_uint8 = np.reshape(data, (data.shape[0] // 3, 3)).astype(np.int16).T
		fst_uint12 = (fst_uint8) + ((mid_uint8 % 16) << 8)
		snd_uint12 = (mid_uint8 >> 4)  + (lst_uint8 << 4)
		adcData = np.reshape(np.concatenate((fst_uint12[:, None], snd_uint12[:, None]), axis=1), 2 * fst_uint12.shape[0])
		adcData = (adcData<<(16-12))>>(16-12)
		fp[0, int(i*2e6):int((i+1)*2e6)] = adcData

	return 0