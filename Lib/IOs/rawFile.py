from ast import Return
import numpy as np

def readRawFile(fileName, bitvalid_number=12, count=-1, offset=0):
	# offset is in bytes but count is in samples
	adcData = np.fromfile(fileName, dtype='int16', count=count, offset=offset)
	adcData = (adcData<<(16-bitvalid_number))>>(16-bitvalid_number)
	print('ADC Data: {} samples'.format(adcData.size))
	print('ADC Data Min/Max: ',adcData.min(),adcData.max(), type(adcData[0]))
	return adcData

def writeRawFile(fileName, data):
	fp = np.memmap(fileName, mode='w+', dtype=np.dtype('<h'), shape=(1, data.size))
	fp[:] = data[:]

def readAdcDUMP(fileName, count=-1, offset=0):
	data = np.fromfile(fileName, dtype='uint8', count=count, offset=offset)
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
	fp.flush()

	return 0

def convert_8_to_16_bit(data):
	if (type(data[0]) != np.uint8 and type(data[0]) != np.int8):
		return -1
	
	data1, data2 = np.reshape(data, (data.shape[0] // 2, 2)).astype(np.int16).T

	outData1 = (data1) + ((data2 & 0x0f) * 256)
	outData1 = (outData1<<4)>>4

	return outData1

def convert_12_to_16_bit(data):
	if (type(data[0]) != np.uint8 and type(data[0]) != np.int8):
		return -1
		
	data1, data2, data3 = np.reshape(data, (data.shape[0] // 3, 3)).astype(np.int16).T

	outData1 = (data1 & 0xff) + ((data2 & 0x0f)<<8)
	outData2 = (data2 & 0xf0)>>4  + ((data3 & 0xff)<<4)

	outData = np.array([outData1, outData2]).ravel()
	outData = (outData<<4)>>4

	return outData

def convert_16_to_12_bit(data):
	if (type(data[0]) != np.uint16 and type(data[0]) != np.int16):
		return -1

	data1, data2 = np.reshape(data, (data.shape[0] // 2, 2)).astype(np.int16).T
		
	outData1 = (data1 & 0x00ff).astype(np.uint8)
	outData2 = ((data1 & 0x0f00)/256 + (data2 & 0x000f)*16).astype(np.uint8)
	outData3 = ((data2 & 0x0ff0)/16).astype(np.uint8)

	outData = np.array([outData1, outData2, outData3]).ravel()

	return outData
