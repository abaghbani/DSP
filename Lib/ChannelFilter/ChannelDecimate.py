import numpy as np

from .Constant import *
C = Constant()

def decimator_fix(data, Fs, Fmix, hb_flt, bitWidth, calcType):

	data_mix_p = np.array([dd*np.exp((np.arange(dd.size)*2j*np.pi*Fmix/Fs)+1j*0.06287) for dd in data])
	data_mix_n = np.array([dd*np.exp((np.arange(dd.size)*2j*np.pi*-Fmix/Fs)+1j*0.06287) for dd in data])
	data_mix = np.concatenate([np.vstack((dp, dn)) for dp, dn in zip(data_mix_p, data_mix_n)])
	
	data_flt = np.array([(np.convolve((hb_flt*bitWidth).astype(calcType), dd, 'same')/bitWidth)[::2] for dd in data_mix])

	return (data_flt.real).astype(calcType)+1j*(data_flt.imag).astype(calcType)

def ChannelDecimate_hdl(data, bitWidth = 2**17, calcType = 'int64'):
	
	#####################################################################################
	# stage 0 : input: 1 @ 240MSps, Fmix = 58.5MHz, output: 1 stream @ 120MSps
	#####################################################################################
	dataFlt0 = decimator_fix(data.reshape(1,-1)*1.0e2, 240.0, 58.5, C.hbFlt0, bitWidth, calcType)
	dataFlt0 = dataFlt0[0].reshape(1,-1)
	
	#####################################################################################
	# stage 1 : input: 1 @ 120MSps, Fmix = 24.0MHz, output: 2+1 stream @ 60MSps
	#####################################################################################
	dataFlt1_1 = decimator_fix(dataFlt0, 120, 24.0, C.hbFlt1, bitWidth, calcType)
	dataFlt1_2 = decimator_fix(dataFlt0, 120, 0.0, C.hbFlt1, bitWidth, calcType)
	dataFlt1_2 = dataFlt1_2[0].reshape(1,-1)

	#####################################################################################
	# stage 2 : input: 2+1 @ 60MSps, Fmix = 8.0MHz, output: 4+1 stream @ 30MSps
	#####################################################################################
	dataFlt2_1 = decimator_fix(dataFlt1_1, 60, 8.0, C.hbFlt2, bitWidth, calcType)
	dataFlt2_2 = decimator_fix(dataFlt1_2, 60, 0.0, C.hbFlt2, bitWidth, calcType)
	dataFlt2 = np.vstack((dataFlt2_1[0:2], dataFlt2_2[0].reshape(1,-1), dataFlt2_1[2:4]))

	#####################################################################################
	# stage 3 : input: 5 @ 30MSps, Fmix = 4.0MHz, output: 10 stream @ 15MSps
	#####################################################################################
	dataFlt3 = decimator_fix(dataFlt2, 30, 4.0, C.hbFlt3, bitWidth, calcType)

	#####################################################################################
	# stage 4 : input: 10 @ 15MSps, Fmix = 2.0MHz, output: 20 stream @ 7.5MSps (+ 80 stream @ 7.5MSps)
	#####################################################################################
	dataFlt4 = decimator_fix(dataFlt3, 15, 2.0, C.hbFlt4, bitWidth, calcType)

	dataFlt4_4M_0 = decimator_fix(dataFlt3, 15, 0.5, C.hbFlt4_4M, bitWidth, calcType)
	dataFlt4_4M_1 = decimator_fix(dataFlt3, 15, 1.5, C.hbFlt4_4M, bitWidth, calcType)
	dataFlt4_4M_2 = decimator_fix(dataFlt3, 15, 2.5, C.hbFlt4_4M, bitWidth, calcType)
	dataFlt4_4M_3 = decimator_fix(dataFlt3, 15, 3.5, C.hbFlt4_4M, bitWidth, calcType)
	
	## sorting 80 ch from 0 to 79
	data_4M = np.zeros((80, dataFlt4_4M_0[0].size), dtype=type(dataFlt4_4M_0[0][0]))
	for i in range(10):
		data_4M[8*i+0] = dataFlt4_4M_3[2*i+0]
		data_4M[8*i+1] = dataFlt4_4M_2[2*i+0]
		data_4M[8*i+2] = dataFlt4_4M_1[2*i+0]
		data_4M[8*i+3] = dataFlt4_4M_0[2*i+0]
		data_4M[8*i+4] = dataFlt4_4M_0[2*i+1]
		data_4M[8*i+5] = dataFlt4_4M_1[2*i+1]
		data_4M[8*i+6] = dataFlt4_4M_2[2*i+1]
		data_4M[8*i+7] = dataFlt4_4M_3[2*i+1]

	#####################################################################################
	# stage 5 : input: 20 @ 7.5MSps, Fmix = 1.0MHz, output: 40 stream @ 3.75MSps (+ 80 stream @ 3.75MSps)
	#####################################################################################
	dataFlt5 = decimator_fix(dataFlt4, 7.5, 1.0, C.hbFlt5, bitWidth, calcType)

	dataFlt5_2M_0 = decimator_fix(dataFlt4, 7.5, 0.5, C.hbFlt5_2M, bitWidth, calcType)
	dataFlt5_2M_1 = decimator_fix(dataFlt4, 7.5, 1.5, C.hbFlt5_2M, bitWidth, calcType)

	## sorting 80 ch from 0 to 79 
	data_2M = np.zeros((80, dataFlt5_2M_0[0].size), dtype=type(dataFlt5_2M_0[0][0]))
	for i in range(20):
		data_2M[4*i+0] = dataFlt5_2M_1[2*i+0]
		data_2M[4*i+1] = dataFlt5_2M_0[2*i+0]
		data_2M[4*i+2] = dataFlt5_2M_0[2*i+1]
		data_2M[4*i+3] = dataFlt5_2M_1[2*i+1]
	
	#####################################################################################
	# stage 6 : input: 40 @ 3.75MSps, Fmix = 0.5MHz, output: 80 stream @ 1.875MSps
	#####################################################################################
	dataFlt6 = decimator_fix(dataFlt5, 3.75, 0.5, C.hbFlt6, bitWidth, calcType)
	data_1M = dataFlt6
	
	return data_4M, data_2M, data_1M

def decimator(data, Fs, Fmix, Fc, n_tap, bitWidth, calcType):
	
	data_mix_p = np.array([dd*np.exp((np.arange(dd.size)*2j*np.pi*Fmix/Fs)+1j*0.06287) for dd in data])
	data_mix_n = np.array([dd*np.exp((np.arange(dd.size)*2j*np.pi*-Fmix/Fs)+1j*0.06287) for dd in data])
	data_mix = np.concatenate([np.vstack((dp, dn)) for dp, dn in zip(data_mix_p, data_mix_n)])
	
	hbFlt = (GenHalfBandFilterCoeff(n_tap, Fc, Fs)*bitWidth).astype(calcType)
	data_flt = np.array([(np.convolve(hbFlt, dd, 'same')/bitWidth)[::2] for dd in data_mix])

	return (data_flt.real).astype(calcType)+1j*(data_flt.imag).astype(calcType)

def ChannelDecimate(data, bitWidth = 2**17, calcType = np.int64):

	#####################################################################################
	# stage 0 : input: 1 @ 240MSps, Fmix = 58.5MHz, Fc=40.2MHz, output: 1 stream @ 120MSps
	#####################################################################################
	dataFlt0 = decimator(data.reshape(1,-1)*1.0e2, 240.0, 58.5, 40.8, 18, bitWidth, calcType)

	#####################################################################################
	# stage 1 : input: 1 @ 120MSps, Fmix = 20.0MHz, Fc=20.2MHz, output: 2 stream @ 60MSps
	#####################################################################################
	dataFlt1 = decimator(dataFlt0[0].reshape(1,-1), 120, 20.0, 20.8, 18, bitWidth, calcType)

	#####################################################################################
	# stage 2 : input: 2 @ 60MSps, Fmix = 10.0MHz, Fc=10.2MHz, output: 4 stream @ 30MSps
	#####################################################################################
	dataFlt2 = decimator(dataFlt1, 60, 10.0, 10.8, 18, bitWidth, calcType)

	#####################################################################################
	# stage 3 : input: 4 @ 30MSps, Fmix = 5.0MHz, Fc=5.2MHz, output: 8 stream @ 15MSps
	#####################################################################################
	dataFlt3 = decimator(dataFlt2, 30, 5.0, 5.8, 18, bitWidth, calcType)
		
	#####################################################################################
	# stage 4 : input: 8 @ 15MSps, Fmix = 2.0MHz, Fc=3.2MHz, output: 16 stream @ 7.5MSps
	#####################################################################################
	dataFlt4_4M_0 = decimator(dataFlt3, 15, 0.5, 2.8, 26, bitWidth, calcType)
	dataFlt4_4M_1 = decimator(dataFlt3, 15, 1.5, 2.8, 26, bitWidth, calcType)
	dataFlt4_4M_2 = decimator(dataFlt3, 15, 2.5, 2.8, 26, bitWidth, calcType)
	dataFlt4_4M_3 = decimator(dataFlt3, 15, 3.5, 2.8, 26, bitWidth, calcType)
	dataFlt4_4M_4 = decimator(dataFlt3, 15, 4.5, 2.8, 26, bitWidth, calcType)
	
	dataFlt4 = decimator(dataFlt3, 15, 2.0, 3.2, 18, bitWidth, calcType)
	
	## sorting 80 ch from 0 to 79
	data_4M = np.zeros((80, dataFlt4_4M_0[0].size), dtype=type(dataFlt4_4M_0[0][0]))
	for i in range(8):
		data_4M[10*i+0] = dataFlt4_4M_4[2*i+0]
		data_4M[10*i+1] = dataFlt4_4M_3[2*i+0]
		data_4M[10*i+2] = dataFlt4_4M_2[2*i+0]
		data_4M[10*i+3] = dataFlt4_4M_1[2*i+0]
		data_4M[10*i+4] = dataFlt4_4M_0[2*i+0]
		data_4M[10*i+5] = dataFlt4_4M_0[2*i+1]
		data_4M[10*i+6] = dataFlt4_4M_1[2*i+1]
		data_4M[10*i+7] = dataFlt4_4M_2[2*i+1]
		data_4M[10*i+8] = dataFlt4_4M_3[2*i+1]
		data_4M[10*i+9] = dataFlt4_4M_4[2*i+1]
		
	#####################################################################################
	# stage 5 : input: 16 @ 7.5MSps, Fmix = 1.0MHz, Fc=1.2MHz, output: 32 stream @ 3.75MSps
	#####################################################################################
	dataFlt5_2M_0 = decimator(dataFlt4, 7.5, 0.5, 1.2, 26, bitWidth, calcType)
	dataFlt5_2M_1 = decimator(dataFlt4, 7.5, 1.5, 1.2, 26, bitWidth, calcType)
	dataFlt5_2M_2 = decimator(dataFlt4, 7.5, 2.5, 1.2, 26, bitWidth, calcType)
	
	dataFlt5 = decimator(dataFlt4, 7.5, 1.0, 1.7, 18, bitWidth, calcType)

	## sorting 80 ch from 0 to 79 (we have totally 3*32=96 ch that some are copy of eachother)
	data_2M = np.zeros((80, dataFlt5_2M_0[0].size), dtype=type(dataFlt5_2M_0[0][0]))
	for i in range(8):
		data_2M[10*i+0] = dataFlt5_2M_2[4*i+0]
		data_2M[10*i+1] = dataFlt5_2M_1[4*i+0]
		data_2M[10*i+2] = dataFlt5_2M_0[4*i+0]
		data_2M[10*i+3] = dataFlt5_2M_0[4*i+1]
		data_2M[10*i+4] = dataFlt5_2M_1[4*i+1]
		data_2M[10*i+5] = dataFlt5_2M_2[4*i+1]
		#data_2M[10*i+x] = dataFlt5_2M_2[4*i+2]		it's a same copy of data[4], no need to keep
		#data_2M[10*i+x] = dataFlt5_2M_1[4*i+2]		it's a same copy of data[5], no need to keep
		data_2M[10*i+6] = dataFlt5_2M_0[4*i+2]
		data_2M[10*i+7] = dataFlt5_2M_0[4*i+3]
		data_2M[10*i+8] = dataFlt5_2M_1[4*i+3]
		data_2M[10*i+9] = dataFlt5_2M_2[4*i+3]

	#####################################################################################
	# stage 6 : input: 32 @ 3.75MSps, Fmix = 0.5MHz, Fc=0.7MHz, output: 64 stream @ 1.875MSps
	#####################################################################################
	dataFlt6_0 = decimator(dataFlt5, 3.75, 0.5, 0.7, 26, bitWidth, calcType)
	dataFlt6_1 = decimator(dataFlt5, 3.75, 1.5, 0.7, 26, bitWidth, calcType)
	
	## sorting 80 ch from 0 to 79 (we have totally 2*64=128 ch that some are copy of eachother)
	data_1M = np.zeros((80, dataFlt6_0[0].size), dtype=type(dataFlt6_0[0][0]))
	for i in range(8):
		data_1M[10*i+0] = dataFlt6_1[8*i+0]
		data_1M[10*i+1] = dataFlt6_0[8*i+0]
		data_1M[10*i+2] = dataFlt6_0[8*i+1]
		data_1M[10*i+3] = dataFlt6_1[8*i+1]
		#data_1M[10*i+x] = dataFlt6_1[8*i+2]
		#data_1M[10*i+x] = dataFlt6_0[8*i+2]
		data_1M[10*i+4] = dataFlt6_0[8*i+3]
		data_1M[10*i+5] = dataFlt6_1[8*i+3]
		#data_1M[10*i+x] = dataFlt6_1[8*i+4]
		#data_1M[10*i+x] = dataFlt6_0[8*i+4]
		data_1M[10*i+6] = dataFlt6_0[8*i+5]
		data_1M[10*i+7] = dataFlt6_1[8*i+5]
		#data_1M[10*i+x] = dataFlt6_1[8*i+6]
		#data_1M[10*i+x] = dataFlt6_0[8*i+6]
		data_1M[10*i+8] = dataFlt6_0[8*i+7]
		data_1M[10*i+9] = dataFlt6_1[8*i+7]
	
	return data_4M, data_2M, data_1M
