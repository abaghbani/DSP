import numpy as np

from .Constant import *
C = Constant()

def ChannelDecimate_v1(AdcStream):

	# Fs = C.AdcSamplingFrequency
	Fs = 240.0e+6
	data = AdcStream
	n = np.arange(data.size)
	bitWidth = 2**17
	calcType = 'int64'

	# input data : [-98M (2402) ... -19M (2481)],   [[2402:2481]-2260(rf_mixer)]sampling@240MHz => nyquist1 left side = -98:-19

	# mixer: (I+jQ)*exp(j*2pi*fmix*n) = [I*cos(2*pi*fmix*n)-Q*sin(2pi*fmix*n)]+j[I*sin(2pi*fmix*n)+Q*cos(2pi*fmix*n)]
	# mixed_I = I*cos(fmix) - Q*sin(fmix)
	# mixed_Q = Q*cos(fmix) + I*sin(fmix)

	# stage 0 : Fs = 240MHz, Fmix = 58.5MHz, Fc=40.2MHz channel=1 [output = -39.5M:39.5M]
	fmix0 = 58.5e+6
	cosMix0 = (np.cos((n*2*np.pi*fmix0/Fs)+0.06287)*bitWidth).astype(calcType)
	sinMix0 = (np.sin((n*2*np.pi*fmix0/Fs)+0.06287)*bitWidth).astype(calcType)
	dataMix0 = [[0]*n for i in range(2)]
	dataMix0[0] = np.multiply(data, cosMix0)//(2**(17-6))#bitWidth   # input data is 12-bits and here we convert it to 12+6 bits
	dataMix0[1] = np.multiply(data, sinMix0)//(2**(17-6))#bitWidth
	
	hbFlt0 = (C.hbFlt0*bitWidth).astype(calcType)
	dataFlt0 = [[0]*n for i in range(2)]
	for i in range(1*2):
		dataFlt0[i] = (np.convolve(hbFlt0,dataMix0[i],'same')/bitWidth).astype(calcType)
		dataFlt0[i] = dataFlt0[i][::2]

	# stage 1 : Fs = 120MHz, Fmix = 24.0MHz, Fc=16.2MHz channel=2+1
	fmix1 = 24.0e+6
	n=np.arange(dataFlt0[0].size)
	cosMix1 = (np.cos((n*2*np.pi*fmix1/(Fs/2))+0.06287)*bitWidth).astype(calcType)
	sinMix1 = (np.sin((n*2*np.pi*fmix1/(Fs/2))+0.06287)*bitWidth).astype(calcType)
	dataMix1 = [[0]*n for i in range(6)]
	for i in range(2//2):
		dataMix1[i*4]   = ((np.multiply(dataFlt0[i*2],   cosMix1) - np.multiply(dataFlt0[i*2+1], sinMix1))/bitWidth).astype(calcType)
		dataMix1[i*4+1] = ((np.multiply(dataFlt0[i*2+1], cosMix1) + np.multiply(dataFlt0[i*2],   sinMix1))/bitWidth).astype(calcType)
		dataMix1[i*4+2] = ((np.multiply(dataFlt0[i*2],   cosMix1) + np.multiply(dataFlt0[i*2+1], sinMix1))/bitWidth).astype(calcType)
		dataMix1[i*4+3] = ((np.multiply(dataFlt0[i*2+1], cosMix1) - np.multiply(dataFlt0[i*2],   sinMix1))/bitWidth).astype(calcType)
	dataMix1[4] = dataFlt0[0]
	dataMix1[5] = dataFlt0[1]

	hbFlt1 = (C.hbFlt1*bitWidth).astype(calcType)
	dataFlt1 = [[0]*n for i in range(6)]
	for i in range(3*2):
		dataFlt1[i] = (np.convolve(hbFlt1,dataMix1[i],'same')/bitWidth).astype(calcType)
		dataFlt1[i] = dataFlt1[i][::2]

	# stage 2 : Fs = 60MHz, Fmix = 8.0MHz, Fc=8.2MHz channel=4+1
	fmix2 = 8.0e+6
	n=np.arange(dataFlt1[0].size)
	cosMix2 = (np.cos((n*2*np.pi*fmix2/(Fs/4))+0.06287)*bitWidth).astype(calcType)
	sinMix2 = (np.sin((n*2*np.pi*fmix2/(Fs/4))+0.06287)*bitWidth).astype(calcType)
	dataMix2 = [[0]*n for i in range(10)]
	for i in range(4//2):
		dataMix2[i*6]   = ((np.multiply(dataFlt1[i*2],   cosMix2) - np.multiply(dataFlt1[i*2+1], sinMix2))/bitWidth).astype(calcType)
		dataMix2[i*6+1] = ((np.multiply(dataFlt1[i*2+1], cosMix2) + np.multiply(dataFlt1[i*2],   sinMix2))/bitWidth).astype(calcType)
		dataMix2[i*6+2] = ((np.multiply(dataFlt1[i*2],   cosMix2) + np.multiply(dataFlt1[i*2+1], sinMix2))/bitWidth).astype(calcType)
		dataMix2[i*6+3] = ((np.multiply(dataFlt1[i*2+1], cosMix2) - np.multiply(dataFlt1[i*2],   sinMix2))/bitWidth).astype(calcType)
	dataMix2[4] = dataFlt1[4]
	dataMix2[5] = dataFlt1[5]

	hbFlt2 = (C.hbFlt2*bitWidth).astype(calcType)
	dataFlt2 = [[0]*n for i in range(10)]
	for i in range(5*2):
		dataFlt2[i] = (np.convolve(hbFlt2,dataMix2[i],'same')/bitWidth).astype(calcType)
		dataFlt2[i] = dataFlt2[i][::2]

	# stage 3 : Fs = 30MHz, Fmix = 4.0MHz, Fc=4.2MHz channel=8+2
	fmix3 = 4.0e+6
	n=np.arange(dataFlt2[0].size)
	cosMix3 = (np.cos((n*2*np.pi*fmix3/(Fs/8))+0.06287)*bitWidth).astype(calcType)
	sinMix3 = (np.sin((n*2*np.pi*fmix3/(Fs/8))+0.06287)*bitWidth).astype(calcType)
	dataMix3 = [[0]*n for i in range(20)]
	for i in range(10//2):
		dataMix3[i*4]   = ((np.multiply(dataFlt2[i*2],   cosMix3) - np.multiply(dataFlt2[i*2+1], sinMix3))/bitWidth).astype(calcType)
		dataMix3[i*4+1] = ((np.multiply(dataFlt2[i*2+1], cosMix3) + np.multiply(dataFlt2[i*2],   sinMix3))/bitWidth).astype(calcType)
		dataMix3[i*4+2] = ((np.multiply(dataFlt2[i*2],   cosMix3) + np.multiply(dataFlt2[i*2+1], sinMix3))/bitWidth).astype(calcType)
		dataMix3[i*4+3] = ((np.multiply(dataFlt2[i*2+1], cosMix3) - np.multiply(dataFlt2[i*2],   sinMix3))/bitWidth).astype(calcType)

	hbFlt3 = (C.hbFlt3*bitWidth).astype(calcType)
	dataFlt3 = [[0]*n for i in range(20)]
	for i in range(10*2):
		dataFlt3[i] = (np.convolve(hbFlt3,dataMix3[i],'same')/bitWidth).astype(calcType)
		dataFlt3[i] = dataFlt3[i][::2]

	# stage 4 : Fs = 15MHz, Fmix = 2.0MHz, Fc=2.2MHz channel=16+4
	fmix4 = 2.0e+6
	n=np.arange(dataFlt3[0].size)
	cosMix4 = (np.cos((n*2*np.pi*fmix4/(Fs/16))+0.06287)*bitWidth).astype(calcType)
	sinMix4 = (np.sin((n*2*np.pi*fmix4/(Fs/16))+0.06287)*bitWidth).astype(calcType)
	dataMix4 = [[0]*n for i in range(40)]
	for i in range(20//2):
		dataMix4[i*4]   = ((np.multiply(dataFlt3[i*2],   cosMix4) - np.multiply(dataFlt3[i*2+1], sinMix4))/bitWidth).astype(calcType)
		dataMix4[i*4+1] = ((np.multiply(dataFlt3[i*2+1], cosMix4) + np.multiply(dataFlt3[i*2],   sinMix4))/bitWidth).astype(calcType)
		dataMix4[i*4+2] = ((np.multiply(dataFlt3[i*2],   cosMix4) + np.multiply(dataFlt3[i*2+1], sinMix4))/bitWidth).astype(calcType)
		dataMix4[i*4+3] = ((np.multiply(dataFlt3[i*2+1], cosMix4) - np.multiply(dataFlt3[i*2],   sinMix4))/bitWidth).astype(calcType)

	hbFlt4 = (C.hbFlt4*bitWidth).astype(calcType)
	dataFlt4 = [[0]*n for i in range(40)]
	for i in range(20*2):
		dataFlt4[i] = (np.convolve(hbFlt4,dataMix4[i],'same')/bitWidth).astype(calcType)
		dataFlt4[i] = dataFlt4[i][::2]

	# stage 4 (4M) : Fs = 15MHz, Fmix = 2.0MHz, Fc=2.2MHz channel=16+4
	fmix4_3p5 = 3.5e+6
	fmix4_2p5 = 2.5e+6
	fmix4_1p5 = 1.5e+6
	fmix4_0p5 = 0.5e+6
	n=np.arange(dataFlt3[0].size)
	cosMix4_3p5 = (np.cos((n*2*np.pi*fmix4_3p5/(Fs/16))-0.06287)*bitWidth).astype(calcType)
	sinMix4_3p5 = (np.sin((n*2*np.pi*fmix4_3p5/(Fs/16))-0.06287)*bitWidth).astype(calcType)
	cosMix4_2p5 = (np.cos((n*2*np.pi*fmix4_2p5/(Fs/16))-0.06287)*bitWidth).astype(calcType)
	sinMix4_2p5 = (np.sin((n*2*np.pi*fmix4_2p5/(Fs/16))-0.06287)*bitWidth).astype(calcType)
	cosMix4_1p5 = (np.cos((n*2*np.pi*fmix4_1p5/(Fs/16))-0.06287)*bitWidth).astype(calcType)
	sinMix4_1p5 = (np.sin((n*2*np.pi*fmix4_1p5/(Fs/16))-0.06287)*bitWidth).astype(calcType)
	cosMix4_0p5 = (np.cos((n*2*np.pi*fmix4_0p5/(Fs/16))-0.06287)*bitWidth).astype(calcType)
	sinMix4_0p5 = (np.sin((n*2*np.pi*fmix4_0p5/(Fs/16))-0.06287)*bitWidth).astype(calcType)
	dataMix4_4M = [[0]*n for i in range(80*2)]
	for i in range(20//2):
		dataMix4_4M[i*16]    = ((np.multiply(dataFlt3[i*2],   cosMix4_3p5) - np.multiply(dataFlt3[i*2+1], sinMix4_3p5))/bitWidth).astype(calcType)
		dataMix4_4M[i*16+1]  = ((np.multiply(dataFlt3[i*2+1], cosMix4_3p5) + np.multiply(dataFlt3[i*2],   sinMix4_3p5))/bitWidth).astype(calcType)
		dataMix4_4M[i*16+2]  = ((np.multiply(dataFlt3[i*2],   cosMix4_2p5) - np.multiply(dataFlt3[i*2+1], sinMix4_2p5))/bitWidth).astype(calcType)
		dataMix4_4M[i*16+3]  = ((np.multiply(dataFlt3[i*2+1], cosMix4_2p5) + np.multiply(dataFlt3[i*2],   sinMix4_2p5))/bitWidth).astype(calcType)
		dataMix4_4M[i*16+4]  = ((np.multiply(dataFlt3[i*2],   cosMix4_1p5) - np.multiply(dataFlt3[i*2+1], sinMix4_1p5))/bitWidth).astype(calcType)
		dataMix4_4M[i*16+5]  = ((np.multiply(dataFlt3[i*2+1], cosMix4_1p5) + np.multiply(dataFlt3[i*2],   sinMix4_1p5))/bitWidth).astype(calcType)
		dataMix4_4M[i*16+6]  = ((np.multiply(dataFlt3[i*2],   cosMix4_0p5) - np.multiply(dataFlt3[i*2+1], sinMix4_0p5))/bitWidth).astype(calcType)
		dataMix4_4M[i*16+7]  = ((np.multiply(dataFlt3[i*2+1], cosMix4_0p5) + np.multiply(dataFlt3[i*2],   sinMix4_0p5))/bitWidth).astype(calcType)
		dataMix4_4M[i*16+8]  = ((np.multiply(dataFlt3[i*2],   cosMix4_0p5) + np.multiply(dataFlt3[i*2+1], sinMix4_0p5))/bitWidth).astype(calcType)
		dataMix4_4M[i*16+9]  = ((np.multiply(dataFlt3[i*2+1], cosMix4_0p5) - np.multiply(dataFlt3[i*2],   sinMix4_0p5))/bitWidth).astype(calcType)
		dataMix4_4M[i*16+10] = ((np.multiply(dataFlt3[i*2],   cosMix4_1p5) + np.multiply(dataFlt3[i*2+1], sinMix4_1p5))/bitWidth).astype(calcType)
		dataMix4_4M[i*16+11] = ((np.multiply(dataFlt3[i*2+1], cosMix4_1p5) - np.multiply(dataFlt3[i*2],   sinMix4_1p5))/bitWidth).astype(calcType)
		dataMix4_4M[i*16+12] = ((np.multiply(dataFlt3[i*2],   cosMix4_2p5) + np.multiply(dataFlt3[i*2+1], sinMix4_2p5))/bitWidth).astype(calcType)
		dataMix4_4M[i*16+13] = ((np.multiply(dataFlt3[i*2+1], cosMix4_2p5) - np.multiply(dataFlt3[i*2],   sinMix4_2p5))/bitWidth).astype(calcType)
		dataMix4_4M[i*16+14] = ((np.multiply(dataFlt3[i*2],   cosMix4_3p5) + np.multiply(dataFlt3[i*2+1], sinMix4_3p5))/bitWidth).astype(calcType)
		dataMix4_4M[i*16+15] = ((np.multiply(dataFlt3[i*2+1], cosMix4_3p5) - np.multiply(dataFlt3[i*2],   sinMix4_3p5))/bitWidth).astype(calcType)
		
	hbFlt4_4M = (C.hbFlt4_4M*bitWidth).astype(calcType)
	dataFlt4_4M = [[0]*n for i in range(80*2)]
	for i in range(80*2):
		dataFlt4_4M[i] = (np.convolve(hbFlt4_4M,dataMix4_4M[i],'same')/bitWidth).astype(calcType)
		dataFlt4_4M[i] = dataFlt4_4M[i][::2]

	# stage 5 : Fs = 7.5MHz, Fmix = 1.0MHz, Fc=1.2MHz channel=32+8
	fmix5 = 1.0e+6
	n=np.arange(dataFlt4[0].size)
	cosMix5 = (np.cos((n*2*np.pi*fmix5/(Fs/32))+0.06287)*bitWidth).astype(calcType)
	sinMix5 = (np.sin((n*2*np.pi*fmix5/(Fs/32))+0.06287)*bitWidth).astype(calcType)
	dataMix5 = [[0]*n for i in range(80)]
	for i in range(40//2):
		dataMix5[i*4]   = ((np.multiply(dataFlt4[i*2],   cosMix5) - np.multiply(dataFlt4[i*2+1], sinMix5))/bitWidth).astype(calcType)
		dataMix5[i*4+1] = ((np.multiply(dataFlt4[i*2+1], cosMix5) + np.multiply(dataFlt4[i*2],   sinMix5))/bitWidth).astype(calcType)
		dataMix5[i*4+2] = ((np.multiply(dataFlt4[i*2],   cosMix5) + np.multiply(dataFlt4[i*2+1], sinMix5))/bitWidth).astype(calcType)
		dataMix5[i*4+3] = ((np.multiply(dataFlt4[i*2+1], cosMix5) - np.multiply(dataFlt4[i*2],   sinMix5))/bitWidth).astype(calcType)

	hbFlt5 = (C.hbFlt5*bitWidth).astype(calcType)
	dataFlt5 = [[0]*n for i in range(80)]
	for i in range(40*2):
		dataFlt5[i] = (np.convolve(hbFlt5,dataMix5[i],'same')/bitWidth).astype(calcType)
		dataFlt5[i] = dataFlt5[i][::2]

	# stage 5 (2M) : Fs = 7.5MHz, Fmix = 1.5MHz/-0.5MHz, Fc=1.2MHz channel=32+8
	fmix5_0p5 = 0.5e+6
	fmix5_1p5 = 1.5e+6
	n=np.arange(dataFlt4[0].size)
	cosMix5_0p5 = (np.cos((n*2*np.pi*fmix5_0p5/(Fs/32))+0.06287)*bitWidth).astype(calcType)
	sinMix5_0p5 = (np.sin((n*2*np.pi*fmix5_0p5/(Fs/32))+0.06287)*bitWidth).astype(calcType)
	cosMix5_1p5 = (np.cos((n*2*np.pi*fmix5_1p5/(Fs/32))+0.06287)*bitWidth).astype(calcType)
	sinMix5_1p5 = (np.sin((n*2*np.pi*fmix5_1p5/(Fs/32))+0.06287)*bitWidth).astype(calcType)
	dataMix5_2M = [[0]*n for i in range(80*2)]
	for i in range(40//2):
		dataMix5_2M[i*8+0] = ((np.multiply(dataFlt4[i*2],   cosMix5_1p5) - np.multiply(dataFlt4[i*2+1], sinMix5_1p5))/bitWidth).astype(calcType)
		dataMix5_2M[i*8+1] = ((np.multiply(dataFlt4[i*2+1], cosMix5_1p5) + np.multiply(dataFlt4[i*2],   sinMix5_1p5))/bitWidth).astype(calcType)
		dataMix5_2M[i*8+2] = ((np.multiply(dataFlt4[i*2],   cosMix5_0p5) - np.multiply(dataFlt4[i*2+1], sinMix5_0p5))/bitWidth).astype(calcType)
		dataMix5_2M[i*8+3] = ((np.multiply(dataFlt4[i*2+1], cosMix5_0p5) + np.multiply(dataFlt4[i*2],   sinMix5_0p5))/bitWidth).astype(calcType)
		dataMix5_2M[i*8+4] = ((np.multiply(dataFlt4[i*2],   cosMix5_0p5) + np.multiply(dataFlt4[i*2+1], sinMix5_0p5))/bitWidth).astype(calcType)
		dataMix5_2M[i*8+5] = ((np.multiply(dataFlt4[i*2+1], cosMix5_0p5) - np.multiply(dataFlt4[i*2],   sinMix5_0p5))/bitWidth).astype(calcType)
		dataMix5_2M[i*8+6] = ((np.multiply(dataFlt4[i*2],   cosMix5_1p5) + np.multiply(dataFlt4[i*2+1], sinMix5_1p5))/bitWidth).astype(calcType)
		dataMix5_2M[i*8+7] = ((np.multiply(dataFlt4[i*2+1], cosMix5_1p5) - np.multiply(dataFlt4[i*2],   sinMix5_1p5))/bitWidth).astype(calcType)

	hbFlt5_2M = (C.hbFlt5_2M*bitWidth).astype(calcType)
	dataFlt5_2M = [[0]*n for i in range(80*2)]
	for i in range(80*2):
		dataFlt5_2M[i] = (np.convolve(hbFlt5_2M,dataMix5_2M[i],'same')/bitWidth).astype(calcType)
		dataFlt5_2M[i] = dataFlt5_2M[i][::2]

	# stage 6 : Fs = 3.75MHz, Fmix = 0.5MHz, Fc=0.7MHz channel=64+16
	fmix6 = 0.5e+6
	n=np.arange(dataFlt5[0].size)
	cosMix6 = (np.cos((n*2*np.pi*fmix6/(Fs/64))+0.06287)*bitWidth).astype(calcType)
	sinMix6 = (np.sin((n*2*np.pi*fmix6/(Fs/64))+0.06287)*bitWidth).astype(calcType)
	dataMix6 = [[0]*n for i in range(160)]
	for i in range(80//2):
		dataMix6[i*4]   = ((np.multiply(dataFlt5[i*2],   cosMix6) - np.multiply(dataFlt5[i*2+1], sinMix6))/bitWidth).astype(calcType)
		dataMix6[i*4+1] = ((np.multiply(dataFlt5[i*2+1], cosMix6) + np.multiply(dataFlt5[i*2],   sinMix6))/bitWidth).astype(calcType)
		dataMix6[i*4+2] = ((np.multiply(dataFlt5[i*2],   cosMix6) + np.multiply(dataFlt5[i*2+1], sinMix6))/bitWidth).astype(calcType)
		dataMix6[i*4+3] = ((np.multiply(dataFlt5[i*2+1], cosMix6) - np.multiply(dataFlt5[i*2],   sinMix6))/bitWidth).astype(calcType)

	hbFlt6 = (C.hbFlt6*bitWidth).astype(calcType)
	dataFlt6 = [[0]*n for i in range(160)]
	for i in range(80*2):
		dataFlt6[i] = (np.convolve(hbFlt6,dataMix6[i],'same')/bitWidth).astype(calcType)
		dataFlt6[i] = dataFlt6[i][::2]
	
	return dataFlt4_4M, dataFlt5_2M, dataFlt6

def decimator(data, Fs, Fmix, Fc, bitWidth, calcType):
	
	data_mix_p = np.array([dd*np.exp((np.arange(dd.size)*2j*np.pi*Fmix/Fs)+1j*0.06287) for dd in data])
	data_mix_n = np.array([dd*np.exp((np.arange(dd.size)*2j*np.pi*-Fmix/Fs)+1j*0.06287) for dd in data])
	data_mix = np.concatenate([np.vstack((dp, dn)) for dp, dn in zip(data_mix_p, data_mix_n)])
	
	hbFlt = (GenHalfBandFilterCoeff(18, Fc, Fs)*bitWidth).astype(calcType)
	data_flt = np.array([(np.convolve(hbFlt, dd, 'same')/bitWidth)[::2] for dd in data_mix])

	return (data_flt.real).astype(calcType)+1j*(data_flt.imag).astype(calcType)

def ChannelDecimate(data, bitWidth = 2**17, calcType = np.int64):

	#####################################################################################
	# stage 0 : input: 1 @ 240MSps, Fmix = 58.5MHz, Fc=40.2MHz, output: 1 stream @ 120MSps
	#####################################################################################
	dataFlt0 = decimator(data.reshape(1,-1)*1.0e2, 240.0, 58.5, 40.8, bitWidth, calcType)

	#####################################################################################
	# stage 1 : input: 1 @ 120MSps, Fmix = 20.0MHz, Fc=20.2MHz, output: 2 stream @ 60MSps
	#####################################################################################
	dataFlt1 = decimator(dataFlt0[0].reshape(1,-1), 120, 20.0, 20.8, bitWidth, calcType)

	#####################################################################################
	# stage 2 : input: 2 @ 60MSps, Fmix = 10.0MHz, Fc=10.2MHz, output: 4 stream @ 30MSps
	#####################################################################################
	dataFlt2 = decimator(dataFlt1, 60, 10.0, 10.8, bitWidth, calcType)

	#####################################################################################
	# stage 3 : input: 4 @ 30MSps, Fmix = 5.0MHz, Fc=5.2MHz, output: 8 stream @ 15MSps
	#####################################################################################
	dataFlt3 = decimator(dataFlt2, 30, 5.0, 5.8, bitWidth, calcType)
		
	#####################################################################################
	# stage 4 : input: 8 @ 15MSps, Fmix = 2.0MHz, Fc=3.2MHz, output: 16 stream @ 7.5MSps
	#####################################################################################
	dataFlt4_4M_0 = decimator(dataFlt3, 15, 0.5, 2.8, bitWidth, calcType)
	dataFlt4_4M_1 = decimator(dataFlt3, 15, 1.5, 2.8, bitWidth, calcType)
	dataFlt4_4M_2 = decimator(dataFlt3, 15, 2.5, 2.8, bitWidth, calcType)
	dataFlt4_4M_3 = decimator(dataFlt3, 15, 3.5, 2.8, bitWidth, calcType)
	dataFlt4_4M_4 = decimator(dataFlt3, 15, 4.5, 2.8, bitWidth, calcType)
	
	dataFlt4 = decimator(dataFlt3, 15, 2.0, 3.2, bitWidth, calcType)
	
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
	dataFlt5_2M_0 = decimator(dataFlt4, 7.5, 0.5, 1.2, bitWidth, calcType)
	dataFlt5_2M_1 = decimator(dataFlt4, 7.5, 1.5, 1.2, bitWidth, calcType)
	dataFlt5_2M_2 = decimator(dataFlt4, 7.5, 2.5, 1.2, bitWidth, calcType)
	
	dataFlt5 = decimator(dataFlt4, 7.5, 1.0, 1.7, bitWidth, calcType)

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
	dataFlt6_0 = decimator(dataFlt5, 3.75, 0.5, 0.7, bitWidth, calcType)
	dataFlt6_1 = decimator(dataFlt5, 3.75, 1.5, 0.7, bitWidth, calcType)
	
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
