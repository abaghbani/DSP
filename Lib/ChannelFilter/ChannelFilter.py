import numpy as np

import Filter as fd

from .Constant import *
C = Constant()

def ChannelFilter(data4M, data2M, data1M, channel, type, bitWidth = 2**17, calcType = np.int64):

	if type == C.ChannelFilterType.Gfsk1M:
		chFltGfsk1M = (C.chFltGfsk1M*bitWidth).astype(calcType)
		dataLowRateI = (np.convolve(chFltGfsk1M,data1M[channel].real,'same')/bitWidth).astype(calcType)
		dataLowRateQ = (np.convolve(chFltGfsk1M,data1M[channel].imag,'same')/bitWidth).astype(calcType)
		upsample_rate = 4
		fs = 1.875 * upsample_rate
	elif type == C.ChannelFilterType.Dpsk1M:
		chFltDpsk1M = (C.chFltDpsk1M*bitWidth).astype(calcType)
		dataLowRateI = (np.convolve(chFltDpsk1M,data1M[channel].real,'same')/bitWidth).astype(calcType)
		dataLowRateQ = (np.convolve(chFltDpsk1M,data1M[channel].imag,'same')/bitWidth).astype(calcType)
		upsample_rate = 4
		fs = 1.875 * upsample_rate
	elif type == C.ChannelFilterType.Gfsk2M:
		chFltGfsk2M = (C.chFltGfsk2M*bitWidth).astype(calcType)
		dataLowRateI = (np.convolve(chFltGfsk2M,data2M[channel].real,'same')/bitWidth).astype(calcType)
		dataLowRateQ = (np.convolve(chFltGfsk2M,data2M[channel].imag,'same')/bitWidth).astype(calcType)
		upsample_rate = 4
		fs = 3.75 * upsample_rate
	elif type == C.ChannelFilterType.Dpsk4M.ch1M:
		chFltDpskHdr1M = (C.chFltDpskHdr1M*bitWidth).astype(calcType)
		dataLowRateI = (np.convolve(chFltDpskHdr1M,data4M[channel].real,'same')/bitWidth).astype(calcType)
		dataLowRateQ = (np.convolve(chFltDpskHdr1M,data4M[channel].imag,'same')/bitWidth).astype(calcType)
		upsample_rate = 4
		fs = 7.5 * upsample_rate
	elif type == C.ChannelFilterType.Dpsk4M.ch2M:
		chFltDpskHdr2M = (C.chFltDpskHdr2M*bitWidth).astype(calcType)
		dataLowRateI = (np.convolve(chFltDpskHdr2M,data4M[channel].real,'same')/bitWidth).astype(calcType)
		dataLowRateQ = (np.convolve(chFltDpskHdr2M,data4M[channel].imag,'same')/bitWidth).astype(calcType)
		upsample_rate = 4
		fs = 7.5 * upsample_rate
	elif type == C.ChannelFilterType.Dpsk4M.ch4M:
		chFltDpskHdr4M = (C.chFltDpskHdr4M*bitWidth).astype(calcType)
		dataLowRateI = (np.convolve(chFltDpskHdr4M,data4M[channel].real,'same')/bitWidth).astype(calcType)
		dataLowRateQ = (np.convolve(chFltDpskHdr4M,data4M[channel].imag,'same')/bitWidth).astype(calcType)
		upsample_rate = 4
		fs = 7.5 * upsample_rate
	elif type == C.ChannelFilterType.Hdt2M:
		chFltHdt2M = (C.chFltHdt2M*bitWidth).astype(calcType)
		dataLowRateI = (np.convolve(chFltHdt2M, data2M[channel].real, 'same') / bitWidth).astype(calcType)
		dataLowRateQ = (np.convolve(chFltHdt2M, data2M[channel].imag, 'same') / bitWidth).astype(calcType)
		upsample_rate = 4
		fs = 3.75 * upsample_rate
	elif type == C.ChannelFilterType.Hdt4M:
		chFltHdt4M = (C.chFltHdt4M*bitWidth).astype(calcType)
		dataLowRateI = (np.convolve(chFltHdt4M, data4M[channel].real, 'same') / bitWidth).astype(calcType)
		dataLowRateQ = (np.convolve(chFltHdt4M, data4M[channel].imag, 'same') / bitWidth).astype(calcType)
		upsample_rate = 4
		fs = 7.5 * upsample_rate

	dataI = fd.cicFilter_upsample(dataLowRateI.astype(calcType), upsample_rate)
	dataQ = fd.cicFilter_upsample(dataLowRateQ.astype(calcType), upsample_rate)

	return dataI, dataQ, fs

def ChannelFilter_unused(data4M, data2M, data1M, channel, type):

	bitWidth = 2**17
	calcType = 'int64'
	
	if type == C.ChannelFilterType.Gfsk1M:
		chFltGfsk1M = (C.chFltGfsk1M*bitWidth).astype(calcType)
		dataLowRateI = (np.convolve(chFltGfsk1M,data1M[channel*2],'same')/bitWidth).astype(calcType)
		dataLowRateQ = (np.convolve(chFltGfsk1M,data1M[channel*2+1],'same')/bitWidth).astype(calcType)
		fs = 7.5
	elif type == C.ChannelFilterType.Dpsk1M:
		chFltDpsk1M = (C.chFltDpsk1M*bitWidth).astype(calcType)
		dataLowRateI = (np.convolve(chFltDpsk1M,data1M[channel*2],'same')/bitWidth).astype(calcType)
		dataLowRateQ = (np.convolve(chFltDpsk1M,data1M[channel*2+1],'same')/bitWidth).astype(calcType)
		fs = 7.5
	elif type == C.ChannelFilterType.Gfsk2M:
		chFltGfsk2M = (C.chFltGfsk2M*bitWidth).astype(calcType)
		dataLowRateI = (np.convolve(chFltGfsk2M,data2M[channel*2],'same')/bitWidth).astype(calcType)
		dataLowRateQ = (np.convolve(chFltGfsk2M,data2M[channel*2+1],'same')/bitWidth).astype(calcType)
		fs = 15.0
	elif type == C.ChannelFilterType.Dpsk4M.ch1M:
		chFltDpskHdr1M = (C.chFltDpskHdr1M*bitWidth).astype(calcType)
		dataLowRateI = (np.convolve(chFltDpskHdr1M,data4M[channel*2],'same')/bitWidth).astype(calcType)
		dataLowRateQ = (np.convolve(chFltDpskHdr1M,data4M[channel*2+1],'same')/bitWidth).astype(calcType)
		fs = 30.0
	elif type == C.ChannelFilterType.Dpsk4M.ch2M:
		chFltDpskHdr2M = (C.chFltDpskHdr2M*bitWidth).astype(calcType)
		dataLowRateI = (np.convolve(chFltDpskHdr2M,data4M[channel*2],'same')/bitWidth).astype(calcType)
		dataLowRateQ = (np.convolve(chFltDpskHdr2M,data4M[channel*2+1],'same')/bitWidth).astype(calcType)
		fs = 30.0
	elif type == C.ChannelFilterType.Dpsk4M.ch4M:
		chFltDpskHdr4M = (C.chFltDpskHdr4M*bitWidth).astype(calcType)
		dataLowRateI = (np.convolve(chFltDpskHdr4M,data4M[channel*2],'same')/bitWidth).astype(calcType)
		dataLowRateQ = (np.convolve(chFltDpskHdr4M,data4M[channel*2+1],'same')/bitWidth).astype(calcType)
		fs = 30.0
	elif type == C.ChannelFilterType.Hdt:
		chFltHdt4M = (C.chFltHdt4M*bitWidth).astype(calcType)
		dataLowRateI = (np.convolve(chFltHdt4M,data4M[channel*2],'same')/bitWidth).astype(calcType)
		dataLowRateQ = (np.convolve(chFltHdt4M,data4M[channel*2+1],'same')/bitWidth).astype(calcType)
		fs = 30.0

	dataI = fd.cicFilter(dataLowRateI.astype(calcType))
	dataQ = fd.cicFilter(dataLowRateQ.astype(calcType))
	return dataI, dataQ, fs

