import numpy as np

from Spectrum.Constant import Constant
from Spectrum.ModemLib import ModemLib
C = Constant
myLib = ModemLib(0)

def ChannelFilter(data4M, data2M, data1M, channel, type):

	bitWidth = 2**17
	calcType = 'int32'
	
	if type == C.ChannelFilterType.Gfsk1M:
		chFltGfsk1M = (C.chFltGfsk1M*bitWidth).astype(calcType)
		dataLowRateI = np.convolve(chFltGfsk1M,data1M[channel*2],'same')//bitWidth
		dataLowRateQ = np.convolve(chFltGfsk1M,data1M[channel*2+1],'same')//bitWidth
		fs = 7.5
	elif type == C.ChannelFilterType.Dpsk1M:
		chFltDpsk1M = (C.chFltDpsk1M*bitWidth).astype(calcType)
		dataLowRateI = np.convolve(chFltDpsk1M,data1M[channel*2],'same')//bitWidth
		dataLowRateQ = np.convolve(chFltDpsk1M,data1M[channel*2+1],'same')//bitWidth
		fs = 7.5
	elif type == C.ChannelFilterType.Gfsk2M:
		chFltGfsk2M = (C.chFltGfsk2M*bitWidth).astype(calcType)
		dataLowRateI = np.convolve(chFltGfsk2M,data2M[channel*2],'same')//bitWidth
		dataLowRateQ = np.convolve(chFltGfsk2M,data2M[channel*2+1],'same')//bitWidth
		fs = 15.0
	elif type == C.ChannelFilterType.Dpsk4M.ch1M:
		chFltDpskHdr1M = (C.chFltDpskHdr1M*bitWidth).astype(calcType)
		dataLowRateI = np.convolve(chFltDpskHdr1M,data4M[channel*2],'same')//bitWidth
		dataLowRateQ = np.convolve(chFltDpskHdr1M,data4M[channel*2+1],'same')//bitWidth
		fs = 30.0
	elif type == C.ChannelFilterType.Dpsk4M.ch2M:
		chFltDpskHdr2M = (C.chFltDpskHdr2M*bitWidth).astype(calcType)
		dataLowRateI = np.convolve(chFltDpskHdr2M,data4M[channel*2],'same')//bitWidth
		dataLowRateQ = np.convolve(chFltDpskHdr2M,data4M[channel*2+1],'same')//bitWidth
		fs = 30.0
	elif type == C.ChannelFilterType.Dpsk4M.ch4M:
		chFltDpskHdr4M = (C.chFltDpskHdr4M*bitWidth).astype(calcType)
		dataLowRateI = np.convolve(chFltDpskHdr4M,data4M[channel*2],'same')//bitWidth
		dataLowRateQ = np.convolve(chFltDpskHdr4M,data4M[channel*2+1],'same')//bitWidth
		fs = 30.0

	dataI = myLib.cicFilter(dataLowRateI.astype(calcType))
	dataQ = myLib.cicFilter(dataLowRateQ.astype(calcType))
	return dataI, dataQ, fs
