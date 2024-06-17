import numpy as np
import matplotlib.pyplot as plt

import RfModel as rf
import Spectrum as sp
import ChannelFilter as cf

from .DpskModulation import *
from .DpskDemodulation import *
from .Constant import *

def DpskTransmitter(payload, channel, modulation_type, snr):
	baseband = DpskModulation(payload, modulation_type)
	symbol_rate = SymbolRate(modulation_type)
	upsample_rate = cf.Constant.AdcSamplingFrequency/symbol_rate
	baseband_up = np.sqrt(upsample_rate)*rf.UpSampling(baseband, upsample_rate)
	
	## modeling carrier frequency/phase offset in transmitter
	freq_offset = 0.0001
	phase_offset = 3*np.pi/8
	tx_mixer = rf.Mixer(baseband_up, (channel+cf.Constant.IfMixerFrequency+freq_offset)/cf.Constant.AdcSamplingFrequency, phase_offset)
	tx_sig = tx_mixer.real + rf.WhiteNoise(tx_mixer, snr)

	#sp.fftPlot(baseband.real, baseband.imag, n=2)
	#sp.fftPlot(baseband_up.real, baseband_up.imag, n=2)
	#sp.fftPlot(tx_mixer.real, tx_mixer.imag, n=2, fs=cf.Constant.AdcSamplingFrequency)
	#sp.fftPlot(tx_sig.real, fs=cf.Constant.AdcSamplingFrequency)
	
	return tx_sig

def ChannelFilterSelection(symbol_rate):
	if symbol_rate == 1:
		return cf.Constant.ChannelFilterType.Dpsk4M.ch1M
		#return cf.Constant.ChannelFilterType.Dpsk1M
	elif symbol_rate == 2:
		return cf.Constant.ChannelFilterType.Dpsk4M.ch2M
	elif symbol_rate == 4:
		return cf.Constant.ChannelFilterType.Dpsk4M.ch4M

def DpskReceiver(adcSamples, channel, modulation_type):
	(data4M, data2M, data1M) = cf.ChannelDecimate(adcSamples)
	(dataI, dataQ, fs) = cf.ChannelFilter(data4M, data2M, data1M, channel, ChannelFilterSelection(SymbolRate(modulation_type)))
	(rssi, sync, valid, data) = DpskDemodulation(dataI, dataQ, fs, modulation_type)
	return rssi, sync, valid, data
	
def DpskModem(channel, byte_number, modType, snr):
	payload = (np.random.rand(byte_number)*256).astype(np.uint8)
	tx_Sig = DpskTransmitter(payload, channel, modType, snr)

	lnaGain = 0.9*(2**15)/np.abs(tx_Sig).max()
	adcData = (lnaGain*tx_Sig).astype('int16')
	adcData = adcData//16

	fp = np.memmap('dpsk_ch'+str(channel)+'_type'+str(modType)+'.bin', mode='w+', dtype=np.dtype('<h'),shape=(1,adcData.size))
	fp[:] = adcData[:]

	print('transmit byte number=',byte_number)
	print('ADC Data: {} samples'.format(adcData.size))
	print('ADC Data Min/Max: ',adcData.min(),adcData.max(), type(adcData[0]))

	(rssi, sync, valid, data) = DpskReceiver(adcData, channel, modType)
	##CompareData(payload, rssi, sync, valid, data, modType)
