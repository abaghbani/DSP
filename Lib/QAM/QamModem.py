import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

import Spectrum as sp
import ChannelFilter as cf
import Filter as fd
import RfModel as rf

from .QamModulation import *
from .QamDemodulation import *
from .Constant import *
C = Constant()

def QamTransmitter(channel, payload, block_number, mod_type, snr):
	lts_seq = int(np.random.random(1)*16)
	txBaseband, fs = modulation(payload, block_number, mod_type, lts_seq)

	# first step of upsampling (2Msps -> 16Msps)
	fs *= 8
	txBaseband = txBaseband.repeat(8)
	# b = fd.rrc_2(141, 0.4, 8)
	b = fd.rrc(141, 0.4, 2.4, 16.0)
	txBaseband = np.convolve(b, txBaseband, 'same')
	
	# second step of upsampling (16Msps -> 240Msps)
	fs_RF = cf.Constant.AdcSamplingFrequency
	# tx_upsampled = rf.UpSampling(txBaseband, fs_RF//fs)	
	data_upsampled = txBaseband.repeat(fs_RF//fs)
	b = signal.firwin(141, 1/120, fs=1)
	tx_upsampled = np.convolve(b, data_upsampled, mode='same')
	
	freq_offset = (np.random.random(1)-0.5)/5	# offset -0.1 to +0.1 (-100KHz to +100KHz)
	phase_offset = 0 # (np.random.random(1)-0.5)*2*np.pi # offset -pi to +pi
	tx_mixer = rf.Mixer(tx_upsampled, cf.Constant.IfMixerFrequency+channel+freq_offset, phase_offset, fs_RF)
	tx_sig = tx_mixer.real + rf.WhiteNoise(tx_mixer, snr)

	print(f'transmitter: payload={payload.size=} bytes, freq_offset={float(freq_offset*1000):.3f} KHz, phase_offset={phase_offset*180/np.pi} Deg, {lts_seq=}')
	#plt.plot(txBaseband.real, txBaseband.imag, 'bo')
	#plt.grid()
	#plt.show()
	
	# sp.fftPlot(txBaseband.real, txBaseband.imag, n=2, fs=fs)
	# sp.fftPlot(tx_upsampled.real, tx_upsampled.imag, n=2, fs=fs_RF)
	# sp.fftPlot(tx_mixer.real, tx_mixer.imag, n=2, fs=fs_RF)
	# sp.fftPlot(tx_sig, fs=fs_RF)

	return tx_sig

def QamReceiver(adcSamples, channel, modulation_type):
	(data4M, data2M, data1M) = cf.ChannelDecimate(adcSamples, bitWidth = 1, calcType = 'float')
	# (dataI, dataQ, fs) = cf.ChannelFilter(data4M, data2M, data1M, channel, cf.Constant.ChannelFilterType.Hdt4M)
	
	bitWidth = 1 # 2**17
	calcType = 'float'
	
	chFltGfsk2M = np.array([-1.6098022461e-03, -2.5405883789e-03, +5.9661865234e-03, +4.1198730469e-03, -1.7410278320e-02, -8.3923339844e-04, +3.7895202637e-02, -1.5983581543e-02, -6.7192077637e-02, +6.0684204102e-02, +1.0366058350e-01, -1.6591644287e-01, -1.6375732422e-01, +5.1779174805e-01, +1.0000000000e+00, +5.1779174805e-01, -1.6375732422e-01, -1.6591644287e-01, +1.0366058350e-01, +6.0684204102e-02, -6.7192077637e-02, -1.5983581543e-02, +3.7895202637e-02, -8.3923339844e-04, -1.7410278320e-02, +4.1198730469e-03, +5.9661865234e-03, -2.5405883789e-03, -1.6098022461e-03])
	chFltHdt2M = np.array([-0.00191084, 0.00300883, -0.00323079, -0.00656127, 0.01445852, -0.01023644, -0.02178773, 0.04277787, -0.02036393, -0.06704888, 0.10807287, -0.02968232, -0.26372720, 0.38023531, 1.00000000, 0.38023531, -0.26372720, -0.02968232, 0.10807287, -0.06704888, -0.02036393, 0.04277787, -0.02178773, -0.01023644, 0.01445852, -0.00656127, -0.00323079, 0.00300883, -0.00191084])
	
	fs = 1.875 * 2
	def GenFIRCoeff(nTap, Fcut_off, Ftrans, Fs):
		b = signal.firls(nTap+1, np.array([0., Fcut_off-Ftrans, Fcut_off+Ftrans, fs/2]), [1, 1.4, 0, 0], fs=Fs)
		return b
	
	hdt2m_2p4MHz = GenFIRCoeff(48, 1.4, .22, fs)
	# hdt2m_2p4MHz /= hdt2m_2p4MHz[14]
	# w, h = signal.freqz(hdt2m_2p4MHz)
	# cm.ModemLib().plot_amp(w, h, fs, 'hdt2m_BW2p4MHz', False)
	# print('hdt2m_BW2p4MHz = ' + np.array2string(hdt2m_2p4MHz, separator=', ', formatter={'float':lambda x: " %1.8e" % x}, max_line_width=1000))

	chFlt = (hdt2m_2p4MHz*bitWidth).astype(calcType)
	dataLowRateI = (np.convolve(chFlt, data2M[channel].real, 'same') / bitWidth).astype(calcType)
	dataLowRateQ = (np.convolve(chFlt, data2M[channel].imag, 'same') / bitWidth).astype(calcType)
	dataI = fd.cicFilter(dataLowRateI)
	dataQ = fd.cicFilter(dataLowRateQ)
	fs = 15.0


	Demodulation(dataI+1j*dataQ, fs, modulation_type)
	
def QamModem(channel, byte_number, block_number, modulation_type, snr):
	payload = (np.random.rand(byte_number)*256).astype(np.uint8)
	IfSig = QamTransmitter(channel, payload, block_number, modulation_type, snr)

	# lnaGain = 0.9*(2**15)/np.abs(IfSig).max()
	# adcData = (IfSig*lnaGain).astype('int16')
	adcData = IfSig
	
	print(f'ADC Data: {adcData.size} samples, Min = {adcData.min()}, Max = {adcData.max()}, type={type(adcData[0])}')

	#print(f'transmit payload : {[hex(val)[2:] for val in payload]}')

	QamReceiver(adcData, channel, modulation_type)

	return adcData

def qam_modem_baseband(byte_number, block_number, mod_type, snr):
	payload = (np.random.rand(byte_number)*256).astype(np.uint8)
	lts_seq = int(np.random.random(1)*16)
	txBaseband, fs = modulation(payload, block_number, mod_type, lts_seq)

	fs *= 8
	txBaseband = txBaseband.repeat(8)
	# b = fd.rrc_2(141, 0.4, 8)
	b = fd.rrc(141, 0.4, 2.4, fs)
	txBaseband = np.convolve(b, txBaseband, 'same')

	freq_offset = (np.random.random(1)-0.5)/5	# offset -0.1 to +0.1 (-100KHz to +100KHz)
	phase_offset = (np.random.random(1)-0.5)*2*np.pi # offset -pi to +pi

	tx_mixer = rf.Mixer(txBaseband, freq_offset, phase_offset, fs)
	noise = rf.WhiteNoise(tx_mixer, snr)
	tx_sig = tx_mixer + (noise+1j*noise)

	# tx_sig *= (2**16)/np.abs(tx_sig).max()
	# rx_sig = (tx_sig.real//2)+1j*(tx_sig.imag//2)
	print(f'transmitter: payload={payload.size=} bytes, freq_offset={float(freq_offset*1000):.3f} KHz, phase_offset={float(phase_offset*180/np.pi):.2f} Deg, {lts_seq=}')
	Demodulation(tx_sig, fs, mod_type)
