import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import logging as log

import Spectrum as sp
import RfModel as rf
import ClockRecovery as cr

from .WpcTransmitter import *
from .WpcReceiver import *
from .WpcPacket import *
from .WpcCommon import *
from .WpcBER import *
from .WpcHdlModel import *
import WpcReceiverShift as demod_shift

def CompareAskData(txData, rxData, len):
	start_index_rx = 0
	for i in range(12, rxData.size):
		if np.all(rxData[i-12:i] == np.array([1,0,1,0,1,0,1,0,1,0,1,1])):
			start_index_rx = i
			tx_phase = 1
			break
		if np.all(rxData[i-12:i] == np.array([0,1,0,1,0,1,0,1,0,1,0,0])):
			start_index_rx = i
			tx_phase = 0
			break
	start_index_tx = 0
	for i in range(12, txData.size):
		if np.all(txData[i-12:i] == np.array([1,0,1,0,1,0,1,0,1,0,1,1])):
			rx_phase = 1
			start_index_tx = i
			break
		if np.all(txData[i-12:i] == np.array([0,1,0,1,0,1,0,1,0,1,0,0])):
			start_index_tx = i
			rx_phase = 0
			break

	if start_index_tx and start_index_rx:
		print(f'preamble is detected @index = tx: {start_index_tx}--{tx_phase}  rx: {start_index_rx}--{rx_phase}')
	else:
		print(f'preamble is not detected')
		return -1
	
	print(f'transmit data size = {txData.size-start_index_tx}  -- reseived data size = {rxData.size-start_index_rx}')
	error = 0
	error_index = []
	rxData = (-1*rxData)+1 if rx_phase != tx_phase else rxData

	if rxData.size-start_index_rx < 2*len or txData.size-start_index_tx < 2*len:
		print('Error: packet is not fully captured.')
		return -1

	for i in range(np.min([txData.size-start_index_tx, rxData.size-start_index_rx, 2*len])):
		if txData[start_index_tx+i] != rxData[start_index_rx+i]:
			error += 1
			error_index.append(i)
	print(f'number of error = {error}')
	#if error != 0:
	#	print(error_index)
	return error

def AskBERtest(packet_len, fop, SNRdb, model='new'):

	fs = 5.0e6
	modulation_index = 0.95
	bit_rate = 2.0e3
	amplitude = 1.0

	txData = WpcPacketGenerator(packet_len)
	sig = WpcModulation(txData, bit_rate*2, 'ask', modulation_index, amplitude, fop, fs)
	
	signal_power = np.mean(abs(sig**2))
	sigma2 = signal_power * 10**(-SNRdb/10)
	print (f'RX Signal power: {signal_power:.4f}, Noise power: {sigma2:.4f}   ===   SNR: {SNRdb} dB')
	noise_delay = int(39*fs/(2*bit_rate))
	sig[noise_delay:] += rf.whiteNoiseGen(sig.size-noise_delay, sigma2)

	data_format = 'float64'
	sig = (sig*(2**12)).astype(data_format)
	if model == 'new':
		data_flt = WpcFrontendFiltering(sig, fs, 10, type=data_format)
		ask_data, rssi = WpcAskDemodulator(data_flt, fs, type=data_format)
	elif model == 'shift':
		data_flt, temp, fs_low  = demod_shift.WpcFrontendFiltering(sig, fs, fop, 10, type=data_format)
		ask_data, rssi = demod_shift.WpcAskDemodulator(data_flt, fs_low, type=data_format)
	#rxData, rxIndex_ask = cr.Early_late(ask_data, 125, 10, plot_data=False)
	rxData, rxIndex_ask = cr.Early_late(ask_data, 25, 2, plot_data=False)

	error = 0
	if rxData.size:
		error = CompareAskData(txData, rxData, packet_len)
	else:
		print('no packet received')
	
	return error
	
def FskBERtest(packet_len, fop, SNRdb, model='new'):
	fs = 5.0e6
	modulation_index = 0.95
	bit_rate = fop/512
	amplitude = 1.0

	txData = WpcPacketGenerator(packet_len)
	sig = WpcModulation(txData, bit_rate*2, 'fsk', modulation_index, amplitude, fop, fs)
								
	signal_power = np.mean(abs(sig**2))
	sigma2 = signal_power * 10**(-SNRdb/10)
	print (f'RX Signal power: {signal_power:.4f}, Noise power: {sigma2:.4f}   ===   SNR: {SNRdb} dB')
	noise_delay = int(39*fs/(2*bit_rate))
	sig[noise_delay:] += rf.whiteNoiseGen(sig.size-noise_delay, sigma2)

	data_format = 'float64'
	sig = (sig*(2**12)).astype(data_format)
	if model == 'new':
		data_flt = WpcFrontendFiltering(sig, fs, 10, type=data_format)
		fsk_data, fsk_index, period =  WpcFskDemodulator(data_flt, fs, type=data_format)
		rxData, rxIndex_fsk = cr.Early_late(fsk_data, 8, 1)
	elif model == 'shift':
		data_flt, temp, fs_low  = demod_shift.WpcFrontendFiltering(sig, fs, fop, 10, type=data_format)
		fsk_data = demod_shift.WpcFskDemodulator(data_flt, fs_low, fop, type=data_format)
		rxData, rxIndex_ask = cr.Early_late(fsk_data, int((fs_low*256)/fop), 50)

	error = 0
	if rxData.size:
		error = CompareAskData(txData, rxData, packet_len)
	else:
		print('no packet received')

	return error