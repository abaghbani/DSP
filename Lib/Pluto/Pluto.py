import numpy as np
import matplotlib.pyplot as plt
import adi
import time

## pyadi-iio should be installed (dependency: pylibiio)

# rx gain mode could be "manual" for AGC enable: "slow_attack", "fast_attack"
# rx gain in manual mode : valid range is 0 to 74.5 dB
# tx gain : valid range is -90 to 0 dB
pluto_connection_ip = "ip:192.168.2.1"
pluto_connection_usb = 'usb:1.24.5'
pluto_connection_auto = 'ip:pluto.local'

class Pluto:
	
	def __init__(self, var):
		if var == 0:
			self.sdr = adi.Pluto(pluto_connection_ip)
		elif var == 1:
			self.sdr = adi.Pluto(pluto_connection_usb)		
		elif var == 2:
			self.sdr = adi.Pluto(pluto_connection_auto)
		elif var == 3:
			self.sdr = adi.Pluto()
	
	def info(self):
		print(self.sdr)
	
	def init(self, fs, bw, tx_lo, rx_lo):
		self.sdr.sample_rate = int(fs)
		self.sdr.tx_rf_bandwidth = int(bw)
		self.sdr.rx_rf_bandwidth = int(bw)
		self.sdr.tx_lo = int(tx_lo)
		self.sdr.rx_lo = int(rx_lo)

		self.sdr.tx_hardwaregain_chan0 = 0.0 #valid range is -90 to 0 dB
		self.sdr.gain_control_mode_chan0 = "slow_attack"

	def transmit(self, samples, loop_en):
		# if ch0 is enable: first 2 bytes is ch0.real, next 2 bytes is ch0.imag (12-bits-signed)
		# if ch1 is enable: third 2 bytes is ch1.real, next 2 bytes is ch1.imag (12-bits-signed)
		self.sdr.tx_destroy_buffer()
		self.sdr.tx_cyclic_buffer = loop_en
		self.sdr.tx(samples)

	def Read(self, SampleRate, BandWidth, LoFrequency, Gain, SampleNumber):
		self.sdr.sample_rate = int(SampleRate)
		self.sdr.rx_rf_bandwidth = int(BandWidth)
		self.sdr.rx_lo = int(LoFrequency)
		self.sdr.gain_control_mode_chan0 = "manual"
		self.sdr.rx_hardwaregain_chan0 = Gain
		
		sliceBufSize = 1024*100
		sliceNumber = int(np.ceil(SampleNumber/sliceBufSize))
		self.sdr.rx_buffer_size = sliceBufSize
		samples= np.zeros((sliceNumber, sliceBufSize), dtype='complex')
		
		start_time = time.time()
		for i in range(sliceNumber):
			samples[i] = self.sdr.rx()
		end_time = time.time()
		
		fs = self.sdr.sample_rate
		rxSamples = np.hstack(samples)
		print('sample freq: ', fs, 'sample size: ', rxSamples.size, 'required time(s): ', rxSamples.size/fs, 'capturing time(s): ', end_time-start_time)
		return rxSamples, fs

	def Write(self, SampleRate, BandWidth, LoFrequency, Gain, Samples, ota=0):
		self.sdr.sample_rate = int(SampleRate)
		self.sdr.tx_rf_bandwidth = int(BandWidth)
		self.sdr.tx_lo = int(LoFrequency)
		self.sdr.tx_hardwaregain_chan0 = Gain
		self.sdr.tx_cyclic_buffer = False
		print(self.sdr)
		start_time = time.time()
		self.sdr.tx(Samples)
		end_time = time.time()
		if ota != 0:
			while((end_time-start_time) < ota):
				end_time = time.time()
			self.sdr.tx_destroy_buffer()	
		
		fs = self.sdr.sample_rate
		print(f'sample freq: , {fs}, transmitting over the air time(s): , {end_time-start_time:.3f}')
		return fs
	
	def ReadWrite(self, SampleRate, BandWidth, LoFrequency, txGain, rxGain, txSamples, rxSampleNumber):
		self.sdr.sample_rate = int(SampleRate)
		
		self.sdr.rx_rf_bandwidth = int(BandWidth)
		self.sdr.rx_lo = int(LoFrequency)
		self.sdr.gain_control_mode_chan0 = "manual"
		self.sdr.rx_hardwaregain_chan0 = rxGain
		
		self.sdr.tx_rf_bandwidth = int(BandWidth)
		self.sdr.tx_lo = int(LoFrequency)
		self.sdr.tx_hardwaregain_chan0 = txGain
		
		self.sdr.tx_cyclic_buffer = True
		self.sdr.tx(txSamples)
		
		sliceBufSize = 1024*100
		sliceNumber = int(np.ceil(rxSampleNumber/sliceBufSize))
		self.sdr.rx_buffer_size = sliceBufSize
		rxSamples= np.zeros((sliceNumber, sliceBufSize), dtype='complex')
		
		## dump read to make buffer empty (not necessary, just to be safe)
		for i in range(3):
			rxSamples[i] = self.sdr.rx()

		start_time = time.time()
		for i in range(sliceNumber):
			rxSamples[i] = self.sdr.rx()
		end_time = time.time()
		
		self.sdr.tx_destroy_buffer()		
		
		fs = self.sdr.sample_rate
		rxSamples = np.hstack(rxSamples)
		print('sample freq: ', fs, 'sample size: ', rxSamples.size, 'required time(s): ', rxSamples.size/fs, 'capturing time(s): ', end_time-start_time)
		return rxSamples, fs

	def ExtraCommand(self):
		#loopback: Set loopback mode. Options are: 0 (Disable), 1 (Digital), 2 (RF)
		self.sdr.loopback = 0

	def DDS(self):
		self.sdr.dds_single_tone(80e3, 0.9)	
	def DDS_stop(self):
		self.sdr.disable_dds()