import iio

## pyadi-iio should be installed (dependency: pylibiio)

# rx gain mode could be "manual" for AGC enable: "slow_attack", "fast_attack"
# rx gain in manual mode : valid range is 0 to 74.5 dB
# tx gain : valid range is -90 to 0 dB
pluto_connection_auto = 'ip:sdr.local'

class sdr:
	
	def __init__(self):
		ctx = iio.Context("ip:sdr.local")
		self.phy = ctx.find_device("ad9361-phy")

	def get_rx_lo(self):
		rx_lo = self.phy.find_channel("altvoltage0", True)
		return rx_lo.attrs["frequency"].value

	def get_tx_lo(self):
		tx_lo = self.phy.find_channel("altvoltage1", True)
		return tx_lo.attrs["frequency"].value

	def set_rx_lo(self, freq):
		rx_lo = self.phy.find_channel("altvoltage0", True)
		rx_lo.attrs["frequency"].value = str(int(freq))

	def set_tx_lo(self, freq):
		tx_lo = self.phy.find_channel("altvoltage1", True)
		tx_lo.attrs["frequency"].value = str(int(freq))

	def set_sample_rate(self, fs):
		self.phy.find_channel("voltage0").attrs["sampling_frequency"].value = str(int(fs))

	def set_band_width(self, bw):
		self.phy.find_channel("voltage0").attrs["rf_bandwidth"].value = str(int(bw))

	def set_filter_config(self, filename):
		with open(filename, "r") as f:
			fir_cfg = f.read()

		self.phy.attrs["filter_fir_config"].value = fir_cfg
		self.phy.attrs["trx_fir_enable"].value = "1"   # enable FIRs
		self.phy.attrs["initialize"].value = "1"       # reinitialize

	def disable_filter_config(self):
		self.phy.attrs["trx_fir_enable"].value = "0"

	def info(self):
		for ch in self.phy.channels:
			print(f"Channel: {ch.id}")
			if hasattr(ch, "attrs"):
				for attr_name, attr in ch.attrs.items():
					try:
						print(f"  {attr_name} = {attr.value}")
					except Exception as e:
						print(f"  {attr_name} = <error: {e}>")

	def init(self, SampleRate, BandWidth, tx_LoFrequency, rx_LoFrequency):
		self.set_sample_rate(SampleRate)
		self.set_band_width(BandWidth)
		self.set_tx_lo(tx_LoFrequency)
		self.set_rx_lo(rx_LoFrequency)


'''
| Channel                                        | Direction    | Meaning                                                                   |
| ---------------------------------------------- | ------------ | ------------------------------------------------------------------------- |
| `altvoltage0`                                  | Output       | RX LO synthesizer (local oscillator frequency)                            |
| `altvoltage1`                                  | Output       | TX LO synthesizer                                                         |
| `out`                                          | Output       | Global control attributes (often PLL or calibration)                      |
| `temp0`                                        | Input        | On-chip temperature sensor                                                |
| `voltage0`, `voltage1`, `voltage2`, `voltage3` | Input/output | RX/TX I/Q channels — each usually appears twice (once input, once output) |



Channel: altvoltage0
  frequency = 2401999998
  frequency_available = [70000000 1 6000000000]
Channel: altvoltage1
  frequency = 2441000000
  frequency_available = [46875001 1 6000000000]
Channel: voltage0
  gain_control_mode = slow_attack
  gain_control_mode_available = manual fast_attack slow_attack hybrid
  hardwaregain = 71.000000 dB
  hardwaregain_available = [-3 1 71]
  rf_bandwidth = 20000000
  rf_bandwidth_available = [200000 1 56000000]
  sampling_frequency = 56000000
  sampling_frequency_available = [2083333 1 61440000]
Channel: voltage0
  hardwaregain = 0.000000 dB
  hardwaregain_available = [-89.750000 0.250000 0.000000]
  rf_bandwidth = 56000000
  rf_bandwidth_available = [200000 1 40000000]
  rf_port_select = A
  rf_port_select_available = A B
  sampling_frequency = 56000000
  sampling_frequency_available = [2083333 1 61440000]
Channel: voltage1
  hardwaregain = 0.000000 dB
  hardwaregain_available = [-89.750000 0.250000 0.000000]
  rf_bandwidth = 56000000
  rf_bandwidth_available = [200000 1 40000000]
  rf_port_select = A
  rf_port_select_available = A B
  sampling_frequency = 56000000
  sampling_frequency_available = [2083333 1 61440000]
Channel: voltage1
  bb_dc_offset_tracking_en = 1
  filter_fir_en = 0
  gain_control_mode = slow_attack
  gain_control_mode_available = manual fast_attack slow_attack hybrid
  hardwaregain = 71.000000 dB
  hardwaregain_available = [-3 1 71]
  quadrature_tracking_en = 1
  rf_bandwidth = 20000000
  rf_bandwidth_available = [200000 1 56000000]
  rf_dc_offset_tracking_en = 1
  rf_port_select = A_BALANCED
  rf_port_select_available = A_BALANCED B_BALANCED C_BALANCED A_N A_P B_N B_P C_N C_P TX_MONITOR1 TX_MONITOR2 TX_MONITOR1_2
  rssi = 116.25 dB
  sampling_frequency = 56000000
  sampling_frequency_available = [2083333 1 61440000]
Channel: voltage2
  filter_fir_en = 0
  raw = 306
  rf_bandwidth = 56000000
  rf_bandwidth_available = [200000 1 40000000]
  rf_port_select_available = A B
  sampling_frequency = 56000000
  sampling_frequency_available = [2083333 1 61440000]
  scale = 1.000000
Channel: voltage2
  bb_dc_offset_tracking_en = 1
  filter_fir_en = 0
  gain_control_mode_available = manual fast_attack slow_attack hybrid
  offset = 57
  quadrature_tracking_en = 1
  raw = 0
  rf_bandwidth = 20000000
  rf_bandwidth_available = [200000 1 56000000]
  rf_dc_offset_tracking_en = 1
  rf_port_select_available = A_BALANCED B_BALANCED C_BALANCED A_N A_P B_N B_P C_N C_P TX_MONITOR1 TX_MONITOR2 TX_MONITOR1_2
  sampling_frequency = 56000000
  sampling_frequency_available = [2083333 1 61440000]
  scale = 0.305250
Channel: voltage3
  filter_fir_en = 0
  raw = 306
  rf_bandwidth = 56000000
  rf_bandwidth_available = [200000 1 40000000]
  rf_port_select_available = A B
  sampling_frequency = 56000000
  sampling_frequency_available = [2083333 1 61440000]
  scale = 1.000000

calib_mode = auto
calib_mode_available = auto manual manual_tx_quad tx_quad rf_dc_offs rssi_gain_step
dcxo_tune_coarse = <error: [Errno 19] No such device>
dcxo_tune_coarse_available = [0 0 0]
dcxo_tune_fine = <error: [Errno 19] No such device>
dcxo_tune_fine_available = [0 0 0]
ensm_mode = fdd
ensm_mode_available = sleep wait alert fdd pinctrl pinctrl_fdd_indep
filter_fir_config = FIR Rx: 0,0 Tx: 0,0
gain_table_config = <error: [Errno 5] Input/output error>        
multichip_sync = <error: [Errno 13] Permission denied>
rssi_gain_step_error = lna_error: 0 0 0 0
mixer_error: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
gain_step_calib_reg_val: 0 0 0 0 0
rx_path_rates = BBPLL:896000000 ADC:448000000 R2:224000000 R1:112000000 RF:56000000 RXSAMP:56000000
trx_rate_governor = nominal
trx_rate_governor_available = nominal highest_osr
tx_path_rates = BBPLL:896000000 DAC:224000000 T2:112000000 T1:56000000 TF:56000000 TXSAMP:56000000
xo_correction = 40000000
xo_correction_available = [39992000 1 40008000]
'''
