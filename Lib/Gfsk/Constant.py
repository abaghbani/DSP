
class Constant:
	
	GfskBT = 0.5
	GfskBrModulationIndex = [0.28, 0.35]
	GfskBleModulationIndex = [0.45, 0.55]
	GfskPreamble = [1,-1]*8
	
	class GfskModulationType:
		Gfsk1M = 1
		Gfsk2M = 2
	
	## Gfsk demodulator
	ModeRateThreshold = 11 # fix me! dependancy of sample rate (this is for 15Msps)
	FrequencyAvrageMaximum = 10
	
