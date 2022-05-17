import numpy as np
import scipy.signal as signal

class Constant:
	
	## Rf-If parameteres
	AdcSamplingFrequency = 240.0e6
	IfMixerFrequency = -98.0e6	
	
	class ChannelFilterType:
		Gfsk1M = 1
		Gfsk2M = 2
		Dpsk1M = 3
		class Dpsk4M:
			ch1M = 4
			ch2M = 5
			ch4M = 6
		
	## Halfband fliter
	hbFlt0 = np.array([0.0041656494, 0.0, -0.0141029358, 0.0, 0.0363273621, 0.0, -0.0873146057, 0.0, 0.3115997314, 0.5, 0.3115997314, 0.0, -0.0873146057, 0.0, 0.0363273621, 0.0, -0.0141029358, 0.0, 0.0041656494])
	# filter level 1,2 and 3 for non-HDR design
	# hbFlt1 = np.array([+0.0016479492, +0.0000000000, -0.0087661743, +0.0000000000, +0.0289764404, +0.0000000000, -0.0807418823, +0.0000000000, +0.3089523315, +0.5000000000, +0.3089523315, +0.0000000000, -0.0807418823, +0.0000000000, +0.0289764404, +0.0000000000, -0.0087661743, +0.0000000000, +0.0016479492])
	# hbFlt2 = np.array([+0.0019836426, +0.0000000000, -0.0096817017, +0.0000000000, +0.0303916931, +0.0000000000, -0.0820884705, +0.0000000000, +0.3095092773, +0.5000000000, +0.3095092773, +0.0000000000, -0.0820884705, +0.0000000000, +0.0303916931, +0.0000000000, -0.0096817017, +0.0000000000, +0.0019836426])
	# hbFlt3 = np.array([+0.0029792786, +0.0000000000, -0.0119438171, +0.0000000000, +0.0335807800, +0.0000000000, -0.0849761963, +0.0000000000, +0.3106803894, +0.5000000000, +0.3106803894, +0.0000000000, -0.0849761963, +0.0000000000, +0.0335807800, +0.0000000000, -0.0119438171, +0.0000000000, +0.0029792786])
	
	# filter level 1,2 and 3 for HDR design
	hbFlt1 = np.array([-0.0008544922, +0.0000000000, +0.0043449402, +0.0000000000, -0.0138931274, +0.0000000000, +0.0356407166, +0.0000000000, -0.0865516663, +0.0000000000, +0.3112716675, +0.5000000000, +0.3112716675, +0.0000000000, -0.0865516663, +0.0000000000, +0.0356407166, +0.0000000000, -0.0138931274, +0.0000000000, +0.0043449402, +0.0000000000, -0.0008544922])
	hbFlt2 = np.array([-0.0018386841, +0.0000000000, +0.0067291260, +0.0000000000, -0.0177612305, +0.0000000000, +0.0401802063, +0.0000000000, -0.0902671814, +0.0000000000, +0.3127098083, +0.5000000000, +0.3127098083, +0.0000000000, -0.0902671814, +0.0000000000, +0.0401802063, +0.0000000000, -0.0177612305, +0.0000000000, +0.0067291260, +0.0000000000, -0.0018386841])
	hbFlt3 = np.array([+0.0010070801, +0.0000000000, -0.0019111633, +0.0000000000, +0.0036125183, +0.0000000000, -0.0062103271, +0.0000000000, +0.0100135803, +0.0000000000, -0.0154953003, +0.0000000000, +0.0234680176, +0.0000000000, -0.0356101990, +0.0000000000, +0.0562667847, +0.0000000000, -0.1015205383, +0.0000000000, +0.3167572021, +0.5000000000, +0.3167572021, +0.0000000000, -0.1015205383, +0.0000000000, +0.0562667847, +0.0000000000, -0.0356101990, +0.0000000000, +0.0234680176, +0.0000000000, -0.0154953003, +0.0000000000, +0.0100135803, +0.0000000000, -0.0062103271, +0.0000000000, +0.0036125183, +0.0000000000, -0.0019111633, +0.0000000000, +0.0010070801])
	
	hbFlt4 = np.array([-0.0008430481, +0.0000000000, +0.0026893616, +0.0000000000, -0.0065727234, +0.0000000000, +0.0136947632, +0.0000000000, -0.0260772705, +0.0000000000, +0.0482292175, +0.0000000000, -0.0961418152, +0.0000000000, +0.3148651123, +0.5000000000, +0.3148651123, +0.0000000000, -0.0961418152, +0.0000000000, +0.0482292175, +0.0000000000, -0.0260772705, +0.0000000000, +0.0136947632, +0.0000000000, -0.0065727234, +0.0000000000, +0.0026893616, +0.0000000000, -0.0008430481])
	hbFlt4_4M = np.array([+0.0017395020, +0.0000000000, -0.0050888062, +0.0000000000, +0.0118675232, +0.0000000000, -0.0241470337, +0.0000000000, +0.0465126038, +0.0000000000, -0.0949554443, +0.0000000000, +0.3144378662, +0.5000000000, +0.3144378662, +0.0000000000, -0.0949554443, +0.0000000000, +0.0465126038, +0.0000000000, -0.0241470337, +0.0000000000, +0.0118675232, +0.0000000000, -0.0050888062, +0.0000000000, +0.0017395020])
	hbFlt5 = np.array([+0.0029792786, +0.0000000000, -0.0119438171, +0.0000000000, +0.0335807800, +0.0000000000, -0.0849761963, +0.0000000000, +0.3106803894, +0.5000000000, +0.3106803894, +0.0000000000, -0.0849761963, +0.0000000000, +0.0335807800, +0.0000000000, -0.0119438171, +0.0000000000, +0.0029792786])
	hbFlt5_2M = np.array([+0.0004425049, 0.0000000000, -0.0022277832, 0.0000000000, +0.0071182251, 0.0000000000, -0.0179901123, 0.0000000000, +0.0402183533, 0.0000000000, -0.0902061462, 0.0000000000, +0.3126716614, 0.5000000000, +0.3126716614, 0.0000000000, -0.0902061462, 0.0000000000, +0.0402183533, 0.0000000000, -0.0179901123, 0.0000000000, +0.0071182251, 0.0000000000, -0.0022277832, 0.0000000000, +0.0004425049])
	hbFlt6 = np.array([+0.0017395020, +0.0000000000, -0.0050888062, +0.0000000000, +0.0118675232, +0.0000000000, -0.0241470337, +0.0000000000, +0.0465126038, +0.0000000000, -0.0949554443, +0.0000000000, +0.3144378662, +0.5000000000, +0.3144378662, +0.0000000000, -0.0949554443, +0.0000000000, +0.0465126038, +0.0000000000, -0.0241470337, +0.0000000000, +0.0118675232, +0.0000000000, -0.0050888062, +0.0000000000, +0.0017395020])
	
	## Channel fliter
	chFltGfsk1M = np.array([-2.8228759766e-04, +4.9667358398e-03, -3.9443969727e-03, -1.2725830078e-02, +2.1667480469e-02, +1.4610290527e-02, -6.1141967773e-02, +1.3160705566e-02, +1.2137603760e-01, -1.1454010010e-01, -2.0558166504e-01, +4.6765136719e-01, +1.0000000000e+00, +4.6765136719e-01, -2.0558166504e-01, -1.1454010010e-01, +1.2137603760e-01, +1.3160705566e-02, -6.1141967773e-02, +1.4610290527e-02, +2.1667480469e-02, -1.2725830078e-02, -3.9443969727e-03, +4.9667358398e-03, -2.8228759766e-04])
	chFltDpsk1M = np.array([+3.5095214844e-004, -4.4250488281e-004, +4.5776367188e-005, +1.9226074219e-003, -5.3482055664e-003, +4.9438476563e-003, +2.3345947266e-003, -1.9256591797e-002, +5.5557250977e-002, -3.8421630859e-002, -1.9121551514e-001, +4.1075134277e-001, +1.0000000000e+000, +4.1075134277e-001, -1.9121551514e-001, -3.8421630859e-002, +5.5557250977e-002, -1.9256591797e-002, +2.3345947266e-003, +4.9438476563e-003, -5.3482055664e-003, +1.9226074219e-003, +4.5776367188e-005, -4.4250488281e-004, +3.5095214844e-004])
	chFltGfsk2M = np.array([-0.0016098022, -0.0025405884, +0.0059661865, +0.0041198730, -0.0174102783, -0.0008392334, +0.0378952026, -0.0159835815, -0.0671920776, +0.0606842041, +0.1036605835, -0.1659164429, -0.1637573242, +0.5177917480, +1.0000000000, +0.5177917480, -0.1637573242, -0.1659164429, +0.1036605835, +0.0606842041, -0.0671920776, -0.0159835815, +0.0378952026, -0.0008392334, -0.0174102783, +0.0041198730, +0.0059661865, -0.0025405884, -0.0016098022])
	chFltDpskHdr1M = np.array([+4.5776367188e-04, +1.6021728516e-04, +9.1552734375e-05, -6.8664550781e-05, -2.6702880859e-04, -4.2724609375e-04, -4.5013427734e-04, -2.5939941406e-04, +1.4495849609e-04, +6.7138671875e-04, +1.1444091797e-03, +1.3046264648e-03, +9.6130371094e-04, +6.1035156250e-05, -1.2588500977e-03, -2.5939941406e-03, -3.4255981445e-03, -3.2577514648e-03, -1.8463134766e-03, +6.1798095703e-04, +3.4713745117e-03, +5.6838989258e-03, +6.1569213867e-03, +4.2037963867e-03, -9.1552734375e-05, -5.6152343750e-03, -1.0322570801e-02, -1.1802673340e-02, -8.0337524414e-03, +1.6174316406e-03, +1.5708923340e-02, +3.0471801758e-02, +4.0382385254e-02, +3.9489746094e-02, +2.3307800293e-02, -9.1857910156e-03, -5.4046630859e-02, -1.0188293457e-01, -1.3883209229e-01, -1.4897155762e-01, -1.1775207520e-01, -3.5835266113e-02, +9.7755432129e-02, +2.7394866943e-01, +4.7405242920e-01, +6.7238616943e-01, +8.4072113037e-01, +9.5360565186e-01, +1.0000000000e+00, +9.5360565186e-01, +8.4072113037e-01, +6.7238616943e-01, +4.7405242920e-01, +2.7394866943e-01, +9.7755432129e-02, -3.5835266113e-02, -1.1775207520e-01, -1.4897155762e-01, -1.3883209229e-01, -1.0188293457e-01, -5.4046630859e-02, -9.1857910156e-03, +2.3307800293e-02, +3.9489746094e-02, +4.0382385254e-02, +3.0471801758e-02, +1.5708923340e-02, +1.6174316406e-03, -8.0337524414e-03, -1.1802673340e-02, -1.0322570801e-02, -5.6152343750e-03, -9.1552734375e-05, +4.2037963867e-03, +6.1569213867e-03, +5.6838989258e-03, +3.4713745117e-03, +6.1798095703e-04, -1.8463134766e-03, -3.2577514648e-03, -3.4255981445e-03, -2.5939941406e-03, -1.2588500977e-03, +6.1035156250e-05, +9.6130371094e-04, +1.3046264648e-03, +1.1444091797e-03, +6.7138671875e-04, +1.4495849609e-04, -2.5939941406e-04, -4.5013427734e-04, -4.2724609375e-04, -2.6702880859e-04, -6.8664550781e-05, +9.1552734375e-05, +1.6021728516e-04, +4.5776367188e-04])
	chFltDpskHdr2M = np.array([-1.1444091797e-04, -7.6293945313e-06, +6.1035156250e-05, +6.8664550781e-05, -2.2888183594e-05, -1.2969970703e-04, -9.1552734375e-05, +9.9182128906e-05, +2.1362304688e-04, +3.8146972656e-05, -2.6702880859e-04, -3.1280517578e-04, +9.9182128906e-05, +5.3405761719e-04, +3.8146972656e-04, -3.4332275391e-04, -7.8582763672e-04, -2.2888183594e-04, +8.3923339844e-04, +1.0070800781e-03, -3.0517578125e-04, -1.7318725586e-03, -1.2664794922e-03, +1.0910034180e-03, +2.6626586914e-03, +1.0681152344e-03, -2.3269653320e-03, -3.1585693359e-03, +6.6375732422e-04, +5.1498413086e-03, +3.8299560547e-03, -3.8452148438e-03, -9.5825195313e-03, -4.7454833984e-03, +7.5531005859e-03, +1.2512207031e-02, +3.2806396484e-04, -1.7257690430e-02, -1.3870239258e-02, +2.0210266113e-02, +5.3802490234e-02, +3.3348083496e-02, -5.8746337891e-02, -1.5763092041e-01, -1.3954925537e-01, +8.3534240723e-02, +4.6975708008e-01, +8.4246063232e-01, +1.0000000000e+00, +8.4246063232e-01, +4.6975708008e-01, +8.3534240723e-02, -1.3954925537e-01, -1.5763092041e-01, -5.8746337891e-02, +3.3348083496e-02, +5.3802490234e-02, +2.0210266113e-02, -1.3870239258e-02, -1.7257690430e-02, +3.2806396484e-04, +1.2512207031e-02, +7.5531005859e-03, -4.7454833984e-03, -9.5825195313e-03, -3.8452148438e-03, +3.8299560547e-03, +5.1498413086e-03, +6.6375732422e-04, -3.1585693359e-03, -2.3269653320e-03, +1.0681152344e-03, +2.6626586914e-03, +1.0910034180e-03, -1.2664794922e-03, -1.7318725586e-03, -3.0517578125e-04, +1.0070800781e-03, +8.3923339844e-04, -2.2888183594e-04, -7.8582763672e-04, -3.4332275391e-04, +3.8146972656e-04, +5.3405761719e-04, +9.9182128906e-05, -3.1280517578e-04, -2.6702880859e-04, +3.8146972656e-05, +2.1362304688e-04, +9.9182128906e-05, -9.1552734375e-05, -1.2969970703e-04, -2.2888183594e-05, +6.8664550781e-05, +6.1035156250e-05, -7.6293945313e-06, -1.1444091797e-04])
	chFltDpskHdr4M = np.array([-3.8146972656e-05, +2.2888183594e-05, -1.5258789063e-05, -7.6293945313e-06, +3.8146972656e-05, -4.5776367188e-05, +2.2888183594e-05, +2.2888183594e-05, -7.6293945313e-05, +9.1552734375e-05, -4.5776367188e-05, -5.3405761719e-05, +1.5258789063e-04, -1.6784667969e-04, +6.8664550781e-05, +1.1444091797e-04, -2.8991699219e-04, +3.1280517578e-04, -1.0681152344e-04, -2.2125244141e-04, +5.1879882813e-04, -5.4931640625e-04, +1.5258789063e-04, +4.3487548828e-04, -9.0026855469e-04, +9.3841552734e-04, -2.2125244141e-04, -8.6975097656e-04, +1.6021728516e-03, -1.5792846680e-03, +3.4332275391e-04, +1.7471313477e-03, -3.0364990234e-03, +2.7008056641e-03, -4.8828125000e-04, -3.6315917969e-03, +6.4086914063e-03, -4.9896240234e-03, +2.8228759766e-04, +8.4915161133e-03, -1.6601562500e-02, +1.1497497559e-02, +4.3106079102e-03, -2.9174804688e-02, +7.2143554688e-02, -4.4242858887e-02, -2.0447540283e-01, +4.1822052002e-01, +1.0000000000e+00, +4.1822052002e-01, -2.0447540283e-01, -4.4242858887e-02, +7.2143554688e-02, -2.9174804688e-02, +4.3106079102e-03, +1.1497497559e-02, -1.6601562500e-02, +8.4915161133e-03, +2.8228759766e-04, -4.9896240234e-03, +6.4086914063e-03, -3.6315917969e-03, -4.8828125000e-04, +2.7008056641e-03, -3.0364990234e-03, +1.7471313477e-03, +3.4332275391e-04, -1.5792846680e-03, +1.6021728516e-03, -8.6975097656e-04, -2.2125244141e-04, +9.3841552734e-04, -9.0026855469e-04, +4.3487548828e-04, +1.5258789063e-04, -5.4931640625e-04, +5.1879882813e-04, -2.2125244141e-04, -1.0681152344e-04, +3.1280517578e-04, -2.8991699219e-04, +1.1444091797e-04, +6.8664550781e-05, -1.6784667969e-04, +1.5258789063e-04, -5.3405761719e-05, -4.5776367188e-05, +9.1552734375e-05, -7.6293945313e-05, +2.2888183594e-05, +2.2888183594e-05, -4.5776367188e-05, +3.8146972656e-05, -7.6293945313e-06, -1.5258789063e-05, +2.2888183594e-05, -3.8146972656e-05])

	def GenMixerCoeff(nSample, Fmix, Fs):
		n=np.arange(nSample)
		cosMix = np.cos((n*2*np.pi*Fmix/Fs)+0.06287)
		sinMix = np.sin((n*2*np.pi*Fmix/Fs)+0.06287)
		return cosMix, sinMix

	Fs = 240.0e6
	cosMix0, sinMix0 = GenMixerCoeff(160, 58.5e6, Fs)
	cosMix1, sinMix1 = GenMixerCoeff(5, 24.0e6, Fs/2)
	cosMix2, sinMix2 = GenMixerCoeff(15, 8.0e6, Fs/4)
	cosMix3, sinMix3 = GenMixerCoeff(15, 4.0e6, Fs/8)
	cosMix4, sinMix4 = GenMixerCoeff(15, 2.0e6, Fs/16)
	cosMix5, sinMix5 = GenMixerCoeff(15, 1.0e6, Fs/32)
	cosMix6, sinMix6 = GenMixerCoeff(15, 0.5e6, Fs/64)
	
	cosMix4_3p5, sinMix4_3p5 = GenMixerCoeff(30, -3.5e6, Fs/16)
	cosMix4_2p5, sinMix4_2p5 = GenMixerCoeff( 6, -2.5e6, Fs/16)
	cosMix4_1p5, sinMix4_1p5 = GenMixerCoeff(10, -1.5e6, Fs/16)
	cosMix4_0p5, sinMix4_0p5 = GenMixerCoeff(30, -0.5e6, Fs/16)
	
	cosMix5_0p5, sinMix5_0p5 = GenMixerCoeff(15, -0.5e6, Fs/32)
	cosMix5_1p5, sinMix5_1p5 = GenMixerCoeff(15, 1.5e6, Fs/32)

	def GenHalfBandFilterCoeff(nTap, Fcut, Fs):
		b = signal.remez(nTap+1, np.array([0., Fcut/Fs, 0.5-Fcut/Fs, 0.5]), [1,0])
		b[abs(b) <= 1e-4] = 0.0 # force all even coef to be exact zero (they are close to zero)
		b[int(nTap/2)] = 0.5 # force center coef to be exact 0.5 (it is close to 0.5)
		return b

	hbFlt0_gen = GenHalfBandFilterCoeff(18, 40.8, 240.0)
	hbFlt1_gen = GenHalfBandFilterCoeff(18, 16.8, 120.0)
	hbFlt2_gen = GenHalfBandFilterCoeff(18, 8.8, 60.0)
	hbFlt3_gen = GenHalfBandFilterCoeff(18, 4.8, 30.0)
	hbFlt4_gen = GenHalfBandFilterCoeff(30, 2.8, 15.0)
	hbFlt5_gen = GenHalfBandFilterCoeff(18, 1.2, 7.5)
	hbFlt6_gen = GenHalfBandFilterCoeff(26, 0.7, 3.75)

	def GenFIRCoeff(nTap, Fcut, Fs):
		b = signal.remez(nTap+1, np.array([0., Fcut/Fs, 0.5-Fcut/Fs, 0.5]), [1,0])
		return b


	#chFltGfsk1M_gen = GenFIRCoeff(24, 0.6, 1.875)
	#chFltDpsk1M_gen = GenFIRCoeff(24, 0.7, 1.875)
	#chFltGfsk1M_gen = GenFIRCoeff(28, 1.2, 3.75)
