import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

sys.path.insert(1, sys.path[0]+'\..\Common')
from ModemLib			import ModemLib
myLib = ModemLib(0)

if len(sys.argv) > 1:
	filename = sys.argv[1]
else:
	filename = 'data2.bttraw'

if len(sys.argv) > 2:
	RequiredChannel = int(sys.argv[2])
else:
	RequiredChannel = 0

Fs = 240.0e+6

data = np.memmap(filename, mode='r', dtype=np.dtype('<h'))
# data = data[6500000:6700000]
# data = data[215000:400000]
print 'Data: {} samples'.format(data.size)
data = data << (16 - 12)
data = data >> (16 - 12)
print 'Data Min/Max: ',data.min(),data.max(), type(data[0])

n=np.arange(data.size)
bitWidth = 2**17
calcType = 'int32'

# input data : [-98M (2402) ... -19M (2481)],   [[2402:2481]-2260(rf_mixer)]sampling@240MHz => nyquist1 left side = -98:-19

# mixer:
# mixed_I = I*cos(fmix) - Q*sin(fmix)
# mixed_Q = Q*cos(fmix) + I*sin(fmix)

# stage 0 : Fs = 240MHz, Fmix = 58.5MHz, Fc=40.2MHz channel=1 [output = -39.5M:39.5M]
fmix0 = 58.5e+6
cosMix0 = (np.cos((n*2*np.pi*fmix0/Fs)+0.06287)*bitWidth).astype(calcType)
sinMix0 = (np.sin((n*2*np.pi*fmix0/Fs)+0.06287)*bitWidth).astype(calcType)
dataMix0 = [[0]*n for i in range(2)]
dataMix0[0] = np.multiply(data, cosMix0)/(2**(17-6))#bitWidth   # input data is 12-bits and here we convert it to 12+6 bits
dataMix0[1] = np.multiply(data, sinMix0)/(2**(17-6))#bitWidth

hbFlt0 = np.array([0.0041656494, 0.0, -0.0141029358, 0.0, 0.0363273621, 0.0, -0.0873146057, 0.0, 0.3115997314, 0.5, 0.3115997314, 0.0, -0.0873146057, 0.0, 0.0363273621, 0.0, -0.0141029358, 0.0, 0.0041656494])
hbFlt0 = (hbFlt0*bitWidth).astype(calcType)
dataFlt0 = [[0]*n for i in range(2)]
for i in range(1*2):
	# dataFlt0[i] = (signal.filtfilt(hbFlt0,1,dataMix0[i])/bitWidth).astype(calcType)
	dataFlt0[i] = np.convolve(hbFlt0,dataMix0[i],'same')/bitWidth
	dataFlt0[i] = dataFlt0[i][::2]

# stage 1 : Fs = 120MHz, Fmix = 24.0MHz, Fc=16.2MHz channel=2+1
fmix1 = 24.0e+6
n=np.arange(dataFlt0[0].size)
cosMix1 = (np.cos((n*2*np.pi*fmix1/(Fs/2))+0.06287)*bitWidth).astype(calcType)
sinMix1 = (np.sin((n*2*np.pi*fmix1/(Fs/2))+0.06287)*bitWidth).astype(calcType)
dataMix1 = [[0]*n for i in range(6)]
for i in range(2/2):
	dataMix1[i*4]   = (np.multiply(dataFlt0[i*2],   cosMix1) - np.multiply(dataFlt0[i*2+1], sinMix1))/bitWidth
	dataMix1[i*4+1] = (np.multiply(dataFlt0[i*2+1], cosMix1) + np.multiply(dataFlt0[i*2],   sinMix1))/bitWidth
	dataMix1[i*4+2] = (np.multiply(dataFlt0[i*2],   cosMix1) + np.multiply(dataFlt0[i*2+1], sinMix1))/bitWidth
	dataMix1[i*4+3] = (np.multiply(dataFlt0[i*2+1], cosMix1) - np.multiply(dataFlt0[i*2],   sinMix1))/bitWidth
dataMix1[4] = dataFlt0[0]
dataMix1[5] = dataFlt0[1]

# hbFlt1 = np.array([+0.0016479492, +0.0000000000, -0.0087661743, +0.0000000000, +0.0289764404, +0.0000000000, -0.0807418823, +0.0000000000, +0.3089523315, +0.5000000000, +0.3089523315, +0.0000000000, -0.0807418823, +0.0000000000, +0.0289764404, +0.0000000000, -0.0087661743, +0.0000000000, +0.0016479492])
hbFlt1 = np.array([-0.0008544922, +0.0000000000, +0.0043449402, +0.0000000000, -0.0138931274, +0.0000000000, +0.0356407166, +0.0000000000, -0.0865516663, +0.0000000000, +0.3112716675, +0.5000000000, +0.3112716675, +0.0000000000, -0.0865516663, +0.0000000000, +0.0356407166, +0.0000000000, -0.0138931274, +0.0000000000, +0.0043449402, +0.0000000000, -0.0008544922])
hbFlt1 = (hbFlt1*bitWidth).astype(calcType)
dataFlt1 = [[0]*n for i in range(6)]
for i in range(3*2):
	dataFlt1[i] = np.convolve(hbFlt1,dataMix1[i],'same')/bitWidth
	dataFlt1[i] = dataFlt1[i][::2]

# stage 2 : Fs = 60MHz, Fmix = 8.0MHz, Fc=8.2MHz channel=4+1
fmix2 = 8.0e+6
n=np.arange(dataFlt1[0].size)
cosMix2 = (np.cos((n*2*np.pi*fmix2/(Fs/4))+0.06287)*bitWidth).astype(calcType)
sinMix2 = (np.sin((n*2*np.pi*fmix2/(Fs/4))+0.06287)*bitWidth).astype(calcType)
dataMix2 = [[0]*n for i in range(10)]
for i in range(4/2):
	dataMix2[i*4]   = (np.multiply(dataFlt1[i*2],   cosMix2) - np.multiply(dataFlt1[i*2+1], sinMix2))/bitWidth
	dataMix2[i*4+1] = (np.multiply(dataFlt1[i*2+1], cosMix2) + np.multiply(dataFlt1[i*2],   sinMix2))/bitWidth
	dataMix2[i*4+2] = (np.multiply(dataFlt1[i*2],   cosMix2) + np.multiply(dataFlt1[i*2+1], sinMix2))/bitWidth
	dataMix2[i*4+3] = (np.multiply(dataFlt1[i*2+1], cosMix2) - np.multiply(dataFlt1[i*2],   sinMix2))/bitWidth
dataMix2[8] = dataFlt1[4]
dataMix2[9] = dataFlt1[5]

# hbFlt2 = np.array([+0.0019836426, +0.0000000000, -0.0096817017, +0.0000000000, +0.0303916931, +0.0000000000, -0.0820884705, +0.0000000000, +0.3095092773, +0.5000000000, +0.3095092773, +0.0000000000, -0.0820884705, +0.0000000000, +0.0303916931, +0.0000000000, -0.0096817017, +0.0000000000, +0.0019836426])
hbFlt2 = np.array([-0.0018386841, +0.0000000000, +0.0067291260, +0.0000000000, -0.0177612305, +0.0000000000, +0.0401802063, +0.0000000000, -0.0902671814, +0.0000000000, +0.3127098083, +0.5000000000, +0.3127098083, +0.0000000000, -0.0902671814, +0.0000000000, +0.0401802063, +0.0000000000, -0.0177612305, +0.0000000000, +0.0067291260, +0.0000000000, -0.0018386841])
hbFlt2 = (hbFlt2*bitWidth).astype(calcType)
dataFlt2 = [[0]*n for i in range(10)]
for i in range(5*2):
	dataFlt2[i] = np.convolve(hbFlt2,dataMix2[i],'same')/bitWidth
	dataFlt2[i] = dataFlt2[i][::2]

# stage 3 : Fs = 30MHz, Fmix = 4.0MHz, Fc=4.2MHz channel=8+2
fmix3 = 4.0e+6
n=np.arange(dataFlt2[0].size)
cosMix3 = (np.cos((n*2*np.pi*fmix3/(Fs/8))+0.06287)*bitWidth).astype(calcType)
sinMix3 = (np.sin((n*2*np.pi*fmix3/(Fs/8))+0.06287)*bitWidth).astype(calcType)
dataMix3 = [[0]*n for i in range(20)]
for i in range(10/2):
	dataMix3[i*4]   = (np.multiply(dataFlt2[i*2],   cosMix3) - np.multiply(dataFlt2[i*2+1], sinMix3))/bitWidth
	dataMix3[i*4+1] = (np.multiply(dataFlt2[i*2+1], cosMix3) + np.multiply(dataFlt2[i*2],   sinMix3))/bitWidth
	dataMix3[i*4+2] = (np.multiply(dataFlt2[i*2],   cosMix3) + np.multiply(dataFlt2[i*2+1], sinMix3))/bitWidth
	dataMix3[i*4+3] = (np.multiply(dataFlt2[i*2+1], cosMix3) - np.multiply(dataFlt2[i*2],   sinMix3))/bitWidth

# hbFlt3 = np.array([+0.0029792786, +0.0000000000, -0.0119438171, +0.0000000000, +0.0335807800, +0.0000000000, -0.0849761963, +0.0000000000, +0.3106803894, +0.5000000000, +0.3106803894, +0.0000000000, -0.0849761963, +0.0000000000, +0.0335807800, +0.0000000000, -0.0119438171, +0.0000000000, +0.0029792786])
hbFlt3 = np.array([+0.0010070801, +0.0000000000, -0.0019111633, +0.0000000000, +0.0036125183, +0.0000000000, -0.0062103271, +0.0000000000, +0.0100135803, +0.0000000000, -0.0154953003, +0.0000000000, +0.0234680176, +0.0000000000, -0.0356101990, +0.0000000000, +0.0562667847, +0.0000000000, -0.1015205383, +0.0000000000, +0.3167572021, +0.5000000000, +0.3167572021, +0.0000000000, -0.1015205383, +0.0000000000, +0.0562667847, +0.0000000000, -0.0356101990, +0.0000000000, +0.0234680176, +0.0000000000, -0.0154953003, +0.0000000000, +0.0100135803, +0.0000000000, -0.0062103271, +0.0000000000, +0.0036125183, +0.0000000000, -0.0019111633, +0.0000000000, +0.0010070801])
hbFlt3 = (hbFlt3*bitWidth).astype(calcType)
dataFlt3 = [[0]*n for i in range(20)]
for i in range(10*2):
	dataFlt3[i] = np.convolve(hbFlt3,dataMix3[i],'same')/bitWidth
	dataFlt3[i] = dataFlt3[i][::2]

# stage 4 : Fs = 15MHz, Fmix = 2.0MHz, Fc=2.2MHz channel=16+4
fmix4 = 2.0e+6
n=np.arange(dataFlt3[0].size)
cosMix4 = (np.cos((n*2*np.pi*fmix4/(Fs/16))+0.06287)*bitWidth).astype(calcType)
sinMix4 = (np.sin((n*2*np.pi*fmix4/(Fs/16))+0.06287)*bitWidth).astype(calcType)
dataMix4 = [[0]*n for i in range(40)]
for i in range(20/2):
	dataMix4[i*4]   = (np.multiply(dataFlt3[i*2],   cosMix4) - np.multiply(dataFlt3[i*2+1], sinMix4))/bitWidth
	dataMix4[i*4+1] = (np.multiply(dataFlt3[i*2+1], cosMix4) + np.multiply(dataFlt3[i*2],   sinMix4))/bitWidth
	dataMix4[i*4+2] = (np.multiply(dataFlt3[i*2],   cosMix4) + np.multiply(dataFlt3[i*2+1], sinMix4))/bitWidth
	dataMix4[i*4+3] = (np.multiply(dataFlt3[i*2+1], cosMix4) - np.multiply(dataFlt3[i*2],   sinMix4))/bitWidth

hbFlt4 = np.array([-0.0008430481, +0.0000000000, +0.0026893616, +0.0000000000, -0.0065727234, +0.0000000000, +0.0136947632, +0.0000000000, -0.0260772705, +0.0000000000, +0.0482292175, +0.0000000000, -0.0961418152, +0.0000000000, +0.3148651123, +0.5000000000, +0.3148651123, +0.0000000000, -0.0961418152, +0.0000000000, +0.0482292175, +0.0000000000, -0.0260772705, +0.0000000000, +0.0136947632, +0.0000000000, -0.0065727234, +0.0000000000, +0.0026893616, +0.0000000000, -0.0008430481])
hbFlt4 = (hbFlt4*bitWidth).astype(calcType)
dataFlt4 = [[0]*n for i in range(40)]
for i in range(20*2):
	dataFlt4[i] = np.convolve(hbFlt4,dataMix4[i],'same')/bitWidth
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
for i in range(20/2):
	dataMix4_4M[i*16]    = (np.multiply(dataFlt3[i*2],   cosMix4_3p5) - np.multiply(dataFlt3[i*2+1], sinMix4_3p5))/bitWidth
	dataMix4_4M[i*16+1]  = (np.multiply(dataFlt3[i*2+1], cosMix4_3p5) + np.multiply(dataFlt3[i*2],   sinMix4_3p5))/bitWidth
	dataMix4_4M[i*16+2]  = (np.multiply(dataFlt3[i*2],   cosMix4_2p5) - np.multiply(dataFlt3[i*2+1], sinMix4_2p5))/bitWidth
	dataMix4_4M[i*16+3]  = (np.multiply(dataFlt3[i*2+1], cosMix4_2p5) + np.multiply(dataFlt3[i*2],   sinMix4_2p5))/bitWidth
	dataMix4_4M[i*16+4]  = (np.multiply(dataFlt3[i*2],   cosMix4_1p5) - np.multiply(dataFlt3[i*2+1], sinMix4_1p5))/bitWidth
	dataMix4_4M[i*16+5]  = (np.multiply(dataFlt3[i*2+1], cosMix4_1p5) + np.multiply(dataFlt3[i*2],   sinMix4_1p5))/bitWidth
	dataMix4_4M[i*16+6]  = (np.multiply(dataFlt3[i*2],   cosMix4_0p5) - np.multiply(dataFlt3[i*2+1], sinMix4_0p5))/bitWidth
	dataMix4_4M[i*16+7]  = (np.multiply(dataFlt3[i*2+1], cosMix4_0p5) + np.multiply(dataFlt3[i*2],   sinMix4_0p5))/bitWidth
	dataMix4_4M[i*16+8]  = (np.multiply(dataFlt3[i*2],   cosMix4_0p5) + np.multiply(dataFlt3[i*2+1], sinMix4_0p5))/bitWidth
	dataMix4_4M[i*16+9]  = (np.multiply(dataFlt3[i*2+1], cosMix4_0p5) - np.multiply(dataFlt3[i*2],   sinMix4_0p5))/bitWidth
	dataMix4_4M[i*16+10] = (np.multiply(dataFlt3[i*2],   cosMix4_1p5) + np.multiply(dataFlt3[i*2+1], sinMix4_1p5))/bitWidth
	dataMix4_4M[i*16+11] = (np.multiply(dataFlt3[i*2+1], cosMix4_1p5) - np.multiply(dataFlt3[i*2],   sinMix4_1p5))/bitWidth
	dataMix4_4M[i*16+12] = (np.multiply(dataFlt3[i*2],   cosMix4_2p5) + np.multiply(dataFlt3[i*2+1], sinMix4_2p5))/bitWidth
	dataMix4_4M[i*16+13] = (np.multiply(dataFlt3[i*2+1], cosMix4_2p5) - np.multiply(dataFlt3[i*2],   sinMix4_2p5))/bitWidth
	dataMix4_4M[i*16+14] = (np.multiply(dataFlt3[i*2],   cosMix4_3p5) + np.multiply(dataFlt3[i*2+1], sinMix4_3p5))/bitWidth
	dataMix4_4M[i*16+15] = (np.multiply(dataFlt3[i*2+1], cosMix4_3p5) - np.multiply(dataFlt3[i*2],   sinMix4_3p5))/bitWidth
	
hbFlt4_4M = np.array([+0.0017395020, +0.0000000000, -0.0050888062, +0.0000000000, +0.0118675232, +0.0000000000, -0.0241470337, +0.0000000000, +0.0465126038, +0.0000000000, -0.0949554443, +0.0000000000, +0.3144378662, +0.5000000000, +0.3144378662, +0.0000000000, -0.0949554443, +0.0000000000, +0.0465126038, +0.0000000000, -0.0241470337, +0.0000000000, +0.0118675232, +0.0000000000, -0.0050888062, +0.0000000000, +0.0017395020])
hbFlt4_4M = (hbFlt4_4M*bitWidth).astype(calcType)
dataFlt4_4M = [[0]*n for i in range(80*2)]
for i in range(80*2):
	dataFlt4_4M[i] = np.convolve(hbFlt4_4M,dataMix4_4M[i],'same')/bitWidth
	dataFlt4_4M[i] = dataFlt4_4M[i][::2]

# stage 5 : Fs = 7.5MHz, Fmix = 1.0MHz, Fc=1.2MHz channel=32+8
fmix5 = 1.0e+6
n=np.arange(dataFlt4[0].size)
cosMix5 = (np.cos((n*2*np.pi*fmix5/(Fs/32))+0.06287)*bitWidth).astype(calcType)
sinMix5 = (np.sin((n*2*np.pi*fmix5/(Fs/32))+0.06287)*bitWidth).astype(calcType)
dataMix5 = [[0]*n for i in range(80)]
for i in range(40/2):
	dataMix5[i*4]   = (np.multiply(dataFlt4[i*2],   cosMix5) - np.multiply(dataFlt4[i*2+1], sinMix5))/bitWidth
	dataMix5[i*4+1] = (np.multiply(dataFlt4[i*2+1], cosMix5) + np.multiply(dataFlt4[i*2],   sinMix5))/bitWidth
	dataMix5[i*4+2] = (np.multiply(dataFlt4[i*2],   cosMix5) + np.multiply(dataFlt4[i*2+1], sinMix5))/bitWidth
	dataMix5[i*4+3] = (np.multiply(dataFlt4[i*2+1], cosMix5) - np.multiply(dataFlt4[i*2],   sinMix5))/bitWidth

hbFlt5 = np.array([+0.0029792786, +0.0000000000, -0.0119438171, +0.0000000000, +0.0335807800, +0.0000000000, -0.0849761963, +0.0000000000, +0.3106803894, +0.5000000000, +0.3106803894, +0.0000000000, -0.0849761963, +0.0000000000, +0.0335807800, +0.0000000000, -0.0119438171, +0.0000000000, +0.0029792786])
hbFlt5 = (hbFlt5*bitWidth).astype(calcType)
dataFlt5 = [[0]*n for i in range(80)]
for i in range(40*2):
	dataFlt5[i] = np.convolve(hbFlt5,dataMix5[i],'same')/bitWidth
	dataFlt5[i] = dataFlt5[i][::2]

# stage 5 (2M) : Fs = 7.5MHz, Fmix = 1.5MHz/-0.5MHz, Fc=1.2MHz channel=32+8
fmix5_0p5 = 0.5e+6
fmix5_1p5 = 1.5e+6
n=np.arange(dataFlt4[0].size)
cosMix5_0p5 = (np.cos((n*2*np.pi*fmix5_0p5/(Fs/32))+0.06287)*bitWidth).astype(calcType)
sinMix5_0p5 = (np.sin((n*2*np.pi*fmix5_0p5/(Fs/32))+0.06287)*bitWidth).astype(calcType)
cosMix5_1p5 = (np.cos((n*2*np.pi*fmix5_1p5/(Fs/32))+0.06287)*bitWidth).astype(calcType)
sinMix5_1p5 = (np.sin((n*2*np.pi*fmix5_1p5/(Fs/32))+0.06287)*bitWidth).astype(calcType)
dataMix5_2M = [[0]*n for i in range(80)]
for i in range(40/2):
	dataMix5_2M[i*4]   = (np.multiply(dataFlt4[i*2],   cosMix5_1p5) - np.multiply(dataFlt4[i*2+1], sinMix5_1p5))/bitWidth
	dataMix5_2M[i*4+1] = (np.multiply(dataFlt4[i*2+1], cosMix5_1p5) + np.multiply(dataFlt4[i*2],   sinMix5_1p5))/bitWidth
	dataMix5_2M[i*4+2] = (np.multiply(dataFlt4[i*2],   cosMix5_0p5) + np.multiply(dataFlt4[i*2+1], sinMix5_0p5))/bitWidth
	dataMix5_2M[i*4+3] = (np.multiply(dataFlt4[i*2+1], cosMix5_0p5) - np.multiply(dataFlt4[i*2],   sinMix5_0p5))/bitWidth

hbFlt5_2M = np.array([+0.0004425049, 0.0000000000, -0.0022277832, 0.0000000000, +0.0071182251, 0.0000000000, -0.0179901123, 0.0000000000, +0.0402183533, 0.0000000000, -0.0902061462, 0.0000000000, +0.3126716614, 0.5000000000, +0.3126716614, 0.0000000000, -0.0902061462, 0.0000000000, +0.0402183533, 0.0000000000, -0.0179901123, 0.0000000000, +0.0071182251, 0.0000000000, -0.0022277832, 0.0000000000, +0.0004425049])
hbFlt5_2M = (hbFlt5_2M*bitWidth).astype(calcType)
dataFlt5_2M = [[0]*n for i in range(80)]
for i in range(40*2):
	dataFlt5_2M[i] = np.convolve(hbFlt5_2M,dataMix5_2M[i],'same')/bitWidth
	dataFlt5_2M[i] = dataFlt5_2M[i][::2]

# stage 6 : Fs = 3.75MHz, Fmix = 0.5MHz, Fc=0.7MHz channel=64+16
fmix6 = 0.5e+6
n=np.arange(dataFlt5[0].size)
cosMix6 = (np.cos((n*2*np.pi*fmix6/(Fs/64))+0.06287)*bitWidth).astype(calcType)
sinMix6 = (np.sin((n*2*np.pi*fmix6/(Fs/64))+0.06287)*bitWidth).astype(calcType)
dataMix6 = [[0]*n for i in range(160)]
for i in range(80/2):
	dataMix6[i*4]   = (np.multiply(dataFlt5[i*2],   cosMix6) - np.multiply(dataFlt5[i*2+1], sinMix6))/bitWidth
	dataMix6[i*4+1] = (np.multiply(dataFlt5[i*2+1], cosMix6) + np.multiply(dataFlt5[i*2],   sinMix6))/bitWidth
	dataMix6[i*4+2] = (np.multiply(dataFlt5[i*2],   cosMix6) + np.multiply(dataFlt5[i*2+1], sinMix6))/bitWidth
	dataMix6[i*4+3] = (np.multiply(dataFlt5[i*2+1], cosMix6) - np.multiply(dataFlt5[i*2],   sinMix6))/bitWidth

hbFlt6 = np.array([+0.0017395020, +0.0000000000, -0.0050888062, +0.0000000000, +0.0118675232, +0.0000000000, -0.0241470337, +0.0000000000, +0.0465126038, +0.0000000000, -0.0949554443, +0.0000000000, +0.3144378662, +0.5000000000, +0.3144378662, +0.0000000000, -0.0949554443, +0.0000000000, +0.0465126038, +0.0000000000, -0.0241470337, +0.0000000000, +0.0118675232, +0.0000000000, -0.0050888062, +0.0000000000, +0.0017395020])
hbFlt6 = (hbFlt6*bitWidth).astype(calcType)
dataFlt6 = [[0]*n for i in range(160)]
for i in range(80*2):
	dataFlt6[i] = np.convolve(hbFlt6,dataMix6[i],'same')/bitWidth
	dataFlt6[i] = dataFlt6[i][::2]

# channel filter :
chFltGfsk1M = np.array([-2.8228759766e-04, +4.9667358398e-03, -3.9443969727e-03, -1.2725830078e-02, +2.1667480469e-02, +1.4610290527e-02, -6.1141967773e-02, +1.3160705566e-02, +1.2137603760e-01, -1.1454010010e-01, -2.0558166504e-01, +4.6765136719e-01, +1.0000000000e+00, +4.6765136719e-01, -2.0558166504e-01, -1.1454010010e-01, +1.2137603760e-01, +1.3160705566e-02, -6.1141967773e-02, +1.4610290527e-02, +2.1667480469e-02, -1.2725830078e-02, -3.9443969727e-03, +4.9667358398e-03, -2.8228759766e-04])
chFltDpsk = np.array([+3.5095214844e-004, -4.4250488281e-004, +4.5776367188e-005, +1.9226074219e-003, -5.3482055664e-003, +4.9438476563e-003, +2.3345947266e-003, -1.9256591797e-002, +5.5557250977e-002, -3.8421630859e-002, -1.9121551514e-001, +4.1075134277e-001, +1.0000000000e+000, +4.1075134277e-001, -1.9121551514e-001, -3.8421630859e-002, +5.5557250977e-002, -1.9256591797e-002, +2.3345947266e-003, +4.9438476563e-003, -5.3482055664e-003, +1.9226074219e-003, +4.5776367188e-005, -4.4250488281e-004, +3.5095214844e-004])
chFltGfsk1M = (chFltGfsk1M*bitWidth).astype(calcType)
chFltDpsk = (chFltDpsk*bitWidth).astype(calcType)
dataGfsk1M = [[0]*n for i in range(160)]
dataDpsk = [[0]*n for i in range(160)]
for i in range(80*2):
	dataGfsk1M[i] = np.convolve(chFltGfsk1M,dataFlt6[i],'same')/bitWidth
	dataDpsk[i]   = np.convolve(chFltDpsk,dataFlt6[i],'same')/bitWidth
chFltGfsk2M = np.array([-0.0016098022, -0.0025405884, +0.0059661865, +0.0041198730, -0.0174102783, -0.0008392334, +0.0378952026, -0.0159835815, -0.0671920776, +0.0606842041, +0.1036605835, -0.1659164429, -0.1637573242, +0.5177917480, +1.0000000000, +0.5177917480, -0.1637573242, -0.1659164429, +0.1036605835, +0.0606842041, -0.0671920776, -0.0159835815, +0.0378952026, -0.0008392334, -0.0174102783, +0.0041198730, +0.0059661865, -0.0025405884, -0.0016098022])
chFltGfsk2M = (chFltGfsk2M*bitWidth).astype(calcType)
dataGfsk2M = [[0]*n for i in range(80)]
for i in range(40*2):
	dataGfsk2M[i] = np.convolve(chFltGfsk2M,dataFlt5_2M[i],'same')/bitWidth

chFltDpskHdr1M = np.array([+4.5776367188e-04, +1.6021728516e-04, +9.1552734375e-05, -6.8664550781e-05, -2.6702880859e-04, -4.2724609375e-04, -4.5013427734e-04, -2.5939941406e-04, +1.4495849609e-04, +6.7138671875e-04, +1.1444091797e-03, +1.3046264648e-03, +9.6130371094e-04, +6.1035156250e-05, -1.2588500977e-03, -2.5939941406e-03, -3.4255981445e-03, -3.2577514648e-03, -1.8463134766e-03, +6.1798095703e-04, +3.4713745117e-03, +5.6838989258e-03, +6.1569213867e-03, +4.2037963867e-03, -9.1552734375e-05, -5.6152343750e-03, -1.0322570801e-02, -1.1802673340e-02, -8.0337524414e-03, +1.6174316406e-03, +1.5708923340e-02, +3.0471801758e-02, +4.0382385254e-02, +3.9489746094e-02, +2.3307800293e-02, -9.1857910156e-03, -5.4046630859e-02, -1.0188293457e-01, -1.3883209229e-01, -1.4897155762e-01, -1.1775207520e-01, -3.5835266113e-02, +9.7755432129e-02, +2.7394866943e-01, +4.7405242920e-01, +6.7238616943e-01, +8.4072113037e-01, +9.5360565186e-01, +1.0000000000e+00, +9.5360565186e-01, +8.4072113037e-01, +6.7238616943e-01, +4.7405242920e-01, +2.7394866943e-01, +9.7755432129e-02, -3.5835266113e-02, -1.1775207520e-01, -1.4897155762e-01, -1.3883209229e-01, -1.0188293457e-01, -5.4046630859e-02, -9.1857910156e-03, +2.3307800293e-02, +3.9489746094e-02, +4.0382385254e-02, +3.0471801758e-02, +1.5708923340e-02, +1.6174316406e-03, -8.0337524414e-03, -1.1802673340e-02, -1.0322570801e-02, -5.6152343750e-03, -9.1552734375e-05, +4.2037963867e-03, +6.1569213867e-03, +5.6838989258e-03, +3.4713745117e-03, +6.1798095703e-04, -1.8463134766e-03, -3.2577514648e-03, -3.4255981445e-03, -2.5939941406e-03, -1.2588500977e-03, +6.1035156250e-05, +9.6130371094e-04, +1.3046264648e-03, +1.1444091797e-03, +6.7138671875e-04, +1.4495849609e-04, -2.5939941406e-04, -4.5013427734e-04, -4.2724609375e-04, -2.6702880859e-04, -6.8664550781e-05, +9.1552734375e-05, +1.6021728516e-04, +4.5776367188e-04])
chFltDpskHdr2M = np.array([-1.1444091797e-04, -7.6293945313e-06, +6.1035156250e-05, +6.8664550781e-05, -2.2888183594e-05, -1.2969970703e-04, -9.1552734375e-05, +9.9182128906e-05, +2.1362304688e-04, +3.8146972656e-05, -2.6702880859e-04, -3.1280517578e-04, +9.9182128906e-05, +5.3405761719e-04, +3.8146972656e-04, -3.4332275391e-04, -7.8582763672e-04, -2.2888183594e-04, +8.3923339844e-04, +1.0070800781e-03, -3.0517578125e-04, -1.7318725586e-03, -1.2664794922e-03, +1.0910034180e-03, +2.6626586914e-03, +1.0681152344e-03, -2.3269653320e-03, -3.1585693359e-03, +6.6375732422e-04, +5.1498413086e-03, +3.8299560547e-03, -3.8452148438e-03, -9.5825195313e-03, -4.7454833984e-03, +7.5531005859e-03, +1.2512207031e-02, +3.2806396484e-04, -1.7257690430e-02, -1.3870239258e-02, +2.0210266113e-02, +5.3802490234e-02, +3.3348083496e-02, -5.8746337891e-02, -1.5763092041e-01, -1.3954925537e-01, +8.3534240723e-02, +4.6975708008e-01, +8.4246063232e-01, +1.0000000000e+00, +8.4246063232e-01, +4.6975708008e-01, +8.3534240723e-02, -1.3954925537e-01, -1.5763092041e-01, -5.8746337891e-02, +3.3348083496e-02, +5.3802490234e-02, +2.0210266113e-02, -1.3870239258e-02, -1.7257690430e-02, +3.2806396484e-04, +1.2512207031e-02, +7.5531005859e-03, -4.7454833984e-03, -9.5825195313e-03, -3.8452148438e-03, +3.8299560547e-03, +5.1498413086e-03, +6.6375732422e-04, -3.1585693359e-03, -2.3269653320e-03, +1.0681152344e-03, +2.6626586914e-03, +1.0910034180e-03, -1.2664794922e-03, -1.7318725586e-03, -3.0517578125e-04, +1.0070800781e-03, +8.3923339844e-04, -2.2888183594e-04, -7.8582763672e-04, -3.4332275391e-04, +3.8146972656e-04, +5.3405761719e-04, +9.9182128906e-05, -3.1280517578e-04, -2.6702880859e-04, +3.8146972656e-05, +2.1362304688e-04, +9.9182128906e-05, -9.1552734375e-05, -1.2969970703e-04, -2.2888183594e-05, +6.8664550781e-05, +6.1035156250e-05, -7.6293945313e-06, -1.1444091797e-04])
chFltDpskHdr4M = np.array([-3.8146972656e-05, +2.2888183594e-05, -1.5258789063e-05, -7.6293945313e-06, +3.8146972656e-05, -4.5776367188e-05, +2.2888183594e-05, +2.2888183594e-05, -7.6293945313e-05, +9.1552734375e-05, -4.5776367188e-05, -5.3405761719e-05, +1.5258789063e-04, -1.6784667969e-04, +6.8664550781e-05, +1.1444091797e-04, -2.8991699219e-04, +3.1280517578e-04, -1.0681152344e-04, -2.2125244141e-04, +5.1879882813e-04, -5.4931640625e-04, +1.5258789063e-04, +4.3487548828e-04, -9.0026855469e-04, +9.3841552734e-04, -2.2125244141e-04, -8.6975097656e-04, +1.6021728516e-03, -1.5792846680e-03, +3.4332275391e-04, +1.7471313477e-03, -3.0364990234e-03, +2.7008056641e-03, -4.8828125000e-04, -3.6315917969e-03, +6.4086914063e-03, -4.9896240234e-03, +2.8228759766e-04, +8.4915161133e-03, -1.6601562500e-02, +1.1497497559e-02, +4.3106079102e-03, -2.9174804688e-02, +7.2143554688e-02, -4.4242858887e-02, -2.0447540283e-01, +4.1822052002e-01, +1.0000000000e+00, +4.1822052002e-01, -2.0447540283e-01, -4.4242858887e-02, +7.2143554688e-02, -2.9174804688e-02, +4.3106079102e-03, +1.1497497559e-02, -1.6601562500e-02, +8.4915161133e-03, +2.8228759766e-04, -4.9896240234e-03, +6.4086914063e-03, -3.6315917969e-03, -4.8828125000e-04, +2.7008056641e-03, -3.0364990234e-03, +1.7471313477e-03, +3.4332275391e-04, -1.5792846680e-03, +1.6021728516e-03, -8.6975097656e-04, -2.2125244141e-04, +9.3841552734e-04, -9.0026855469e-04, +4.3487548828e-04, +1.5258789063e-04, -5.4931640625e-04, +5.1879882813e-04, -2.2125244141e-04, -1.0681152344e-04, +3.1280517578e-04, -2.8991699219e-04, +1.1444091797e-04, +6.8664550781e-05, -1.6784667969e-04, +1.5258789063e-04, -5.3405761719e-05, -4.5776367188e-05, +9.1552734375e-05, -7.6293945313e-05, +2.2888183594e-05, +2.2888183594e-05, -4.5776367188e-05, +3.8146972656e-05, -7.6293945313e-06, -1.5258789063e-05, +2.2888183594e-05, -3.8146972656e-05])
chFltDpskHdr1M = (chFltDpskHdr1M*bitWidth).astype(calcType)
chFltDpskHdr2M = (chFltDpskHdr2M*bitWidth).astype(calcType)
chFltDpskHdr4M = (chFltDpskHdr4M*bitWidth).astype(calcType)
dataDpskHdr1M = [[0]*n for i in range(160)]
dataDpskHdr2M = [[0]*n for i in range(160)]
dataDpskHdr4M = [[0]*n for i in range(160)]
for i in range(80*2):
	dataDpskHdr1M[i] = np.convolve(chFltDpskHdr1M,dataFlt4_4M[i],'same')/bitWidth
	dataDpskHdr2M[i] = np.convolve(chFltDpskHdr2M,dataFlt4_4M[i],'same')/bitWidth
	dataDpskHdr4M[i] = np.convolve(chFltDpskHdr4M,dataFlt4_4M[i],'same')/bitWidth

##  0-31		=>  0-31
## 32-63		=> 48-79
## 64-79		=> 32-47
requiredChConverted = RequiredChannel if RequiredChannel >= 0 and RequiredChannel <= 31 else RequiredChannel+32 if RequiredChannel >= 32 and RequiredChannel <= 47 else RequiredChannel-16 if RequiredChannel >= 48 and RequiredChannel <= 79 else -1

# upsampling of required channel
dataGfsk1MModI = myLib.cicFilter(dataGfsk1M[requiredChConverted*2])
dataGfsk1MModQ = myLib.cicFilter(dataGfsk1M[requiredChConverted*2+1])

dataGfsk2MModI = myLib.cicFilter(dataGfsk2M[int(requiredChConverted/2)*2])
dataGfsk2MModQ = myLib.cicFilter(dataGfsk2M[int(requiredChConverted/2)*2+1])

dataDpskModI = myLib.cicFilter(dataDpsk[requiredChConverted*2])
dataDpskModQ = myLib.cicFilter(dataDpsk[requiredChConverted*2+1])

dataDpskHdr1MModI = myLib.cicFilter(dataDpskHdr1M[int(requiredChConverted)*2])
dataDpskHdr1MModQ = myLib.cicFilter(dataDpskHdr1M[int(requiredChConverted)*2+1])
dataDpskHdr2MModI = myLib.cicFilter(dataDpskHdr2M[int(requiredChConverted)*2])
dataDpskHdr2MModQ = myLib.cicFilter(dataDpskHdr2M[int(requiredChConverted)*2+1])
dataDpskHdr4MModI = myLib.cicFilter(dataDpskHdr4M[requiredChConverted*2])
dataDpskHdr4MModQ = myLib.cicFilter(dataDpskHdr4M[requiredChConverted*2+1])

# saving output data
np.save('gfsk1MData', [dataGfsk1MModI, dataGfsk1MModQ])
np.save('gfsk2MData', [dataGfsk2MModI, dataGfsk2MModQ])

np.save('dpskData', [dataDpskModI, dataDpskModQ])

np.save('dpskHdr1MData', [dataDpskHdr1MModI, dataDpskHdr1MModQ])
np.save('dpskHdr2MData', [dataDpskHdr2MModI, dataDpskHdr2MModQ])
np.save('dpskHdr4MData', [dataDpskHdr4MModI, dataDpskHdr4MModQ])

# for i in range(10):
	# myLib.fftPlot(dataDpskHdr2M[i*2], dataDpskHdr2M[i*2+1], n=2, fs=Fs, index=i)
	# myLib.fftPlot(dataFlt5_2M[i*2], dataFlt5_2M[i*2+1], n=2, fs=Fs, index=i)

##### debugging
# plt.subplot(211)
# plt.plot(dataGfsk2M[int(requiredChConverted/2)*2])
# plt.subplot(212)
# plt.plot(dataGfsk2MModI)
# plt.show()
# plt.plot(dataGfsk1MModQ)
# plt.show()
