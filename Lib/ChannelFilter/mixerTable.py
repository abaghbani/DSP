import numpy as np
import matplotlib.pyplot as plt

Fs = 240.0e+6

fmix0 = 58.5e+6
n=np.arange(160)
cosMix0=np.cos((n*2*np.pi*fmix0/Fs)+0.06287)
sinMix0=np.sin((n*2*np.pi*fmix0/Fs)+0.06287)
print "Mix0 : ", cosMix0, "\n\r"
print "Mix0 : ", sinMix0, "\n\r"

fmix1 = 24.0e+6
n=np.arange(5)
cosMix1=np.cos((n*2*np.pi*fmix1/(Fs/2))+0.06287)
sinMix1=np.sin((n*2*np.pi*fmix1/(Fs/2))+0.06287)
print "Mix1 : ", cosMix1, "\n\r"
print "Mix1 : ", sinMix1, "\n\r"

fmix2 = 8.0e+6
n=np.arange(15)
cosMix2=np.cos((n*2*np.pi*fmix2/(Fs/4))+0.06287)
sinMix2=np.sin((n*2*np.pi*fmix2/(Fs/4))+0.06287)
print "Mix2 : ", cosMix2, "\n\r"
print "Mix2 : ", sinMix2, "\n\r"

fmix3 = 4.0e+6
n=np.arange(15)
cosMix3=np.cos((n*2*np.pi*fmix3/(Fs/8))+0.06287)
sinMix3=np.sin((n*2*np.pi*fmix3/(Fs/8))+0.06287)
print "Mix3 : ", cosMix3, "\n\r"
print "Mix3 : ", sinMix3, "\n\r"

fmix4_2p0 = 2.0e+6
fmix4_3p5 = -3.5e+6
fmix4_2p5 = -2.5e+6
fmix4_1p5 = -1.5e+6
fmix4_0p5 = -0.5e+6
n=np.arange(15)
cosMix4_2p0=np.cos((n*2*np.pi*fmix4_2p0/(Fs/16))+0.06287)
sinMix4_2p0=np.sin((n*2*np.pi*fmix4_2p0/(Fs/16))+0.06287)
n=np.arange(30)
cosMix4_3p5=np.cos((n*2*np.pi*fmix4_3p5/(Fs/16))-0.06287)
sinMix4_3p5=np.sin((n*2*np.pi*fmix4_3p5/(Fs/16))-0.06287)
n=np.arange(6)
cosMix4_2p5=np.cos((n*2*np.pi*fmix4_2p5/(Fs/16))-0.06287)
sinMix4_2p5=np.sin((n*2*np.pi*fmix4_2p5/(Fs/16))-0.06287)
n=np.arange(10)
cosMix4_1p5=np.cos((n*2*np.pi*fmix4_1p5/(Fs/16))-0.06287)
sinMix4_1p5=np.sin((n*2*np.pi*fmix4_1p5/(Fs/16))-0.06287)
n=np.arange(30)
cosMix4_0p5=np.cos((n*2*np.pi*fmix4_0p5/(Fs/16))-0.06287)
sinMix4_0p5=np.sin((n*2*np.pi*fmix4_0p5/(Fs/16))-0.06287)
print "Mix4_cos2p0MHz : ", cosMix4_2p0, "\n\r"
print "Mix4_sin2p0MHz : ", sinMix4_2p0, "\n\r"
print "Mix4_cos3p5MHz : ", cosMix4_3p5, "\n\r"
print "Mix4_sin3p5MHz : ", sinMix4_3p5, "\n\r"
print "Mix4_cos2p5MHz : ", cosMix4_2p5, "\n\r"
print "Mix4_sin2p5MHz : ", sinMix4_2p5, "\n\r"
print "Mix4_cos1p5MHz : ", cosMix4_1p5, "\n\r"
print "Mix4_sin1p5MHz : ", sinMix4_1p5, "\n\r"
print "Mix4_cos0p5MHz : ", cosMix4_0p5, "\n\r"
print "Mix4_sin0p5MHz : ", sinMix4_0p5, "\n\r"


fmix5_1p0 = 1.0e+6
fmix5_0p5 = -0.5e+6 # in vhdl we use Fmix5_0p5MHz = -0.5MHz
fmix5_1p5 = 1.5e+6
n=np.arange(15)
cosMix5_1p0=np.cos((n*2*np.pi*fmix5_1p0/(Fs/32))+0.06287)
sinMix5_1p0=np.sin((n*2*np.pi*fmix5_1p0/(Fs/32))+0.06287)
cosMix5_0p5=np.cos((n*2*np.pi*fmix5_0p5/(Fs/32))+0.06287)
sinMix5_0p5=np.sin((n*2*np.pi*fmix5_0p5/(Fs/32))+0.06287)
cosMix5_1p5=np.cos((n*2*np.pi*fmix5_1p5/(Fs/32))+0.06287)
sinMix5_1p5=np.sin((n*2*np.pi*fmix5_1p5/(Fs/32))+0.06287)
print "Mix5_cos1p0MHz : ", cosMix5_1p0, "\n\r"
print "Mix5_sin1p0MHz : ", sinMix5_1p0, "\n\r"
print "Mix5_cos0p5MHz : ", cosMix5_0p5, "\n\r"
print "Mix5_sin0p5MHz : ", sinMix5_0p5, "\n\r"
print "Mix5_cos1p5MHz : ", cosMix5_1p5, "\n\r"
print "Mix5_sin1p5MHz : ", sinMix5_1p5, "\n\r"


fmix6 = 0.5e+6
n=np.arange(15)
cosMix6=np.cos((n*2*np.pi*fmix6/(Fs/64))+0.06287)
sinMix6=np.sin((n*2*np.pi*fmix6/(Fs/64))+0.06287)
print "Mix6_cos0p5MHz : ", cosMix6, "\n\r"
print "Mix6_sin0p5MHz : ", sinMix6, "\n\r"
