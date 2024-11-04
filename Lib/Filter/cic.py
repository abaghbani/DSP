import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

# CIC filter
def cicFilter(inSig):
    x = np.zeros(3)
    y = np.zeros(3)
    midSig = np.zeros(4*inSig.size)
    outSig = np.zeros(4*inSig.size)

    for i in range(inSig.size):
        midSig[i*4]=inSig[i]-x[0]-x[1]-x[2]
        x[2]=inSig[i]-x[0]-x[1]
        x[1]=inSig[i]-x[0]
        x[0]=inSig[i]

    for i in range(4*inSig.size):
        outSig[i]=midSig[i]+y[0]+y[1]+y[2]
        y[2]=midSig[i]+y[0]+y[1]+y[2]
        y[1]=midSig[i]+y[0]+y[1]
        y[0]=midSig[i]+y[0]
    
    return outSig

def Downsampling_CIC_n3(inSig, down_rate=4):
    x = np.zeros(3, dtype=inSig.dtype)
    y = np.zeros(3, dtype=inSig.dtype)
    midSig = np.zeros(inSig.size, dtype=inSig.dtype)
    outSig = np.zeros(inSig.size//down_rate, dtype=inSig.dtype)

    for i in range(inSig.size):
        midSig[i]=inSig[i]+x[0]+x[1]+x[2]
        x[2]=inSig[i]+x[0]+x[1]+x[2]
        x[1]=inSig[i]+x[0]+x[1]
        x[0]=inSig[i]+x[0]
    
    for i in range(inSig.size//down_rate):
        outSig[i]=midSig[i*down_rate]-y[0]-y[1]-y[2]
        y[2]=midSig[i*down_rate]-y[0]-y[1]
        y[1]=midSig[i*down_rate]-y[0]
        y[0]=midSig[i*down_rate]
    
    return outSig

def Downsampling_CIC_n4(inSig, down_rate):
    # N: number of stage (for both decimator and integrator), by Xilinx Vivado between 3 and 6 (in this function = 4)
    # R: decimation rate, between 4 and 8192 (int this function = down_rate)
    # M: differential delay, between 1 and 2 (usually is 1, in this function = 1)
    # filter gain = (MR)^N = (down_rate)^4
    x = np.zeros(4, dtype=inSig.dtype)
    y = np.zeros(4, dtype=inSig.dtype)
    midSig = np.zeros(inSig.size, dtype=inSig.dtype)
    outSig = np.zeros(inSig.size//down_rate, dtype=inSig.dtype)

    for i in range(inSig.size):
        midSig[i]=inSig[i]+x[0]+x[1]+x[2]+x[3]
        x[3]=inSig[i]+x[0]+x[1]+x[2]+x[3]
        x[2]=inSig[i]+x[0]+x[1]+x[2]
        x[1]=inSig[i]+x[0]+x[1]
        x[0]=inSig[i]+x[0]
    
    for i in range(inSig.size//down_rate):
        outSig[i]=midSig[i*down_rate]-(y[0]+y[1]+y[2]+y[3])
        y[3]=midSig[i*down_rate]-(y[0]+y[1]+y[2])
        y[2]=midSig[i*down_rate]-(y[0]+y[1])
        y[1]=midSig[i*down_rate]-y[0]
        y[0]=midSig[i*down_rate]
    
    return outSig
