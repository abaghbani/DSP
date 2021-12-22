import numpy as np
import matplotlib.pyplot as plt
import logging as log

def ManchesterEncoder(data, init_value=0):
	ret_val = np.empty(2*data.size, dtype='bool')
	ret_val[0] = ~init_value 
	ret_val[1] = init_value if data[0] else ~init_val
	for i in range(1, data.size):
		ret_val[2*i] = ~ret_val[2*i-1]
		ret_val[2*i+1] = ret_val[2*i-1] if data[i] else ~ret_val[2*i-1]

	return ret_val

def ManchesterDecoder(data):
	ret_val = np.empty(data.size//2, dtype='bool')
	ret_err = np.zeros(data.size//2, dtype='bool')
	for i in range(ret_val.size):
		ret_val[i] = (data[2*i] != data[2*i+1])
		ret_err[i] = (data[2*i] == data[2*i-1])
	return ret_val, ret_err

def BitsToByte(bits):
	ret_val = 0
	for i in range(8):
		ret_val += bits[i]*np.power(2,i)
		
	return ret_val

def ParityCheck(bits):
	# this function check parity even for ask and parity odd for fsk
	ret_val = 0
	for i in range(bits.size):
		ret_val ^= bits[i]
	return ret_val

def Sign(dd):
	return 1 if dd>=0 else -1

def Clip(dd, lim):
	return lim if dd>lim else (-lim if dd<-lim else dd)

