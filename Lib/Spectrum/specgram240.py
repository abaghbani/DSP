#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
	data = np.memmap(sys.argv[1], mode='r', dtype=np.dtype('<h'))
	data = (data << 4)>>4
	# data = data[350000:1000000]
	dataMix = np.multiply(data, np.cos((np.arange(data.size)*2*np.pi*120.0e+6/240.0e+6)+0.06287))
	plt.specgram(dataMix, NFFT=1024, Fc=2380e6, Fs=240e+6)
	plt.show()