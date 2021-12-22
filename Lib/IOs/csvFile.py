import numpy as np

def readCsvFile(filename):

	ret_val = np.loadtxt(filename, delimiter=",")
	return ret_val