import numpy as np
import pandas as pd

def readCsvFile(filename):
	ret_val = np.loadtxt(filename, delimiter=",")
	return ret_val

def read_csv(filename):
	data = pd.read_csv(filename).to_numpy()
	return data
