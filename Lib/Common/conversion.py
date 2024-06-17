import numpy as np

def npvect_to_hexstr(data):
	return np.vectorize(hex)(data)

def npvect_to_hex(data):
	return np.vectorize(np.base_repr)(data, 16)

def hexvect_to_str(data, width):
	return np.char.add('0x',np.char.rjust(data, width, '0'))

def npvect_to_floatstr(data, format="%1.8f"):
	return np.array2string(data, separator=', ', formatter={'float':lambda x: format % x}, max_line_width=25000)

