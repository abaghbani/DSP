import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import logging as log
import datetime

import Spectrum as sp
import Filter as fd
import IOs
import Common as cm
import RfModel as rf

if __name__=="__main__":
	
	log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG)
	log.getLogger('matplotlib.font_manager').disabled = True
	log.getLogger('PIL').setLevel(log.INFO)

	print('S: Specgram of sampled data')
	print('H: Histogram to Jpeg of sampled data')
	print('A: FFT plot')
	print('D: Debugging')
	print('X: Exit')
	print('>> ')

	path = 'D:/Documents/Samples/'
	
	while True:
		c = IOs.get_console_key()
		cm.prRedbgGreen('Command <<'+str(c)+'>> is running:')

		if c == 's':
			filename = IOs.get_file_from_path(path+'HDT/UPF76/', def_file=0, extension='bin')
			adcData = IOs.readRawFile(filename)
			# adcData = adcData[int(240*63e3):int(240*66e3)]
			adcData = adcData[int(240*0):int(240*200e3)]
			print('ADC Data Min/Max: ',adcData.min(),adcData.max(), type(adcData[0]))
			sp.specPlot(adcData)
			
		elif c == 'h':
			filename = IOs.get_file_from_path(path+'./', def_file=0)
			adcData = IOs.readRawFile(filename)
			sp.histogram2jpeg(adcData)
			
		elif c == 'a':
			filename = IOs.get_file_from_path(path+'./', def_file=0)
			adcData = IOs.readRawFile(filename)
			sp.fftPlot(adcData)				

		elif c == 'e':
			filename = IOs.get_file_from_path(path+'HDT/samsung/', extension='.bin')
			data = np.fromfile(filename, dtype='uint16', count=100*1024*1024)
			outData = IOs.convert_16_to_12_bit(data)
			outData.tofile(path+'bv1-replay-dump.bin', sep='')
			
		elif c == 'f':
			filename = IOs.get_file_from_path(path+'HDT/ebq/', extension='.bin')
			data = np.fromfile(filename, dtype='uint8', count=60*1024*1024)
			data = data[:int(240*3*10000)]
			data = IOs.convert_12_to_16_bit(data)
			sp.specPlot(data)

		elif c == 'd':
			# def equations(vars):
			# 	x, y = vars
			# 	eq1 = x**2 + y**2 - 4
			# 	eq2 = x**2 - y - 1
			# 	return [eq1, eq2]
			# initial_guesses = [1, 1]  # Initial guesses for x and y

			# from scipy.optimize import fsolve
			# # Solve the system
			# solution = fsolve(equations, initial_guesses)
			# print("Solution to the system:", solution)

			# # Print the results
			# x, y = solution
			# print(f"Solved values are x = {x:.2f} and y = {y:.2f}")

			# # Verify the solution by substituting it back into the equations
			# print("Verification:")
			# print(f"f1(x, y) = {x**2 + y**2 - 4:.2f}")
			# print(f"f2(x, y) = {x**2 - y - 1:.2f}")
			
			# x_sol, y_sol = solution
			# x = np.linspace(-3, 3, 400)
			# y = np.linspace(-3, 3, 400)
			# X, Y = np.meshgrid(x, y)

			# # Define the equations for plotting
			# Z1 = X**2 + Y**2 - 4
			# Z2 = X**2 - Y - 1

			# # Plot the contours
			# plt.figure(figsize=(8, 6))
			# plt.contour(X, Y, Z1, levels=np.linspace(0,5), colors='blue', label='x^2 + y^2 - 4')
			# # plt.contour(X, Y, Z2, levels=[0], colors='red', label='x^2 - y - 1')
			# # plt.plot(x_sol, y_sol, 'go', label='Solution')
			# plt.xlabel('x')
			# plt.ylabel('y')
			# plt.title('2D Visualization of Nonlinear Equations')
			# plt.legend()
			# plt.grid(True)
			# plt.show()
			
			print('\n','='*30,'\n', 'end of debug.')
			
		elif c == 'x':
			break

	print('Exit')
