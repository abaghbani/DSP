import numpy as np
import matplotlib.pyplot as plt

def dds(input_phase, enable_plot=False):

	## x = 2pi/(2**20) * x_uint20 => x_uint20 = x1*(2**12) + x2*(2**6) + x3
	## sin(x) ~= [(2**18)*sin(x1) + x2*pi*(2**9)*cos(x1)/(2**4) + x2*x3*(-pi*pi)*sin(x1)/(2**14) ] / (2**18)
	## sin(A+B+C) = sin(A)cos(B)cos(C) + cos(A)sin(B)cos(C) - sin(A)sin(B)sin(C) + cos(A)cos(B)sin(C)
	
	def plot_var(x, y):
		plt.plot(x, label = 'table 1')
		plt.plot(y, label = 'sin')
		plt.legend()
		plt.show()
		plt.plot(y-x)
		plt.show()

	from ctypes import c_uint32 as uint32

	x1 = np.array([3221, 9652, 16078, 22495, 28897, 35282, 41646, 47985, 54295, 60573, 66813, 73014, 79171, 85280, 91337, 97340, 103283, 109165, 114981, 120728, 126402, 132000, 137518, 142953, 148303, 153563, 158730, 163802, 168775, 173647, 178414, 183074, 187623, 192059, 196380, 200582, 204663, 208622, 212454, 216159, 219733, 223175, 226482, 229654, 232686, 235579, 238330, 240937, 243399, 245714, 247882, 249900, 251767, 253483, 255047, 256456, 257711, 258811, 259755, 260543, 261173, 261647, 261962, 262120, 262120, 261962, 261647, 261173, 260543, 259755, 258811, 257711, 256456, 255047, 253483, 251767, 249900, 247882, 245714, 243399, 240937, 238330, 235579, 232686, 229654, 226482, 223175, 219733, 216159, 212454, 208622, 204663, 200582, 196380, 192059, 187623, 183074, 178414, 173647, 168775, 163802, 158730, 153563, 148303, 142953, 137518, 132000, 126402, 120728, 114981, 109165, 103283, 97340, 91337, 85280, 79171, 73014, 66813, 60573, 54295, 47985, 41646, 35282, 28897, 22495, 16078, 9652, 3221, -3213, -9644, -16070, -22487, -28889, -35274, -41638, -47977, -54287, -60565, -66805, -73006, -79163, -85272, -91329, -97332, -103275, -109157, -114973, -120720, -126394, -131992, -137510, -142945, -148295, -153555, -158722, -163794, -168767, -173639, -178406, -183066, -187615, -192051, -196372, -200574, -204655, -208614, -212446, -216151, -219725, -223167, -226474, -229646, -232678, -235571, -238322, -240929, -243391, -245706, -247874, -249892, -251759, -253475, -255039, -256448, -257703, -258803, -259747, -260535, -261165, -261639, -261954, -262112, -262112, -261954, -261639, -261165, -260535, -259747, -258803, -257703, -256448, -255039, -253475, -251759, -249892, -247874, -245706, -243391, -240929, -238322, -235571, -232678, -229646, -226474, -223167, -219725, -216151, -212446, -208614, -204655, -200574, -196372, -192051, -187615, -183066, -178406, -173639, -168767, -163794, -158722, -153555, -148295, -142945, -137510, -131992, -126394, -120720, -114973, -109157, -103275, -97332, -91329, -85272, -79163, -73006, -66805, -60565, -54287, -47977, -41638, -35274, -28889, -22487, -16070, -9644, -3213])
	x2 = np.array([1608, 1607, 1605, 1603, 1599, 1594, 1588, 1581, 1574, 1565, 1555, 1545, 1533, 1521, 1508, 1493, 1478, 1462, 1445, 1428, 1409, 1390, 1369, 1348, 1326, 1304, 1280, 1256, 1231, 1205, 1178, 1151, 1123, 1095, 1065, 1036, 1005, 974, 942, 910, 877, 844, 810, 776, 741, 705, 670, 634, 597, 560, 523, 486, 448, 410, 372, 333, 294, 256, 216, 177, 138, 99, 59, 20, -20, -59, -99, -138, -177, -216, -256, -294, -333, -372, -410, -448, -486, -523, -560, -597, -634, -670, -705, -741, -776, -810, -844, -877, -910, -942, -974, -1005, -1036, -1065, -1095, -1123, -1151, -1178, -1205, -1231, -1256, -1280, -1304, -1326, -1348, -1369, -1390, -1409, -1428, -1445, -1462, -1478, -1493, -1508, -1521, -1533, -1545, -1555, -1565, -1574, -1581, -1588, -1594, -1599, -1603, -1605, -1607, -1608, -1608, -1607, -1605, -1603, -1599, -1594, -1588, -1581, -1574, -1565, -1555, -1545, -1533, -1521, -1508, -1493, -1478, -1462, -1445, -1428, -1409, -1390, -1369, -1348, -1326, -1304, -1280, -1256, -1231, -1205, -1178, -1151, -1123, -1095, -1065, -1036, -1005, -974, -942, -910, -877, -844, -810, -776, -741, -705, -670, -634, -597, -560, -523, -486, -448, -410, -372, -333, -294, -256, -216, -177, -138, -99, -59, -20, 20, 59, 99, 138, 177, 216, 256, 294, 333, 372, 410, 448, 486, 523, 560, 597, 634, 670, 705, 741, 776, 810, 844, 877, 910, 942, 974, 1005, 1036, 1065, 1095, 1123, 1151, 1178, 1205, 1231, 1256, 1280, 1304, 1326, 1348, 1369, 1390, 1409, 1428, 1445, 1462, 1478, 1493, 1508, 1521, 1533, 1545, 1555, 1565, 1574, 1581, 1588, 1594, 1599, 1603, 1605, 1607, 1608])
	x3 = np.array([0, 0, -1, -1, -1, -1, -2, -2, -2, -2, -3, -3, -3, -3, -3, -4, -4, -4, -4, -5, -5, -5, -5, -5, -6, -6, -6, -6, -6, -7, -7, -7, -7, -7, -7, -8, -8, -8, -8, -8, -8, -8, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -8, -8, -8, -8, -8, -8, -8, -7, -7, -7, -7, -7, -7, -6, -6, -6, -6, -6, -5, -5, -5, -5, -5, -4, -4, -4, -4, -3, -3, -3, -3, -3, -2, -2, -2, -2, -1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 8, 8, 8, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0])
	
	skew_half_bit = 0.0
	## this 0.999 is just to compress this table to fit final calc in 16 bit
	y1 = np.rint(0.999*np.power(2, 18)*np.sin(2*np.pi*(np.arange(256)+skew_half_bit)/256)).astype(int)
	y2 = np.rint(np.pi*np.power(2, 9)*np.cos(2*np.pi*(np.arange(256)+skew_half_bit)/256)).astype(int)
	y3 = np.rint(-np.pi*np.pi*np.sin(2*np.pi*(np.arange(256)+skew_half_bit)/256)).astype(int)
	
	if enable_plot:
		print('x1', np.array2string(x1, separator=', '))
		print('x2', np.array2string(x2, separator=', '))
		print('x3', np.array2string(x3, separator=', '))
		print('y1', np.array2string(y1, separator=', '))
		print('y2', np.array2string(y2, separator=', '))
		print('y3', np.array2string(y3, separator=', '))

		plot_var(x1, y1)
		plot_var(x2, y2)
		plot_var(x3, y3)
				
	phase = uint32(int(input_phase * (2**20))).value
				
	phase_1 = phase >> 12
	phase_2 = (phase & 0xfff) >> 6
	phase_3 = (phase & 0x3f)

	correction_1 = 0
	correction_2 = 4
	correction_3 = 14
					
	#calc_1 = x1[phase_1] >> correction_1
	#calc_2 = (x2[phase_1] * phase_2) >> correction_2
	#calc_3 = (x3[phase_1] * phase_2 * phase_3) >> correction_3
					
	calc_1 = (y1[phase_1]) // (2**correction_1)
	calc_2 = (y2[phase_1] * phase_2) // (2**correction_2)
	calc_3 = (y3[phase_1] * phase_2 * phase_3) // (2**correction_3)
					
	sin_value = calc_1+calc_2+calc_3
	
	if enable_plot:
		print(f'{hex(phase)=}, {hex(phase_1)=}, {hex(phase_2)=}, {hex(phase_3)=}')
		print(f'{calc_1=}, {calc_2=}, {calc_3=}')
		print(f'out = {sin_value}')

	return sin_value