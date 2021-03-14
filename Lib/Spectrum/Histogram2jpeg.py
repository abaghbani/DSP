import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from PIL import Image, ImageDraw, ImageFont

#M  =1e+6	#MHz scale
#G  =1e+9	#MHz scale
#PI =np.pi

def getpalette(cmap):
	cm = list((255.*np.array(list(map(lambda x: cmap(x)[0:3], np.linspace(0., 1.,256)))).ravel()).astype('int'))
	cm[2] = 0; #black is black
	return cm

def norm255(x):
	return ((x - x.min())*255.0 / (x.max() - x.min())).astype(np.uint8)

def spec_hist(data, NBIT=16, NFFT=4096, Pmin = -120.0, Pmax=3.0, aspect=1080.0/1920.0):
	FS = Pmax-Pmin
	W = data.shape[0]
	H = int(W * aspect);
	hst = np.ones((data.shape[0],H), dtype=np.int)
	data = 10.*np.log10(data)	#to dB
	y = data
	y = y*H/FS + H;
	y -= Pmax *H/FS;							# keep little upper gap
	y[y>H-1] = H-1;
	y[y<0]   = 0;
	for i in range(W):
		hst[i,:] = np.histogram(y[i,:], bins=H, range=(0,H-1))[0]
	
	print('Cumulative Spectrum shape: {0[0]}:{0[1]} | max:{1}'.format(hst.shape, hst.max()))
	return hst[:,1:]+1

def histogram2jpeg(fileName):

	aspect = float(1080.0/1920.0);
	Pmax = int(3);
	Pmin = int(-90);
	NFFT = int(4096);
	overlap = float(0.5);
	
	NBPS = 12
	
	FS_span = Pmax - Pmin

	dataTemp = np.memmap(fileName, mode='r', dtype=np.dtype('<h'))
	# data = dataTemp[480*640:480*1660]
	data = dataTemp
	print('Data: {} samples'.format(data.size))
		
	print('Data Min/Max: ',data.min(),data.max())
	data = data << (16 - NBPS)
	data = data >> (16 - NBPS)
	print('Data Min/Max: ',data.min(),data.max())
	if data.min()   <= -2**(NBPS-1)    : print('Warning: MIN Saturated !!!!!')
	elif data.min() <  -2**(NBPS-1)+16 : print('Note: MIN close to be saturated !!!!!')

	if data.max()   >=  2**(NBPS-1)-1  : print('Warning: MAX Saturated !!!!!')
	elif data.max() >   2**(NBPS-1)-16 : print('Note: MAX close to be saturated !!!!!')
	
	data = data.astype(float) * (2**-(NBPS-1));	#scale to 0dBFS
	data = np.multiply(data, np.cos((np.arange(data.size)*2*np.pi*120.0/240.0)+0.06287))

	spectrum, unused1, unused2 = mlab.specgram(data, NFFT=NFFT, window = np.blackman(NFFT), noverlap=NFFT*overlap)
	spectrum /= NFFT/8;

	print('Spectrogram shape: {0[0]}:{0[1]}'.format(spectrum.shape))
	cum_hist = spec_hist(spectrum, NFFT=NFFT, Pmin=Pmin, aspect=aspect)
	
	
	im = Image.fromarray( norm255(np.log2(cum_hist)) )
	im.putpalette(getpalette(plt.cm.gnuplot2))
	im = im.convert('RGBA').rotate(90, expand=True)
	bg = Image.new('RGBA', im.size, 'black')
	draw = ImageDraw.Draw(bg)
	FONTSIZE = 12
	font = ImageFont.truetype("arial.ttf", FONTSIZE)
	for i in range(Pmax, FS_span, 5):	# --- power level grid
		y = i*(float(bg.size[1])/FS_span)
		c = "green" if ((i-Pmax)%20 == 0) else (32,32,32,255)
		draw.line((0, y, bg.size[0], y), fill=c)
		t = "{}".format(i-Pmax);
		w,h = font.getsize(t)
		draw.text((0, y-h*1.2), t, font=font)


	for i in range(81):					# --- channels grid
		# x = (float(bg.size[0])/120)*(98.0 - i)
		x = (float(bg.size[0])/120)*(i+22)
		c = (32,64,32,255) if (i%5 == 0) else (32,32,32,255)
		draw.line((x, 0, x, bg.size[1]), fill=c)
		t = "{}".format(i);
		w,h = font.getsize(t)
		if (i%5 == 0) :draw.text((x-w/2, 0), t, font=font)

	del draw

	mask = im.convert('L')
	mask = mask.point(lambda x: 255 if x<2 else 0, '1')
	im.paste(bg, mask = mask )

	im = im.convert("RGB")
	im.save(fileName[:-7]+'.jpg', quality=95)
	im.show()
