import sys
import numpy as np
import sounddevice as sd
import scipy.signal as signal
import matplotlib.pyplot as plt

sys.path.insert(1, sys.path[0]+'\..')
from ModemLib			import ModemLib
myLib = ModemLib(0)

fs = 44100
f1 = 440
f2 = 580
f3 = 880
n = np.arange(0.5*fs)
x1 = np.sin(2*np.pi*f1*n/fs)
x2 = np.sin(2*np.pi*f2*n/fs)
x3 = np.sin(2*np.pi*f3*n/fs)

x = x1+x2+x3

# Ensure that highest value is in 16-bit range
# audio = x * (2**15 - 1) / np.max(np.abs(x))
# Convert to 16-bit data
# audio = audio.astype(np.int16)
# sd.play(audio, fs)
# sd.wait()

sd.play(x, fs)
sd.wait()

plt.plot(x)
plt.show()

myLib.fftPlot(x, fs=fs)