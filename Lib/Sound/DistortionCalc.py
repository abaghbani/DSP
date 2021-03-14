import numpy as np
import scipy.signal as signal

def rms_flat(a):
    return np.sqrt(np.mean(np.absolute(a)**2))

def find_range(f, x):
    for i in np.arange(x+1, len(f)):
        if f[i+1] >= f[i]:
            uppermin = i
            break
    for i in np.arange(x-1, 0, -1):
        if f[i] <= f[i-1]:
            lowermin = i + 1
            break
    return (lowermin, uppermin)

def rms_harmony(f, x):
    total_rms_harmony = 0
    for i in np.arange(2*x, len(f), x):
        total_rms_harmony += np.abs(f[i])**2
    
    return (np.sqrt(total_rms_harmony))

def Thd(samples, fs = 1.0):
    
    samples -= np.mean(samples)
    windowed = samples * signal.blackmanharris(len(samples))

    f = np.fft.rfft(windowed)
    i = np.argmax(np.abs(f))
    ThdValue = rms_harmony(f, i) / np.abs(f[i])

    print('Frequency: %f Hz' % ((fs//2 + 1) * (i / len(f))))
    print("THD:       %.4f%% or %.2f dB" % (ThdValue * 100, 20 * np.log10(ThdValue)))

    return(ThdValue)

def ThdN(samples, fs = 1.0):
    
    samples -= np.mean(samples)
    windowed = samples * signal.blackmanharris(len(samples))

    f = np.fft.rfft(windowed)
    i = np.argmax(np.abs(f))
    
    lowermin, uppermin = find_range(abs(f), i)
    f[lowermin: uppermin] = 0
    noise = np.fft.irfft(f)
    ThdValue = rms_flat(noise) / rms_flat(windowed)
    
    print('Frequency: %f Hz' % ((fs//2 + 1) * (i / len(f))))
    print("THD+N:     %.4f%% or %.2f dB" % (ThdValue * 100, 20 * np.log10(ThdValue)))

    return(ThdValue)
