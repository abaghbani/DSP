import numpy as np
import wave

def WriteToWav(fileName, samples, fs = 44100):

    data = np.array(samples*(32767.0 / np.max(np.abs(samples))), 'int16').tostring()
    file = wave.open(fileName, 'wb')
    file.setparams((1, 2, fs, len(samples), 'NONE', 'uncompressed'))
    file.writeframes(data)
    file.close()