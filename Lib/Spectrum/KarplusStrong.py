import numpy as np

def KarplusStrong(nSamples, sampleRate, frequency):

    buf = np.array(np.random.rand(int(sampleRate/frequency)) - 0.5)
    samples = np.array([0]*nSamples, 'float32')
    for i in range(nSamples):
        samples[i] = buf[0]
        avg = 0.995*0.5*(buf[0] + buf[1])
        buf[:-1] = buf[1:]
        buf[-1] = avg
    return (samples)