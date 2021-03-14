import numpy as np
import scipy as scipy
import scipy.signal as signal
import matplotlib.pyplot as plt
from collections import defaultdict


class OFDM:
    pass
ofdm = OFDM()
ofdm.K = 1024                      # Number of OFDM subcarriers
ofdm.Kon = 600                     # Number of switched-on subcarriers
ofdm.CP = 128                      # Number of samples in the CP
ofdm.ofdmSymbolsPerFrame = 5       # N, number of payload symbols in each frame
ofdm.L = ofdm.K//2                 # Parameter L, denotes the length of one repeated part of the preamble

def random_qam(ofdm):
    qam = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    return np.random.choice(qam, size=(ofdm.Kon), replace=True)

def ofdm_modulate(ofdm, qam):
    assert (len(qam) == ofdm.Kon)
    fd_data = np.zeros(ofdm.K, dtype=complex)
    off = (ofdm.K - ofdm.Kon)//2
    fd_data[off:(off+len(qam))] = qam  # modulate in the center of the frequency
    fd_data = np.fft.fftshift(fd_data)
    symbol = np.fft.ifft(fd_data) * np.sqrt(ofdm.K)
    return np.hstack([symbol[-ofdm.CP:], symbol])

qam_preamble = np.sqrt(2)*random_qam(ofdm)
qam_preamble[::2] = 0   # delete every second data to make the preamble 2-periodic

# preamble = ofdm_modulate(ofdm, qam_preamble)
# plt.subplot(121)
# plt.plot(abs(preamble))
# plt.subplot(122)
# f = np.linspace(-ofdm.K/2, ofdm.K/2, 4*len(preamble), endpoint=False)
# plt.plot(f, 20*np.log10(abs(np.fft.fftshift(np.fft.fft(preamble, 4*len(preamble))/np.sqrt(len(preamble))))))
# plt.show()

def createFrame(ofdm, qam_preamble=None):
    if qam_preamble is None:
        qam_preamble = np.sqrt(2)*random_qam(ofdm)
        qam_preamble[::2] = 0
    preamble = ofdm_modulate(ofdm, qam_preamble)
    
    payload = np.hstack([ofdm_modulate(ofdm, random_qam(ofdm)) for _ in range(ofdm.ofdmSymbolsPerFrame)])
    return np.hstack([preamble, payload])

# frame = createFrame(ofdm, qam_preamble=qam_preamble*2)
# plt.plot(abs(frame))

def addCFO(signal, cfo):  # Add carrier frequency offset (unused in this notebook)
    return signal * np.exp(2j*np.pi*cfo*np.arange(len(signal)))

def addSTO(signal, sto):  # add some time offset
    return np.hstack([np.zeros(sto), signal])

def addNoise(signal, sigma2):  # add AWGN
    noise = np.sqrt(sigma2/2) * (np.random.randn(len(signal)) + 1j*np.random.randn(len(signal)))
    return signal + noise

def addChannel(signal, h):       # add some multipath impulse response (unused in this notebook)
    return scipy.signal.lfilter(h, (1,), signal)

x = createFrame(ofdm, qam_preamble=qam_preamble*2)
sto = ofdm.K//2
r = addNoise(addSTO(x, sto), 0.5)
# plt.plot(abs(r))
# plt.show()

############################################


def calcP_R_M(rx_signal, L):
    rx1 = rx_signal[:-L]
    rx2 = rx_signal[L:]
    mult = rx1.conj() * rx2
    square = abs(rx1**2)
    
    zi = np.zeros(L-1)
    
    a_P = (1, -1)
    b_P = np.zeros(L); b_P[0] = 1; b_P[-1] = -1
    P = scipy.signal.lfilter(b_P, a_P, mult) / L
    R = scipy.signal.lfilter(b_P, a_P, square) / L
    
    Pr = P[L:]
    Rr = R[L:]
    M = abs(Pr/Rr)**2
    return Pr, Rr, M  # throw away first L samples, as they are not correct due to filter causality

M_dopt = defaultdict(list)
M_doutside = defaultdict(list)
SNRs = np.linspace(-10, 30, 21)
for SNR in SNRs:
    for i in range(100):
        tx_signal = createFrame(ofdm, qam_preamble=None)
        sigma_s2 = np.mean(abs(tx_signal**2))
        sigma_n2 = sigma_s2 * 10**(-SNR/10.)
        sto = 1000
        cfo = 0.05/ofdm.K
        rx_signal = addNoise(addCFO(addSTO(tx_signal, sto), cfo), sigma_n2)
        P, R, M = calcP_R_M(rx_signal, ofdm.L)
        M_dopt[SNR].append(M[sto])
        M_doutside[SNR].append(M[sto+ofdm.K])

def calc_MeanStd(SNRs, measurement):
    mean = np.array([np.mean(measurement[SNR]) for SNR in SNRs])
    std = np.array([np.std(measurement[SNR]) for SNR in SNRs])
    return mean, std
mean_opt, std_opt = calc_MeanStd(SNRs, M_dopt)

# Plot the measured curves
plt.plot(SNRs, mean_opt, label='Simulated', color='blue', lw=3)
plt.plot(SNRs, mean_opt+3*std_opt, 'b--')
plt.plot(SNRs, mean_opt-3*std_opt, 'b--')

# Plot the theoretic curves
rho = 10**(-SNRs/10)
mu = 1/(1+rho)**2
std = np.sqrt(2*((1+mu)*rho+(1+2*mu)*rho**2)/(ofdm.L*(1+rho)**4))
#print (std)
plt.plot(SNRs, mu, label='Theory', color='r', lw=2)
plt.plot(SNRs, mu + 3*std, color='r', lw=2, ls='--')
plt.plot(SNRs, mu - 3*std, color='r', lw=2, ls='--')
plt.show()

