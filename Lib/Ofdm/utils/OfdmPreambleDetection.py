import numpy as np
import scipy as scipy
import scipy.signal as signal
import matplotlib.pyplot as plt



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
# metric calculation
############################################

## P(d) = Sum(r[d+m]'.r[d+m+L])

# Method 1 to calculate P(d)
d_set = np.arange(0, sto + ofdm.K)
P1 = np.zeros(len(d_set), dtype=complex)
for i, d in enumerate(d_set):
    P1[i] = sum(r[d+m].conj()*r[d+m+ofdm.L] for m in range(ofdm.L))

plt.plot(d_set, abs(P1))
# plt.show()

# Method 2 to calculate P(d)
P0 = sum(r[m].conj()*r[m+ofdm.L] for m in range(ofdm.L))  # initialize P[0] with the default method
def calcP_method2(r, d_set, P0):
    P2 = np.zeros(len(d_set), dtype=complex)
    P2[0] = P0
    for d in d_set[:-1]:
        P2[d+1] = P2[d] + r[d+ofdm.L].conj()*r[d+2*ofdm.L] - r[d].conj()*r[d+ofdm.L]
    return P2

P2 = calcP_method2(r, d_set, P0)

plt.plot(d_set, abs(P2), 'b--', lw=3, label='$P(d)$ (equation (6))')
plt.plot(d_set, abs(P1), 'r', label='$P(d)$ (equation (5))')
# plt.show()

P2_0 = calcP_method2(r, d_set, P0=0)
plt.plot(d_set, abs(P2_0), 'b--', lw=3, label='$P(d)$ (equation (6) with $P(0)=0$)')
plt.plot(d_set, abs(P1), 'r', label='$P(d)$ (equation (5))')
# plt.show()

# Method 3 to calculate P(d)
def calcP_method3(r):
    L = ofdm.L
    b_P = np.zeros(L)
    b_P[0] = 1; b_P[L-1] = -1
    a_P = (1, -1)
    
    # Implements r[d-L] * r[d], assuming r[d<0] = 0
    v_bar = np.hstack([np.zeros(L), r[L:].conj() * r[:-L]]) 
    P_bar = scipy.signal.lfilter(b_P, a_P, v_bar)
    return P_bar

P_bar = calcP_method3(r)
plt.plot(abs(P_bar), label='$\\bar{P}(d)$')
plt.plot(abs(P1), label='$P(d)$')

b = np.zeros(ofdm.L)
b[0] = 1
b[ofdm.L-1] = -1
a = (1,-1)
impulse = np.zeros(4*ofdm.L)
impulse[500] = 1

plt.plot(np.arange(len(impulse)), scipy.signal.lfilter(b, a, impulse))

def calcR_method1(r, d_set):
    # calculation based on the iterative method
    R = np.zeros(len(d_set))
    for i, d in enumerate(d_set):
        R[i] = sum(abs(r[d+m+ofdm.L])**2 for m in range(ofdm.L))
    return R

def calcR_method2(r, d_set, R0):
    # calculation based on non-causal recursive expression
    R = np.zeros(len(d_set))
    R[0] = R0
    for d in d_set[:-1]:
        R[d+1] = R[d] + abs(r[d+2*ofdm.L])**2-abs(r[d+ofdm.L])**2
    return R

def calcR_method3(r):
    # calculation based on IIR filter expression
    b = np.zeros(ofdm.L); b[0] = 1; b[-1] = -1
    a = (1,-1)
    return scipy.signal.lfilter(b, a, abs(r)**2)

R1 = calcR_method1(r, d_set)
R2 = calcR_method2(r, d_set, R1[0])
R_bar = calcR_method3(r)

plt.plot(abs(R1), 'b--', lw=3, label='R, method 1')
plt.plot(abs(R2), 'r', label='R, method 2')
plt.plot(abs(R_bar), 'g', label='R, method 3')
plt.annotate(s='', xy=(0,4*270), xytext=(2*ofdm.L,4*270), arrowprops=dict(arrowstyle='<-', shrinkA=0,shrinkB=0))
# plt.show()

M = abs(P1)**2/R1**2
M_bar = abs(P_bar)**2/R_bar**2

plt.plot(abs(r), label='$r[n]$', color='cyan')
plt.plot(M, label='$M(d)$')
plt.plot(M_bar, label='$\\bar{M}(d)$')
plt.annotate(s='', xy=(sto,1), xytext=(sto+2*ofdm.L,1), arrowprops=dict(arrowstyle='<-', shrinkA=0,shrinkB=0))
# plt.show()




