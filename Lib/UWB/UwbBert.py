import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import logging as log

from Spectrum import freqPlot as freqPlt
from RfModel import RfTransceiver as rfTrx
from ClockRecovery import ClockRecovery as cr

from UWB import UwbTransmitter as uwb_tx

## CSUWB_BELOW_SPARSITY(cr,Np,Nf,ch,num_channel)
## CSUWB_BELOW_SPARSITY: Compressed Sensing based Ultra-wideband system with
##  sensing matrix operating below the sparsity level of 35%
##  Inputs
## crs Compression Ratio (M/N), default [.05,.10,.20]
## Np Number of pilot symbols, default 1
## Nf Number of frames in each symbol, default 5
## CMN Type of channel according to the Reference 21,default 1
## num_channel Number of channel realizations, default 4000
## Outputs
##  Bit Error rate performance of CCS-UWB and PCS-UWB for
##  different comprestion ratios
## SNR The range of Signal to Noise ratio
## cber CCS-UWB bit error rate for different compression ratios
## pber PCS-UWB bit error rate for different compression ratios
## Description
## CSUWB_BELOW_SPARSITY gives BER performance graph for conventional
## compressed sensing ultra-wideband (CCS-UWB) and proposed compressed
## sensing ultra-wideband (PCS-UWB) for different compression ratios (M/N)

def UwbBert(crs=[0.05, 0.1, 0.2], Np=1, Nf=5, CMN=1, num_channels=4000):
	
	####################PARAMETERS ####################################
	Fs=20e9			# Digital frequency
	fp=2e9			# Pulse frequency
	Tc=10e-9		# Chip Width
	Tf=200			# Frame duration in Nanoseconds
	Ni=1000			# Number of information symbols
	Ns=Np+Ni		# Total number of symbols
	SNR=np.arange(-11, 9, 1)	# SNR in dB
	
	# PULSE GENERATION
	t, p = uwb_tx.gmonopuls(fp, Fs)
	
	#CHIP GENERATION
	#chiptime = np.arange(0, (len(t)-1)/Fs, 1/Fs];				# Chip samples
	#chip=np.zeros(length(chiptime),1);
	#chip(1:length(p))=(p'); # UWB Chip
	#chip = np.concatenate((p, np.zeros(5)), axis=None)
	chip = p
	
	########## UWB CHANNEL ##########################################
	#uwb_h=uwb_channel(CMN,num_channels)		# Reference 21
	
	############# SIMULATION ###########################################
	#for cr in range(1, len(crs)):			# Iteration for each compression ratio
	#	for snrdb in range(1, len(SNR)):	# Iteration for each SNR
	#		for z in range(1, num_channels):		# Number of channel realizations
	
	cr = 0
	snrdb = -15
	z = 0
	
	############ FRAME GENERATION #####################################
	#frame1=conv(uwb_h(:,num_channels),chip)
	#frame=frame1(1:1200)
	#N=length(frame)
	frame = chip

	sig = np.hstack([frame]*Nf)

	######### TRANSMITTED SIGNAL ###################################
	Mod = (np.random.rand(sig.size) >= 0.5).astype('int')
	s3 = np.multiply(sig, Mod)
	s = np.concatenate((np.hstack([sig]*Np), np.hstack([s3]*Ni)), axis=None)

	############ RX SIGNAL GENERATION #################################
	signal_power = np.mean(abs(s**2))
	sigma2 = signal_power * 10**(snrdb/10)  # calculate noise power based on signal power and SNR
	print ("RX Signal power: %.4f, Noise power: %.4f" % (signal_power, sigma2))
	noiseSignal = np.sqrt(sigma2/2) * (np.random.randn(s.size)-0.5)
	r = s + noiseSignal

	#plt.plot(s)
	#plt.plot(r)
	#plt.show()
	#freqPlt.fftPlot(r, fs=Fs)

	############Reconstruction Methods%############
	N = frame.size
	M = int(crs[cr]*N)
	cy=np.zeros(M)
	py=np.zeros(M*Nf)
	cfrmrec=np.zeros(N)
	pfrmrec=np.zeros(N)
	pPhi=np.zeros((M*Nf,N))
	cPhi=np.random.randn(M,N)
	for i in range(Nf):
		if i==0:
			pPhi[i*M:(i+1)*M,0:N]=cPhi
		else:
			pPhi[i*M:(i+1)*M,0:N] = np.random.randn(M,N)
				
	############# RECEIVER #############################################
	rp = np.zeros((N, 1))
	for i in range(Np):
		for j in range(Nf):
			rp[0:N] = r[i*sig.size+j*N:i*sig.size+(j+1)*N]
			cy = cy + cPhi*rp # CCS-UWB
			py[j*M:(j+1)*M] = pPhi[j*M:(j+1)*M, :N] * rp # PCS-UWB
				
	########### RECONSTRUCTION ######################################
	[cfrmrec, iters, activationHist] = SolveOMP(cPhi,cy/Nf,N,[],0,0)
	[pfrmrec, iters, activationHist] = SolveOMP(pPhi,py,N,300,[],0,0,1e-4)
				
	###############Demodulation#################
	for n in range(Ni):
		for j in range(Nf):
			cfrm[:N,j] = r[(n+Np)*sig.size+j*N:(n+Np)*sig.size+(j+1)*N] * cfrmrec
			cfrmsum[1,j] = sum(cfrm[:,j])
			pfrm[1:N,j]=r[(n+Np)*sig.size+j*N : (n+Np)*sig.size+(j+1)*N] * pfrmrec
			pfrmsum[1,j]=sum(pfrm[:,j])
		csymb[n] = sign(sum(cfrmsum[1,:]/N))
		psymb[n] = sign(sum(pfrmsum[1,:]/N))
				
	############ ERROR CALCULATION ###################################
	ccnterr[z,snrdb] = np.find(Mod-csymb).size
	pcnterr[z,snrdb] = np.find(Mod-psymb).size
		
############# BIT ERROR RATE ######################################
	cber[cr,1:snrdb] = np.sum[ccnterr[:,:]]/(z*(Ni));		# BER for CCS-UWB
	pber[cr,1:snrdb] = np.sum[pcnterr[:,:]]/(z*(Ni));		# BER for PCS-UWB
	
	############ PERFORMANCE PLOTS ####################################
	#figure,
	#semilogy(SNR,cber(1,1:snrdb),'k-*',SNR,pber(1,1:snrdb),'k-s',SNR,cber(2,1:snrdb),'k:+',SNR,pber(2,1:snrdb),'k:s',SNR,cber(3,1:snrdb),'k--+',SNR,pber(3,1:snrdb),'k--s')
	#title('BER Performance with Compression ratio (M/N) less than 35%')
	#xlabel('SNR (dB)')
	#ylabel('Average BER')
	#legend( 'CCS-UWB 5%', 'PCS-UWB 5%','CCS-UWB 10%', 'PCS-UWB 10%','CCS-UWB 20%', 'PCS-UWB 20%')
	#axis([-11 9 1e-5 1])
	
	return SNR, cber, pber
