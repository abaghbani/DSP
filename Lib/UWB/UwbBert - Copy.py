import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import logging as log

from Spectrum import freqPlot as fp
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
	chip = np.concatenate((p, np.zeros(5)), axis=None)
	
	plt.plot(p)
	plt.show()
	plt.plot(chip)
	plt.show()

	########## UWB CHANNEL ##########################################
	#uwb_h=uwb_channel(CMN,num_channels)		# Reference 21
	
	############# SIMULATION ###########################################
	for cr in range(1, len(crs)):			# Iteration for each compression ratio
		for snrdb in range(1, len(SNR)):	# Iteration for each SNR
			for z in range(1, num_channels):		# Number of channel realizations
	
				############ FRAME GENERATION #####################################
				#frame1=conv(uwb_h(:,num_channels),chip)
				#frame=frame1(1:1200)
				#N=length(frame)
				frame = chip
				N = frame.size
				
				############## PILOT SIGNAL #########################################
				sig=np.zeros(Nf*N)
				for j in range(1, Nf):
					sig[(j-1)*N:j*N] = frame
				
				s2 = np.zeros(Np*sig.size)
				for n in range(1, Np):
					s2[(n-1)*sig.size:n*sig.size] = sig
				
				plt.plot(s2)
				plt.show()

				######### TRANSMITTED SIGNAL ###################################
				s = np.zeros(Ni*sig.size)
				s[:Np*sig.size] = s2
				Mod = np.sign(np.random.rand(sig.size))			# Modulation
				s3 = np.multiply(sig, Mod)
				s[Np*sig.size:(Np+1)*sig.size]=s3		# Modulated TX signal
				

				plt.plot(s)
				plt.show()

				############ RX SIGNAL GENERATION #################################
	#			rt=SNR(snrdb);						# SNR value
	#			len=length(s);						# Length of TX signal
	#			awg=awgnnoise(rt,s,len,fp,Fs);		# AWG Noise generation
	#			r=s+awg;							# RX signal with AWG Noise
				
	#			signal_power = np.mean(abs(IfSignal**2))
	#			sigma2 = signal_power * 10**(-SNRdb/10)  # calculate noise power based on signal power and SNR
	#			print ("RX Signal power: %.4f, Noise power: %.4f" % (signal_power, sigma2))
	#			noiseSignal = np.sqrt(sigma2/2) * (np.random.randn(s.size)-0.5)
				
	#			############Reconstruction Methods%############
	#			M=single(round(crs(cr)*N));			# Number of reduced dimention
	#			cy=zeros(M,1);
	#			py=zeros(M*Nf,1);
	#			cfrmrec=zeros(N,1);
	#			pfrmrec=zeros(N,1);
	#			pPhi=zeros(M*Nf,N);
	#			cPhi=randn(M,N);
	#			for l in range(1, Nf):
	#				if l==1:
	#					pPhi((l-1)*M+1:l*M,1:N)=cPhi
	#				else:
	#					pPhi((l-1)*M+1:l*M,1:N)=randn(M,N)
				
	#			############# RECEIVER #############################################
	#			for n in range(1, Np):
	#				for j in range(1, Nf):
	#					rp(1:N,1)=r((n-1)*length(sig)+(j-1)*N+1:(n-1)*length(sig)+j*N,1)
	#					cy=cy+cPhi*rp # CCS-UWB
	#					py((j-1)*M+1:j*M,1)= pPhi((j-1)*M+1:j*M,1:N)*rp(1:N) # PCS-UWB
				
	#			########### RECONSTRUCTION ######################################
	#			[cfrmrec(1:N,1),iters, activationHist]=SolveOMP(cPhi,cy/Nf,N,[],0,0)
	#			[pfrmrec(1:N,1),iters, activationHist]=SolveOMP(pPhi,py,N,300,[],0,0,1e-4)
				
	#			###############Demodulation#################
	#			for n in range(1, Ni):
	#				for j in range(1, Nf):
	#					cfrm(1:N,j)=r((n+Np-1)*length(sig)+(j-1)*N+1:(n+Np-1)*length(sig)+j*N).*cfrmrec
	#					cfrmsum(1,j)=sum(cfrm(:,j))
	#					pfrm(1:N,j)=r((n+Np-1)*length(sig)+(j-1)*N+1:(n+Np-1)*length(sig)+j*N).*pfrmrec
	#					pfrmsum(1,j)=sum(pfrm(:,j))
	#				csymb(n)=sign(sum(cfrmsum(1,:)/N))
	#				psymb(n)=sign(sum(pfrmsum(1,:)/N))
				
	#			############ ERROR CALCULATION ###################################
	#			ccnterr(z,snrdb)=length(find(Mod-csymb'))
	#			pcnterr(z,snrdb)=length(find(Mod-psymb'))
		
	#	############# BIT ERROR RATE ######################################
	#	cber(cr,1:snrdb)=sum(ccnterr(:,:))/(z*(Ni));		# BER for CCS-UWB
	#	pber(cr,1:snrdb)=sum(pcnterr(:,:))/(z*(Ni));		# BER for PCS-UWB
	
	############ PERFORMANCE PLOTS ####################################
	#figure,
	#semilogy(SNR,cber(1,1:snrdb),'k-*',SNR,pber(1,1:snrdb),'k-s',SNR,cber(2,1:snrdb),'k:+',SNR,pber(2,1:snrdb),'k:s',SNR,cber(3,1:snrdb),'k--+',SNR,pber(3,1:snrdb),'k--s')
	#title('BER Performance with Compression ratio (M/N) less than 35%')
	#xlabel('SNR (dB)')
	#ylabel('Average BER')
	#legend( 'CCS-UWB 5%', 'PCS-UWB 5%','CCS-UWB 10%', 'PCS-UWB 10%','CCS-UWB 20%', 'PCS-UWB 20%')
	#axis([-11 9 1e-5 1])
	
	return SNR, cber, pber
