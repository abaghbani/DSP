function [SNR cber pber]= CSUWB_BELOW_SPARSITY(crs,Np,Nf,CMN,num_channels)
%CSUWB_BELOW_SPARSITY(cr,Np,Nf,ch,num_channel)
%CSUWB_BELOW_SPARSITY: Compressed Sensing based Ultra-wideband system with
% sensing matrix operating below the sparsity level of 35%
% Inputs
%crs Compression Ratio (M/N), default [.05,.10,.20]
%Np Number of pilot symbols, default 1
%Nf Number of frames in each symbol, default 5
%CMN Type of channel according to the Reference 21,default 1
%num_channel Number of channel realizations, default 4000
%Outputs
% Bit Error rate performance of CCS-UWB and PCS-UWB for
% different comprestion ratios
%SNR The range of Signal to Noise ratio
%cber CCS-UWB bit error rate for different compression ratios
%pber PCS-UWB bit error rate for different compression ratios
%Description
%CSUWB_BELOW_SPARSITY gives BER performance graph for conventional
%compressed sensing ultra-wideband (CCS-UWB) and proposed compressed
%sensing ultra-wideband (PCS-UWB) for different compression ratios (M/N)
if nargin<5
num_channels=4000;
end
if nargin<4
CMN=1;
end
if nargin<3
Nf=5;
end
if nargin<2
Np=1
end
if nargin<1
crs=[0.05 0.1 0.2]
end
%###################PARAMETERS ####################################
Fs=20e9; %Digital frequency
fp=2e9; %Pulse frequency
Tc=10e-9; %Chip Width
Tf=200; %Frame duration in Nanoseconds
Ni=1000; %Number of information symbols
Ns=Np+Ni; %Total number of symbols
SNR=[-11:1:9]; % SNR in dB
%############PULSE GENERATION ######################################
tc = gmonopuls('cutoff', fp); % Pulse width parameter
Tp = -2*tc : 1/Fs : 2*tc; % Actual pulse width
p = gmonopuls(Tp,fp); % UWB Pulse
%############CHIP GENERATION #######################################
chiptime=0:1/Fs:Tp-1/Fs; % Chip samples
chip=zeros(length(chiptime),1);
chip(1:length(p))=(p'); % UWB Chip
%########## UWB CHANNEL ##########################################
uwb_h=uwb_channel(CMN,num_channels); % Reference 21
%############# SIMULATION ###########################################
for cr=1:length(crs) % Iteration for each compression ratio
for snrdb=1:length(SNR) % Iteration for each SNR
for z=1:num_channels % Number of channel realizations
%############ FRAME GENERATION #####################################
frame1=conv(uwb_h(:,num_channels),chip);
frame=frame1(1:1200);
N=length(frame);
%############## PILOT SIGNAL #########################################
sig=zeros(Nf*N,1);
for j=1:Nf
sig((j-1)*N+1:j*N,1)=frame;
end
for n=1:Np
s2(1:length(sig),n)=sig;
end
%######### TRANSMITTED SIGNAL ###################################
s=single(zeros(Ns*length(sig),1));
s(1:Np*length(sig))=s2(:);
Mod=(sign(randn(Ni,1))); % Modulation
s3=single(sig*Mod(1:Ni,1)');
s(Np*length(sig)+1:Ns*length(sig),1)=s3(:); % Modulated TX signal
clear s3;
%############ RX SIGNAL GENERATION #################################
rt=SNR(snrdb); % SNR value
len=length(s); % Length of TX signal
awg=awgnnoise(rt,s,len,fp,Fs); %AWG Noise generation
r=s+awg; % RX signal with AWG Noise
clear s;
%############Reconstruction Methods%############
M=single(round(crs(cr)*N)); % Number of reduced dimention
cy=zeros(M,1);
py=zeros(M*Nf,1);
cfrmrec=zeros(N,1);
pfrmrec=zeros(N,1);
pPhi=zeros(M*Nf,N);
cPhi=randn(M,N);
for l=1:Nf
if l==1;
pPhi((l-1)*M+1:l*M,1:N)=cPhi;
else
pPhi((l-1)*M+1:l*M,1:N)=randn(M,N);
end
end
%############# RECEIVER #############################################
for n=1:Np
for j=1:Nf
rp(1:N,1)=r((n-1)*length(sig)+(j-1)*N+1:(n-1)*length(sig)+j*N,1);
cy=cy+cPhi*rp; % CCS-UWB
py((j-1)*M+1:j*M,1)= pPhi((j-1)*M+1:j*M,1:N)*rp(1:N); % PCS-UWB
%########### RECONSTRUCTION ######################################
end
end
[cfrmrec(1:N,1),iters, activationHist]=SolveOMP(cPhi,cy/Nf,N,[],0,0);
[pfrmrec(1:N,1),iters, activationHist]=SolveOMP(pPhi,py,N,300,[],0,0,1e-4);
%###############Demodulation#################
for n=1:Ni
for j=1:Nf
cfrm(1:N,j)=r((n+Np-1)*length(sig)+(j-1)*N+1:(n+Np-1)*length(sig)+j*N).*cfrmrec;
cfrmsum(1,j)=sum(cfrm(:,j));
pfrm(1:N,j)=r((n+Np-1)*length(sig)+(j-1)*N+1:(n+Np-1)*length(sig)+j*N).*pfrmrec;
pfrmsum(1,j)=sum(pfrm(:,j));
end
csymb(n)=sign(sum(cfrmsum(1,:)/N));
psymb(n)=sign(sum(pfrmsum(1,:)/N));
end
clear r;
clear ynew;
clear frmrec;
clear frmrec2;
%############ ERROR CALCULATION ###################################
ccnterr(z,snrdb)=length(find(Mod-csymb'));
pcnterr(z,snrdb)=length(find(Mod-psymb'));
end
end
%############# BIT ERROR RATE ######################################
cber(cr,1:snrdb)=sum(ccnterr(:,:))/(z*(Ni)); %BER for CCS-UWB
pber(cr,1:snrdb)=sum(pcnterr(:,:))/(z*(Ni)); %BER for PCS-UWB
end
%########### PERFORMANCE PLOTS ####################################
figure,
semilogy(SNR,cber(1,1:snrdb),'k-*',SNR,pber(1,1:snrdb),'k-s',SNR,cber(2,1:snrdb),'k:+',SNR,pber(2,1:snrdb),'k:s',SNR,cber(3,1:snrdb),'k--+',SNR,pber(3,1:snrdb),'k--s')
title('BER Performance with Compression ratio (M/N) less than 35%')
xlabel('SNR (dB)')
ylabel('Average BER')
legend( 'CCS-UWB 5%', 'PCS-UWB 5%','CCS-UWB 10%', 'PCS-UWB 10%','CCS-UWB 20%', 'PCS-UWB 20%')
axis([-11 9 1e-5 1])

