function [SNR cberNf1 pberNf1 cberNf2 pberNf2]=CSUWB_DIFF_FRAMES(crs,Np,Nf1,Nf2,CMN,num_channels)
%CSUWB_DIFF_FRAMES(cr,Np,Nf1,Nf2,ch,num_channel)
%CSUWB_DIFF_FRAMES: Compressed Sensing based Ultra-wideband system with
% sensing matrix operating for different number of
% frames per symbol for different compression ratios
%Inputs
%crs Compression Ratio (M/N), default [0.1 0.2]
%Np Number of pilot symbols, default 1
%Nf1 Number of frames in each symbol, default 15
%Nf2 Number of frames in each symbol, default 5, Nf1>Nf2
%CMN Type of channel according to the Reference 21,default 1
%num_channel Number of channel realizations, default 4000
%Outputs
% Bit Error rate performance of CCS-UWB and PCS-UWB for
% different comprestion ratios
%SNR The range of Signal to Noise ratio
%cberNf1 CCS-UWB Nf1 bit error rate for different compression ratios
%pberNf1 PCS-UWB Nf1 bit error rate for different compression ratios
%cberNf2 CCS-UWB Nf2 bit error rate for different compression ratios
%pberNf2 PCS-UWB Nf2 bit error rate for different compression ratios
%Description
%CSUWB_DIFF_FRAMES gives BER performance graph for conventional
%compressed sensing ultra-wideband (CCS-UWB) and proposed compressed
%sensing ultra-wideband (PCS-UWB) for different frames per symbols Nf1 and Nf2
%for different compression ratios (M/N)
if nargin<6
num_channels=4;
end
if nargin<5
CMN=1;
end
if nargin<4
Nf2=5;
end
if nargin<3
Nf1=15;
end
if nargin<2
Np=1
end
if nargin<1
crs=[0.1 0.2]
end
%num_channels=4;CMN=1; Nf1=15;Nf2=5;Np=1;crs=[0.01 0.05]
%################PARAMETERS #######################################
Fs=20e9; %Digital frequency
fp=2e9; %Pulse frequency
Tc=10e-9; %Chip Width
Tf=200; %Frame duration in Nanoseconds
Ni=10; %Number of information symbols
Ns=Np+Ni; %Total number of symbols
SNR=[-15 :1:5]; % SNR in dB
%############PULSE GENERATION ######################################
tc = gmonopuls('cutoff', fp); % Pulse width parameter
Tp = -2*tc : 1/Fs : 2*tc; % Actual pulse width
p = gmonopuls(Tp,fp); % UWB Pulse
%###########CHIP GENERATION #######################################
chiptime=0:1/Fs:Tp-1/Fs; % Chip samples
chip=zeros(length(chiptime),1);
chip(1:length(p))=(p'); % UWB Chip
%############ UWB CHANNEL ##########################################
uwb_h=uwb_channel(CMN,num_channels); % Reference 21
%############ SIMULATION ###########################################
for cr=1:length(crs) % Iteration for each compression ratio
for snrdb=1:length(SNR) % Iteration for each SNR
for z=1:num_channels % Number of channel realizations
%############ FRAME GENERATION #####################################
frame1=conv(uwb_h(:,num_channels),chip);
frame=frame1(1:1200);
N=length(frame);
%############## PILOT SIGNAL #########################################
sig=zeros(Nf1*N,1);
for j=1:Nf1
sig((j-1)*N+1:j*N,1)=frame;
end
for n=1:Np
s2(1:length(sig),n)=sig;
end
%############ TRANSMITTED SIGNAL ###################################
s=single(zeros(Ns*length(sig),1));
s(1:Np*length(sig))=s2(:);
Mod=(sign(randn(Ni,1))); % Modulation
s3=single(sig*Mod(1:Ni,1)');
s(Np*length(sig)+1:Ns*length(sig),1)=s3(:); % Modulated TX signal
clear s3;
%############RX SIGNAL GENERATION #################################
rt=SNR(snrdb); % SNR value
len=length(s); % Length of TX signal
awg=awgnnoise(rt,s,len,fp,Fs); %AWG Noise generation
r=s+awg; % RX signal with AWG Noise
clear s;
%############Reconstruction Methods%############
M=single(round(crs(cr)*N)); % Number of reduced dimention
cyNf1=zeros(M,1);
pyNf1=zeros(M*Nf1,1);
cyNf2=zeros(M,1);
pyNf2=zeros(M*Nf2,1);
cfrmrecNf1=zeros(N,1);
pfrmrecNf1=zeros(N,1);
cfrmrecNf2=zeros(N,1);
pfrmrecNf2=zeros(N,1);
pPhi=zeros(M*Nf1,N);
cPhi=randn(M,N);
for l=1:Nf1
if l==1;
pPhi((l-1)*M+1:l*M,1:N)=cPhi;
else
pPhi((l-1)*M+1:l*M,1:N)=randn(M,N);
end
end
%############# RECEIVER #############################################
for n=1:Np
for j=1:Nf1
rp(1:N,1)=r((n-1)*length(sig)+(j-1)*N+1:(n-1)*length(sig)+j*N,1);
%%%%%%%%%%%% Compressed Sampling for ALL Methods%%%%%%%%%%%%%%
if j<=Nf2
cyNf2=cyNf2+cPhi*rp; % Conventional method's Compresed Sampled output
pyNf2((j-1)*M+1:j*M,1)= pPhi((j-1)*M+1:j*M,1:N)*rp(1:N);
Phinf((j-1)*M+1:j*M,1:N)=pPhi((j-1)*M+1:j*M,1:N);
end
cyNf1=cyNf1+cPhi*rp; % Conventional method's Compresed Sampled output
pyNf1((j-1)*M+1:j*M,1)= pPhi((j-1)*M+1:j*M,1:N)*rp(1:N);
%########## RECONSTRUCTION ######################################
end
end
[cfrmrecNf2(1:N,1),iters, activationHist]=SolveOMP(cPhi,cyNf2/Nf2,N,[],0,0); % Reconstructed frame (CM)
[pfrmrecNf2(1:N,1),iters, activationHist]=SolveOMP(Phinf,pyNf2,N,300,[],0,0,1e-4);
[cfrmrecNf1(1:N,1),iters, activationHist]=SolveOMP(cPhi,cyNf1/Nf1,N,[],0,0); % Reconstructed frame (CM)
[pfrmrecNf1(1:N,1),iters, activationHist]=SolveOMP(pPhi,pyNf1,N,300,[],0,0,1e-4);
%###############Demodulation#################
for n=1:Ni
for j=1:Nf1
if j<=Nf2
cfrmNf2(1:N,j)=r((n+Np-1)*length(sig)+(j-1)*N+1:(n+Np-1)*length(sig)+j*N).*cfrmrecNf2;
cfrmsumNf2(1,j)=sum(cfrmNf2(:,j));
pfrmNf2(1:N,j)=r((n+Np-1)*length(sig)+(j-1)*N+1:(n+Np-1)*length(sig)+j*N).*pfrmrecNf2;
pfrmsumNf2(1,j)=sum(pfrmNf2(:,j));
cfrmNf1(1:N,j)=r((n+Np-1)*length(sig)+(j-1)*N+1:(n+Np-1)*length(sig)+j*N).*cfrmrecNf1;
cfrmsumNf1(1,j)=sum(cfrmNf1(:,j));
pfrmNf1(1:N,j)=r((n+Np-1)*length(sig)+(j-1)*N+1:(n+Np-1)*length(sig)+j*N).*pfrmrecNf1;
pfrmsumNf1(1,j)=sum(pfrmNf1(:,j));
end
end
csymbNf1(n)=sign(sum(cfrmsumNf1(1,:)/N));
psymbNf1(n)=sign(sum(pfrmsumNf1(1,:)/N));
csymbNf2(n)=sign(sum(cfrmsumNf2(1,:)/N));
psymbNf2(n)=sign(sum(pfrmsumNf2(1,:)/N));
end
clear r;
%###########ERROR CALCULATION ###################################
ccnterrNf1(z,snrdb)=length(find(Mod-csymbNf1'));
pcnterrNf1(z,snrdb)=length(find(Mod-psymbNf1'));
ccnterrNf2(z,snrdb)=length(find(Mod-csymbNf2'));
pcnterrNf2(z,snrdb)=length(find(Mod-psymbNf2'));
end
end
%########### BIT ERROR RATE ######################################
cberNf1(cr,1:snrdb)=sum(ccnterrNf1(:,:))/(z*(Ni));
pberNf1(cr,1:snrdb)=sum(pcnterrNf1(:,:))/(z*(Ni));
cberNf2(cr,1:snrdb)=sum(ccnterrNf2(:,:))/(z*(Ni));
pberNf2(cr,1:snrdb)=sum(pcnterrNf2(:,:))/(z*(Ni));
end
semilogy(SNR,cberNf2(1,1:snrdb),'k-+',SNR,cberNf1(1,1:snrdb),'k-s',SNR,cberNf2(2,1:snrdb),'k:+',SNR,cberNf1(2,1:snrdb),'k:s',SNR,pberNf2(1,1:snrdb),'k--+',SNR,pberNf1(1,1:snrdb),'k--s',SNR,pberNf2(2,1:snrdb),'k-.+',SNR,pberNf1(2,1:snrdb),'k-.s')
title('10%')
xlabel('SNR (dB)')
ylabel('Average BER')
axis([-15 5 .00005 1])