%% This script converts hrtfs to hrirs
clear; close all; clc;

%% setting up parameters
low_freq = 25;
up_freq = 12000;
freq_interp = 25;
low_phi = 0;
up_phi = 355;
phi_interp = 5;
low_theta = 0;
up_theta = 135;
theta_interp = 5;

numFreqs = floor((up_freq-low_freq)/freq_interp)+1;
numHorizontalSrcs = floor((up_phi-low_phi)/phi_interp)+1;
numVerticalSrcs = floor((up_theta-low_theta)/theta_interp)+1;
c = 343.21;

coeffs = zeros(1,numFreqs);
for i = 1 : numFreqs
    wavNum = 2*pi*freq_interp*i/c;
    coeffs(i) = ptSrc(1.0,wavNum,[1,0,0],[0,0,0]);
end

%% read files
folder = '/media/ziqi/HardDisk/Lab/BEM_GPU_FLOAT_PRECISION/MATLAB/';
filename = 'left_hrtfs_pt';
format = '(%f,%f) ';
path = [folder,filename];
fileID = fopen(path,'r');
temp = fscanf(fileID,format);
left_hrtfs = zeros(numHorizontalSrcs+numVerticalSrcs,numFreqs);
for i = 1 : numHorizontalSrcs+numVerticalSrcs
    for j = 1 : numFreqs
        idx = (i-1)*numFreqs+j;
        x = temp(2*(idx-1)+1);
        y = temp(2*(idx-1)+2);
        left_hrtfs(i,j) = complex(x,y);
    end
end
left_hrtfs = left_hrtfs./coeffs;
fclose(fileID);

filename = 'right_hrtfs_pt';
format = '(%f,%f) ';
path = [folder,filename];
fileID = fopen(path,'r');
temp = fscanf(fileID,format);
right_hrtfs = zeros(numHorizontalSrcs+numVerticalSrcs,numFreqs);
for i = 1 : numHorizontalSrcs+numVerticalSrcs
    for j = 1 : numFreqs
        idx = (i-1)*numFreqs+j;
        x = temp(2*(idx-1)+1);
        y = temp(2*(idx-1)+2);
        right_hrtfs(i,j) = complex(x,y);
    end
end
right_hrtfs = right_hrtfs./coeffs;
fclose(fileID);

%% convert hrtfs to hrirs
% adding an element for the 0Hz
temp = left_hrtfs;
left_hrtfs = ones(size(left_hrtfs,1),size(left_hrtfs,2)+1);
left_hrtfs(:,2:end) = temp;

temp = right_hrtfs;
right_hrtfs = ones(size(right_hrtfs,1),size(right_hrtfs,2)+1);
right_hrtfs(:,2:end) = temp;

% adding another half of the frquency content
temp = conj(flip(left_hrtfs,2));
left_hrtfs_full_freq = [left_hrtfs,temp(:,1:end-1)];
temp = conj(flip(right_hrtfs,2));
right_hrtfs_full_freq  = [right_hrtfs,temp(:,1:end-1)];

left_hrirs = real(ifft(left_hrtfs_full_freq,[],2));
right_hrirs = real(ifft(right_hrtfs_full_freq,[],2));