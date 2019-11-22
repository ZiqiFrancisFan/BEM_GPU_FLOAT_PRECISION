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

%% read files
folder = '../';
filename = 'left_hrtfs';
format = '(%f,%f) ';
path = [folder,filename];
fileID = fopen(path,'r');
temp = fscanf(fileID,format);
left_hrtfs = zeros(numHorizontalSrcs+numVerticalSrcs,numFreqs);
right_hrtfs = zeros(numHorizontalSrcs+numVerticalSrcs,numFreqs);
for i = 1 : numHorizontalSrcs+numVerticalSrcs
    for j = 1 : numFreqs
        idx = (i-1)*numFreqs+j;
        x = temp(2*(idx-1)+1);
        y = temp(2*(idx-1)+2);
        left_hrtfs(i,j) = complex(x,y);
    end
end
fclose(fileID);

filename = 'right_hrtfs';
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
fclose(fileID);