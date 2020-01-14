%% This script converts hrtfs to hrirs
clear; close all; clc;

%% setting up parameters
low_freq = 100;
up_freq = 22000;
freq_interp = 100;
low_phi = 0;
up_phi = 355;
phi_interp = 5;
low_theta = 5;
up_theta = 135;
theta_interp = 5;

numFreqs = floor((up_freq-low_freq)/freq_interp)+1;
numHorizontalSrcs = floor((up_phi-low_phi)/phi_interp)+1;
numVerticalSrcs = 2*(floor((up_theta-low_theta)/theta_interp)+1);
c = 343.21; %speed of sound

coeffs = zeros(1,numFreqs);

%% setting up locations
radius = 1;
locs = zeros(numHorizontalSrcs+numVerticalSrcs,3);
for i = 1 : numHorizontalSrcs
    theta = pi/2; % theta = pi/2 denotes the horizontal plane 
    phi = phi_interp*(i-1)*(pi/180);
    x = radius*sin(theta)*cos(phi);
    y = radius*sin(theta)*sin(phi);
    z = 0;
    locs(i,:) = [x,y,z];
end

for i = 1 : numVerticalSrcs
    if i <= numVerticalSrcs/2
        phi = 0;
        theta = low_theta+theta_interp*(i-1)*(pi/180);
    else
        phi = pi;
        theta = low_theta+theta_interp*(i-numVerticalSrcs/2-1)*(pi/180);
    end
    x = radius*sin(theta)*cos(phi);
    y = radius*sin(theta)*sin(phi);
    z = radius*cos(theta);
    locs(numHorizontalSrcs+i,:) = [x,y,z];
end

qs = 0.1;

%% read files
folder = '/media/ziqi/HardDisk/Lab/BEM_GPU_FLOAT_PRECISION/MATLAB/';
filename = 'left_hrtfs_mp_fabian';
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
fclose(fileID);

% for i = 1 : numHorizontalSrcs+numVerticalSrcs
%     for j = 1 : numFreqs
%         freq = (j-1)*freq_interp+low_freq;
%         wavNum = 2*pi*freq/c;
%         coeff = mpSrc(qs,wavNum,locs(i,:),[0,0,0]);
%         left_hrtfs(i,j) = left_hrtfs(i,j)/abs(coeff);
%     end
% end

filename = 'right_hrtfs_mp_fabian';
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

% for i = 1 : numHorizontalSrcs+numVerticalSrcs
%     for j = 1 : numFreqs
%         freq = (j-1)*freq_interp+low_freq;
%         wavNum = 2*pi*freq/c;
%         coeff = mpSrc(qs,wavNum,locs(i,:),[0,0,0]);
%         right_hrtfs(i,j) = right_hrtfs(i,j)/abs(coeff);
%     end
% end

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

left_hrirs = ifft(left_hrtfs_full_freq,[],2);
right_hrirs = ifft(right_hrtfs_full_freq,[],2);

left_hrirs_max = max(max(abs(left_hrirs)));
right_hrirs_max = max(max(abs(left_hrirs)));
cmax = max(left_hrirs_max,right_hrirs_max);

left_hrirs = left_hrirs/cmax;
right_hrirs = right_hrirs/cmax;

save('left_hrirs.mat','left_hrirs');
save('right_hrirs.mat','right_hrirs');

%% play the horizontal sound
%make input noise
% noise = makePinkNoise(1,44100);
% for i= 1 : numHorizontalSrcs
%     left_channel = conv(noise,left_hrirs(i,:));
%     right_channel = conv(noise, right_hrirs(i,:));
%     binaural = [left_channel',right_channel'];
%     sound(binaural,44100);
%     pause(2.0);
% end

%% plot hrirs
hfig = figure;
hax = axes(hfig);
fs = 44100;
duration = 1.0/fs*(0:size(left_hrirs,2)-1);
for i= 1 : numHorizontalSrcs
    left_hrir = left_hrirs(i,:);
    right_hrir = right_hrirs(i,:);
    plt_l = plot(hax,duration,left_hrir,'r');
    hold(hax,'on');
    plt_r = plot(hax,duration,right_hrir,'b');
    pause(3.0);
    delete(plt_l);
    delete(plt_r);
end

%% plot figures
% freqs = freq_interp*(0:(size(left_hrtfs,2)-1));
% 
% hFig = figure;
% hAx = axes(hFig);
% 
% plot(hAx,freqs,abs(left_hrtfs(1,:)));
% saveas(hFig,'../figure/front_monopole_source.png');
% hFig = figure;
% hAx = axes(hFig);
% semilogx(hAx,freqs,20*log(abs(left_hrtfs(1,:))));
% saveas(hFig,'../figure/front_monopole_source_semilog.png');
% 
