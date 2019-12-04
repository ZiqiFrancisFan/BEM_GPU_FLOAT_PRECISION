%% this code reads occupancy grid from file
clear; close all; clc;

folder = '../data/';
filename = 'vox';

format = '%d ';
path = [folder,filename];
fileID = fopen(path,'r');
temp = fscanf(fileID,format);
fclose(fileID);

dim = round(length(temp)^(1/3));
grid = imbinarize(reshape(temp,[dim,dim,dim]));

hFig = figure('units','normalized','outerposition',[0 0 1 1]);
hAx = axes(hFig);
daspect(hAx,[1,1,1]);
daspect(hAx,[1,1,1]);
for i=1:dim
    imshow(grid(:,:,i));
    
    pause(0.2);
end
