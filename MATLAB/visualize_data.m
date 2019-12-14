%% This script visualizes data generated from the data generator
clear; close all; clc;

%% specify the path
folder = '../data/';
grid_name = 'vox';
field_name_base = 'loudness';

grid_path = [folder,grid_name];
numLoudness = 8;
fig1 = figure;
ax1 = axes(fig1);
daspect(ax1,[1,1,1]);
set(ax1,'visible','off');
fig2 = figure;
ax2 = axes(fig2);
for i=1:numLoudness
    field_name = [field_name_base,'_',num2str(i-1)];
    field_path = [folder,field_name];
    [grid,field] = ProcessRawData(grid_path,field_path,100,-5);
    if i==1
        imagesc(ax2,grid);
        set(ax2,'visible','off');
        daspect(ax2,[1,1,1]);
    end
    caxis(ax1,[-10,10]);
    imagesc(ax1,field);
    set(ax1,'visible','off');
    daspect(ax1,[1,1,1]);
    pause(1.0);
    
end
