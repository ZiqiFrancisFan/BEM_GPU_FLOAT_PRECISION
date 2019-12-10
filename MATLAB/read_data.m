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

dim = round(length(temp)^(1/3));
grid = reshape(temp,[dim,dim,dim]);
for i = 1 : dim
    grid(:,:,i) = ConvertBoundary2SolidObject(grid(:,:,i));
end

folder = '../data/';
filename = 'field';

format = '(%f,%f) ';
path = [folder,filename];
fileID = fopen(path,'r');
temp = fscanf(fileID,format);
for i=1:length(temp)/2
    field(i) = complex(temp(2*i-1),temp(2*i));
end
fclose(fileID);


field = reshape(field,[dim,dim,dim]);
field(find(grid==1)) = 0;




hFig1 = figure('units','normalized','outerposition',[0 0 1 1]);
hAx1 = axes(hFig1);
daspect(hAx1,[1,1,1]);
hFig2 = figure('units','normalized','outerposition',[0 0 1 1]);
hAx2 = axes(hFig2);
daspect(hAx2,[1,1,1]);
for i=1:dim
    imagesc(hAx1,abs(field(:,:,i)));
    daspect(hAx1,[1,1,1]);
    imagesc(hAx2,grid(:,:,i));
    daspect(hAx2,[1,1,1]);
    pause(0.5);
end
